"""Train E2M on TCGA and predict each external cohort, for each integration method."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

import e2m
from e2m.config import load_config
from e2m.evaluation import evaluate_predictions

from cohorts import (
    load_cptac_cmi_data,
    load_hugo_data,
    load_immunopog_data,
    load_liu_data,
    load_metabric_data,
    load_morrison_data,
    load_riaz_data,
    load_van_allen_data,
    mutation_labels,
)
from integration import integrate, umap_coordinates

RUN_ROOT = Path(__file__).resolve().parents[1]

# cohort -> (loader, external batch column in clinical or None, per-site column in clinical or None)
LOADERS = {
    "CPTAC_CMI": (load_cptac_cmi_data, "project.project_id", "cases.primary_site"),
    "IMMUNOPOG": (load_immunopog_data, None, "Cancer"),
    "METABRIC": (load_metabric_data, None, "CANCER_TYPE_DETAILED"),
    "HUGO": (load_hugo_data, None, None),
    "LIU": (load_liu_data, None, None),
    "RIAZ": (load_riaz_data, None, None),
    "VAN_ALLEN": (load_van_allen_data, None, None),
    "MORRISON": (load_morrison_data, None, None),
}


def top_targets(path: Path, top_n: int) -> list[str]:
    """The top_n targets of a TCGA cross-validation metrics file by normalized AUPRC."""
    metrics = pd.read_csv(path, index_col=0)
    return metrics["normalized_auprc"].dropna().sort_values(ascending=False).index[:top_n].tolist()


def evaluate(labels, probabilities, targets, top) -> pd.DataFrame:
    metrics = evaluate_predictions(labels[targets], (probabilities[targets] >= 0.5).astype(int), probabilities[targets], targets)
    if top is not None:
        metrics.insert(0, "top_tcga_target", metrics.index.isin(top))
    return metrics


def site_metrics(labels, probabilities, targets, sites: pd.Series, min_n: int) -> pd.DataFrame:
    rows = []
    for site, samples in sites.dropna().groupby(sites.dropna()).groups.items():
        if len(samples) >= min_n:
            metrics = evaluate(labels.loc[samples], probabilities.loc[samples], targets, None)
            rows.append(metrics.reset_index().assign(site=site, site_n_samples=len(samples)))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize(metrics: pd.DataFrame) -> dict:
    evaluable = metrics[metrics["evaluable"]]
    return {
        "n_evaluable": int(len(evaluable)),
        "median_auprc": float(evaluable["auprc"].median()),
        "median_normalized_auprc": float(evaluable["normalized_auprc"].median()),
        "median_roc_auc": float(evaluable["roc_auc"].median()),
    }


def run_cohort(cohort, settings, config, base, args, methods):
    run = config["run"]
    loader, batch_col, site_col = LOADERS[cohort]
    expression, mutations, clinical, loader_meta, profiled = loader(RUN_ROOT / run["external_root"] / settings["data_dir"])
    batches = clinical[batch_col].astype(str) if batch_col else None

    cancers = base["run"]["cohorts"] if settings["tcga_cancers"] == "all" else settings["tcga_cancers"]
    overrides = {**config["data"], "expression_dataset": settings["tcga_expression"], "expression_transform": "raw", "normalization": "none"}
    tcga = e2m.Dataset.from_tcga(cancers, data_dir=RUN_ROOT / run["tcga_data_dir"], config=RUN_ROOT / run["tcga_config"],
                                 data_overrides=overrides, with_tmb=False)
    targets = tcga.targets.tolist()
    labels = mutation_labels(mutations, profiled, expression.index, targets)
    if "measured_genes" in loader_meta:  # targeted panel: genes outside it were not sequenced
        labels.loc[:, ~labels.columns.isin(loader_meta["measured_genes"])] = float("nan")
    ranking = settings.get("target_ranking")
    top = top_targets(RUN_ROOT / ranking["file"], ranking["top_n"]) if ranking else None

    scales = {"tcga_scale": "counts" if settings["tcga_expression"] == "star_counts" else "tpm",
              "external_scale": settings["scale"], "cpm": base["data"].get("normalization") == "cpm"}
    before = None
    rows = []
    for method in methods:
        started = time.time()
        output = RUN_ROOT / "output" / cohort / method
        output.mkdir(parents=True, exist_ok=True)
        train, test = integrate(tcga.expression, expression, method, external_batches=batches, **scales)

        model = e2m.E2MModel(base).fit(train, tcga.mutations.loc[train.index])
        model.save(output / "model")
        probabilities = model.predict(test)
        metrics = evaluate(labels.loc[test.index], probabilities, targets, top)
        metrics.to_csv(output / "metrics.csv")
        if site_col:
            site_metrics(labels.loc[test.index], probabilities, targets, clinical[site_col].reindex(test.index),
                         run["min_site_samples"]).to_csv(output / "site_metrics.csv", index=False)
        probabilities.to_csv(output / "probabilities.csv.gz")
        labels.loc[test.index].to_csv(output / "labels.csv.gz")
        clinical.reindex(test.index).to_csv(output / "clinical.csv.gz")
        model.embed(train).to_csv(output / "embeddings_tcga.csv.gz")
        model.embed(test).to_csv(output / "embeddings_external.csv.gz")

        if run["umap"] and not args.skip_umap:
            if before is None:
                before = umap_coordinates(*integrate(tcga.expression, expression, "none", **scales), batches, label=cohort)
            before.to_csv(output / "umap_before.csv")
            if method != "none":
                umap_coordinates(train, test, batches, label=cohort).to_csv(output / "umap_after.csv")

        summary = {**summarize(metrics), **({"top_targets": summarize(metrics[metrics["top_tcga_target"]])} if top else {})}
        manifest = {
            "cohort": cohort,
            "method": method,
            "main_method": method == methods[0] and not args.methods,
            "tcga": {"cancers": cancers, "expression": settings["tcga_expression"], "samples": len(train),
                     "preprocessing": tcga.manifest},
            "external": {"samples": len(test), "profiled_samples": int(labels.loc[test.index].notna().any(axis=1).sum()),
                         "scale": settings["scale"], "loader": loader_meta},
            "features": train.shape[1],
            "targets": {"n": len(targets), "rule": config["data"], "top_tcga_targets": top},
            "model": model.metadata,
            "metrics": summary,
            "seconds": round(time.time() - started, 1),
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        rows.append({"cohort": cohort, "method": method, "main": manifest["main_method"], "samples": len(test),
                     "features": train.shape[1], "targets": len(targets), **summarize(metrics)})
        print(f"{cohort} / {method}: {summarize(metrics)}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=RUN_ROOT / "config" / "e2m.yaml")
    parser.add_argument("--cohorts", nargs="+", help="Default: all cohorts in the config.")
    parser.add_argument("--methods", nargs="+", help="Default: each cohort's methods in the config.")
    parser.add_argument("--skip-umap", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = load_config(RUN_ROOT / config["run"]["tcga_config"], {"data": config["data"]})
    e2m.set_verbose()
    rows = []
    for cohort in args.cohorts or list(config["cohorts"]):
        settings = config["cohorts"][cohort]
        rows += run_cohort(cohort, settings, config, base, args, args.methods or settings["methods"])
    summary_path = RUN_ROOT / "output" / "runs_summary.csv"
    if summary_path.exists():
        previous = pd.read_csv(summary_path)
        done = {(r["cohort"], r["method"]) for r in rows}
        rows = [r for r in previous.to_dict("records") if (r["cohort"], r["method"]) not in done] + rows
    pd.DataFrame(rows).to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
