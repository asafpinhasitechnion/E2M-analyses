"""Train XGBoost on TCGA and test it on each external cohort of config/config.yaml, per driver gene and per alteration.

Usage (from Runs/ClinicalDrivers, after src/prepare_cohorts.py):
    python src/run_drivers.py
    python src/run_drivers.py --only metabric difg_glass
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import e2m
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from e2m.config import load_config
from e2m.data import sample_id
from inmoose.pycombat import pycombat_norm
from sklearn.metrics import average_precision_score, roc_auc_score

from alterations import ALTERATIONS, alteration_labels
from driver_genes import CANCER_DRIVER_GENES, CODING_VARIANT_CLASSES
from prepare_cohorts import fetch

RUN_ROOT = Path(__file__).resolve().parents[1]


def load_external(config: dict, cohort: dict):
    """Expression, gene labels and alteration labels of a cohort (data/standardized, from src/prepare_cohorts.py)."""
    folder = RUN_ROOT / config["data"]["standardized_dir"] / cohort["source"]
    read = lambda name: pd.read_csv(folder / name, index_col=0).rename(index=str)
    expression = read("expression.csv.gz")
    # Samples of TCGA patients (difg_glass includes TCGA-* gliomas) are in the training data: left out
    expression = expression[~expression.index.str.startswith("TCGA-")]
    if "filter" in cohort:  # POG570: one cancer type of the pan-cancer study (minus the excluded tumour types)
        clinical = read("clinical_sample.csv.gz").reindex(expression.index)
        keep = clinical[cohort["filter"]["column"]].astype(str).isin(cohort["filter"]["values"])
        if "exclude" in cohort["filter"]:
            keep &= ~clinical[cohort["filter"]["exclude"]["column"]].astype(str).isin(cohort["filter"]["exclude"]["values"])
        expression = expression[keep]
    genes = read("mutations_gene_level.csv.gz").reindex(expression.index)

    # Alteration labels from the coding events, missing where the gene was not sequenced; GEO series have no events
    events_path = folder / "mutations_long.csv.gz"
    if not events_path.exists():
        return expression, genes, pd.DataFrame(index=expression.index)
    events = pd.read_csv(events_path, dtype={"Tumor_Sample_Barcode": str}, low_memory=False)
    alterations = alteration_labels(events, expression.index).astype(float)
    alterations = alterations.where(genes[[gene for gene, _ in ALTERATIONS.values()]].notna().to_numpy())
    return expression, genes, alterations


def load_tcga(config: dict, cohort: dict):
    """TCGA from E2M: linear TPM or FPKM, or CPM for counts, of protein-coding genes; every mutated gene as a label; the cancer
    type as ComBat batch. Alteration labels from E2M's per-cancer MC3 event files, coding events only."""
    data_dir = RUN_ROOT / config["tcga"]["data_dir"]
    overrides = {
        "expression_dataset": f"star_{cohort['tcga_expression_measure']}",
        "expression_transform": "raw",       # linear scale; log1p is applied later per the cohort's config
        "normalization": "cpm" if cohort["tcga_expression_measure"] == "counts" else "none",
        "min_mutation_prevalence": 0,
        "min_mutation_positives": 1,
        "max_mutation_targets": 0,
        "targets": None,
    }
    tcga = e2m.Dataset.from_tcga(cohort["tcga_cancers"], data_dir=data_dir, config=RUN_ROOT / config["tcga"]["config"],
                                 data_overrides=overrides, with_tmb=False)

    xena = load_config(RUN_ROOT / config["tcga"]["config"])["xena"]
    events = pd.concat(
        pd.read_csv(fetch(xena["mutation_event_url_template"].format(cancer=cancer),
                          data_dir / "mutation_events" / xena["mutation_event_file"].format(cancer=cancer)), sep="\t", low_memory=False)
        for cancer in cohort["tcga_cancers"]
    )
    events = events[events["effect"].isin(CODING_VARIANT_CLASSES)].rename(columns={
        "sample": "Tumor_Sample_Barcode", "gene": "Hugo_Symbol", "effect": "Variant_Classification", "start": "Start_Position"})
    # MC3 sample IDs have no vial (TCGA-XX-XXXX-01); the expression IDs do
    alterations = alteration_labels(events, tcga.expression.index.map(sample_id)).set_axis(tcga.expression.index)
    return tcga.expression.astype(float), tcga.mutations, alterations, "TCGA_" + tcga.cancer.astype(str)


INTEGRATIONS = ("combat", "sample_rank")


def integrate(tcga: pd.DataFrame, external: pd.DataFrame, tcga_batches: pd.Series, cohort: dict, method: str = "combat"):
    """log1p on each side as configured (and TCGA z-scored per gene with tcga_zscore), on the shared genes that are finite and vary inside every batch (the TCGA cancer
    types and the cohort; what ComBat needs, used for every method so they see the same genes), then:
    combat: ComBat with those batches; sample_rank: each sample's genes ranked within the sample."""
    if method not in INTEGRATIONS:
        raise ValueError(f"integration must be one of {INTEGRATIONS}")
    scale = lambda x, log: np.log1p(x.clip(lower=0)) if log else x  # missing values stay NaN, so their genes are dropped
    tcga = scale(tcga, cohort["log1p_tcga"])
    if cohort.get("tcga_zscore"):  # the cohort is per-gene z-scores: TCGA the same (genes without variance become NaN)
        tcga = (tcga - tcga.mean()) / tcga.std(ddof=0)
    combined = pd.concat([tcga, scale(external, cohort["log1p_external"])], join="inner").sort_index(axis=1)
    batch = pd.concat([tcga_batches, pd.Series(cohort["cohort_id"], index=external.index)])
    combined = combined.loc[:, np.isfinite(combined).all() & combined.groupby(batch.to_numpy()).var(ddof=0).gt(0).all()]
    if method == "combat":
        # inmoose takes genes x samples
        corrected = pycombat_norm(counts=combined.T.to_numpy(dtype=np.float64), batch=batch.to_numpy())
        combined = pd.DataFrame(np.asarray(corrected).T, index=combined.index, columns=combined.columns)
    else:
        combined = combined.rank(axis=1, pct=True)
    return combined.loc[tcga.index], combined.loc[external.index], batch


def select_targets(targets: list[str], tcga_labels: pd.DataFrame, external_labels: pd.DataFrame, config: dict) -> pd.DataFrame:
    """One row per target with its counts; selected if it has enough positives on both sides."""
    rows, selected = [], []
    for target in targets:
        tcga = tcga_labels[target] if target in tcga_labels else None
        external = external_labels[target].dropna() if target in external_labels else None
        reasons = []
        if tcga is None:
            reasons.append("missing_tcga_label")
        elif tcga.sum() < config["target_selection"]["min_tcga_positive"]:
            reasons.append("too_few_tcga_positives")
        if external is None or external.empty:
            reasons.append("missing_external_label")
        else:
            if external.sum() < config["target_selection"]["min_external_positive"]:
                reasons.append("too_few_external_positives")
            # Same external labels as an earlier (more specific) target, e.g. BRAF V600 any when every V600 is V600E
            same = [t for t in selected if external.equals(external_labels[t].dropna())]
            if same:
                reasons.append(f"same_external_labels_as_{same[0]}")
        if not reasons:
            selected.append(target)
        rows.append({
            "target": target,
            "selected": not reasons,
            "skip_reason": ";".join(reasons),
            "tcga_n_positive": int(tcga.sum()) if tcga is not None else 0,
            "external_n_evaluable": len(external) if external is not None else 0,
            "external_n_positive": int(external.sum()) if external is not None else 0,
            "prevalence": external.mean() if external is not None and len(external) else np.nan,
        })
    return pd.DataFrame(rows)


def fit_predict(tcga_x: pd.DataFrame, tcga_y: pd.Series, external_x: pd.DataFrame, params: dict) -> pd.Series:
    """XGBoost on all TCGA samples, positives weighted by negatives/positives; probabilities for the external samples."""
    y = tcga_y.to_numpy() > 0
    model = xgb.XGBClassifier(**params, eval_metric="logloss", scale_pos_weight=(~y).sum() / y.sum())
    model.fit(tcga_x.to_numpy(dtype=np.float32), y)
    return pd.Series(model.predict_proba(external_x.to_numpy(dtype=np.float32))[:, 1], index=external_x.index)


def score(labels: pd.Series, probability: pd.Series, prevalence: float) -> dict:
    """ROC AUC, AUPRC and normalized AUPRC = (AUPRC - prevalence) / (1 - prevalence) on the samples with a label."""
    y = labels.dropna() > 0
    if y.nunique() < 2:
        return {"evaluable": False}
    auprc = average_precision_score(y, probability[y.index])
    return {"roc_auc": roc_auc_score(y, probability[y.index]), "auprc": auprc,
            "normalized_auprc": (auprc - prevalence) / (1 - prevalence), "evaluable": True}


def run_cohort(config: dict, cohort: dict, output_dir: Path, integration: str) -> list[pd.DataFrame]:
    external_x, external_genes, external_alterations = load_external(config, cohort)
    tcga_x, tcga_genes, tcga_alterations, tcga_batches = load_tcga(config, cohort)
    drivers = CANCER_DRIVER_GENES[cohort["drivers"]]
    modes = {
        "gene": (drivers, tcga_genes, external_genes),
        "alteration": ([label for label, (gene, _) in ALTERATIONS.items() if gene in drivers], tcga_alterations, external_alterations),
    }
    tables = {mode: select_targets(targets, tcga_y, external_y, config) for mode, (targets, tcga_y, external_y) in modes.items()}
    if any(table["selected"].any() for table in tables.values()):
        tcga_c, external_c, batch = integrate(tcga_x, external_x, tcga_batches, cohort, integration)

    results = []
    for mode, (_, tcga_y, external_y) in modes.items():
        table = tables[mode]
        probabilities = pd.DataFrame(index=external_x.index)
        for i in table.index[table["selected"]]:
            target = table.at[i, "target"]
            print(f"  {mode} {target}", flush=True)
            probabilities[target] = fit_predict(tcga_c, tcga_y[target], external_c, config["xgboost"])
            for key, value in score(external_y[target], probabilities[target], table.at[i, "prevalence"]).items():
                table.at[i, key] = value

        out = output_dir / mode / cohort["cohort_id"]
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / "targets.csv", index=False)
        probabilities.to_csv(out / "probabilities.csv.gz")
        manifest = {**cohort, "label_mode": mode, "integration": integration, "external_samples": len(external_x), "tcga_samples": len(tcga_x)}
        if table["selected"].any():
            manifest.update(n_features=tcga_c.shape[1], batches=batch.value_counts().to_dict())
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        table.insert(0, "label_mode", mode)
        for key in ("cancer", "label", "cohort_id"):
            table.insert(0, key, cohort[key])
        table["alteration_definition"] = table["target"].map(lambda t: ALTERATIONS[t][1] if t in ALTERATIONS else "")
        results.append(table)
    return results


def integration_output(method: str) -> Path:
    """Every cohort with one integration: output/integration_<method>/."""
    return RUN_ROOT / "output" / f"integration_{method}"


def main_results(config: dict) -> None:
    """output/summary/clinical_driver_results.csv: each cohort with its main (first) integration."""
    rows = []
    for cohort_id, cohort in config["cohorts"].items():
        method = cohort["integrations"][0]
        table = pd.read_csv(integration_output(method) / "summary" / "clinical_driver_results.csv")
        rows.append(table[table["cohort_id"].eq(cohort_id)].assign(integration=method))
    path = RUN_ROOT / "output" / "summary" / "clinical_driver_results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)


def compare_integrations(integrations: list[str]) -> None:
    """Per-target metrics of every integration side by side: output/summary/integration_comparison.csv, and
    integration_comparison.xlsx with one sheet per cohort."""
    keys = ["cohort_id", "label", "cancer", "label_mode", "target", "alteration_definition", "external_n_positive",
            "external_n_evaluable", "prevalence"]
    metrics = ["normalized_auprc", "auprc", "roc_auc"]
    index = ["cohort_id", "label_mode", "target"]
    tables = []
    for method in integrations:
        table = pd.read_csv(integration_output(method) / "summary" / "clinical_driver_results.csv")
        table = table[table["evaluable"].eq(True)].set_index(index)  # NaN for targets not tested
        if not tables:
            tables.append(table[[k for k in keys if k not in index]])  # the labels are the same for every integration
        tables.append(table[metrics].add_suffix(f"_{method}"))
    wide = pd.concat(tables, axis=1).reset_index()
    wide = wide[keys + [f"{m}_{i}" for m in metrics for i in integrations]]
    wide = wide.dropna(subset=[f"normalized_auprc_{m}" for m in integrations])  # targets run with every integration
    first, other = integrations[0], integrations[1]
    wide[f"normalized_auprc_{first}_minus_{other}"] = wide[f"normalized_auprc_{first}"] - wide[f"normalized_auprc_{other}"]
    wide = wide.sort_values(["cohort_id", "label_mode", f"normalized_auprc_{first}"], ascending=[True, True, False])
    summary_dir = RUN_ROOT / "output" / "summary"
    wide.to_csv(summary_dir / "integration_comparison.csv", index=False)
    with pd.ExcelWriter(summary_dir / "integration_comparison.xlsx") as writer:
        medians = wide.groupby(["cohort_id", "label_mode"])[[c for c in wide.columns if c.startswith("normalized_auprc")]].median()
        medians.assign(n_targets=wide.groupby(["cohort_id", "label_mode"]).size()).round(3).reset_index().to_excel(writer, sheet_name="medians", index=False)
        for cohort_id, rows in wide.groupby("cohort_id", sort=False):
            rows.drop(columns=["cohort_id", "label", "cancer"]).round(3).to_excel(writer, sheet_name=cohort_id[:31], index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", help="Run only these cohorts (keys in config/config.yaml).")
    args = parser.parse_args()
    config = yaml.safe_load((RUN_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    integrations = list(dict.fromkeys(m for cohort in config["cohorts"].values() for m in cohort["integrations"]))
    for integration in integrations:
        run_integration(config, args, integration, integration_output(integration))
    main_results(config)
    if len(integrations) > 1:
        compare_integrations(integrations)


def run_integration(config: dict, args, integration: str, output_dir: Path) -> None:
    print(f"Integration {integration} -> {output_dir}", flush=True)
    results = []
    for cohort_id, cohort in config["cohorts"].items():
        if integration not in cohort["integrations"] or (args.only and cohort_id not in args.only):
            continue
        print(f"Cohort {cohort_id}", flush=True)
        results += run_cohort(config, {"cohort_id": cohort_id, "source": cohort_id, **cohort}, output_dir, integration)

    # One table over cohorts and both label modes (Supplementary Table S3c keeps the evaluable rows)
    columns = ["cohort_id", "label", "cancer", "label_mode", "target", "alteration_definition", "external_n_positive",
               "external_n_evaluable", "prevalence", "roc_auc", "auprc", "normalized_auprc", "evaluable"]
    path = output_dir / "summary" / "clinical_driver_results.csv"
    if args.only and path.exists():  # keep the rows of the cohorts not rerun
        previous = pd.read_csv(path)
        results.insert(0, previous[~previous["cohort_id"].isin(args.only)])
    summary = pd.concat(results, ignore_index=True).reindex(columns=columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)


if __name__ == "__main__":
    main()
