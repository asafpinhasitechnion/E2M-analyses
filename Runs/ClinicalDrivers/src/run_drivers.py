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
    if "filter" in cohort:  # POG570: one cancer type of the pan-cancer study
        values = read("clinical_sample.csv.gz")[cohort["filter"]["column"]].reindex(expression.index).astype(str)
        expression = expression[values.isin(cohort["filter"]["values"])]
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
    """TCGA from E2M: linear TPM, or CPM for counts, of protein-coding genes; every mutated gene as a label; the cancer
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


def combat(tcga: pd.DataFrame, external: pd.DataFrame, tcga_batches: pd.Series, cohort: dict):
    """log1p on each side as configured, then ComBat on the shared genes; batches are the TCGA cancer types and the cohort."""
    scale = lambda x, log: np.log1p(x.fillna(0).clip(lower=0)) if log else x.fillna(0)
    combined = pd.concat([scale(tcga, cohort["log1p_tcga"]), scale(external, cohort["log1p_external"])], join="inner").sort_index(axis=1)
    batch = pd.concat([tcga_batches, pd.Series(cohort["cohort_id"], index=external.index)])
    # ComBat needs genes that are finite and vary inside every batch
    combined = combined.loc[:, np.isfinite(combined).all() & combined.groupby(batch.to_numpy()).var(ddof=0).gt(0).all()]
    # inmoose takes genes x samples
    corrected = pycombat_norm(counts=combined.T.to_numpy(dtype=np.float64), batch=batch.to_numpy())
    corrected = pd.DataFrame(np.asarray(corrected).T, index=combined.index, columns=combined.columns)
    return corrected.loc[tcga.index], corrected.loc[external.index], batch


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


def run_cohort(config: dict, cohort: dict, output_dir: Path) -> list[pd.DataFrame]:
    external_x, external_genes, external_alterations = load_external(config, cohort)
    tcga_x, tcga_genes, tcga_alterations, tcga_batches = load_tcga(config, cohort)
    drivers = CANCER_DRIVER_GENES[cohort["drivers"]]
    modes = {
        "gene": (drivers, tcga_genes, external_genes),
        "alteration": ([label for label, (gene, _) in ALTERATIONS.items() if gene in drivers], tcga_alterations, external_alterations),
    }
    tables = {mode: select_targets(targets, tcga_y, external_y, config) for mode, (targets, tcga_y, external_y) in modes.items()}
    if any(table["selected"].any() for table in tables.values()):
        tcga_c, external_c, batch = combat(tcga_x, external_x, tcga_batches, cohort)

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
        manifest = {**cohort, "label_mode": mode, "external_samples": len(external_x), "tcga_samples": len(tcga_x)}
        if table["selected"].any():
            manifest.update(n_features=tcga_c.shape[1], batches=batch.value_counts().to_dict())
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        table.insert(0, "label_mode", mode)
        for key in ("cancer", "label", "cohort_id"):
            table.insert(0, key, cohort[key])
        table["alteration_definition"] = table["target"].map(lambda t: ALTERATIONS[t][1] if t in ALTERATIONS else "")
        results.append(table)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", help="Run only these cohorts (keys in config/config.yaml).")
    args = parser.parse_args()
    config = yaml.safe_load((RUN_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    output_dir = RUN_ROOT / "output"

    results = []
    for cohort_id, cohort in config["cohorts"].items():
        if args.only and cohort_id not in args.only:
            continue
        print(f"Cohort {cohort_id}", flush=True)
        results += run_cohort(config, {"cohort_id": cohort_id, "source": cohort_id, **cohort}, output_dir)

    # One table over cohorts and both label modes (Supplementary Table S3c keeps the evaluable rows)
    columns = ["cohort_id", "label", "cancer", "label_mode", "target", "alteration_definition", "external_n_positive",
               "external_n_evaluable", "prevalence", "roc_auc", "auprc", "normalized_auprc", "evaluable"]
    summary = pd.concat(results, ignore_index=True).reindex(columns=columns)
    (output_dir / "summary").mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary" / "clinical_driver_results.csv", index=False)


if __name__ == "__main__":
    main()
