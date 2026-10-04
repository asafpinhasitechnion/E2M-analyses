"""Run the clinical-driver transfer for all cohorts in config/config.yaml (gene and alteration modes).

Usage (from Runs/ClinicalDrivers):
    python src/run_drivers.py
    python src/run_drivers.py --only metabric difg_glass
    python src/run_drivers.py --summarize-only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from alterations import alteration_definition_table
from transfer import cohort_plans, read_config, run_cohort, run_path, write_run_summary



def run_xgboost_set(config: dict, plans: list, *, output_root: Path, label_mode: str) -> list[dict]:
    output_root.mkdir(parents=True, exist_ok=True)
    if label_mode == "alteration":
        alteration_definition_table().to_csv(output_root / "alteration_label_definitions.csv", index=False)

    manifests = []
    for plan in plans:
        print(f"XGBoost {label_mode}: {plan.cohort_id}", flush=True)
        manifests.append(run_cohort(config, plan, output_root=output_root, label_mode=label_mode))
    write_run_summary(output_root, manifests)
    return manifests


def collect_metrics(model_root: Path, *, model_type: str, label_mode: str) -> pd.DataFrame:
    rows = []
    for manifest_path in sorted(model_root.glob("*/manifest.json")):
        cohort_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selection_path = cohort_dir / "qc" / "target_selection.csv"
        metrics_path = cohort_dir / "metrics" / "gene_metrics.csv"
        selection = pd.read_csv(selection_path) if selection_path.exists() else pd.DataFrame()
        metrics = pd.read_csv(metrics_path, index_col=0) if metrics_path.exists() else pd.DataFrame()

        target_col = "target_label" if label_mode == "alteration" else "gene"
        if not selection.empty:
            selection = selection.rename(columns={"gene": target_col, "reason": "skip_reason"})
        if not metrics.empty:
            metrics = metrics.rename(columns={"gene": target_col})
            merged = selection.merge(metrics, on=target_col, how="left", suffixes=("_selection", ""))
        else:
            merged = selection.copy()
        if merged.empty:
            continue

        for key in (
            "cohort_id",
            "label",
            "cancer",
            "source_cohort_id",
            "tcga_expression_measure",
            "external_expression_type",
            "normalization_applied",
            "n_features",
        ):
            merged[key] = manifest.get(key)
        merged["model_type"] = model_type
        merged["label_mode"] = label_mode
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_summaries(result_root: Path) -> None:
    summary_dir = result_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    gene = collect_metrics(result_root / "xgboost_gene", model_type="xgboost_gene", label_mode="gene")
    alteration = collect_metrics(result_root / "xgboost_alteration", model_type="xgboost_alteration", label_mode="alteration")
    if not gene.empty:
        gene.to_csv(summary_dir / "xgboost_gene_target_metrics.csv", index=False)
    if not alteration.empty:
        alteration.to_csv(summary_dir / "xgboost_alteration_target_metrics.csv", index=False)

    write_consolidated_results(gene, alteration, summary_dir)


def write_consolidated_results(gene: pd.DataFrame, alteration: pd.DataFrame, summary_dir: Path) -> None:
    """Write one explicit clinical-driver results table spanning gene- and alteration-level models."""
    defs = alteration_definition_table().rename(
        columns={"label": "target", "description": "alteration_definition"}
    )
    frames = []
    if not gene.empty:
        g = gene.copy()
        g["target"] = g["gene"]
        g["alteration_definition"] = ""
        frames.append(g)
    if not alteration.empty:
        a = alteration.copy()
        a["target"] = a["target_label"]
        a = a.merge(defs[["target", "alteration_definition"]], on="target", how="left")
        frames.append(a)
    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True, sort=False)
    columns = [
        "cohort_id",
        "label",
        "cancer",
        "model_type",
        "label_mode",
        "target",
        "alteration_definition",
        "external_n_positive",
        "external_n_evaluable",
        "prevalence",
        "roc_auc",
        "auprc",
        "normalized_auprc",
        "evaluable",
    ]
    columns = [col for col in columns if col in combined.columns]
    combined[columns].sort_values(
        ["model_type", "cohort_id", "normalized_auprc"],
        ascending=[True, True, False],
        na_position="last",
    ).to_csv(summary_dir / "clinical_driver_results.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    parser.add_argument("--only", nargs="+", help="Run only these cohorts.")
    parser.add_argument("--output-dir", default="output", help="Output root, relative to Runs/ClinicalDrivers.")
    parser.add_argument("--summarize-only", action="store_true", help="Only rebuild the summary tables.")
    args = parser.parse_args()

    config = read_config(args.config)
    result_root = run_path(args.output_dir)
    result_root.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        plans = [p for p in cohort_plans(config) if not args.only or p.cohort_id in args.only]
        for mode in ("gene", "alteration"):
            run_xgboost_set(config, plans, output_root=result_root / f"xgboost_{mode}", label_mode=mode)

    write_summaries(result_root)
    manifest = {
        "completed_at_unix": time.time(),
        "branches": {
            mode: len(list((result_root / mode).glob("*/manifest.json")))
            for mode in ("xgboost_gene", "xgboost_alteration")
        },
    }
    (result_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
