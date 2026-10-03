"""Train on TCGA, transfer to each external cohort, and save predictions, metrics and embeddings.

Usage (from Runs/External):
    python src/run_external.py                      # all cohorts in config/config.yaml
    python src/run_external.py --only METABRIC HUGO
"""
from __future__ import annotations

import argparse
import math
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline import (
    RUN_ROOT,
    align_mutations,
    available_device,
    choose_target_genes,
    compute_gene_metrics,
    compute_site_metrics,
    integration_coordinates,
    load_cohort_data,
    model_config,
    normalize_for_method,
    print_gpu_status,
    read_config,
    reset_index_named,
    run_path,
    target_gene_list,
    train_external_multitask_model,
    write_dataframe,
    write_json,
)


def target_settings(config: dict[str, Any], cohort: str, n_train: int) -> dict[str, Any]:
    tcfg = config["target_selection"]
    min_samples = int(tcfg.get("min_tcga_mutated_samples", 10))
    min_prevalence = float(tcfg.get("min_tcga_mutation_prevalence", 0.0) or 0.0)
    restrict = target_gene_list(config, cohort)
    return {
        "min_tcga_mutated_samples": min_samples,
        "min_tcga_mutation_prevalence": min_prevalence,
        "effective_min_tcga_mutated_samples": max(min_samples, int(math.ceil(min_prevalence * max(n_train, 1)))),
        "restrict_to_genes": restrict,
    }


def metrics_summary(metrics: pd.DataFrame | None) -> dict[str, Any] | None:
    if metrics is None or metrics.empty:
        return None
    ev = metrics[metrics["evaluable"].astype(bool)]
    if ev.empty:
        return {"n_evaluable": 0}
    return {
        "n_evaluable": int(len(ev)),
        "median_auprc": float(ev["auprc"].median()),
        "median_normalized_auprc": float(ev["normalized_auprc"].median()),
        "median_roc_auc": float(ev["roc_auc"].median()),
    }


def run_cohort(config: dict[str, Any], cohort: str, out_dir: Path, *, skip_integration: bool) -> dict[str, Any]:
    ccfg = config["cohorts"][cohort]
    method = ccfg["method"]
    t0 = time.time()

    data = load_cohort_data(config, cohort)
    target_genes, target_table = choose_target_genes(config, data)
    print(f"  targets={len(target_genes)} train={data.tcga_expression.shape} test={data.external_expression.shape}")

    integ_before = None
    if not skip_integration:
        integ_before = integration_coordinates(
            data.tcga_expression,
            data.external_expression,
            test_batches=data.test_batches,
            external_label=cohort,
        )

    train_expr, test_expr, features, norm_label = normalize_for_method(config, data, method)
    print(f"  normalization: {norm_label} -> features={len(features)}")

    integ_after = None
    if not skip_integration:
        integ_after = integration_coordinates(
            train_expr,
            test_expr,
            test_batches=data.test_batches.reindex(test_expr.index) if data.test_batches is not None else None,
            external_label=cohort,
        )

    train_mut, test_mut = align_mutations(data, target_genes, train_expr, test_expr)
    result = train_external_multitask_model(
        train_expression=train_expr,
        train_mutations=train_mut,
        test_expression=test_expr,
        target_genes=target_genes,
        config=model_config(config),
        random_seed=int(config.get("random_seed", 42)),
        extract_embeddings=True,
    )

    metrics = site_metrics = None
    if test_mut is not None:
        metrics = compute_gene_metrics(test_mut, result.predictions, result.probabilities, target_genes)
        if data.site_col is not None and data.site_col in data.clinical.columns:
            site_metrics = compute_site_metrics(
                test_mut,
                result.predictions,
                result.probabilities,
                data.clinical,
                target_genes,
                site_col=data.site_col,
                min_site_n=int(config["clinical"]["min_site_samples"]),
            )

    write_dataframe(target_table, out_dir / "qc" / "target_gene_selection.csv")
    write_dataframe(reset_index_named(train_expr), out_dir / "processed" / "tcga_train_expression.csv.gz")
    write_dataframe(reset_index_named(test_expr), out_dir / "processed" / "external_expression.csv.gz")
    write_dataframe(reset_index_named(train_mut), out_dir / "processed" / "tcga_train_mutations.csv.gz")
    if test_mut is not None:
        write_dataframe(reset_index_named(test_mut), out_dir / "processed" / "external_mutations.csv.gz")
    clin_external = data.clinical.loc[test_expr.index.intersection(data.clinical.index)]
    write_dataframe(reset_index_named(clin_external), out_dir / "processed" / "external_clinical.csv.gz")
    if integ_before is not None:
        write_dataframe(reset_index_named(integ_before), out_dir / "integration" / "before_normalization.csv")
    if integ_after is not None:
        write_dataframe(reset_index_named(integ_after), out_dir / "integration" / "after_normalization.csv")
    write_dataframe(reset_index_named(result.train_embeddings), out_dir / "embeddings" / "tcga_train_embeddings.csv.gz")
    write_dataframe(reset_index_named(result.test_embeddings), out_dir / "embeddings" / "external_embeddings.csv.gz")
    write_dataframe(reset_index_named(result.predictions), out_dir / "predictions" / "predictions.csv.gz")
    write_dataframe(reset_index_named(result.probabilities), out_dir / "predictions" / "probabilities.csv.gz")
    if metrics is not None:
        write_dataframe(metrics, out_dir / "metrics" / "all_gene_metrics.csv")
    if site_metrics is not None:
        write_dataframe(site_metrics, out_dir / "metrics" / "site_gene_metrics.csv")

    norm_cfg = ccfg.get("normalization", {}) or {}
    manifest = {
        "cohort": cohort,
        "method": method,
        "config_version": config.get("version"),
        "random_seed": int(config.get("random_seed", 42)),
        "device": available_device(),
        "elapsed_s": round(time.time() - t0, 2),
        "normalization": {"method": method, "label": norm_label, **norm_cfg},
        "tcga": {
            "expression_measure": ccfg.get("tcga_expression_measure"),
            "cancer_types": ccfg.get("tcga_cancer_types"),
            "n_samples": int(train_expr.shape[0]),
            "n_genes_input": int(data.tcga_expression.shape[1]),
            "meta": data.tcga_meta,
        },
        "external": {
            "data_dir": ccfg.get("data_dir"),
            "n_samples": int(test_expr.shape[0]),
            "n_genes_input": int(data.external_expression.shape[1]),
            "mutation_labels_available": test_mut is not None,
            "site_column": data.site_col,
            "meta": data.external_meta,
        },
        "features": {"n_features": int(len(features))},
        "targets": {"n_target_genes": int(len(target_genes)), **target_settings(config, cohort, train_expr.shape[0])},
        "model": config["model"]["multitask_nn"],
        "metrics_summary": metrics_summary(metrics),
    }
    write_json(manifest, out_dir / "manifest.json")
    print(f"  saved -> {out_dir}  ({manifest['elapsed_s']:.1f}s)")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=RUN_ROOT / "config" / "config.yaml")
    parser.add_argument("--only", nargs="+", help="Run only these cohorts.")
    parser.add_argument("--output-dir", default="output", help="Output root, relative to Runs/External.")
    parser.add_argument("--force", action="store_true", help="Re-run cohorts that already have a manifest.")
    parser.add_argument("--skip-integration", action="store_true", help="Skip the before/after UMAP coordinates.")
    args = parser.parse_args()

    config = read_config(args.config)
    print_gpu_status()
    output_root = run_path(args.output_dir)
    cohorts = [c for c in config["cohorts"] if not args.only or c in args.only]

    summary_rows = []
    for i, cohort in enumerate(cohorts, 1):
        print(f"\n=== [{i}/{len(cohorts)}] {cohort} / {config['cohorts'][cohort]['method']} ===")
        out_dir = output_root / cohort
        if (out_dir / "manifest.json").exists() and not args.force:
            print("  [skip] manifest exists (use --force to re-run)")
            continue
        try:
            manifest = run_cohort(config, cohort, out_dir, skip_integration=args.skip_integration)
            summary_rows.append({
                "cohort": cohort,
                "method": manifest["method"],
                "norm_label": manifest["normalization"]["label"],
                "n_train": manifest["tcga"]["n_samples"],
                "n_test": manifest["external"]["n_samples"],
                "n_features": manifest["features"]["n_features"],
                "n_targets": manifest["targets"]["n_target_genes"],
                **(manifest["metrics_summary"] or {}),
                "elapsed_s": manifest["elapsed_s"],
                "status": "ok",
            })
        except Exception as exc:
            traceback.print_exc()
            summary_rows.append({"cohort": cohort, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})

    if summary_rows:
        output_root.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame(summary_rows)
        summary.to_csv(output_root / "runs_summary.csv", index=False)
        print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
