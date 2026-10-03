"""Copy the files needed for Figure 5 and Table S4 into output/plot_inputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import config


# Each entry: (target subfolder name inside plot_inputs, source run folder,
#              list of files to copy from that run folder).
BUNDLE: List[tuple] = [
    (
        "admason_AdamsonWeissman2016_GSM2406675_10X001_mc300_simple_wide_run",
        "admason_AdamsonWeissman2016_GSM2406675_10X001_mc300_simple_wide_run",
        [
            "metric_summary_per_gene_folds.csv",
            "oof_cell_folds.csv",
            "oof_probabilities.csv",
            "oof_true_labels.csv",
        ],
    ),
    (
        "admason_AdamsonWeissman2016_GSM2406677_10X005_mc300_simple_wide_run",
        "admason_AdamsonWeissman2016_GSM2406677_10X005_mc300_simple_wide_run",
        [
            "metric_summary_per_gene_folds.csv",
            "oof_cell_folds.csv",
            "oof_probabilities.csv",
            "oof_true_labels.csv",
        ],
    ),
    (
        "admason_AdamsonWeissman2016_GSM2406681_10X010_mc300_simple_wide_run",
        "admason_AdamsonWeissman2016_GSM2406681_10X010_mc300_simple_wide_run",
        [
            "metric_summary_per_gene_folds.csv",
            "oof_cell_folds.csv",
            "oof_probabilities.csv",
            "oof_true_labels.csv",
        ],
    ),
    (
        "McFarlandTsherniak2020_mc300_simple_wide_run_cvgrp_cell_line",
        "McFarlandTsherniak2020_mc300_simple_wide_run_cvgrp_cell_line",
        ["metric_summary_per_gene_folds.csv"],
    ),
    (
        "ReplogleWeissman2022_K562_essential_mc300_simple_wide_run",
        "ReplogleWeissman2022_K562_essential_mc300_simple_wide_run",
        ["metric_summary_per_gene_folds.csv"],
    ),
    (
        "ReplogleWeissman2022_rpe1_mc300_simple_wide_run",
        "ReplogleWeissman2022_rpe1_mc300_simple_wide_run",
        ["metric_summary_per_gene_folds.csv"],
    ),
    (
        "FrangiehIzar2021_RNA_mc800_simple_wide_run",
        "FrangiehIzar2021_RNA_mc800_simple_wide_run",
        [
            "metric_summary_per_gene_folds.csv",
            "metric_summary_per_gene_by_condition.csv",
            "metric_summary_per_gene_by_condition_folds.csv",
        ],
    ),
    (
        "FrangiehIzar2021_RNA_per_gene_mc800_baseline",
        "FrangiehIzar2021_RNA_per_gene_mc800_baseline",
        [
            "metric_summary_per_gene.csv",
            "metric_summary_per_gene_folds.csv",
            "metric_summary_per_gene_by_condition.csv",
            "metric_summary_per_gene_by_condition_folds.csv",
        ],
    ),
    (
        "TianKampmann2021_CRISPRa_two_guide_per-gene_mcg100_simple_wide_run",
        "TianKampmann2021_CRISPRa_two_guide_per-gene_mcg100_simple_wide_run",
        ["directional_metrics_per_gene.csv"],
    ),
    (
        "ZhaoSims2021_mc300_simple_wide_run_cvgrp_sample",
        "ZhaoSims2021_mc300_simple_wide_run_cvgrp_sample",
        [
            "oof_cell_folds.csv",
            "oof_probabilities_selected_targets.csv",
            "pr_curve_points_selected_targets.csv",
            "pr_curve_summary_selected_targets.csv",
            "run_metadata.json",
        ],
    ),
    # CCLE has its own subfolder name (no leading run-folder prefix).
    (
        "CCLE",
        "ccle_tmb_and_multitask",
        [
            "cell_level_xgboost.csv",
            "model_level_xgboost.csv",
            "meta_xgboost.csv",
            "multitask_fold_metrics_50.csv",
            "multitask_summary_50.csv",
            "multitask_fold_metrics_100.csv",
            "multitask_summary_100.csv",
            "multitask_fold_metrics_150.csv",
            "multitask_summary_150.csv",
            "multitask_fold_metrics_200.csv",
            "multitask_summary_200.csv",
        ],
    ),
]


def _copy_subset(src_dir: Path, dst_dir: Path, files: Iterable[str]) -> Dict[str, str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    status: Dict[str, str] = {}
    for name in files:
        src = src_dir / name
        dst = dst_dir / name
        if not src.exists():
            status[name] = "missing"
            continue
        shutil.copy2(src, dst)
        status[name] = "ok"
    return status


def main() -> None:
    config.make_dirs()
    plot_inputs = config.PLOT_INPUTS_DIR
    print(f"[bundle] writing into {plot_inputs}")

    manifest: Dict[str, Dict[str, str]] = {}
    for dst_name, src_name, files in BUNDLE:
        src_dir = config.OUTPUTS_ROOT / src_name
        dst_dir = plot_inputs / dst_name
        if not src_dir.exists():
            print(f"  [skip] {dst_name}: source missing ({src_dir})")
            manifest[dst_name] = {f: "source missing" for f in files}
            continue
        status = _copy_subset(src_dir=src_dir, dst_dir=dst_dir, files=files)
        manifest[dst_name] = status
        print(f"  [ok]   {dst_name}: {sum(v == 'ok' for v in status.values())}/{len(status)} files")

    (plot_inputs / "_bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[bundle] manifest -> {plot_inputs / '_bundle_manifest.json'}")
