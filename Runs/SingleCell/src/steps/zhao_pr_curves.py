"""Per-fold precision-recall curves for the selected Zhao/Sims targets."""

from __future__ import annotations

import config
from datasets.zhao import extract_zhao_pr_curves


RUN_FOLDER = "ZhaoSims2021_mc300_simple_wide_run_cvgrp_sample"
DATA_FILE = "ZhaoSims2021.h5ad"
TARGETS = ("etoposide", "panobinostat")


def main() -> None:
    run_dir = config.OUTPUTS_ROOT / RUN_FOLDER
    data_path = config.DATA_DIR / DATA_FILE
    if not run_dir.exists():
        print(f"[skip] missing run dir from stage01: {run_dir}")
        return
    if not data_path.exists():
        print(f"[skip] missing data file: {data_path}")
        return

    print(f"[zhao_pr] extracting PR curves for {TARGETS} -> {run_dir}")
    extract_zhao_pr_curves(run_dir=run_dir, data_file=data_path, targets=TARGETS)
