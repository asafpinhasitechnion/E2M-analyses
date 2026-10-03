"""Tian/Kampmann two-guide transfer: train on one guide, test on the other."""

from __future__ import annotations

from types import SimpleNamespace

import config
from datasets.tian import run_one_file


FILE_NAME = "TianKampmann2021_CRISPRa.h5ad"
MIN_CELLS_PER_GUIDE = 100
MODE = "per-gene"            # or "multitask"
GENE_COL = "perturbation"
GUIDE_COL = "guide_id"
CONTROL_LABELS = "control"


def main() -> None:
    config.make_dirs()
    data_path = config.DATA_DIR / FILE_NAME
    if not data_path.exists():
        print(f"[skip] file missing: {data_path}")
        return

    args = SimpleNamespace(
        data_folder=config.DATA_DIR,
        output_root=config.OUTPUTS_ROOT,
        mode=MODE,
        gene_col=GENE_COL,
        guide_col=GUIDE_COL,
        control_labels=CONTROL_LABELS,
        min_cells_per_guide=MIN_CELLS_PER_GUIDE,
        preset=config.MULTITASK_PRESET,
        model_json=None,
        seed=config.RANDOM_STATE,
        quiet=False,
        preprocess_target_sum=config.PREPROCESS_TARGET_SUM,
    )
    run_one_file(args, FILE_NAME)
