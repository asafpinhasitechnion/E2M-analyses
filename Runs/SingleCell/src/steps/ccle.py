"""CCLE TMB regression and per-gene mutation multitask prediction."""

from __future__ import annotations

import config
from datasets.ccle import load_ccle_adata, run_log_tmb_cv, run_multitask_mutation_cv


TOP_N_VALUES = (50, 100, 150, 200)
REGRESSOR = "xgboost"


def main() -> None:
    config.make_dirs()
    if not config.CCLE_ADATA.exists() or not config.CCLE_MUTATIONS_CSV.exists():
        print(f"[skip] missing CCLE inputs in {config.CCLE_DIR}")
        return

    out_dir = config.OUTPUTS_ROOT / "ccle_tmb_and_multitask"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ccle] loading expression + mapping mutations...")
    adata = load_ccle_adata(
        adata_path=config.CCLE_ADATA,
        mutations_csv=config.CCLE_MUTATIONS_CSV,
        model_column="Model_ID",
        preprocess=True,
    )
    print(f"[ccle] cells={adata.n_obs} features={adata.n_vars}")

    print(f"[ccle] log-TMB regression CV...")
    run_log_tmb_cv(
        adata=adata,
        out_dir=out_dir,
        model_column="Model_ID",
        n_splits=config.CV_FOLDS,
        seed=config.RANDOM_STATE,
        regressor=REGRESSOR,
        model_params=config.XGBOOST_PARAMS if REGRESSOR == "xgboost" else None,
    )

    for top_n in TOP_N_VALUES:
        print(f"[ccle] multitask top {top_n} genes...")
        run_multitask_mutation_cv(
            adata=adata,
            mutations_csv=config.CCLE_MUTATIONS_CSV,
            out_dir=out_dir,
            top_n=top_n,
            preset_name=config.MULTITASK_PRESET,
            n_splits=config.CV_FOLDS,
            seed=config.RANDOM_STATE,
        )

    print(f"[ccle] done -> {out_dir}")
