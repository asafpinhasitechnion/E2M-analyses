"""Settings from config/config.yaml and paths for the single-cell analysis (relative to Runs/SingleCell)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml

RUN_ROOT: Path = Path(__file__).resolve().parents[1]
with open(RUN_ROOT / "config" / "config.yaml", "r", encoding="utf-8") as _handle:
    _cfg = yaml.safe_load(_handle)

DATA_DIR: Path = RUN_ROOT / "data"
CCLE_DIR: Path = DATA_DIR / "CCLE"
OUTPUTS_ROOT: Path = RUN_ROOT / "output" / "runs"
PLOT_INPUTS_DIR: Path = RUN_ROOT / "output" / "plot_inputs"

RANDOM_STATE: int = int(_cfg["random_state"])
CV_FOLDS: int = int(_cfg["cv_folds"])
MULTITASK_PRESET: str = _cfg["multitask_preset"]
PREPROCESS_TARGET_SUM: float = float(_cfg["preprocess_target_sum"])

DATASET_FILES: dict[str, str] = dict(_cfg["dataset_files"])
CCLE_ADATA = CCLE_DIR / _cfg["ccle_files"]["adata"]
CCLE_MUTATIONS_CSV = CCLE_DIR / _cfg["ccle_files"]["mutations"]

ZENODO_RECORD_ID = str(_cfg["zenodo_record_id"])


def _xgb_has_cuda() -> bool:
    """Return True if the installed XGBoost was built with CUDA support."""
    try:
        import xgboost as xgb

        info_fn = getattr(xgb, "build_info", None)
        if info_fn is not None:
            return bool(info_fn().get("USE_CUDA", 0))
    except Exception:
        return False
    return False


# XGBoost runs on GPU when available; set SINGLE_CELL_USE_GPU=0/1 to force CPU/GPU.
_use_gpu_env = os.environ.get("SINGLE_CELL_USE_GPU", "auto").lower()
if _use_gpu_env in {"1", "true", "yes", "on"}:
    USE_GPU: bool = True
elif _use_gpu_env in {"0", "false", "no", "off"}:
    USE_GPU = False
else:
    USE_GPU = _xgb_has_cuda()

XGBOOST_PARAMS = {
    "n_estimators": int(_cfg["xgboost"]["n_estimators"]),
    "learning_rate": float(_cfg["xgboost"]["learning_rate"]),
    "random_state": RANDOM_STATE,
    "n_jobs": int(_cfg["xgboost"]["n_jobs"]),
}
if USE_GPU:
    XGBOOST_PARAMS["device"] = "cuda"


def make_dirs() -> None:
    OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)
    PLOT_INPUTS_DIR.mkdir(parents=True, exist_ok=True)


def describe_runtime() -> str:
    gpu = "CUDA" if USE_GPU else "CPU"
    return f"runtime: xgboost={gpu}, cv_folds={CV_FOLDS}, random_state={RANDOM_STATE}"
