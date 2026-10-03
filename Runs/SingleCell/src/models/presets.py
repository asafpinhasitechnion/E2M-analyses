"""Multitask hyperparameter preset used for all single-cell runs."""

from __future__ import annotations

from typing import Any, Dict, Tuple


# Figure 5 perturbation runs use one wide multitask preset. Inputs are already
# library-normalized and log-transformed, so no additional fold-level z-scoring is
# applied. Positive-label weighting is retained because perturbation labels are
# sparse and uneven across targets.
SIMPLE_WIDE_RUN: Dict[str, Any] = {
    "hidden_dims": (512, 256),
    "dropout": 0.3,
    "learning_rate": 1.5e-4,
    "weight_decay": 8e-4,
    "batch_size": 1024,
    "epochs": 100,
    "early_stopping_patience": 6,
    "val_split": 0.2,
    "use_pos_weight": True,
    "positive_label_bias": 1.0,
    "max_pos_weight": 30.0,
    "min_pos_weight": 1.0,
    "use_lr_scheduler": True,
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_patience": 2,
    "min_lr": 1e-6,
    "gradient_clip_norm": 1.0,
    "early_stopping_metric": "val_auprc",
    "decision_threshold": 0.5,
    "input_scaling": "none",
}


MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "simple_wide_run": dict(SIMPLE_WIDE_RUN),
    "simple_wide": dict(SIMPLE_WIDE_RUN),
}

PRESET_NAMES: Tuple[str, ...] = tuple(sorted(MODEL_PRESETS.keys()))
