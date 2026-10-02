"""Training functions for mutation prediction models."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from evaluation.metrics import combine_fold_predictions, evaluate_multilabel


def run_kfold_training(
    model,
    X,
    Y,
    k: int = 5,
    output_dir: str | Path = "results/",
    config_meta: Dict | None = None,
    random_state: int | None = 42,
    label: str | None = None,
    save_models: bool = False,
):
    """K-fold cross-validation with per-fold artifacts."""
    mutation_names = (
        Y.columns if isinstance(Y, pd.DataFrame) else [f"mutation_{i}" for i in range(Y.shape[1])]
    )
    sample_ids = (
        Y.index.to_numpy() if isinstance(Y, pd.DataFrame) else np.arange(Y.shape[0])
    )

    X_values = X.values if isinstance(X, pd.DataFrame) else X
    Y_values = Y.values if isinstance(Y, pd.DataFrame) else Y

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_values), start=1):
        print(f"\nFold {fold}/{k}...")

        model_fold = copy.deepcopy(model)

        X_train, X_test = X_values[train_idx], X_values[test_idx]
        Y_train, Y_test = Y_values[train_idx], Y_values[test_idx]
        test_ids = sample_ids[test_idx]

        model_fold.fit(X_train, Y_train)
        Y_pred, Y_prob = model_fold.predict(X_test)

        fold_df = evaluate_multilabel(Y_test, Y_pred, Y_prob, mutation_names)
        fold_metrics.append(fold_df.assign(fold=fold))

        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True)
        fold_df.to_csv(fold_dir / "metrics.csv")

        preds_df = pd.DataFrame(Y_pred, columns=mutation_names, index=test_ids)
        probs_df = pd.DataFrame(Y_prob, columns=mutation_names, index=test_ids)
        preds_df.to_csv(fold_dir / "predictions.csv")
        probs_df.to_csv(fold_dir / "probabilities.csv")

        if save_models:
            model_fold.save(fold_dir)

    stacked = pd.concat(fold_metrics)
    summary = stacked.groupby(level=0).agg(['mean', 'std'])
    summary.columns = ["_".join(col).strip() for col in summary.columns]
    summary.to_csv(output_dir / "summary.csv")

    combine_fold_predictions(
        kfold_dir=output_dir,
        output_dir=output_dir,
    )

    if isinstance(Y, pd.DataFrame):
        from evaluation.metrics import compute_metrics_from_combined
        try:
            combined_predictions_dir = output_dir / "combined_predictions"
            if combined_predictions_dir.exists():
                print("\nComputing overall metrics from combined predictions...")
                compute_metrics_from_combined(
                    combined_predictions_dir=combined_predictions_dir,
                    y_true=Y,
                    output_dir=output_dir / "metrics",
                )
        except Exception as e:
            print(f"Warning: Could not compute metrics from combined predictions: {e}")

    if config_meta:
        with open(output_dir / "meta.json", "w") as f:
            json.dump(config_meta, f, indent=2)

    print("K-fold evaluation complete.")
    return summary
