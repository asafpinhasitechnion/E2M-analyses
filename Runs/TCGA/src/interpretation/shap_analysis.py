"""SHAP analysis for tree-based model interpretation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save_shap_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    sample_ids: list[str] | pd.Index | None = None,
    top_n: int = 10,
) -> None:
    """Save a compact summary of SHAP values instead of the full matrix."""
    n_samples, n_features = shap_values.shape

    if sample_ids is None:
        sample_ids = [f"sample_{i}" for i in range(n_samples)]
    elif isinstance(sample_ids, pd.Index):
        sample_ids = sample_ids.tolist()

    summary_rows = []

    for sample_idx, sample_id in enumerate(sample_ids):
        sample_shap = shap_values[sample_idx, :]

        top_indices = np.argsort(np.abs(sample_shap))[-top_n:][::-1]

        for rank, feat_idx in enumerate(top_indices, start=1):
            feat_name = feature_names[feat_idx]
            shap_val = sample_shap[feat_idx]

            summary_rows.append({
                'sample_id': sample_id,
                'feature': feat_name,
                'shap_value': shap_val,
                'rank': rank,  # Rank 1 = highest absolute SHAP value for this sample
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(['sample_id', 'rank'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    feature_summary = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.mean(np.abs(shap_values), axis=0),
        'mean_shap': np.mean(shap_values, axis=0),
    })

    feature_summary = feature_summary[feature_summary['mean_abs_shap'] > 0].copy()
    feature_summary = feature_summary.sort_values('mean_abs_shap', ascending=False)

    feature_summary_path = output_path.parent / f"{output_path.stem}_feature_summary.csv"
    feature_summary.to_csv(feature_summary_path, index=False)
