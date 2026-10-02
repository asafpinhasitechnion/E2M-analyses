"""Metrics computation for multi-label classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_multilabel(y_true, y_pred, y_prob, mutation_names):
    """Evaluate multi-label classification performance with comprehensive metrics."""
    results = {}
    for i, mutation in enumerate(mutation_names):
        y_true_gene = y_true[:, i]
        y_pred_gene = y_pred[:, i]
        y_prob_gene = y_prob[:, i]

        tp = np.sum((y_true_gene == 1) & (y_pred_gene == 1))
        tn = np.sum((y_true_gene == 0) & (y_pred_gene == 0))
        fp = np.sum((y_true_gene == 0) & (y_pred_gene == 1))
        fn = np.sum((y_true_gene == 1) & (y_pred_gene == 0))

        gene_metrics = {}

        try:
            gene_metrics['f1'] = f1_score(y_true_gene, y_pred_gene)
        except (ValueError, ZeroDivisionError):
            gene_metrics['f1'] = None

        try:
            gene_metrics['roc_auc'] = roc_auc_score(y_true_gene, y_prob_gene)
        except ValueError:
            gene_metrics['roc_auc'] = None

        try:
            gene_metrics['auprc'] = average_precision_score(y_true_gene, y_prob_gene)
        except ValueError:
            gene_metrics['auprc'] = None

        try:
            gene_metrics['accuracy'] = accuracy_score(y_true_gene, y_pred_gene)
        except (ValueError, ZeroDivisionError):
            gene_metrics['accuracy'] = None

        try:
            gene_metrics['precision'] = precision_score(y_true_gene, y_pred_gene, zero_division=0)
        except ValueError:
            gene_metrics['precision'] = None

        try:
            gene_metrics['recall'] = recall_score(y_true_gene, y_pred_gene, zero_division=0)
        except ValueError:
            gene_metrics['recall'] = None

        try:
            if (tn + fp) > 0:
                gene_metrics['specificity'] = tn / (tn + fp)
            else:
                gene_metrics['specificity'] = None
        except (ValueError, ZeroDivisionError):
            gene_metrics['specificity'] = None

        try:
            gene_metrics['mcc'] = matthews_corrcoef(y_true_gene, y_pred_gene)
        except ValueError:
            gene_metrics['mcc'] = None

        n_samples = len(y_true_gene)
        if n_samples > 0:
            gene_metrics['prevalence'] = np.sum(y_true_gene == 1) / n_samples
        else:
            gene_metrics['prevalence'] = None

        results[mutation] = gene_metrics

    return pd.DataFrame(results).T


def _load_and_align_predictions(
    predictions_path: str | Path,
    probabilities_path: str | Path,
    y_true: pd.DataFrame,
    drop_fold_column: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load predictions and probabilities and align them with y_true."""
    predictions_path = Path(predictions_path)
    probabilities_path = Path(probabilities_path)

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    if not probabilities_path.exists():
        raise FileNotFoundError(f"Probabilities file not found: {probabilities_path}")

    y_pred = pd.read_csv(predictions_path, index_col=0)
    y_prob = pd.read_csv(probabilities_path, index_col=0)

    if drop_fold_column:
        if 'fold' in y_pred.columns:
            y_pred = y_pred.drop(columns=['fold'])
        if 'fold' in y_prob.columns:
            y_prob = y_prob.drop(columns=['fold'])

    common_index = y_true.index.intersection(y_pred.index).intersection(y_prob.index)
    if len(common_index) == 0:
        raise ValueError("No common sample IDs found between y_true, y_pred, and y_prob")

    y_true_aligned = y_true.loc[common_index]
    y_pred = y_pred.loc[common_index]
    y_prob = y_prob.loc[common_index]

    common_genes = y_true_aligned.columns.intersection(y_pred.columns).intersection(y_prob.columns)
    if len(common_genes) == 0:
        raise ValueError("No common gene columns found between y_true, y_pred, and y_prob")

    y_pred = y_pred[common_genes]
    y_prob = y_prob[common_genes]
    gene_names = common_genes.tolist()

    return y_pred, y_prob, gene_names


def _save_metrics_results(results: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    results.to_csv(csv_path)
    print(f"\nSaved metrics to: {csv_path}")


def compute_metrics_from_combined(
    combined_predictions_dir: str | Path,
    y_true: pd.DataFrame,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Compute overall metrics from combined fold predictions."""
    combined_predictions_dir = Path(combined_predictions_dir)

    if not combined_predictions_dir.exists():
        raise FileNotFoundError(f"Combined predictions directory not found: {combined_predictions_dir}")

    predictions_path = combined_predictions_dir / "predictions.csv"
    probabilities_path = combined_predictions_dir / "probabilities.csv"

    print(f"Processing combined predictions from: {combined_predictions_dir}")

    y_pred, y_prob, gene_names = _load_and_align_predictions(
        predictions_path=predictions_path,
        probabilities_path=probabilities_path,
        y_true=y_true,
        drop_fold_column=True,
    )

    common_index = y_pred.index
    y_true = y_true.loc[common_index][gene_names]

    print(f"\nComputing overall metrics...")
    print(f"   Samples: {len(common_index)}")
    print(f"   Genes: {len(gene_names)}")

    metrics_df = evaluate_multilabel(
        y_true=y_true.values,
        y_pred=y_pred.values,
        y_prob=y_prob.values,
        mutation_names=gene_names,
    )

    if output_dir is not None:
        _save_metrics_results(results=metrics_df, output_dir=Path(output_dir))

    return metrics_df


def combine_fold_predictions(
    kfold_dir: str | Path,
    output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine predictions and probabilities from k-fold cross-validation results."""
    kfold_dir = Path(kfold_dir)

    if not kfold_dir.exists():
        raise FileNotFoundError(f"K-fold directory not found: {kfold_dir}")

    fold_dirs = sorted([d for d in kfold_dir.iterdir() if d.is_dir() and d.name.startswith('fold_')])

    if len(fold_dirs) == 0:
        raise ValueError(f"No fold directories found in {kfold_dir}")

    print(f"Combining predictions from {len(fold_dirs)} folds...")

    combined_preds_list = []
    combined_probs_list = []

    for fold_dir in fold_dirs:
        preds_path = fold_dir / "predictions.csv"
        probs_path = fold_dir / "probabilities.csv"

        if not preds_path.exists() or not probs_path.exists():
            print(f"   Warning: Missing files in {fold_dir}. Skipping...")
            continue

        preds_df = pd.read_csv(preds_path, index_col=0)
        probs_df = pd.read_csv(probs_path, index_col=0)

        fold_num = fold_dir.name.replace('fold_', '')
        try:
            fold_num = int(fold_num)
        except ValueError:
            pass

        preds_df = preds_df.copy()
        probs_df = probs_df.copy()
        preds_df['fold'] = fold_num
        probs_df['fold'] = fold_num

        combined_preds_list.append(preds_df)
        combined_probs_list.append(probs_df)

    if len(combined_preds_list) == 0:
        raise ValueError("No valid fold predictions found to combine")

    combined_preds = pd.concat(combined_preds_list, axis=0)
    combined_probs = pd.concat(combined_probs_list, axis=0)

    print(f"   Combined {len(combined_preds)} samples from {len(combined_preds_list)} folds")

    if output_dir is not None:
        output_dir = Path(output_dir)
        combined_dir = output_dir / "combined_predictions"
        combined_dir.mkdir(parents=True, exist_ok=True)

        combined_preds.to_csv(combined_dir / "predictions.csv")
        combined_probs.to_csv(combined_dir / "probabilities.csv")

        print(f"   Saved to: {combined_dir}")

    return combined_preds, combined_probs
