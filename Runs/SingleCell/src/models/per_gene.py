from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import issparse
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

def _to_model_matrix(x):
    """Keep sparse expression sparse for tree models."""
    return x.copy() if issparse(x) else np.asarray(x)


def infer_cell_line_name(input_filename: str) -> str:
    stem = Path(input_filename).stem
    # Preferred inference rule: take first two tokens split by underscore.
    parts = stem.split("_")
    name = "_".join(parts[:2]) if len(parts) >= 2 else stem
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return name or "unknown_cell_line"


def get_genes_by_min_cells(
    adata,
    min_cells_per_gene: int = 100,
    gene_col: str = "gene",
    exclude: Tuple[str, ...] = ("non-targeting",),
) -> List[str]:
    counts = adata.obs[gene_col].value_counts()
    for g in exclude:
        if g in counts.index:
            counts = counts.drop(g)
    return counts[counts >= min_cells_per_gene].index.tolist()


def build_gene_dataset(
    adata,
    target_gene: str,
    gene_col: str = "gene",
    negative_mode: str = "all",
    non_target_label: str = "non-targeting",
    selected_genes: Optional[List[str]] = None,
):
    gene_values = adata.obs[gene_col].astype(str)
    mode_aliases = {
        "all_other_cells": "all",
        "selected_genes_plus_non_targeting": "selected_plus_non_targeting",
        "current_plus_non_targeting": "non_targeting",
    }
    negative_mode = mode_aliases.get(negative_mode, negative_mode)

    pos_mask = gene_values == target_gene
    if negative_mode == "all":
        neg_mask = gene_values != target_gene
    elif negative_mode == "non_targeting":
        neg_mask = gene_values == non_target_label
    elif negative_mode == "selected_plus_non_targeting":
        if selected_genes is None:
            raise ValueError(
                "selected_genes must be provided when negative_mode='selected_plus_non_targeting'"
            )
        eligible_negative_labels = set(selected_genes) | {non_target_label}
        neg_mask = gene_values.isin(eligible_negative_labels) & (gene_values != target_gene)
    else:
        raise ValueError(
            "negative_mode must be one of "
            "{'all', 'selected_plus_non_targeting', 'non_targeting'} "
            "or aliases {'all_other_cells', 'selected_genes_plus_non_targeting', 'current_plus_non_targeting'}"
        )

    select_mask = pos_mask | neg_mask
    adata_sub = adata[select_mask].copy()
    y = (adata_sub.obs[gene_col].astype(str) == target_gene).astype(np.int8).values
    X = _to_model_matrix(adata_sub.X)
    return X, y, adata_sub.obs_names.to_numpy()


def get_model(
    model_type: str,
    random_state: int = 42,
    model_params: Optional[Dict] = None,
):
    model_type = model_type.lower()
    model_params = dict(model_params or {})

    if model_type == "xgboost":
        params = {"random_state": random_state}
        params.update(model_params)
        return xgb.XGBClassifier(**params)

    raise ValueError("model_type must be 'xgboost'")


def _safe_metric(metric_fn, *args, **kwargs) -> float:
    try:
        return float(metric_fn(*args, **kwargs))
    except Exception:
        return float("nan")


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    return {
        "roc_auc": _safe_metric(roc_auc_score, y_true, y_prob),
        "average_precision": _safe_metric(average_precision_score, y_true, y_prob),
        "f1": _safe_metric(f1_score, y_true, y_pred),
        "accuracy": _safe_metric(accuracy_score, y_true, y_pred),
        "precision": _safe_metric(precision_score, y_true, y_pred, zero_division=0),
        "recall": _safe_metric(recall_score, y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "mcc": _safe_metric(matthews_corrcoef, y_true, y_pred),
        "prevalence": float(np.mean(y_true == 1)),
    }


def per_gene_cv(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
    model_type: str = "xgboost",
    model_params: Optional[Dict] = None,
    verbose: bool = False,
    gene_label: str = "",
):
    n_pos = int(y.sum())
    n_neg = int((1 - y).sum())
    min_class = min(n_pos, n_neg)
    if min_class < 2:
        raise ValueError(f"Not enough samples for CV. positives={n_pos}, negatives={n_neg}")

    n_splits_eff = min(n_splits, min_class)
    skf = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)

    pred_rows = []
    metric_rows = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        fit_params = dict(model_params or {})
        if verbose:
            device = fit_params.get("device") or fit_params.get("device_type") or "default"
            print(
                f"[per_gene] {gene_label} fold {fold}/{n_splits_eff}: "
                f"train={len(tr_idx)} valid={len(va_idx)} device={device}",
                flush=True,
            )
        t0 = time.time()
        clf = get_model(
            model_type=model_type,
            random_state=random_state,
            model_params=fit_params,
        )
        try:
            clf.fit(X[tr_idx], y[tr_idx])
        except Exception as exc:
            is_xgb_gpu_error = (
                model_type.lower() == "xgboost"
                and isinstance(exc, xgb.core.XGBoostError)
                and str(fit_params.get("device", "")).lower().startswith("cuda")
            )
            if is_xgb_gpu_error:
                print(
                    f"[per_gene] {gene_label} fold {fold}/{n_splits_eff}: "
                    f"CUDA XGBoost failed ({exc}); retrying this fold on CPU.",
                    flush=True,
                )
                retry_params = dict(fit_params)
                retry_params["device"] = "cpu"
                clf = get_model(
                    model_type=model_type,
                    random_state=random_state,
                    model_params=retry_params,
                )
                clf.fit(X[tr_idx], y[tr_idx])
            else:
                raise

        y_true = y[va_idx]
        y_prob = clf.predict_proba(X[va_idx])[:, 1]
        fold_metrics = compute_binary_metrics(y_true=y_true, y_prob=y_prob)
        if verbose:
            print(
                f"[per_gene] {gene_label} fold {fold}/{n_splits_eff}: "
                f"auprc={fold_metrics['average_precision']:.4f} "
                f"roc_auc={fold_metrics['roc_auc']:.4f} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

        metric_rows.append(
            {
                "fold": fold,
                "n_train": int(len(tr_idx)),
                "n_valid": int(len(va_idx)),
                "n_pos_valid": int(y_true.sum()),
                "n_neg_valid": int((1 - y_true).sum()),
                **fold_metrics,
            }
        )
        pred_rows.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "sample_id": sample_ids[va_idx],
                    "y_true": y_true,
                    "y_pred_proba": y_prob,
                }
            )
        )

    preds_df = pd.concat(pred_rows, ignore_index=True)
    fold_metrics_df = pd.DataFrame(metric_rows)

    oof_metrics = compute_binary_metrics(
        y_true=preds_df["y_true"].values,
        y_prob=preds_df["y_pred_proba"].values,
    )
    return preds_df, fold_metrics_df, oof_metrics, n_splits_eff, n_pos, n_neg


def _default_run_name(
    cell_line_name: str,
    model_type: str,
    negative_mode: str,
    top_n_genes: int,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{cell_line_name}_{model_type}_{negative_mode}_top{top_n_genes}"


def run_per_gene_benchmark(
    adata,
    genes: List[str],
    output_root: Path,
    input_file: str,
    cell_line_name: str,
    model_type: str = "xgboost",
    gene_col: str = "gene",
    negative_mode: str = "all",
    non_target_label: str = "non-targeting",
    n_splits: int = 5,
    random_state: int = 42,
    model_params: Optional[Dict] = None,
    run_name: Optional[str] = None,
    verbose: bool = True,
    condition_col: Optional[str] = None,
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = run_name or _default_run_name(cell_line_name, model_type, negative_mode, len(genes))
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    all_gene_summaries = []
    all_fold_metrics = []
    all_condition_metrics = []

    condition_by_cell = None
    if condition_col is not None:
        if condition_col not in adata.obs.columns:
            raise ValueError(f"condition_col={condition_col!r} not found in adata.obs")
        condition_by_cell = adata.obs[condition_col].astype(str)

    for gene_idx, gene in enumerate(genes, start=1):
        X, y, sample_ids = build_gene_dataset(
            adata=adata,
            target_gene=gene,
            gene_col=gene_col,
            negative_mode=negative_mode,
            non_target_label=non_target_label,
            selected_genes=genes,
        )
        if verbose:
            matrix_type = "sparse" if issparse(X) else "dense"
            print(
                f"[per_gene] {gene_idx}/{len(genes)} {gene}: "
                f"samples={len(y)} positives={int(y.sum())} negatives={int((1 - y).sum())} "
                f"matrix={matrix_type} shape={X.shape}",
                flush=True,
            )
        preds_df, fold_metrics_df, oof_metrics, n_splits_eff, n_pos, n_neg = per_gene_cv(
            X=X,
            y=y,
            sample_ids=sample_ids,
            n_splits=n_splits,
            random_state=random_state,
            model_type=model_type,
            model_params=model_params,
            verbose=verbose,
            gene_label=f"{gene_idx}/{len(genes)} {gene}",
        )

        if condition_by_cell is not None:
            preds_df["condition"] = condition_by_cell.reindex(preds_df["sample_id"]).to_numpy()
            for condition, cond_df in preds_df.groupby("condition", dropna=False):
                y_true_cond = cond_df["y_true"].to_numpy()
                y_prob_cond = cond_df["y_pred_proba"].to_numpy()
                cond_metrics = compute_binary_metrics(y_true=y_true_cond, y_prob=y_prob_cond)
                all_condition_metrics.append(
                    {
                        "gene": gene,
                        "condition": condition,
                        "n_samples": int(cond_df.shape[0]),
                        "n_positives": int(y_true_cond.sum()),
                        "n_negatives": int((1 - y_true_cond).sum()),
                        **{f"oof_{k}": float(v) for k, v in cond_metrics.items()},
                    }
                )

        preds_df.to_csv(run_dir / f"predictions_{gene}.csv", index=False)
        fold_metrics_df.to_csv(run_dir / f"fold_metrics_{gene}.csv", index=False)
        if verbose:
            print(
                f"[per_gene] {gene_idx}/{len(genes)} {gene}: "
                f"OOF AUPRC={oof_metrics['average_precision']:.4f} "
                f"ROC_AUC={oof_metrics['roc_auc']:.4f}",
                flush=True,
            )

        metric_cols = [c for c in fold_metrics_df.columns if c not in {"fold", "n_train", "n_valid", "n_pos_valid", "n_neg_valid"}]
        summary_row = {
            "gene": gene,
            "n_samples": int(len(y)),
            "n_positives": int(n_pos),
            "n_negatives": int(n_neg),
            "n_splits_used": int(n_splits_eff),
        }
        for m in metric_cols:
            summary_row[f"{m}_mean"] = float(fold_metrics_df[m].mean())
            summary_row[f"{m}_std"] = float(fold_metrics_df[m].std(ddof=0))
        for k, v in oof_metrics.items():
            summary_row[f"oof_{k}"] = float(v)

        all_gene_summaries.append(summary_row)
        all_fold_metrics.append(fold_metrics_df.assign(gene=gene))

    summary_df = pd.DataFrame(all_gene_summaries).sort_values("oof_average_precision", ascending=False)
    all_folds_df = pd.concat(all_fold_metrics, ignore_index=True)

    summary_df.to_csv(run_dir / "metric_summary_per_gene.csv", index=False)
    all_folds_df.to_csv(run_dir / "metric_summary_per_gene_folds.csv", index=False)
    if all_condition_metrics:
        condition_df = pd.DataFrame(all_condition_metrics).sort_values(
            ["gene", "condition"], kind="stable"
        )
        condition_df.to_csv(run_dir / "metric_summary_per_gene_by_condition.csv", index=False)

    metadata = {
        "run_name": run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": input_file,
        "cell_line_name": cell_line_name,
        "model_type": model_type,
        "model_params_override": model_params,
        "gene_col": gene_col,
        "negative_mode": negative_mode,
        "non_target_label": non_target_label,
        "condition_col": condition_col,
        "n_splits_requested": n_splits,
        "random_state": random_state,
        "top_n_genes": len(genes),
        "genes": genes,
        "n_cells_total": int(adata.n_obs),
        "n_features_total": int(adata.n_vars),
        "cv_split_note": "StratifiedKFold with shuffle=True and fixed random_state per gene-specific dataset.",
        "files": {
            "summary": "metric_summary_per_gene.csv",
            "fold_summary": "metric_summary_per_gene_folds.csv",
            "condition_summary": "metric_summary_per_gene_by_condition.csv" if all_condition_metrics else None,
            "per_gene_predictions_pattern": "predictions_<gene>.csv",
            "per_gene_fold_metrics_pattern": "fold_metrics_<gene>.csv",
        },
    }
    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return summary_df, all_folds_df, run_dir, metadata
