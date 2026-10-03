"""Frangieh/Izar metrics per immune condition (control, IFN-γ, co-culture) from saved held-out predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

import config
from models.multitask import compute_binary_metrics

FRANGIEH_MULTITASK = "FrangiehIzar2021_RNA_mc800_simple_wide_run"
FRANGIEH_PER_GENE = "FrangiehIzar2021_RNA_per_gene_mc800_baseline"
CONDITION_COL = "perturbation_2"
CONTROL_LABELS = ("non-targeting", "control")


def multitask_metrics_by_condition(
    run_dir: Path,
    adata_path: Path,
    condition_col: str = CONDITION_COL,
    control_labels: tuple[str, ...] = CONTROL_LABELS,
    decision_threshold: float = 0.5,
) -> Path:
    probs_path = run_dir / "oof_probabilities.csv"
    folds_path = run_dir / "oof_cell_folds.csv"
    if not probs_path.exists():
        raise FileNotFoundError(f"Missing {probs_path}")
    if not folds_path.exists():
        raise FileNotFoundError(f"Missing {folds_path}")

    probs_df = pd.read_csv(probs_path, index_col=0)
    folds_df = pd.read_csv(folds_path)
    if not np.array_equal(probs_df.index.astype(str), folds_df["sample_id"].astype(str)):
        folds_df = folds_df.set_index("sample_id").reindex(probs_df.index).reset_index()

    adata = sc.read_h5ad(adata_path, backed="r")
    if condition_col not in adata.obs.columns:
        raise ValueError(f"{condition_col!r} not in adata.obs")
    conditions = adata.obs.loc[probs_df.index.astype(str), condition_col].astype(str).to_numpy()
    perturbations = folds_df["gene_or_perturbation"].astype(str).to_numpy()
    ctl = set(control_labels)

    genes = list(probs_df.columns)
    rows = []
    for g_idx, gene in enumerate(genes):
        y_true = np.array(
            [(p == gene) and (p not in ctl) for p in perturbations],
            dtype=np.int8,
        )
        y_prob = probs_df[gene].to_numpy(dtype=float)
        for condition in np.unique(conditions):
            mask = conditions == condition
            yt = y_true[mask]
            yp = y_prob[mask]
            m = compute_binary_metrics(yt, yp, threshold=decision_threshold)
            rows.append(
                {
                    "gene": gene,
                    "condition": condition,
                    "n_samples": int(mask.sum()),
                    "n_positives": int(yt.sum()),
                    "n_negatives": int((1 - yt).sum()),
                    **{f"oof_{k}": v for k, v in m.items()},
                }
            )

    out = run_dir / "metric_summary_per_gene_by_condition.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _metrics_row(
    gene: str,
    fold: int,
    condition: str,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict:
    m = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    return {
        "gene": gene,
        "fold": int(fold),
        "condition": condition,
        "n_samples": int(len(y_true)),
        "n_positives": int(np.sum(y_true == 1)),
        "n_negatives": int(np.sum(y_true == 0)),
        **{f"oof_{k}": v for k, v in m.items()},
    }


def multitask_metrics_by_fold(
    run_dir: Path,
    adata_path: Path,
    condition_col: str = CONDITION_COL,
    control_labels: tuple[str, ...] = CONTROL_LABELS,
    decision_threshold: float = 0.5,
) -> pd.DataFrame:
    probs_df = pd.read_csv(run_dir / "oof_probabilities.csv", index_col=0)
    folds_df = pd.read_csv(run_dir / "oof_cell_folds.csv")
    folds_df = folds_df.set_index("sample_id").reindex(probs_df.index.astype(str)).reset_index()

    adata = sc.read_h5ad(adata_path, backed="r")
    conditions = adata.obs.loc[probs_df.index.astype(str), condition_col].astype(str).to_numpy()
    perturbations = folds_df["gene_or_perturbation"].astype(str).to_numpy()
    fold_ids = folds_df["fold"].to_numpy(dtype=int)
    ctl = set(control_labels)

    rows = []
    for gene in probs_df.columns:
        y_true_all = np.array(
            [(p == gene) and (p not in ctl) for p in perturbations],
            dtype=np.int8,
        )
        y_prob_all = probs_df[gene].to_numpy(dtype=float)
        for fold in sorted(np.unique(fold_ids)):
            fold_mask = fold_ids == fold
            for condition in np.unique(conditions[fold_mask]):
                mask = fold_mask & (conditions == condition)
                rows.append(
                    _metrics_row(
                        gene=gene,
                        fold=int(fold),
                        condition=condition,
                        y_true=y_true_all[mask],
                        y_prob=y_prob_all[mask],
                        threshold=decision_threshold,
                    )
                )
    return pd.DataFrame(rows)


def per_gene_metrics_by_fold(
    run_dir: Path,
    decision_threshold: float = 0.5,
) -> pd.DataFrame:
    pred_files = sorted(run_dir.glob("predictions_*.csv"))
    if not pred_files:
        raise FileNotFoundError(f"No predictions_*.csv in {run_dir}")

    rows = []
    for path in pred_files:
        gene = path.stem.replace("predictions_", "", 1)
        preds = pd.read_csv(path)
        if "condition" not in preds.columns:
            raise ValueError(f"{path} missing 'condition' column")
        for (fold, condition), grp in preds.groupby(["fold", "condition"], dropna=False):
            rows.append(
                _metrics_row(
                    gene=gene,
                    fold=int(fold),
                    condition=str(condition),
                    y_true=grp["y_true"].to_numpy(dtype=int),
                    y_prob=grp["y_pred_proba"].to_numpy(dtype=float),
                    threshold=decision_threshold,
                )
            )
    return pd.DataFrame(rows).sort_values(["gene", "fold", "condition"], kind="stable")


def main() -> None:
    adata_path = config.DATA_DIR / config.DATASET_FILES["frangieh"]
    multitask_dir = config.OUTPUTS_ROOT / FRANGIEH_MULTITASK
    per_gene_dir = config.OUTPUTS_ROOT / FRANGIEH_PER_GENE

    threshold = 0.5
    meta_path = multitask_dir / "run_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        threshold = float(meta.get("model_config", {}).get("decision_threshold", 0.5))

    out = multitask_metrics_by_condition(multitask_dir, adata_path, decision_threshold=threshold)
    print(f"  wrote {out}")

    for run_dir, df in (
        (multitask_dir, multitask_metrics_by_fold(multitask_dir, adata_path, decision_threshold=threshold)),
        (per_gene_dir, per_gene_metrics_by_fold(per_gene_dir)),
    ):
        out = run_dir / "metric_summary_per_gene_by_condition_folds.csv"
        df.to_csv(out, index=False)
        print(f"  wrote {out} ({len(df)} rows)")
