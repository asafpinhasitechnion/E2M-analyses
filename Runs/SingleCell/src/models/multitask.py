from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    train_test_split,
)


def infer_cell_line_name(input_filename: str) -> str:
    stem = Path(input_filename).stem
    parts = stem.split("_")
    name = "_".join(parts[:2]) if len(parts) >= 2 else stem
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return name or "unknown_cell_line"


def _safe_metric(metric_fn, *args, **kwargs) -> float:
    try:
        return float(metric_fn(*args, **kwargs))
    except Exception:
        return float("nan")


def compute_binary_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    return {
        "roc_auc": _safe_metric(roc_auc_score, y_true, y_prob),
        # No AUPRC without positives (sklearn would return 0)
        "average_precision": _safe_metric(average_precision_score, y_true, y_prob) if y_true.any() else float("nan"),
        "f1": _safe_metric(f1_score, y_true, y_pred),
        "accuracy": _safe_metric(accuracy_score, y_true, y_pred),
        "precision": _safe_metric(precision_score, y_true, y_pred, zero_division=0),
        "recall": _safe_metric(recall_score, y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "mcc": _safe_metric(matthews_corrcoef, y_true, y_pred),
        "prevalence": float(np.mean(y_true == 1)),
    }


def _to_dense_array(x):
    return x.toarray() if issparse(x) else np.asarray(x)


def _batches(X, Y, batch_size: int, shuffle: bool = False):
    """Mini-batches as tensors. X stays sparse; only one batch at a time is made dense."""
    order = np.random.permutation(X.shape[0]) if shuffle else np.arange(X.shape[0])
    for start in range(0, len(order), batch_size):
        rows = order[start:start + batch_size]
        yield torch.from_numpy(_to_dense_array(X[rows])), torch.from_numpy(Y[rows])


def get_eligible_perturbations(
    adata,
    gene_col: str = "gene",
    min_cells_per_gene: int = 300,
    exclude_labels: Tuple[str, ...] = (),
) -> List[str]:
    """Perturbation names with at least ``min_cells_per_gene`` cells in ``adata`` (exclude_labels excluded from counts)."""
    counts = adata.obs[gene_col].astype(str).value_counts()
    for label in exclude_labels:
        if label in counts.index:
            counts = counts.drop(label)
    return counts[counts >= min_cells_per_gene].index.tolist()


def build_multitask_dataset(
    adata,
    gene_col: str = "gene",
    min_cells_per_gene: int = 300,
    control_labels: Tuple[str, ...] = ("non-targeting", "control"),
    non_target_subsample: Optional[int] = None,
    cv_group_col: Optional[str] = None,
    stratify_col: Optional[str] = None,
    random_state: int = 42,
    verbose: bool = True,
):
    """Build X, Y: eligible targets (full cell coverage) plus rows labeled in ``control_labels``."""
    if cv_group_col is not None and cv_group_col not in adata.obs.columns:
        raise ValueError(f"cv_group_col={cv_group_col!r} not found in adata.obs")

    perturbation_series = adata.obs[gene_col].astype(str)

    genes = get_eligible_perturbations(
        adata=adata,
        gene_col=gene_col,
        min_cells_per_gene=min_cells_per_gene,
        exclude_labels=control_labels,
    )
    if len(genes) == 0:
        raise ValueError("No perturbations passed the min_cells_per_gene filter.")

    if verbose:
        print(
            f"[multitask] Eligible targets: {len(genes)} perturbations "
            f"(min_cells_per_gene={min_cells_per_gene}); adata n_obs={adata.n_obs}",
            flush=True,
        )

    ctl_set = set(control_labels)
    selected_labels = set(genes) | ctl_set
    select_mask = perturbation_series.isin(selected_labels).values

    adata_sub = adata[select_mask].copy()
    perturbation_series = adata_sub.obs[gene_col].astype(str)
    is_ctl = perturbation_series.isin(ctl_set).values
    n_tgt = int((~is_ctl).sum())
    n_nt = int(is_ctl.sum())
    if verbose:
        print(
            f"[multitask] After label filter: n_cells={adata_sub.n_obs} "
            f"(perturbation={n_tgt}, control={n_nt})",
            flush=True,
        )

    if non_target_subsample is not None:
        if non_target_subsample < 1:
            raise ValueError("non_target_subsample must be >= 1 when set.")
        is_nt = perturbation_series.isin(ctl_set).values
        nt_pos = np.flatnonzero(is_nt)
        tgt_pos = np.flatnonzero(~is_nt)
        n_nt = int(nt_pos.size)
        n_take = min(int(non_target_subsample), n_nt)
        rng = np.random.RandomState(random_state)
        if n_take < n_nt:
            picked = rng.choice(nt_pos, size=n_take, replace=False)
            keep_rows = np.sort(np.concatenate([tgt_pos, picked]))
            adata_sub = adata_sub[keep_rows].copy()
            perturbation_series = adata_sub.obs[gene_col].astype(str)
        is_ctl2 = perturbation_series.isin(ctl_set).values
        n_tgt2 = int((~is_ctl2).sum())
        n_nt2 = int(is_ctl2.sum())
        if verbose:
            print(
                f"[multitask] After non-target subsample (cap={non_target_subsample}): "
                f"n_cells={adata_sub.n_obs} (perturbation={n_tgt2}, control={n_nt2})",
                flush=True,
            )

    X = csr_matrix(adata_sub.X, dtype=np.float32)
    sample_ids = adata_sub.obs_names.to_numpy()

    Y = np.zeros((adata_sub.n_obs, len(genes)), dtype=np.float32)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    for i, g in enumerate(perturbation_series.values):
        idx = gene_to_idx.get(g)
        if idx is not None:
            Y[i, idx] = 1.0

    prevalence = Y.mean(axis=0)
    n_cells_final = int(adata_sub.n_obs)
    if verbose:
        print(
            f"[multitask] Design matrix: X {X.shape}, Y {Y.shape} (n_cells={n_cells_final})",
            flush=True,
        )
    cv_groups: Optional[np.ndarray] = None
    if cv_group_col is not None:
        cv_groups = adata_sub.obs[cv_group_col].astype(str).to_numpy()
    stratify_y: Optional[np.ndarray] = None
    if stratify_col is not None:
        if stratify_col not in adata_sub.obs.columns:
            raise ValueError(f"stratify_col={stratify_col!r} not found in adata.obs")
        stratify_y = pd.factorize(adata_sub.obs[stratify_col].astype(str))[0].astype(np.int64)
    return X, Y, sample_ids, genes, prevalence, n_cells_final, cv_groups, stratify_y


# Map values equal to these strings are not gene columns (e.g. placeholder ``control``).
_MULTIGEN_SKIP_VALUES = frozenset({"control"})


def _multigene_targets(
    pert_label: str,
    perturbation_to_genes: Mapping[str, Sequence[str]],
) -> List[str]:
    raw = perturbation_to_genes.get(pert_label)
    if not raw:
        return []
    return [g for g in raw if g not in _MULTIGEN_SKIP_VALUES]


def build_multitask_dataset_multigene(
    adata,
    gene_col: str,
    perturbation_to_genes: Mapping[str, Sequence[str]],
    min_cells_per_gene: int = 300,
    control_labels: Tuple[str, ...] = ("non-targeting", "control"),
    non_target_subsample: Optional[int] = None,
    cv_group_col: Optional[str] = None,
    stratify_col: Optional[str] = None,
    random_state: int = 42,
    verbose: bool = True,
):
    """Multi-label Y: columns are **gene symbols**; each row can have multiple 1s."""
    if cv_group_col is not None and cv_group_col not in adata.obs.columns:
        raise ValueError(f"cv_group_col={cv_group_col!r} not found in adata.obs")

    ctl_set = frozenset(control_labels)
    perturbation_series = adata.obs[gene_col].astype(str)

    from collections import Counter

    gene_counts: Counter = Counter()
    for p in perturbation_series.values:
        ps = str(p)
        if ps in ctl_set:
            continue
        for g in _multigene_targets(ps, perturbation_to_genes):
            gene_counts[g] += 1

    genes = sorted([g for g, c in gene_counts.items() if c >= min_cells_per_gene])
    if len(genes) == 0:
        raise ValueError("No gene symbols passed the min_cells_per_gene filter (multigene map).")

    if verbose:
        print(
            f"[multitask] Eligible gene targets: {len(genes)} symbols "
            f"(min_cells_per_gene={min_cells_per_gene}); adata n_obs={adata.n_obs}",
            flush=True,
        )

    eligible_set = set(genes)
    select_mask = np.zeros(adata.n_obs, dtype=bool)
    for i, p in enumerate(perturbation_series.values):
        ps = str(p)
        if ps in ctl_set:
            select_mask[i] = True
            continue
        rt = set(_multigene_targets(ps, perturbation_to_genes))
        if rt & eligible_set:
            select_mask[i] = True

    adata_sub = adata[select_mask].copy()
    perturbation_series = adata_sub.obs[gene_col].astype(str)
    is_ctl = np.array([str(p) in ctl_set for p in perturbation_series.values], dtype=bool)
    n_tgt = int((~is_ctl).sum())
    n_nt = int(is_ctl.sum())
    if verbose:
        print(
            f"[multitask] After label filter: n_cells={adata_sub.n_obs} "
            f"(with mapped targets={n_tgt}, control={n_nt})",
            flush=True,
        )

    if non_target_subsample is not None:
        if non_target_subsample < 1:
            raise ValueError("non_target_subsample must be >= 1 when set.")
        is_nt = is_ctl
        nt_pos = np.flatnonzero(is_nt)
        tgt_pos = np.flatnonzero(~is_nt)
        n_nt = int(nt_pos.size)
        n_take = min(int(non_target_subsample), n_nt)
        rng = np.random.RandomState(random_state)
        if n_take < n_nt:
            picked = rng.choice(nt_pos, size=n_take, replace=False)
            keep_rows = np.sort(np.concatenate([tgt_pos, picked]))
            adata_sub = adata_sub[keep_rows].copy()
            perturbation_series = adata_sub.obs[gene_col].astype(str)
            is_ctl = np.array([str(p) in ctl_set for p in perturbation_series.values], dtype=bool)
        n_tgt2 = int((~is_ctl).sum())
        n_nt2 = int(is_ctl.sum())
        if verbose:
            print(
                f"[multitask] After non-target subsample (cap={non_target_subsample}): "
                f"n_cells={adata_sub.n_obs} (with targets={n_tgt2}, control={n_nt2})",
                flush=True,
            )

    X = csr_matrix(adata_sub.X, dtype=np.float32)
    sample_ids = adata_sub.obs_names.to_numpy()

    Y = np.zeros((adata_sub.n_obs, len(genes)), dtype=np.float32)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    for i, p in enumerate(perturbation_series.values):
        ps = str(p)
        if ps in ctl_set:
            continue
        for g in _multigene_targets(ps, perturbation_to_genes):
            idx = gene_to_idx.get(g)
            if idx is not None:
                Y[i, idx] = 1.0

    prevalence = Y.mean(axis=0)
    n_cells_final = int(adata_sub.n_obs)
    if verbose:
        print(
            f"[multitask] Design matrix: X {X.shape}, Y {Y.shape} (n_cells={n_cells_final})",
            flush=True,
        )
    cv_groups: Optional[np.ndarray] = None
    if cv_group_col is not None:
        cv_groups = adata_sub.obs[cv_group_col].astype(str).to_numpy()
    stratify_y: Optional[np.ndarray] = None
    if stratify_col is not None:
        if stratify_col not in adata_sub.obs.columns:
            raise ValueError(f"stratify_col={stratify_col!r} not found in adata.obs")
        stratify_y = pd.factorize(adata_sub.obs[stratify_col].astype(str))[0].astype(np.int64)
    return X, Y, sample_ids, genes, prevalence, n_cells_final, cv_groups, stratify_y


class MultiTaskPerturbationNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [512, 256]
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(prev, output_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.head(z)

    def encode(self, x):
        return self.encoder(x)


@dataclass
class MultiTaskConfig:
    hidden_dims: Tuple[int, ...] = (256, 128)
    dropout: float = 0.5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    batch_size: int = 1024
    epochs: int = 80
    early_stopping_patience: int = 6
    val_split: float = 0.2
    use_pos_weight: bool = False
    positive_label_bias: float = 1.0
    max_pos_weight: float = 50.0
    min_pos_weight: float = 1.0
    use_lr_scheduler: bool = True
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 2
    min_lr: float = 1e-6
    gradient_clip_norm: float = 1.0
    early_stopping_metric: str = "val_loss"  # one of {"val_loss", "val_auprc", "val_roc_auc"}
    decision_threshold: float = 0.5
    input_scaling: str = "none"  # one of {"none", "zscore"}
    scaling_eps: float = 1e-6


def _fit_zscore_scaler(X: np.ndarray, eps: float = 1e-6):
    mean = X.mean(axis=0, keepdims=True).astype(np.float32)
    std = X.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, np.float32(eps))
    return mean, std


def _apply_input_scaling(
    X_train_split: np.ndarray,
    X_val_split: np.ndarray,
    X_valid: np.ndarray,
    cfg: MultiTaskConfig,
):
    mode = str(cfg.input_scaling).lower()
    if mode in {"none", ""}:
        return X_train_split, X_val_split, X_valid
    if mode == "zscore":
        mean, std = _fit_zscore_scaler(X_train_split, eps=cfg.scaling_eps)
        return (
            ((X_train_split - mean) / std).astype(np.float32),
            ((X_val_split - mean) / std).astype(np.float32),
            ((X_valid - mean) / std).astype(np.float32),
        )
    raise ValueError("input_scaling must be one of {'none', 'zscore'}")


def _build_model(input_dim: int, output_dim: int, cfg: MultiTaskConfig, device: torch.device):
    model = MultiTaskPerturbationNet(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=list(cfg.hidden_dims),
        dropout=cfg.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    return model, optimizer


def _compute_pos_weight(
    y_train: np.ndarray,
    device: torch.device,
    min_pos_weight: float = 1.0,
    max_pos_weight: float = 200.0,
):
    pos = y_train.sum(axis=0)
    neg = y_train.shape[0] - pos
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(pos > 0, neg / np.maximum(pos, 1e-8), 1.0)
    w = np.clip(w, min_pos_weight, max_pos_weight).astype(np.float32)
    return torch.from_numpy(w).to(device)


def _compute_multitask_val_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute macro validation metrics across multitask targets."""
    auprc_list = []
    rocauc_list = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = y_prob[:, i]
        # skip degenerate columns in this split
        if np.unique(yt).size < 2:
            continue
        m = compute_binary_metrics(yt, yp)
        if not np.isnan(m["average_precision"]):
            auprc_list.append(m["average_precision"])
        if not np.isnan(m["roc_auc"]):
            rocauc_list.append(m["roc_auc"])

    return {
        "val_auprc": float(np.mean(auprc_list)) if len(auprc_list) > 0 else float("nan"),
        "val_roc_auc": float(np.mean(rocauc_list)) if len(rocauc_list) > 0 else float("nan"),
    }


def _fit_one_fold(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_valid: np.ndarray,
    Y_valid: np.ndarray,
    cfg: MultiTaskConfig,
    random_state: int,
    verbose: bool = True,
    fold_label: str = "",
):
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(
            f"[multitask] {fold_label} device={device} | "
            f"cells in fold train={X_train.shape[0]}, fold valid={X_valid.shape[0]} | "
            f"early_stop_metric={cfg.early_stopping_metric}",
            flush=True,
        )
    model, optimizer = _build_model(X_train.shape[1], Y_train.shape[1], cfg, device)
    pos_weight = (
        _compute_pos_weight(
            Y_train,
            device,
            min_pos_weight=cfg.min_pos_weight,
            max_pos_weight=cfg.max_pos_weight,
        )
        if cfg.use_pos_weight
        else None
    )
    if pos_weight is not None and cfg.positive_label_bias != 1.0:
        pos_weight = pos_weight * float(cfg.positive_label_bias)
    criterion = (
        nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if pos_weight is not None
        else nn.BCEWithLogitsLoss()
    )
    scheduler = None
    if cfg.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.lr_scheduler_factor,
            patience=cfg.lr_scheduler_patience,
            min_lr=cfg.min_lr,
        )

    x_tr, x_val, y_tr, y_val = train_test_split(
        X_train,
        Y_train,
        test_size=cfg.val_split,
        random_state=random_state,
        shuffle=True,
    )

    x_tr, x_val, X_valid = _apply_input_scaling(
        X_train_split=x_tr,
        X_val_split=x_val,
        X_valid=X_valid,
        cfg=cfg,
    )

    best_state = None
    if cfg.early_stopping_metric == "val_loss":
        best_score = float("inf")
    else:
        best_score = float("-inf")
    no_improve = 0
    history_rows = []
    log_every = 1 if cfg.epochs <= 24 else (5 if cfg.epochs <= 80 else 10)

    for epoch in range(cfg.epochs):
        model.train()
        tr_loss_sum = 0.0
        tr_count = 0
        for bx, by in _batches(x_tr, y_tr, cfg.batch_size, shuffle=True):
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            if cfg.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_norm)
            optimizer.step()
            tr_loss_sum += loss.item() * bx.shape[0]
            tr_count += bx.shape[0]
        train_loss = tr_loss_sum / max(tr_count, 1)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        val_prob_chunks = []
        val_true_chunks = []
        with torch.no_grad():
            for bx, by in _batches(x_val, y_val, cfg.batch_size):
                bx = bx.to(device)
                by = by.to(device)
                logits = model(bx)
                loss = criterion(logits, by)
                val_loss_sum += loss.item() * bx.shape[0]
                val_count += bx.shape[0]
                val_prob_chunks.append(torch.sigmoid(logits).cpu().numpy())
                val_true_chunks.append(by.cpu().numpy())
        val_loss = val_loss_sum / max(val_count, 1)
        val_probs = np.vstack(val_prob_chunks)
        val_true = np.vstack(val_true_chunks)
        val_metric_dict = _compute_multitask_val_metrics(val_true, val_probs)
        if scheduler is not None:
            scheduler.step(val_loss)

        if cfg.early_stopping_metric == "val_loss":
            current_score = float(val_loss)
            improved = (current_score + 1e-6 < best_score) or best_state is None
        elif cfg.early_stopping_metric == "val_roc_auc":
            current_score = float(val_metric_dict["val_roc_auc"])
            improved = (current_score > best_score + 1e-6) or best_state is None
        else:
            current_score = float(val_metric_dict["val_auprc"])
            improved = (current_score > best_score + 1e-6) or best_state is None

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_auprc": float(val_metric_dict["val_auprc"]),
                "val_roc_auc": float(val_metric_dict["val_roc_auc"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if verbose and (
            epoch == 0
            or (epoch + 1) % log_every == 0
            or epoch == cfg.epochs - 1
        ):
            print(
                f"[multitask] {fold_label} epoch {epoch + 1}/{cfg.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_auprc={val_metric_dict['val_auprc']:.4f} "
                f"val_roc_auc={val_metric_dict['val_roc_auc']:.4f}",
                flush=True,
            )

        if improved:
            best_score = current_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= cfg.early_stopping_patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose and history_rows:
        best_ap = max(float(r["val_auprc"]) for r in history_rows)
        print(
            f"[multitask] {fold_label} training finished: {len(history_rows)} epoch(s), "
            f"best val_auprc={best_ap:.4f}",
            flush=True,
        )

    if verbose:
        print(f"[multitask] {fold_label} scoring OOF split ({X_valid.shape[0]} cells)…", flush=True)

    probs_list = []
    emb_list = []
    with torch.no_grad():
        model.eval()
        for bx, _ in _batches(X_valid, Y_valid, cfg.batch_size):
            bx = bx.to(device)
            logits = model(bx)
            probs = torch.sigmoid(logits).cpu().numpy()
            emb = model.encode(bx).cpu().numpy()
            probs_list.append(probs)
            emb_list.append(emb)

    probs_valid = np.vstack(probs_list)
    embeddings_valid = np.vstack(emb_list)
    history_df = pd.DataFrame(history_rows)
    return model, probs_valid, embeddings_valid, history_df


def _extract_head_weights_df(
    model: MultiTaskPerturbationNet,
    genes: Sequence[str],
    fold: int,
) -> pd.DataFrame:
    """Return one row per target with the full output-head weight vector."""
    weight = model.head.weight.detach().cpu().numpy().astype(np.float32)
    bias = model.head.bias.detach().cpu().numpy().astype(np.float32)
    hidden_dim = int(weight.shape[1])
    rows = []
    for idx, gene in enumerate(genes):
        row = {
            "gene": gene,
            "fold": int(fold),
            "bias": float(bias[idx]),
        }
        for h in range(hidden_dim):
            row[f"w_{h}"] = float(weight[idx, h])
        rows.append(row)
    return pd.DataFrame(rows)


def _default_run_name(cell_line_name: str, n_genes: int) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{cell_line_name}_multitask_top{n_genes}"


def run_multitask_benchmark(
    adata,
    output_root: Path,
    input_file: str,
    cell_line_name: Optional[str] = None,
    gene_col: str = "gene",
    min_cells_per_gene: int = 300,
    control_labels: Tuple[str, ...] = ("non-targeting", "control"),
    non_target_subsample: Optional[int] = None,
    cv_group_col: Optional[str] = None,
    n_splits: int = 5,
    random_state: int = 42,
    model_params: Optional[Dict] = None,
    run_name: Optional[str] = None,
    verbose: bool = True,
    perturbation_to_genes: Optional[Mapping[str, Sequence[str]]] = None,
    stratified_group_cv: bool = False,
    stratify_col: Optional[str] = None,
    condition_col: Optional[str] = None,
):
    """K-fold CV on cells (default) or group-wise CV if ``cv_group_col`` is set (e.g. library, sample)."""
    cfg_dict = dict(model_params or {})
    cfg = MultiTaskConfig(**{k: v for k, v in cfg_dict.items() if hasattr(MultiTaskConfig, k)})

    stratify_for_builder = (stratify_col or gene_col) if stratified_group_cv else None
    if stratified_group_cv and cv_group_col is None:
        raise ValueError("stratified_group_cv=True requires cv_group_col (group column for CV).")

    if perturbation_to_genes is not None:
        X, Y, sample_ids, genes, prevalence, n_selected_cells, cv_groups, stratify_y = (
            build_multitask_dataset_multigene(
                adata=adata,
                gene_col=gene_col,
                perturbation_to_genes=perturbation_to_genes,
                min_cells_per_gene=min_cells_per_gene,
                control_labels=control_labels,
                non_target_subsample=non_target_subsample,
                cv_group_col=cv_group_col,
                stratify_col=stratify_for_builder,
                random_state=random_state,
                verbose=verbose,
            )
        )
    else:
        X, Y, sample_ids, genes, prevalence, n_selected_cells, cv_groups, stratify_y = (
            build_multitask_dataset(
                adata=adata,
                gene_col=gene_col,
                min_cells_per_gene=min_cells_per_gene,
                control_labels=control_labels,
                non_target_subsample=non_target_subsample,
                cv_group_col=cv_group_col,
                stratify_col=stratify_for_builder,
                random_state=random_state,
                verbose=verbose,
            )
        )
    cell_line = cell_line_name or infer_cell_line_name(input_file)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_name = run_name or _default_run_name(cell_line, len(genes))
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(
            f"[multitask] Benchmark: {input_file} | {cell_line} | device={_dev} | "
            f"X={X.shape} | targets={len(genes)} | n_splits={n_splits} | out={run_dir}",
            flush=True,
        )

    if cv_groups is not None:
        n_grp = int(np.unique(cv_groups).size)
        if n_grp < n_splits:
            raise ValueError(
                f"cv_group_col={cv_group_col!r} has only {n_grp} distinct values but n_splits={n_splits}; "
                "reduce --n-splits or pick another column."
            )
        if stratified_group_cv:
            if stratify_y is None:
                raise RuntimeError("stratified_group_cv set but stratify_y missing (internal error).")
            scol = stratify_col or gene_col
            if verbose:
                print(
                    f"[multitask] CV: StratifiedGroupKFold on obs['{cv_group_col}'] ({n_grp} groups), "
                    f"stratify=obs['{scol}']",
                    flush=True,
                )
            fold_splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state
            )
            fold_iter = enumerate(
                fold_splitter.split(X, stratify_y, groups=cv_groups), start=1
            )
        else:
            if verbose:
                print(
                    f"[multitask] CV: GroupKFold on obs['{cv_group_col}'] ({n_grp} groups)",
                    flush=True,
                )
            fold_splitter = GroupKFold(n_splits=n_splits)
            fold_iter = enumerate(fold_splitter.split(X, groups=cv_groups), start=1)
    else:
        if verbose:
            print("[multitask] CV: KFold on cells (shuffled)", flush=True)
        fold_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_iter = enumerate(fold_splitter.split(X), start=1)

    oof_probs = np.zeros_like(Y, dtype=np.float32)
    oof_embeddings = np.zeros((X.shape[0], int(cfg.hidden_dims[-1])), dtype=np.float32)
    oof_fold_id = np.full(X.shape[0], -1, dtype=np.int32)
    fold_rows = []
    history_rows = []
    head_weight_rows = []

    for fold, (tr_idx, va_idx) in fold_iter:
        if verbose:
            print(
                f"[multitask] --- CV fold {fold}/{n_splits}: train_n={len(tr_idx)} valid_n={len(va_idx)} ---",
                flush=True,
            )
        model, probs_valid, embeddings_valid, history_df = _fit_one_fold(
            X_train=X[tr_idx],
            Y_train=Y[tr_idx],
            X_valid=X[va_idx],
            Y_valid=Y[va_idx],
            cfg=cfg,
            random_state=random_state + fold,
            verbose=verbose,
            fold_label=f"CV {fold}/{n_splits}",
        )

        fold_head_df = _extract_head_weights_df(model=model, genes=genes, fold=fold)
        fold_head_df.to_csv(run_dir / f"fold_head_weights_{fold}.csv", index=False)
        head_weight_rows.append(fold_head_df)

        oof_probs[va_idx] = probs_valid
        oof_embeddings[va_idx] = embeddings_valid
        oof_fold_id[va_idx] = fold

        y_true_fold = Y[va_idx]
        fold_gene_rows = []
        for g_idx, gene in enumerate(genes):
            metrics = compute_binary_metrics(
                y_true=y_true_fold[:, g_idx],
                y_prob=probs_valid[:, g_idx],
                threshold=cfg.decision_threshold,
            )
            fold_gene_rows.append({"gene": gene, "fold": fold, **metrics})
        fold_gene_df = pd.DataFrame(fold_gene_rows)
        fold_gene_df.to_csv(run_dir / f"fold_metrics_{fold}.csv", index=False)
        fold_rows.append(fold_gene_df)
        if verbose and "average_precision" in fold_gene_df.columns:
            m_ap = float(fold_gene_df["average_precision"].mean())
            print(
                f"[multitask] CV fold {fold}/{n_splits} holdout mean AP (per target)={m_ap:.4f}",
                flush=True,
            )

        history_df = history_df.assign(fold=fold)
        history_df.to_csv(run_dir / f"fold_training_history_{fold}.csv", index=False)
        history_rows.append(history_df)

    summary_rows = []
    for g_idx, gene in enumerate(genes):
        metrics = compute_binary_metrics(
            y_true=Y[:, g_idx],
            y_prob=oof_probs[:, g_idx],
            threshold=cfg.decision_threshold,
        )
        summary_rows.append(
            {
                "gene": gene,
                "prevalence": float(prevalence[g_idx]),
                **{f"oof_{k}": v for k, v in metrics.items()},
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("oof_average_precision", ascending=False)
    all_fold_metrics_df = pd.concat(fold_rows, ignore_index=True)
    all_history_df = pd.concat(history_rows, ignore_index=True)
    all_head_weights_df = pd.concat(head_weight_rows, ignore_index=True)
    all_head_weights_df.to_csv(run_dir / "head_weights_per_fold.csv", index=False)

    if verbose:
        mean_oof_ap = float(np.nanmean(summary_df["oof_average_precision"].astype(float).values))
        print(
            f"[multitask] OOF summary: mean oof_average_precision (per target)={mean_oof_ap:.4f}; "
            f"writing artifacts…",
            flush=True,
        )

    summary_df.to_csv(run_dir / "metric_summary_per_gene.csv", index=False)
    all_fold_metrics_df.to_csv(run_dir / "metric_summary_per_gene_folds.csv", index=False)
    all_history_df.to_csv(run_dir / "training_history_all_folds.csv", index=False)

    cell_conditions: Optional[np.ndarray] = None
    if condition_col is not None:
        if condition_col not in adata.obs.columns:
            raise ValueError(f"condition_col={condition_col!r} not found in adata.obs")
        cell_conditions = adata.obs.loc[sample_ids, condition_col].astype(str).to_numpy()
        condition_rows = []
        for g_idx, gene in enumerate(genes):
            y_g = Y[:, g_idx]
            p_g = oof_probs[:, g_idx]
            for condition in np.unique(cell_conditions):
                mask = cell_conditions == condition
                y_true_cond = y_g[mask]
                y_prob_cond = p_g[mask]
                cond_metrics = compute_binary_metrics(
                    y_true=y_true_cond,
                    y_prob=y_prob_cond,
                    threshold=cfg.decision_threshold,
                )
                condition_rows.append(
                    {
                        "gene": gene,
                        "condition": condition,
                        "n_samples": int(mask.sum()),
                        "n_positives": int(y_true_cond.sum()),
                        "n_negatives": int((1 - y_true_cond).sum()),
                        **{f"oof_{k}": v for k, v in cond_metrics.items()},
                    }
                )
        condition_df = pd.DataFrame(condition_rows)
        condition_df.to_csv(run_dir / "metric_summary_per_gene_by_condition.csv", index=False)
        if verbose:
            print(
                f"[multitask] Wrote condition-stratified OOF metrics: "
                f"{len(condition_df)} rows ({len(genes)} genes x "
                f"{condition_df['condition'].nunique()} conditions)",
                flush=True,
            )

    probs_df = pd.DataFrame(oof_probs, index=sample_ids, columns=genes)
    probs_df.to_csv(run_dir / "oof_probabilities.csv")

    true_df = pd.DataFrame(Y.astype(np.int8), index=sample_ids, columns=genes)
    true_df.to_csv(run_dir / "oof_true_labels.csv")

    emb_cols = [f"emb_{i}" for i in range(oof_embeddings.shape[1])]
    emb_df = pd.DataFrame(oof_embeddings, index=sample_ids, columns=emb_cols)
    emb_df.to_csv(run_dir / "oof_sample_embeddings.csv")

    cell_perturbation = adata.obs.loc[sample_ids, gene_col].astype(str).to_numpy()
    cell_target_count = Y.sum(axis=1).astype(np.int32)
    fold_df = pd.DataFrame(
        {
            "fold": oof_fold_id,
            "gene_or_perturbation": cell_perturbation,
            "n_targets_positive": cell_target_count,
            "is_non_targeting_control": (cell_target_count == 0),
        },
        index=sample_ids,
    )
    if cv_groups is not None:
        fold_df["cv_group"] = cv_groups
    if cell_conditions is not None:
        fold_df["condition"] = cell_conditions
    if stratified_group_cv and stratify_y is not None:
        fold_df["stratify_label_id"] = stratify_y
    fold_df.reset_index(names="sample_id").to_csv(run_dir / "oof_cell_folds.csv", index=False)

    row_label_sum = Y.sum(axis=1)
    n_cells_target = int((row_label_sum > 0).sum())
    n_cells_non_target = int((row_label_sum == 0).sum())

    metadata = {
        "run_name": run_name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_type": "multitask_perturbation",
        "input_file": input_file,
        "cell_line_name": cell_line,
        "gene_col": gene_col,
        "min_cells_per_gene": int(min_cells_per_gene),
        "control_labels": list(control_labels),
        "non_target_subsample": non_target_subsample,
        "cv_group_col": cv_group_col,
        "stratified_group_cv": bool(stratified_group_cv),
        "stratify_col": (stratify_col or gene_col) if stratified_group_cv else None,
        "condition_col": condition_col,
        "n_cv_groups": int(np.unique(cv_groups).size) if cv_groups is not None else None,
        "n_splits": int(n_splits),
        "random_state": int(random_state),
        "n_cells_selected_for_training": int(n_selected_cells),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_targets": int(len(genes)),
        "n_cells_with_eligible_perturbation": n_cells_target,
        "n_cells_non_targeting_included": n_cells_non_target,
        "targets": genes,
        "model_config": {
            "hidden_dims": list(cfg.hidden_dims),
            "dropout": cfg.dropout,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "early_stopping_patience": cfg.early_stopping_patience,
            "val_split": cfg.val_split,
            "use_pos_weight": cfg.use_pos_weight,
            "positive_label_bias": cfg.positive_label_bias,
            "decision_threshold": cfg.decision_threshold,
            "input_scaling": cfg.input_scaling,
            "scaling_eps": cfg.scaling_eps,
        },
        "artifacts": {
            "head_weights_per_fold": "head_weights_per_fold.csv",
            "head_weights_per_fold_split": "fold_head_weights_<fold>.csv",
            "condition_summary": (
                "metric_summary_per_gene_by_condition.csv" if condition_col is not None else None
            ),
        },
        "notes": [
            "Targets = perturbations with >= min_cells_per_gene in full adata; control_labels rows are negatives (optional subsample).",
            "CV: KFold on cells by default; if cv_group_col is set, GroupKFold so train/test do not share groups.",
            "If stratified_group_cv: StratifiedGroupKFold balances stratify_col classes across folds while keeping cv_group_col groups intact.",
            "head_weights_per_fold.csv stores the full output-head weight row per target for each CV fold (bias + w_0..w_{H-1}).",
        ],
        "multigene_perturbation_map": perturbation_to_genes is not None,
    }
    if perturbation_to_genes is not None:
        metadata["notes"].append(
            "Multigene mode: Y columns are gene symbols; controls = obs[gene_col] in control_labels only.",
        )
    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"[multitask] Finished. Results in {run_dir}", flush=True)

    return summary_df, all_fold_metrics_df, emb_df, run_dir, metadata
