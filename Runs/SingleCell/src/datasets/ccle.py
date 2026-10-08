"""CCLE helpers: log-TMB regression CV + per-cell multitask mutation prediction."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    average_precision_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold

import config
from models.multitask import MultiTaskConfig, _fit_one_fold
from models.presets import MODEL_PRESETS


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
# Coding mutations, as for TCGA (missense, nonsense, nonstop, frameshift, in-frame, start codon, splice site).
# DepMap's VariantInfo uses MAF-like names up to 23Q2 (which also list SILENT mutations) and
# sequence-ontology terms joined by "&" from 23Q4 on.
CODING_VARIANTS = {
    "MISSENSE", "NONSENSE", "NONSTOP", "FRAME_SHIFT_DEL", "FRAME_SHIFT_INS", "IN_FRAME_DEL", "IN_FRAME_INS",
    "START_CODON_SNP", "START_CODON_DEL", "START_CODON_INS", "SPLICE_SITE",
    "missense_variant", "stop_gained", "stop_lost", "start_lost", "frameshift_variant", "inframe_insertion",
    "inframe_deletion", "protein_altering_variant", "splice_acceptor_variant", "splice_donor_variant",
}


def load_coding_mutations(mutations_csv: Path) -> pd.DataFrame:
    """DepMap mutation rows in a coding class, with ModelID and HugoSymbol."""
    mut_df = pd.read_csv(mutations_csv, low_memory=False)
    if "ModelID" not in mut_df.columns and "Model_ID" in mut_df.columns:
        mut_df = mut_df.rename(columns={"Model_ID": "ModelID"})
    coding = mut_df["VariantInfo"].astype(str).str.split("&").apply(lambda terms: any(t in CODING_VARIANTS for t in terms))
    valid = mut_df[coding].dropna(subset=["HugoSymbol", "ModelID"]).copy()
    valid["ModelID"] = valid["ModelID"].astype(str)
    valid["HugoSymbol"] = valid["HugoSymbol"].astype(str)
    return valid


def load_ccle_adata(
    adata_path: Path,
    mutations_csv: Path,
    model_column: str = "Model_ID",
    preprocess: bool = True,
) -> "sc.AnnData":
    """Load CCLE expression and attach mutation-derived columns."""
    adata = sc.read_h5ad(adata_path)

    if preprocess:
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.normalize_total(adata, target_sum=config.PREPROCESS_TARGET_SUM)
        sc.pp.log1p(adata)

    valid = load_coding_mutations(mutations_csv)

    model_tmb = dict(valid.groupby("ModelID").size())
    model_gene_n = dict(valid[["ModelID", "HugoSymbol"]].drop_duplicates().groupby("ModelID").size())

    adata.obs[model_column] = adata.obs[model_column].astype(str)
    adata.obs["model_TMB"] = adata.obs[model_column].map(model_tmb).astype(float)
    adata.obs["model_mut_gene_n"] = adata.obs[model_column].map(model_gene_n).astype(float)
    adata.obs["log_model_TMB"] = np.log1p(
        pd.to_numeric(adata.obs["model_TMB"], errors="coerce").clip(lower=0)
    )

    adata = adata[~adata.obs["model_TMB"].isna()].copy()
    return adata


def select_top_mutated_genes(
    mutations_csv: Path,
    top_n: int,
    min_models: int = 20,
) -> List[str]:
    """Return the genes with the most distinct mutated models (>= min_models)."""
    valid = load_coding_mutations(mutations_csv)
    counts = (
        valid[["ModelID", "HugoSymbol"]]
        .drop_duplicates()
        .groupby("HugoSymbol")["ModelID"]
        .nunique()
        .sort_values(ascending=False)
    )
    counts = counts[counts >= int(min_models)]
    return counts.head(int(top_n)).index.astype(str).tolist()


# --------------------------------------------------------------------------- #
# Regression: log-TMB per cell, CV on unique models
# --------------------------------------------------------------------------- #
def _make_regressor(kind: str, model_params: Optional[dict] = None):
    model_params = dict(model_params or {})
    if kind == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(**model_params)
    raise ValueError(f"Unknown regressor kind: {kind!r}")


def run_log_tmb_cv(
    adata,
    out_dir: Path,
    model_column: str = "Model_ID",
    n_splits: int = 5,
    seed: int = 42,
    regressor: str = "xgboost",
    model_params: Optional[dict] = None,
) -> pd.DataFrame:
    """Predict log(TMB) per cell with 5-fold CV at the model level."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = csr_matrix(adata.X, dtype=np.float32)
    cell_models = adata.obs[model_column].astype(str).to_numpy()
    tmb_per_model = (
        adata.obs[[model_column, "log_model_TMB"]]
        .drop_duplicates()
        .set_index(model_column)["log_model_TMB"]
    )
    y_all = adata.obs[model_column].map(tmb_per_model.to_dict()).to_numpy(dtype=float)
    unique_models = np.array(list(tmb_per_model.index))

    y_pred = np.zeros_like(y_all, dtype=float)
    rows = []

    device = "?"
    if isinstance(model_params, dict):
        device = str(model_params.get("device", "cpu"))
    print(
        f"  [log-TMB] regressor={regressor} device={device} "
        f"models={unique_models.size} cells={X.shape[0]} features={X.shape[1]} folds={n_splits}"
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(kf.split(unique_models), start=1):
        tr_mask = np.isin(cell_models, unique_models[tr])
        te_mask = np.isin(cell_models, unique_models[te])
        if int(tr_mask.sum()) == 0 or int(te_mask.sum()) == 0:
            print(f"  [log-TMB] fold {fold}/{n_splits}: skipped (empty split)")
            continue

        t0 = time.time()
        print(
            f"  [log-TMB] fold {fold}/{n_splits}: fitting on "
            f"{int(tr_mask.sum())} cells ({int(np.isin(unique_models, unique_models[tr]).sum())} models) "
            f"-> predicting on {int(te_mask.sum())} cells "
            f"({int(np.isin(unique_models, unique_models[te]).sum())} models)..."
        )
        clf = _make_regressor(kind=regressor, model_params=model_params)
        clf.fit(X[tr_mask], y_all[tr_mask])
        y_pred[te_mask] = clf.predict(X[te_mask])

        mse = mean_squared_error(y_all[te_mask], y_pred[te_mask])
        rmse = float(np.sqrt(mse))
        pear = float(pearsonr(y_all[te_mask], y_pred[te_mask])[0])
        spear = float(spearmanr(y_all[te_mask], y_pred[te_mask])[0])
        r2 = float(r2_score(y_all[te_mask], y_pred[te_mask]))
        dt = time.time() - t0
        print(
            f"  [log-TMB] fold {fold}/{n_splits}: rmse={rmse:.3f} pearson={pear:.3f} "
            f"spearman={spear:.3f} r2={r2:.3f} ({dt:.1f}s)"
        )
        rows.append({"fold": int(fold), "rmse": rmse, "pearson": pear, "spearman": spear, "r2": r2})

    fold_df = pd.DataFrame(rows)
    fold_df.to_csv(out_dir / "log_tmb_fold_metrics.csv", index=False)

    # Cell-level and model-level CSVs (matches old bundle).
    cell_df = pd.DataFrame({
        adata.obs_names.name or "cell_id": adata.obs_names.to_numpy(),
        model_column: cell_models,
        "log_model_TMB": y_all,
        "pred_log_model_TMB": y_pred,
        "model_TMB": adata.obs["model_TMB"].to_numpy(),
    })
    cell_df.to_csv(out_dir / f"cell_level_{regressor}.csv", index=False)

    model_df = (
        cell_df.drop(columns=[adata.obs_names.name or "cell_id"])
        .groupby(model_column)
        .mean(numeric_only=True)
        .reset_index()
    )
    model_df.to_csv(out_dir / f"model_level_{regressor}.csv", index=False)

    meta = pd.DataFrame({
        "field": ["model_column", "true_log_column", "pred_log_column"],
        "value": [model_column, "log_model_TMB", "pred_log_model_TMB"],
    })
    meta.to_csv(out_dir / f"meta_{regressor}.csv", index=False)

    return fold_df


# --------------------------------------------------------------------------- #
# Multitask classifier: top-N mutated genes
# --------------------------------------------------------------------------- #
def run_multitask_mutation_cv(
    adata,
    mutations_csv: Path,
    out_dir: Path,
    top_n: int,
    preset_name: str = "simple_wide_run",
    model_column: str = "Model_ID",
    n_splits: int = 5,
    seed: int = 42,
    min_models_per_gene: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Multitask NN: per-gene mutation probabilities with model-level CV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    top_genes = select_top_mutated_genes(mutations_csv, top_n=top_n, min_models=min_models_per_gene)
    if not top_genes:
        raise RuntimeError(f"No genes selected for top_n={top_n}, min_models_per_gene={min_models_per_gene}")

    pairs = load_coding_mutations(mutations_csv)[["ModelID", "HugoSymbol"]].drop_duplicates()

    cell_models = adata.obs[model_column].astype(str).to_numpy()
    Y = np.zeros((adata.n_obs, len(top_genes)), dtype=np.float32)
    for j, g in enumerate(top_genes):
        mut_models = pairs.loc[pairs["HugoSymbol"] == g, "ModelID"].astype(str).to_numpy()
        Y[:, j] = np.isin(cell_models, mut_models).astype(np.float32)

    X = csr_matrix(adata.X, dtype=np.float32)

    params = dict(MODEL_PRESETS[preset_name])
    params.pop("non_target_subsample", None)
    cfg = MultiTaskConfig(**{k: v for k, v in params.items() if hasattr(MultiTaskConfig, k)})

    unique_models = np.unique(cell_models)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_rows = []
    oof_probs = np.full((adata.n_obs, len(top_genes)), np.nan, dtype=np.float32)
    oof_fold = np.full(adata.n_obs, -1, dtype=np.int32)

    print(
        f"  [multitask-top{top_n}] preset={preset_name} "
        f"genes={len(top_genes)} models={unique_models.size} cells={adata.n_obs} "
        f"features={adata.n_vars} folds={n_splits}"
    )

    for fold, (tr, te) in enumerate(kf.split(unique_models), start=1):
        tr_mask = np.isin(cell_models, unique_models[tr])
        te_mask = np.isin(cell_models, unique_models[te])
        if int(tr_mask.sum()) == 0 or int(te_mask.sum()) == 0:
            print(f"  [multitask-top{top_n}] fold {fold}/{n_splits}: skipped (empty split)")
            continue

        t0 = time.time()
        print(
            f"  [multitask-top{top_n}] fold {fold}/{n_splits}: fitting on "
            f"{int(tr_mask.sum())} cells -> predicting on {int(te_mask.sum())} cells..."
        )
        _, probs, _emb, _hist = _fit_one_fold(
            X_train=X[tr_mask],
            Y_train=Y[tr_mask],
            X_valid=X[te_mask],
            Y_valid=Y[te_mask],
            cfg=cfg,
            random_state=seed + fold,
            verbose=True,
            fold_label=f"CCLE top{top_n} fold {fold}",
        )
        oof_probs[te_mask] = probs
        oof_fold[te_mask] = fold

        gene_aucs = []
        for j, g in enumerate(top_genes):
            yt = Y[te_mask, j].astype(int)
            yp = probs[:, j]
            if np.unique(yt).size < 2:
                continue
            ap = float(average_precision_score(yt, yp))
            roc = float(roc_auc_score(yt, yp))
            gene_aucs.append(ap)
            fold_rows.append({
                "fold": int(fold),
                "gene": g,
                "prevalence": float(np.mean(yt)),
                "roc_auc": roc,
                "auprc": ap,
            })
        dt = time.time() - t0
        mean_ap = float(np.mean(gene_aucs)) if gene_aucs else float("nan")
        print(
            f"  [multitask-top{top_n}] fold {fold}/{n_splits}: "
            f"mean AUPRC across {len(gene_aucs)} testable genes = {mean_ap:.3f} ({dt:.1f}s)"
        )

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(out_dir / f"multitask_fold_metrics_{top_n}.csv", index=False)

    summary = (
        fold_metrics.groupby("gene", as_index=False)
        .agg(prevalence_mean=("prevalence", "mean"),
             auprc_mean=("auprc", "mean"),
             auprc_std=("auprc", "std"),
             roc_auc_mean=("roc_auc", "mean"))
        .sort_values("auprc_mean", ascending=False)
    )
    summary.to_csv(out_dir / f"multitask_summary_{top_n}.csv", index=False)
    return fold_metrics, summary
