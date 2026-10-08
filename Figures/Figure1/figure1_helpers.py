from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import pingouin as pg

mpl.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

from constants import (
    ANNOTATION_FS,
    AXIS_LABEL_FS,
    BEIGE,
    CANCER_COLORS,
    CMAP_BEIGE_PURPLE,
    CMAP_ORANGE_GREEN,
    COLORBAR_LABEL_FS,
    COLORBAR_TICK_FS,
    GOLD,
    LEGEND_FS,
    PURPLE,
    TEAL,
    TICK_FS,
    TITLE_FS,
    make_linear_cmap,
)


def _safe_relative(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def save_panel(fig, path: Path, project_root: Path, dpi: int = 300) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    print(f"saved: {_safe_relative(path, project_root)}")
    return path


def save_table(df: pd.DataFrame, path: Path, project_root: Path, **kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, **kwargs)
    print(f"saved: {_safe_relative(path, project_root)}")
    return path


def load_tmb_results(results_dir: Path):
    results_dir = Path(results_dir)
    pred = pd.read_csv(results_dir / "oof_predictions.csv")
    summary = pd.read_csv(results_dir / "summary.csv")
    fold_metrics = pd.read_csv(results_dir / "fold_metrics.csv")
    metadata_path = results_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    pred = pred.rename(columns={"sample_id": "sample", "cancer": "Cancer", "TMB_true": "TMB_raw_true", "TMB_pred": "TMB_raw_pred"})
    summary = summary.rename(columns={"cancer": "Cancer"})
    if "TMB_true" not in pred.columns:
        pred["TMB_true"] = pred["TMB_raw_true"]
    if "TMB_pred" not in pred.columns:
        pred["TMB_pred"] = pred["TMB_raw_pred"]
    return pred, summary, fold_metrics, metadata


def metric_summary(y_true, y_pred, min_n=3) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[keep], y_pred[keep]
    out = {"n": len(y_true), "spearman": np.nan, "spearman_p": np.nan, "pearson": np.nan, "pearson_p": np.nan}
    if len(y_true) >= min_n and np.std(y_true) > 0 and np.std(y_pred) > 0:
        sp = spearmanr(y_true, y_pred)
        pe = pearsonr(y_true, y_pred)
        out.update(
            {
                "spearman": float(sp.statistic),
                "spearman_p": float(sp.pvalue),
                "pearson": float(pe.statistic),
                "pearson_p": float(pe.pvalue),
            }
        )
    return out


def benjamini_hochberg_fdr(p_values) -> np.ndarray:
    """Return Benjamini-Hochberg FDR-adjusted p-values, preserving NaNs."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.isfinite(p)
    if not np.any(valid):
        return q

    valid_idx = np.flatnonzero(valid)
    order = np.argsort(p[valid])
    ordered_idx = valid_idx[order]
    ordered_p = p[ordered_idx]
    n = len(ordered_p)
    adjusted = ordered_p * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[ordered_idx] = np.clip(adjusted, 0, 1)
    return q


def compute_tmb_purity_partial_correlations(
    predictions_df: pd.DataFrame,
    purity_path: Path,
    min_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pooled and per-cancer TMB partial Spearman correlations."""
    prediction_columns = {"sample", "Cancer", "TMB_log2_true", "TMB_log2_pred"}
    missing_prediction_columns = prediction_columns.difference(predictions_df.columns)
    if missing_prediction_columns:
        raise ValueError(f"TMB predictions are missing columns: {sorted(missing_prediction_columns)}")

    purity_df = pd.read_csv(purity_path, sep="\t")
    purity_columns = {"array", "purity", "call status"}
    missing_purity_columns = purity_columns.difference(purity_df.columns)
    if missing_purity_columns:
        raise ValueError(f"ABSOLUTE purity table is missing columns: {sorted(missing_purity_columns)}")
    if purity_df["array"].duplicated().any():
        raise ValueError("ABSOLUTE purity table contains duplicate array identifiers.")
    purity_df["purity"] = pd.to_numeric(purity_df["purity"], errors="coerce")

    joined_df = predictions_df.copy()
    joined_df["array"] = joined_df["sample"].astype(str).str.replace(
        r"-(\d{2})[A-Z]$", r"-\1", regex=True
    )
    joined_df = joined_df.merge(
        purity_df[["array", "purity", "call status"]],
        on="array",
        how="inner",
        validate="one_to_one",
    ).rename(columns={"call status": "purity_call_status"})
    joined_df = joined_df.dropna(
        subset=["TMB_log2_true", "TMB_log2_pred", "purity"]
    ).copy()

    def correlation_row(cancer: str, frame: pd.DataFrame) -> dict:
        ordinary = spearmanr(frame["TMB_log2_true"], frame["TMB_log2_pred"])
        partial = pg.partial_corr(
            data=frame,
            x="TMB_log2_pred",
            y="TMB_log2_true",
            covar="purity",
            method="spearman",
            alternative="two-sided",
        )
        p_column = "p-val" if "p-val" in partial.columns else "p_val"
        partial_r = float(partial.iloc[0]["r"])
        purity_true = spearmanr(frame["purity"], frame["TMB_log2_true"])
        purity_pred = spearmanr(frame["purity"], frame["TMB_log2_pred"])
        return {
            "Cancer": cancer,
            "n": len(frame),
            "spearman": float(ordinary.statistic),
            "spearman_p": float(ordinary.pvalue),
            "purity_adjusted_spearman": partial_r,
            "purity_adjusted_p": float(partial.iloc[0][p_column]),
            "delta_after_purity_adjustment": partial_r - float(ordinary.statistic),
            "purity_vs_observed_tmb_spearman": float(purity_true.statistic),
            "purity_vs_predicted_tmb_spearman": float(purity_pred.statistic),
        }

    rows = [correlation_row("__POOLED__", joined_df)]
    rows.extend(
        correlation_row(cancer, frame)
        for cancer, frame in joined_df.groupby("Cancer", sort=True)
        if len(frame) >= min_n
    )
    summary_df = pd.DataFrame(rows)
    per_cancer = summary_df["Cancer"].ne("__POOLED__")
    summary_df.loc[per_cancer, "purity_adjusted_fdr"] = benjamini_hochberg_fdr(
        summary_df.loc[per_cancer, "purity_adjusted_p"]
    )
    summary_df["purity_adjusted_fdr_significant"] = (
        summary_df["purity_adjusted_fdr"] < 0.05
    ).astype("boolean")
    summary_df.loc[~per_cancer, "purity_adjusted_fdr_significant"] = pd.NA
    return joined_df, summary_df


def per_cancer_tmb_summary(predictions_df, min_n=10) -> pd.DataFrame:
    rows = []
    for cancer, sub in predictions_df.groupby("Cancer"):
        if len(sub) < min_n:
            continue
        row = {"Cancer": cancer, **metric_summary(sub["TMB_log2_true"], sub["TMB_log2_pred"], min_n=min_n)}
        row["mean_tmb"] = float(sub["TMB_raw_true"].mean())
        row["median_tmb"] = float(sub["TMB_raw_true"].median())
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["spearman_fdr"] = benjamini_hochberg_fdr(out["spearman_p"])
    out["pearson_fdr"] = benjamini_hochberg_fdr(out["pearson_p"])
    out["minus_log10_p"] = -np.log10(out["spearman_p"].clip(lower=1e-300))
    return out.sort_values("spearman", ascending=False).reset_index(drop=True)


def plot_tmb_scatter(
    df,
    project_root: Path,
    true_col="TMB_log2_true",
    pred_col="TMB_log2_pred",
    cancer_col="Cancer",
    figsize=(4, 3.3),
    alpha=0.55,
    point_size=26,
    cmap=CMAP_ORANGE_GREEN,
    vmin=0,
    vmax=1,
    save_path=None,
):
    sub = df[[true_col, pred_col, cancer_col]].dropna().copy()
    cancer_rho = sub.groupby(cancer_col).apply(
        lambda g: spearmanr(g[true_col], g[pred_col]).statistic if len(g) >= 10 else np.nan
    )
    sub["cancer_rho"] = sub[cancer_col].map(cancer_rho)
    x = sub[true_col].to_numpy()
    y = sub[pred_col].to_numpy()
    c = sub["cancer_rho"].to_numpy()
    ok = np.isfinite(c)

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        x[ok],
        y[ok],
        c=c[ok],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=point_size,
        alpha=alpha,
        edgecolors="black",
        linewidths=0.25,
    )
    if np.any(~ok):
        ax.scatter(x[~ok], y[~ok], color="0.75", s=point_size, alpha=alpha, edgecolors="black", linewidths=0.25)

    lo = float(np.nanmin([x.min(), y.min()]))
    hi = float(np.nanmax([x.max(), y.max()]))
    ax.plot([lo, hi], [lo, hi], "--", lw=1, color="0.25", zorder=0)
    overall = spearmanr(x, y)
    if overall.pvalue == 0:
        p_text = f"p < {np.nextafter(0, 1):.1e}"
    else:
        p_text = f"p = {overall.pvalue:.1e}"
    ax.text(
        0.03,
        0.97,
        f"Spearman rho = {overall.statistic:.2f}\n{p_text}\nn = {len(sub):,}",
        transform=ax.transAxes,
        fontsize=ANNOTATION_FS,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.25", alpha=0.8),
    )
    ax.set_xlabel("True log2(TMB + 1)", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("Predicted log2(TMB + 1)", fontsize=AXIS_LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label("Per-cancer Spearman rho", fontsize=COLORBAR_LABEL_FS, rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS)
    fig.tight_layout()
    if save_path:
        save_panel(fig, save_path, project_root)
    return cancer_rho


def plot_per_cancer_corr_p(
    df,
    project_root: Path,
    min_n=10,
    figsize=(3.4, 3.3),
    cmap=CMAP_BEIGE_PURPLE,
    vmax_logp=None,
    point_size=40,
    orientation="horizontal",
    save_path=None,
):
    """Per-cancer TMB Spearman dot plot.

    orientation="horizontal" (default): cancers on y-axis, Spearman rho on x-axis.
    orientation="vertical": cancers on x-axis, Spearman rho on y-axis.
    """
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError("orientation must be 'horizontal' or 'vertical'.")
    # Ascending rho order: horizontal runs bottom-to-top (worst at bottom),
    # vertical runs left-to-right (worst at left, best at right).
    summ = per_cancer_tmb_summary(df, min_n=min_n).sort_values(
        "spearman", ascending=True
    )
    if summ.empty:
        raise ValueError("No cancer types passed min_n filter.")
    if vmax_logp is None:
        vmax_logp = float(summ["minus_log10_p"].max())
    fig, ax = plt.subplots(figsize=figsize)
    pos = np.arange(len(summ))
    color = summ["minus_log10_p"].clip(0, vmax_logp)
    rho_ticks = np.round(np.arange(0.0, 0.81, 0.1), 1)
    if orientation == "horizontal":
        sc = ax.scatter(
            summ["spearman"],
            pos,
            c=color,
            cmap=cmap,
            vmin=0,
            vmax=vmax_logp,
            s=point_size,
            edgecolors="black",
            linewidths=0.35,
        )
        ax.set_yticks(pos)
        ax.set_yticklabels(summ["Cancer"], fontsize=TICK_FS - 1)
        ax.set_xlabel("Spearman rho", fontsize=AXIS_LABEL_FS)
        ax.set_xlim(0, 0.8)
        ax.set_xticks(rho_ticks)
        ax.tick_params(axis="x", labelsize=TICK_FS)
        ax.grid(axis="x", alpha=0.2)
    else:
        sc = ax.scatter(
            pos,
            summ["spearman"],
            c=color,
            cmap=cmap,
            vmin=0,
            vmax=vmax_logp,
            s=point_size,
            edgecolors="black",
            linewidths=0.35,
        )
        ax.set_xticks(pos)
        ax.set_xticklabels(summ["Cancer"], rotation=90, fontsize=TICK_FS - 1)
        ax.set_ylabel("Spearman rho", fontsize=AXIS_LABEL_FS)
        ax.set_ylim(0, 0.8)
        ax.set_yticks(rho_ticks)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.grid(axis="y", alpha=0.2)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.06)
    cbar.set_label("-log10(p)", fontsize=COLORBAR_LABEL_FS, rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS)
    fig.tight_layout()
    if save_path:
        save_panel(fig, save_path, project_root)
    return summ


def plot_tmb_scatter_by_cancer(
    df,
    project_root: Path,
    true_col="TMB_log2_true",
    pred_col="TMB_log2_pred",
    cancer_col="Cancer",
    n_rows=3,
    figsize=(9, 6),
    min_n=10,
    gap=1.5,
    point_size=12,
    alpha=0.6,
    save_path=None,
):
    cancer_order = df.groupby(cancer_col)[true_col].median().sort_values().index.tolist()
    row_cancers = np.array_split(cancer_order, n_rows)
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharey=False)
    if n_rows == 1:
        axes = [axes]
    color_map = {c: CANCER_COLORS[i % len(CANCER_COLORS)] for i, c in enumerate(cancer_order)}

    for ax, cancers in zip(axes, row_cancers):
        x_offset = 0.0
        xticks, labels = [], []
        row = df[df[cancer_col].isin(cancers)]
        yvals = row[pred_col].dropna().to_numpy()
        ylo, yhi = float(yvals.min()), float(yvals.max())
        ypad = max((yhi - ylo) * 0.18, 0.5)
        ax.set_ylim(ylo - 0.05 * ypad, yhi + ypad)
        yr = ax.get_ylim()[1] - ax.get_ylim()[0]

        for i, cancer in enumerate(cancers):
            sub = df[df[cancer_col] == cancer].dropna(subset=[true_col, pred_col])
            if sub.empty:
                continue
            x = sub[true_col].to_numpy()
            y = sub[pred_col].to_numpy()
            x_reset = x - x.min()
            width = max(float(x_reset.max()), 0.5)
            if i % 2 == 0:
                ax.axvspan(x_offset - 0.2, x_offset + width + 0.2, color=TEAL, alpha=0.12, zorder=0)
            ax.scatter(
                x_reset + x_offset,
                y,
                s=point_size,
                alpha=alpha,
                color=color_map[cancer],
                edgecolors="black",
                linewidths=0.2,
            )
            label = f"n={len(sub)}"
            if len(sub) >= min_n:
                sp = spearmanr(sub[true_col], sub[pred_col])
                label = f"rho={sp.statistic:.2f}\np={sp.pvalue:.1e}"
            ax.text(x_offset + width / 2, ax.get_ylim()[1] - 0.06 * yr, label, ha="center", va="top", fontsize=ANNOTATION_FS - 1)
            xticks.append(x_offset + width / 2)
            labels.append(cancer)
            x_offset += width + gap
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=TICK_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xlabel("True log2(TMB + 1), reset within each cancer", fontsize=AXIS_LABEL_FS)
    fig.supylabel("Predicted log2(TMB + 1)", fontsize=AXIS_LABEL_FS, x=0.03)
    fig.tight_layout()
    if save_path:
        save_panel(fig, save_path, project_root)


def plot_raw_tmb_by_cancer(
    df,
    project_root: Path,
    raw_col="TMB_raw_true",
    cancer_col="Cancer",
    figsize=(7.0, 3.0),
    point_size=9,
    alpha=0.35,
    jitter=0.18,
    seed=42,
    save_path=None,
):
    """Scatter individual raw TMB values per cancer with median bars."""
    plot_df = df[[cancer_col, raw_col]].dropna().copy()
    order = plot_df.groupby(cancer_col)[raw_col].median().sort_values(ascending=False).index.tolist()
    pos = {cancer: i for i, cancer in enumerate(order)}
    rng = np.random.default_rng(seed)

    positive = plot_df[raw_col] > 0
    if not positive.all():
        min_positive = plot_df.loc[positive, raw_col].min()
        plot_df.loc[~positive, raw_col] = min_positive / 2

    fig, ax = plt.subplots(figsize=figsize)
    medians = plot_df.groupby(cancer_col)[raw_col].median().reindex(order)
    ax.bar(
        np.arange(len(order)),
        medians.values,
        width=0.72,
        color=BEIGE,
        edgecolor="black",
        linewidth=0.35,
        alpha=0.9,
        zorder=1,
        label="Median",
    )

    for i, cancer in enumerate(order):
        values = plot_df.loc[plot_df[cancer_col] == cancer, raw_col].to_numpy(dtype=float)
        x = pos[cancer] + rng.uniform(-jitter, jitter, size=len(values))
        ax.scatter(x, values, s=point_size, alpha=alpha, color=TEAL, edgecolors="none", zorder=2)

    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=TICK_FS - 1)
    ax.set_ylabel("Coding TMB (log scale)", fontsize=AXIS_LABEL_FS)
    ax.set_xlabel("Cancer type", fontsize=AXIS_LABEL_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.grid(axis="y", alpha=0.2, zorder=0, which="both")
    ax.margins(x=0.01)
    fig.tight_layout()
    if save_path:
        save_panel(fig, save_path, project_root)


def plot_roc_grid(
    df,
    project_root: Path,
    raw_true_col="TMB_raw_true",
    raw_pred_col="TMB_log2_pred",
    cancer_col="Cancer",
    split_quantile=0.5,
    min_n=10,
    n_cols=7,
    figsize_per_panel=(1.25, 1.25),
    save_path=None,
):
    rocs = []
    for cancer, sub in df.groupby(cancer_col):
        sub = sub.dropna(subset=[raw_true_col, raw_pred_col])
        if len(sub) < min_n:
            continue
        vals = sub[raw_true_col].to_numpy()
        if split_quantile == 0.5:
            threshold = np.median(vals)
            y_true = (sub[raw_true_col] > threshold).astype(int)
            y_score = sub[raw_pred_col]
        else:
            lo = np.quantile(vals, split_quantile)
            hi = np.quantile(vals, 1.0 - split_quantile)
            keep = (sub[raw_true_col] <= lo) | (sub[raw_true_col] >= hi)
            y_true = (sub.loc[keep, raw_true_col] >= hi).astype(int)
            y_score = sub.loc[keep, raw_pred_col]
        if len(np.unique(y_true)) < 2 or len(y_true) < min_n:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        rocs.append((cancer, fpr, tpr, auc))
    rocs = sorted(rocs, key=lambda x: x[3], reverse=True)
    n_panels = len(rocs)
    n_cols = min(n_cols, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize_per_panel[0], n_rows * figsize_per_panel[1]), squeeze=False)
    for i, (cancer, fpr, tpr, auc) in enumerate(rocs):
        ax = axes.flat[i]
        ax.plot(fpr, tpr, color=TEAL, lw=1.3)
        ax.plot([0, 1], [0, 1], "--", color="0.7", lw=0.7)
        ax.set_title(cancer, fontsize=ANNOTATION_FS, pad=2)
        ax.text(0.95, 0.05, f"AUC={auc:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=ANNOTATION_FS - 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=TICK_FS - 1, length=2)
    for j in range(n_panels, len(axes.flat)):
        axes.flat[j].set_visible(False)
    fig.supxlabel("FPR", fontsize=AXIS_LABEL_FS, y=0.03)
    fig.supylabel("TPR", fontsize=AXIS_LABEL_FS, x=0.03)
    fig.tight_layout()
    if save_path:
        save_panel(fig, save_path, project_root)


def plot_predicted_tmb_quartile_stratification(
    df: pd.DataFrame,
    project_root: Path,
    raw_true_col: str = "TMB_raw_true",
    pred_col: str = "TMB_log2_pred",
    cancer_col: str = "Cancer",
    quantile: float = 0.25,
    min_n: int = 20,
    min_group_n: int = 3,
    n_rows: int = 2,
    figsize=(7.6, 4.4),
    yscale: str = "log",
    jitter: float = 0.08,
    point_size: float = 8,
    alpha: float = 0.35,
    seed: int = 42,
    save_path=None,
    table_save_path=None,
):
    """Plot observed TMB after stratifying samples by predicted log-TMB quartiles."""
    if not 0 < quantile < 0.5:
        raise ValueError("quantile must be between 0 and 0.5.")

    rows = []
    plot_frames = []
    for cancer, sub in df.groupby(cancer_col):
        sub = sub[[cancer_col, raw_true_col, pred_col]].dropna().copy()
        if len(sub) < min_n:
            continue

        pred_lo = float(np.quantile(sub[pred_col], quantile))
        pred_hi = float(np.quantile(sub[pred_col], 1.0 - quantile))
        if pred_lo >= pred_hi:
            continue

        low = sub.loc[sub[pred_col] <= pred_lo].copy()
        high = sub.loc[sub[pred_col] >= pred_hi].copy()
        if len(low) < min_group_n or len(high) < min_group_n:
            continue

        low_true = low[raw_true_col].to_numpy(dtype=float)
        high_true = high[raw_true_col].to_numpy(dtype=float)
        try:
            mw = mannwhitneyu(high_true, low_true, alternative="two-sided")
            mw_p = float(mw.pvalue)
            stratification_auc = float(mw.statistic / (len(high_true) * len(low_true)))
        except ValueError:
            mw_p = np.nan
            stratification_auc = np.nan

        median_low = float(np.nanmedian(low_true))
        median_high = float(np.nanmedian(high_true))
        rows.append(
            {
                "Cancer": cancer,
                "n_samples": int(len(sub)),
                "n_predicted_bottom_quartile": int(len(low)),
                "n_predicted_top_quartile": int(len(high)),
                "median_observed_tmb_predicted_bottom_quartile": median_low,
                "median_observed_tmb_predicted_top_quartile": median_high,
                "median_observed_tmb_ratio_top_vs_bottom": median_high / median_low if median_low > 0 else np.nan,
                "stratification_auc": stratification_auc,
                "mannwhitney_p": mw_p,
            }
        )

        low["predicted_tmb_group"] = f"Predicted bottom {int(round(100 * quantile))}%"
        high["predicted_tmb_group"] = f"Predicted top {int(round(100 * quantile))}%"
        plot_frames.extend([low, high])

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("No cancer types had enough samples for predicted TMB quartile stratification.")

    summary["mannwhitney_fdr"] = benjamini_hochberg_fdr(summary["mannwhitney_p"])
    summary = summary.sort_values(
        [
            "stratification_auc",
            "median_observed_tmb_ratio_top_vs_bottom",
            "median_observed_tmb_predicted_top_quartile",
        ],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    if table_save_path:
        save_table(summary, table_save_path, project_root=project_root, index=False)

    plot_df = pd.concat(plot_frames, ignore_index=True)
    order = summary["Cancer"].tolist()
    plot_df = plot_df[plot_df[cancer_col].isin(order)].copy()

    positive = plot_df[raw_true_col] > 0
    if not positive.all():
        min_positive = plot_df.loc[positive, raw_true_col].min() if positive.any() else 0.5
        plot_df.loc[~positive, raw_true_col] = min_positive / 2

    rng = np.random.default_rng(seed)
    n_rows = max(1, min(int(n_rows), len(order)))
    row_orders = [list(chunk) for chunk in np.array_split(order, n_rows) if len(chunk)]
    fig, axes = plt.subplots(len(row_orders), 1, figsize=figsize, sharey=True, squeeze=False)
    axes = axes[:, 0]
    offsets = {
        f"Predicted bottom {int(round(100 * quantile))}%": -0.18,
        f"Predicted top {int(round(100 * quantile))}%": 0.18,
    }
    colors = {
        f"Predicted bottom {int(round(100 * quantile))}%": BEIGE,
        f"Predicted top {int(round(100 * quantile))}%": TEAL,
    }

    for ax, row_order in zip(axes, row_orders):
        order_pos = {cancer: i for i, cancer in enumerate(row_order)}
        for group, offset in offsets.items():
            grouped_values = []
            positions = []
            for cancer in row_order:
                values = plot_df.loc[
                    (plot_df[cancer_col] == cancer) & (plot_df["predicted_tmb_group"] == group),
                    raw_true_col,
                ].to_numpy(dtype=float)
                grouped_values.append(values)
                positions.append(order_pos[cancer] + offset)

            box = ax.boxplot(
                grouped_values,
                positions=positions,
                widths=0.30,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "linewidth": 0.9},
                boxprops={"linewidth": 0.45, "edgecolor": "black"},
                whiskerprops={"linewidth": 0.45, "color": "black"},
                capprops={"linewidth": 0.45, "color": "black"},
            )
            for patch in box["boxes"]:
                patch.set_facecolor(colors[group])
                patch.set_alpha(0.88)

            for cancer, values in zip(row_order, grouped_values):
                x = order_pos[cancer] + offset + rng.uniform(-jitter, jitter, size=len(values))
                ax.scatter(
                    x,
                    values,
                    s=point_size,
                    color=colors[group],
                    edgecolors="black",
                    linewidths=0.15,
                    alpha=alpha,
                    zorder=3,
                )

        row_summary = summary.set_index("Cancer").loc[row_order]
        for cancer, row in row_summary.iterrows():
            xpos = order_pos[cancer]
            auc = row["stratification_auc"]
            if np.isfinite(auc):
                ax.text(
                    xpos,
                    0.985,
                    f"AUC={auc:.2f}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=ANNOTATION_FS - 2,
                    clip_on=False,
                )

            fdr = row["mannwhitney_fdr"]
            if not np.isfinite(fdr) or fdr >= 0.05:
                continue
            if fdr < 0.001:
                label = "***"
            elif fdr < 0.01:
                label = "**"
            else:
                label = "*"
            ax.text(
                xpos,
                0.90,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=ANNOTATION_FS,
                clip_on=False,
            )

        if yscale:
            ax.set_yscale(yscale)
        ax.set_xticks(np.arange(len(row_order)))
        ax.set_xticklabels(row_order, rotation=90, fontsize=TICK_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.grid(axis="y", alpha=0.22, zorder=0, which="both")
        ax.margins(x=0.01)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel("Cancer type", fontsize=AXIS_LABEL_FS)
    fig.supylabel("Observed coding TMB", fontsize=AXIS_LABEL_FS, x=0.03)
    if yscale == "log":
        ymin, ymax = axes[0].get_ylim()
        axes[0].set_ylim(ymin, max(ymax, float(plot_df[raw_true_col].max()) * 1.25))

    handles = [
        Patch(facecolor=BEIGE, edgecolor="black", linewidth=0.45, label=f"Predicted bottom {int(round(100 * quantile))}%"),
        Patch(facecolor=TEAL, edgecolor="black", linewidth=0.45, label=f"Predicted top {int(round(100 * quantile))}%"),
    ]
    axes[0].legend(
        handles=handles,
        frameon=False,
        fontsize=LEGEND_FS,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        ncol=2,
        borderaxespad=0,
    )

    fig.tight_layout()
    if save_path:
        save_panel(fig, save_path, project_root)
    return fig, axes, summary


def plot_fold_variability(fold_df, project_root: Path, figsize=(3.3, 2.5), save_path=None):
    fig, ax1 = plt.subplots(figsize=figsize)
    x = np.arange(len(fold_df))
    ax1.plot(x, fold_df["spearman"], marker="o", color=PURPLE, label="Spearman rho")
    if "pearson" in fold_df:
        ax1.plot(x, fold_df["pearson"], marker="o", color=TEAL, label="Pearson r")
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Correlation", fontsize=AXIS_LABEL_FS)
    ax1.set_xlabel("CV fold", fontsize=AXIS_LABEL_FS)
    ax1.set_xticks(x)
    ax1.set_xticklabels(fold_df["fold"].astype(str), fontsize=TICK_FS)
    ax1.tick_params(axis="y", labelsize=TICK_FS)
    ax1.grid(axis="y", alpha=0.2)
    ax2 = ax1.twinx()
    ax2.plot(x, fold_df["rmse"], marker="s", color=GOLD, label="RMSE")
    ax2.set_ylabel("RMSE", fontsize=AXIS_LABEL_FS)
    ax2.tick_params(axis="y", labelsize=TICK_FS)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=LEGEND_FS, loc="center right")
    fig.tight_layout()
    if save_path:
        save_panel(fig, save_path, project_root)


def plot_scatter_corr(x, y, labels=None, xlabel="", ylabel="", color=PURPLE, figsize=(2.7, 2.5), save_path=None, project_root: Path | None = None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    labels = np.asarray(labels)[mask] if labels is not None else None
    sp = spearmanr(x, y)
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(x, y, color=color, edgecolors="black", linewidths=0.35, s=30)
    if labels is not None:
        for xi, yi, lab in zip(x, y, labels):
            ax.text(xi, yi, str(lab), fontsize=ANNOTATION_FS - 2, ha="left", va="bottom")
    ax.text(
        0.38,
        0.25,
        f"Spearman rho = {sp.statistic:.2f}\np = {sp.pvalue:.1e}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOTATION_FS,
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.25", alpha=0.8),
    )
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    fig.tight_layout()
    if save_path:
        if project_root is None:
            raise ValueError("project_root is required when save_path is provided.")
        save_panel(fig, save_path, project_root)


def load_mutation_prediction_summaries(root_candidates: list[Path], project_root: Path):
    root = next((Path(path) for path in root_candidates if Path(path).exists()), None)
    if root is None:
        print("Mutation prediction results not found yet. Expected one of:")
        for path in root_candidates:
            print(" -", _safe_relative(Path(path), project_root))
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    cancer_dict = {}
    for cancer_dir in sorted(root.iterdir()):
        metrics_path = cancer_dir / "cv" / "metrics.csv"
        if metrics_path.exists():
            cancer_dict[cancer_dir.name] = pd.read_csv(metrics_path, index_col=0)

    auprc_dict = {}
    normalized_auprc_dict = {}
    prevalence_dict = {}
    for cancer, df in cancer_dict.items():
        required = {"auprc", "prevalence", "normalized_auprc"}
        if not required.issubset(df.columns):
            continue
        auprc_dict[cancer] = df["auprc"]
        prevalence_dict[cancer] = df["prevalence"]
        normalized_auprc_dict[cancer] = df["normalized_auprc"]

    return cancer_dict, pd.DataFrame(auprc_dict), pd.DataFrame(prevalence_dict), pd.DataFrame(normalized_auprc_dict)


def mutation_cohort_summary(cancer_dict: dict[str, pd.DataFrame], mutation_root: Path) -> pd.DataFrame:
    """Summarize retained mutation-prediction tasks and available outputs."""
    rows = []
    mutation_root = Path(mutation_root)
    for cohort, df in sorted(cancer_dict.items()):
        if not {"auprc", "prevalence", "normalized_auprc"}.issubset(df.columns):
            continue

        norm = df["normalized_auprc"]
        metadata_path = mutation_root / cohort / "cv" / "run_metadata.json"
        metadata = {}
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
            except Exception:
                metadata = {}

        shap_dir = mutation_root / cohort / "shap"
        emb_path = mutation_root / cohort / "embeddings.csv"
        rows.append(
            {
                "cohort": cohort,
                "samples": metadata.get("n_samples", np.nan),
                "expression_features": metadata.get("n_features", np.nan),
                "n_targets": int(df.shape[0]),
                "median_auprc": float(df["auprc"].median()),
                "median_normalized_auprc": float(norm.median()),
                "median_prevalence": float(df["prevalence"].median()),
                "top_auprc_gene": str(df["auprc"].idxmax()),
                "top_auprc": float(df["auprc"].max()),
                "has_shap": shap_dir.is_dir() and any(shap_dir.iterdir()),
                "has_sample_embeddings": emb_path.exists(),
            }
        )
    return pd.DataFrame(rows).sort_values(["cohort"]).reset_index(drop=True)


def tcga_sample_id(barcode) -> str:
    """Map TCGA barcodes to the sample ID used to match expression and MC3 (TCGA-XX-XXXX-NN)."""
    parts = str(barcode).split("-")
    if len(parts) >= 4 and parts[0] == "TCGA":
        return "-".join([parts[0], parts[1], parts[2], parts[3][:2]])
    return str(barcode)


def load_mutation_probabilities(mutation_root: Path, cohort: str) -> pd.DataFrame:
    path = Path(mutation_root) / cohort / "cv" / "oof_probabilities.csv"
    probs = pd.read_csv(path, index_col=0)
    probs.index = [tcga_sample_id(x) for x in probs.index]
    return probs


def load_mc3_gene_labels(mutation_label_dir: Path, cohort: str) -> pd.DataFrame:
    path = Path(mutation_label_dir) / f"{cohort}_mc3_gene_level.txt.gz"
    labels = pd.read_csv(path, sep="\t").rename(columns={"sample": "gene"}).set_index("gene")
    labels = labels.T
    labels.index = [tcga_sample_id(x) for x in labels.index]
    labels = labels.groupby(labels.index).max()
    labels = labels.T.groupby(labels.T.index).max().T
    return (labels.fillna(0).astype(float) > 0).astype(int)


def compute_mutation_rank_sum_tests(
    mutation_root: Path,
    mutation_label_dir: Path,
    min_mutated: int = 3,
    min_wildtype: int = 3,
    skip_cohorts: tuple[str, ...] = ("all", "tmb_prediction"),
) -> pd.DataFrame:
    """Run one-sided rank-sum tests comparing prediction scores in mutated vs wild-type samples."""
    mutation_root = Path(mutation_root)
    mutation_label_dir = Path(mutation_label_dir)
    rows = []

    for cohort_dir in sorted(mutation_root.iterdir()):
        if not cohort_dir.is_dir():
            continue

        cohort = cohort_dir.name
        if cohort in skip_cohorts:
            continue

        prob_path = cohort_dir / "cv" / "oof_probabilities.csv"
        summary_path = cohort_dir / "cv" / "metrics.csv"
        label_path = mutation_label_dir / f"{cohort}_mc3_gene_level.txt.gz"
        if not prob_path.exists() or not summary_path.exists() or not label_path.exists():
            continue

        probs = load_mutation_probabilities(mutation_root, cohort)
        labels = load_mc3_gene_labels(mutation_label_dir, cohort)
        metrics = pd.read_csv(summary_path, index_col=0)

        shared_samples = probs.index.intersection(labels.index)
        shared_genes = probs.columns.intersection(labels.columns)
        probs = probs.loc[shared_samples, shared_genes]
        labels = labels.loc[shared_samples, shared_genes]

        for gene in shared_genes:
            y = labels[gene].astype(int)
            s = probs[gene].astype(float)
            valid = y.notna() & s.notna()
            y = y.loc[valid]
            s = s.loc[valid]

            n_mut = int((y == 1).sum())
            n_wt = int((y == 0).sum())
            p_value = np.nan
            u_stat = np.nan
            auc_from_u = np.nan
            if n_mut >= min_mutated and n_wt >= min_wildtype:
                test = mannwhitneyu(s[y == 1], s[y == 0], alternative="greater", method="auto")
                u_stat = float(test.statistic)
                p_value = float(test.pvalue)
                auc_from_u = u_stat / (n_mut * n_wt)

            metric_row = metrics.loc[gene] if gene in metrics.index else pd.Series(dtype=float)
            auprc = metric_row.get("auprc", np.nan)
            prevalence_metric = metric_row.get("prevalence", np.nan)
            prevalence_observed = n_mut / len(y) if len(y) else np.nan
            norm_auprc = (
                (auprc - prevalence_metric) / (1 - prevalence_metric)
                if pd.notna(auprc) and pd.notna(prevalence_metric) and prevalence_metric < 1
                else np.nan
            )

            rows.append(
                {
                    "cohort": cohort,
                    "gene": gene,
                    "n_samples": int(len(y)),
                    "n_mutated": n_mut,
                    "n_wildtype": n_wt,
                    "prevalence_observed": prevalence_observed,
                    "prevalence": prevalence_metric,
                    "auprc": auprc,
                    "normalized_auprc": norm_auprc,
                    "roc_auc": metric_row.get("roc_auc", np.nan),
                    "mannwhitney_u": u_stat,
                    "auc_from_u": auc_from_u,
                    "rank_sum_p": p_value,
                }
            )

    gene_tests = pd.DataFrame(rows)
    if gene_tests.empty:
        return gene_tests

    gene_tests["rank_sum_fdr_within_cancer"] = np.nan
    for cohort, idx in gene_tests.groupby("cohort").groups.items():
        gene_tests.loc[idx, "rank_sum_fdr_within_cancer"] = benjamini_hochberg_fdr(
            gene_tests.loc[idx, "rank_sum_p"]
        )
    gene_tests["rank_sum_fdr_global"] = benjamini_hochberg_fdr(gene_tests["rank_sum_p"])
    return gene_tests.sort_values(
        ["cohort", "rank_sum_fdr_within_cancer", "rank_sum_p", "gene"],
        na_position="last",
    ).reset_index(drop=True)


def summarize_mutation_rank_sum_tests(gene_tests: pd.DataFrame) -> pd.DataFrame:
    """Build the compact cancer-level rank-sum summary table used for the supplement."""
    if gene_tests.empty:
        return pd.DataFrame(
            columns=[
                "cohort",
                "n_samples",
                "n_target_genes",
                "n_testable_genes",
                "n_raw_p_lt_0_05",
                "fraction_raw_p_lt_0_05",
                "n_fdr_lt_0_05",
                "fraction_fdr_lt_0_05",
                "median_prevalence",
                "median_auprc",
                "median_normalized_auprc",
            ]
        )

    summary = (
        gene_tests.groupby("cohort", as_index=False)
        .agg(
            n_samples=("n_samples", "max"),
            n_target_genes=("gene", "count"),
            n_testable_genes=("rank_sum_p", lambda x: int(np.isfinite(x).sum())),
            n_raw_p_lt_0_05=("rank_sum_p", lambda x: int((x < 0.05).sum())),
            n_fdr_lt_0_05=("rank_sum_fdr_within_cancer", lambda x: int((x < 0.05).sum())),
            median_prevalence=("prevalence", "median"),
            median_auprc=("auprc", "median"),
            median_normalized_auprc=("normalized_auprc", "median"),
        )
    )
    summary["fraction_raw_p_lt_0_05"] = summary["n_raw_p_lt_0_05"] / summary["n_testable_genes"]
    summary["fraction_fdr_lt_0_05"] = summary["n_fdr_lt_0_05"] / summary["n_testable_genes"]
    return summary[
        [
            "cohort",
            "n_samples",
            "n_target_genes",
            "n_testable_genes",
            "n_raw_p_lt_0_05",
            "fraction_raw_p_lt_0_05",
            "n_fdr_lt_0_05",
            "fraction_fdr_lt_0_05",
            "median_prevalence",
            "median_auprc",
            "median_normalized_auprc",
        ]
    ].sort_values(["fraction_fdr_lt_0_05", "n_fdr_lt_0_05"], ascending=False).reset_index(drop=True)


def plot_auprc_barplot(
    df,
    project_root: Path,
    prevalence_col="prevalence",
    auprc_col="auprc",
    figsize=(2.6, 6.5),
    cmap=None,
    vmin=None,
    vmax=None,
    horizontal: bool = True,
    save_path=None,
    color_by: str = "prevalence",
):
    if cmap is None:
        raise ValueError("Pass cmap explicitly.")
    color_options = {
        "prevalence": (prevalence_col, "Prevalence"),
        "auprc": (auprc_col, "AUPRC"),
    }
    if color_by not in color_options:
        raise ValueError("color_by must be either 'prevalence' or 'auprc'.")
    plot_df = df[[prevalence_col, auprc_col]].dropna().copy()
    plot_df = plot_df.sort_values(prevalence_col, ascending=not horizontal)
    genes = plot_df.index.to_numpy()
    auprc = plot_df[auprc_col].to_numpy()
    color_col, colorbar_label = color_options[color_by]
    color_values = plot_df[color_col].to_numpy()
    pos = np.arange(len(genes))
    vmin = float(np.nanmin(color_values)) if vmin is None else vmin
    vmax = float(np.nanmax(color_values)) if vmax is None else vmax
    norm = Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=figsize)

    if horizontal:
        ax.barh(pos, auprc, color=cmap(norm(color_values)), edgecolor="black", linewidth=0.4)
        ax.set_yticks(pos)
        ax.set_yticklabels(genes, fontsize=TICK_FS - 2)
        ax.set_xlabel("AUPRC", fontsize=AXIS_LABEL_FS)
        ax.tick_params(axis="x", labelsize=TICK_FS, length=0)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.5, len(pos) - 0.5)
        ax.margins(y=0)
        ax.invert_yaxis()
    else:
        ax.bar(pos, auprc, color=cmap(norm(color_values)), edgecolor="black", linewidth=0.4)
        ax.set_xticks(pos)
        ax.set_xticklabels(genes, rotation=90, fontsize=TICK_FS - 2)
        ax.set_ylabel("AUPRC", fontsize=AXIS_LABEL_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS, length=0)
        ax.set_xlim(-0.5, len(pos) - 0.5)
        ax.margins(x=0)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=-0.15 if horizontal else 0.02, fraction=0.05)
    cbar.set_label(colorbar_label, fontsize=COLORBAR_LABEL_FS, rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS, length=0)
    fig.tight_layout()
    if save_path:
        save_panel(fig, save_path, project_root)
    return plot_df


def plot_ranked_gene_tiles(
    auprc_df: pd.DataFrame,
    project_root: Path,
    prevalence_df: pd.DataFrame | None = None,
    top_n: int = 15,
    figsize=(6, 3.2),
    cmap=None,
    vmin=None,
    vmax=None,
    interpolation="nearest",
    show_ranks_on_top: bool = True,
    grid: bool = True,
    grid_lw: float = 0.3,
    text_contrast_threshold: float = 0.4,
    xlabel: str = "Rank within cancer (by n-AUPRC)",
    ylabel: str = "Cancer type",
    cbar_label: str = "normalized-AUPRC",
    show_prevalence_dots: bool = True,
    prev_dot_size: float = 30,
    save_path=None,
):
    if auprc_df is None or auprc_df.empty:
        raise ValueError("auprc_df is empty or None.")
    if cmap is None:
        raise ValueError("Pass cmap explicitly.")
    cancer_types = (~auprc_df.isna()).sum().sort_values(ascending=False).index.tolist()
    top_scores = pd.DataFrame(index=range(1, top_n + 1), columns=cancer_types, dtype=float)
    top_genes = pd.DataFrame(index=range(1, top_n + 1), columns=cancer_types, dtype=object)

    for ct in cancer_types:
        s = auprc_df[ct].dropna().sort_values(ascending=False)
        vals = s.values[:top_n]
        genes = s.index.values[:top_n]
        if len(vals) < top_n:
            pad = top_n - len(vals)
            vals = np.concatenate([vals, np.full(pad, np.nan)])
            genes = np.concatenate([genes, np.full(pad, "", dtype=object)])
        top_scores[ct] = vals
        top_genes[ct] = genes

    data = top_scores.T.values
    masked = np.ma.masked_invalid(data)
    cmap_obj = cmap.copy() if isinstance(cmap, Colormap) else plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(alpha=0)
    vmin = float(np.nanmin(data)) if vmin is None else vmin
    vmax = float(np.nanmax(data)) if vmax is None else vmax

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    im = ax.imshow(masked, aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax, interpolation=interpolation, zorder=0)
    ax.set_xticks(np.arange(top_n))
    ax.set_xticklabels([str(i) for i in range(1, top_n + 1)], fontsize=TICK_FS)
    ax.set_yticks(np.arange(len(cancer_types)))
    ax.set_yticklabels(cancer_types, fontsize=TICK_FS - 1)
    if show_ranks_on_top:
        ax.xaxis.tick_top()
    ax.tick_params(axis="x", which="major", labelrotation=0, pad=6, length=0)
    ax.tick_params(axis="y", which="major", pad=6, length=0)
    ax.set_xlabel(xlabel, fontsize=AXIS_LABEL_FS, labelpad=8)
    ax.xaxis.set_label_position("bottom")
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FS, labelpad=8)
    if grid:
        ax.set_xticks(np.arange(-0.5, top_n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(cancer_types), 1), minor=True)
        ax.grid(which="minor", linestyle="-", linewidth=grid_lw)
        ax.tick_params(axis="x", which="minor", top=False, bottom=False, length=0)
        ax.tick_params(axis="y", which="minor", left=False, right=False, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    def _luminance(val):
        r, g, b, _ = cmap_obj((val - vmin) / (vmax - vmin) if vmax > vmin else 0)
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    for i, ct in enumerate(cancer_types):
        for j in range(top_n):
            gene = top_genes.loc[j + 1, ct]
            val = data[i, j]
            if isinstance(gene, str) and gene and np.isfinite(val):
                txt_color = "black" if _luminance(val) > text_contrast_threshold else "white"
                ax.text(j, i, gene, ha="center", va="center", fontsize=ANNOTATION_FS - 2.5, color=txt_color, zorder=4)

    if show_prevalence_dots and prevalence_df is not None:
        xs, ys, cs = [], [], []
        for i, ct in enumerate(cancer_types):
            if ct not in prevalence_df.columns:
                continue
            for j in range(top_n):
                gene = top_genes.loc[j + 1, ct]
                if isinstance(gene, str) and gene and gene in prevalence_df.index:
                    prev = prevalence_df.loc[gene, ct]
                    if pd.notna(prev):
                        xs.append(j + 0.41)
                        ys.append(i + 0.15)
                        cs.append(prev)
        if cs:
            ax.scatter(xs, ys, c=cs, cmap=cmap_obj, vmin=vmin, vmax=vmax, s=prev_dot_size, edgecolors="black", linewidths=0.3, zorder=3)
        ax.legend(
            handles=[
                Patch(facecolor=cmap_obj(0.7), edgecolor="none", label="AUPRC (cell)"),
                Line2D([0], [0], marker="o", color="none", markerfacecolor=cmap_obj(0.7), markeredgecolor="black", markersize=5, label="Prevalence (dot)"),
            ],
            loc="lower right",
            fontsize=LEGEND_FS,
            handlelength=1.2,
            borderpad=0.3,
            labelspacing=0.4,
            frameon=False,
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label, fontsize=COLORBAR_LABEL_FS, rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS, length=0)
    cbar.outline.set_linewidth(0.4)
    fig.tight_layout()
    if save_path:
        save_panel(fig, save_path, project_root)
    return fig, ax, top_scores, top_genes


def dot_matrix_with_marginals(
    auprc_df: pd.DataFrame,
    project_root: Path,
    top_n: int = 15,
    min_cancers_per_gene: int = 2,
    auprc_threshold: float = 0.2,
    max_genes: int = 80,
    figsize=(7.5, 4.2),
    cmap=None,
    dot_size: float = 22,
    cbar_label: str = "AUPRC",
    top_bar_color: str = "#555555",
    left_bar_color: str = "#555555",
    save_path=None,
):
    if cmap is None:
        raise ValueError("Pass cmap explicitly.")
    cancer_types = (auprc_df > auprc_threshold).sum().sort_values(ascending=False).index.tolist()
    membership = pd.DataFrame(False, index=auprc_df.index, columns=cancer_types)
    for ct in cancer_types:
        s = auprc_df[ct].dropna().sort_values(ascending=False).head(top_n)
        membership.loc[s[s > auprc_threshold].index, ct] = True
    membership = membership.loc[:, membership.sum(axis=0) > 0]
    cancer_types = membership.columns.tolist()
    genes = membership.sum(axis=1)
    genes = genes[genes >= min_cancers_per_gene].sort_values(ascending=False).index.tolist()[:max_genes]
    if not genes or not cancer_types:
        raise ValueError("No genes/cancers to show after filtering.")

    mem = membership.loc[genes, cancer_types]
    gene_counts = mem.sum(axis=1)
    gene_mean_auprc = auprc_df.loc[genes, cancer_types].where(mem).mean(axis=1)
    cancer_counts = mem.sum(axis=0)
    genes_sorted = pd.DataFrame({"n_cancers": gene_counts, "mean_auprc": gene_mean_auprc}).sort_values(["n_cancers", "mean_auprc"], ascending=False).index.tolist()
    cancers_sorted = cancer_counts.sort_values(ascending=False).index.tolist()
    mem = mem.loc[genes_sorted, cancers_sorted]

    gi, ci = np.where(mem.values)
    pts = pd.DataFrame(
        {
            "gene": np.array(genes_sorted, dtype=object)[gi],
            "cancer": np.array(cancers_sorted, dtype=object)[ci],
            "auprc": auprc_df.reindex(index=genes_sorted, columns=cancers_sorted).values[gi, ci].astype(float),
        }
    )

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 6], width_ratios=[0.5, 0.0001, 6, 0.12], hspace=0, wspace=0)
    ax_left = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 2])
    ax_mat = fig.add_subplot(gs[1, 2], sharex=ax_top)
    cax = fig.add_subplot(gs[1, 3])

    ax_left.barh(np.arange(len(cancers_sorted)), cancer_counts.loc[cancers_sorted].values, color=left_bar_color)
    ax_left.set_ylim(-0.5, len(cancers_sorted) - 0.5)
    ax_left.invert_yaxis()
    ax_left.set_xlabel("# genes", fontsize=AXIS_LABEL_FS)
    ax_left.tick_params(axis="x", labelsize=TICK_FS)
    ax_left.tick_params(axis="y", left=False, labelleft=False)
    ax_left.grid(True, axis="x", linewidth=0.5, alpha=0.18)

    ax_top.bar(np.arange(len(genes_sorted)), gene_counts.loc[genes_sorted].values, color=top_bar_color, width=0.8)
    ax_top.set_xlim(-0.5, len(genes_sorted) - 0.5)
    ax_top.set_ylabel("# cancers", fontsize=AXIS_LABEL_FS)
    ax_top.tick_params(axis="y", labelsize=TICK_FS)
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_top.grid(True, axis="y", linewidth=0.5, alpha=0.18)

    sc = ax_mat.scatter(gi, ci, c=pts["auprc"].values, s=dot_size, alpha=0.9, cmap=cmap, edgecolors="black", linewidths=0.25)
    ax_mat.set_xlim(-0.5, len(genes_sorted) - 0.5)
    ax_mat.set_ylim(-0.5, len(cancers_sorted) - 0.5)
    ax_mat.invert_yaxis()
    ax_mat.set_xticks(np.arange(len(genes_sorted)))
    ax_mat.set_xticklabels(genes_sorted, rotation=90, fontsize=ANNOTATION_FS - 2)
    ax_mat.set_yticks(np.arange(len(cancers_sorted)))
    ax_mat.set_yticklabels(cancers_sorted, fontsize=TICK_FS - 1)
    ax_mat.set_xlabel("Genes", fontsize=AXIS_LABEL_FS)
    ax_mat.set_ylabel("Cancer type", fontsize=AXIS_LABEL_FS)
    ax_mat.set_xticks(np.arange(-0.5, len(genes_sorted), 1), minor=True)
    ax_mat.set_yticks(np.arange(-0.5, len(cancers_sorted), 1), minor=True)
    ax_mat.grid(which="minor", linewidth=0.5, alpha=0.18)
    ax_mat.tick_params(which="minor", bottom=False, left=False)

    for ax in [ax_left, ax_top, ax_mat]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label(cbar_label, fontsize=COLORBAR_LABEL_FS, rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS)
    if save_path:
        save_panel(fig, save_path, project_root)
    return mem, gene_counts.loc[genes_sorted], cancer_counts.loc[cancers_sorted], pts


def plot_mutation_rank_sum_supplement(
    summary: pd.DataFrame,
    project_root: Path,
    save_path=None,
    fraction_col: str = "fraction_fdr_lt_0_05",
    sig_col: str = "n_fdr_lt_0_05",
    testable_col: str = "n_testable_genes",
    sample_col: str = "n_samples",
    color_col: str = "median_normalized_auprc",
    cohort_col: str = "cohort",
    figsize=(7.2, 3.0),
    min_dot_size: float = 28,
    max_dot_size: float = 310,
    cmap=None,
):
    """Plot per-cancer mutation-target rank-separation summary."""
    plot_df = summary.copy()
    plot_df = plot_df.loc[:, ~plot_df.columns.duplicated()].copy()
    plot_df = plot_df.replace([np.inf, -np.inf], np.nan)
    required = [fraction_col, sig_col, testable_col, sample_col, color_col, cohort_col]
    plot_df = plot_df.dropna(subset=required)
    if plot_df.empty:
        raise ValueError("No rows available for rank-sum summary plot.")

    plot_df[cohort_col] = plot_df[cohort_col].astype(str)
    plot_df = plot_df.sort_values(fraction_col, ascending=False).reset_index(drop=True)

    if cmap is None:
        cmap = make_linear_cmap([BEIGE, TEAL, PURPLE], "beige_teal_purple_rank_sum")

    x = np.arange(len(plot_df))
    y = plot_df[fraction_col].astype(float).to_numpy()
    color_values = plot_df[color_col].astype(float).to_numpy()
    sample_values = plot_df[sample_col].astype(float).to_numpy()

    if np.nanmax(sample_values) == np.nanmin(sample_values):
        sizes = np.full(len(plot_df), (min_dot_size + max_dot_size) / 2)
    else:
        sizes = min_dot_size + (max_dot_size - min_dot_size) * (
            (sample_values - np.nanmin(sample_values))
            / (np.nanmax(sample_values) - np.nanmin(sample_values))
        )

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        x,
        y,
        s=sizes,
        c=color_values,
        cmap=cmap,
        vmin=0,
        vmax=max(0.5, float(np.nanmax(color_values))),
        edgecolor="black",
        linewidth=0.35,
        alpha=0.95,
        zorder=3,
    )

    rotate_counts = len(plot_df) > 25
    for i, row in plot_df.iterrows():
        ax.text(
            i,
            float(row[fraction_col]) + 0.025,
            f"{int(row[sig_col])}/{int(row[testable_col])}",
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_FS - 1,
            rotation=90 if rotate_counts else 0,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(plot_df[cohort_col].tolist(), rotation=90, fontsize=TICK_FS)
    ax.set_ylabel("Fraction significant targets", fontsize=AXIS_LABEL_FS)
    ax.set_xlabel("Cancer type", fontsize=AXIS_LABEL_FS)
    ax.set_ylim(-0.055, min(1.15, max(1.02, float(np.nanmax(y)) + 0.12)))
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.grid(axis="y", alpha=0.22, zorder=0)

    cbar = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.045)
    cbar.set_label("Median normalized AUPRC", fontsize=COLORBAR_LABEL_FS, rotation=270, labelpad=12)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS)

    legend_samples = np.array([np.nanmin(sample_values), np.nanmedian(sample_values), np.nanmax(sample_values)])
    legend_samples = np.unique(np.round(legend_samples).astype(int))
    handles = []
    labels = []
    for value in legend_samples:
        if np.nanmax(sample_values) == np.nanmin(sample_values):
            size = (min_dot_size + max_dot_size) / 2
        else:
            size = min_dot_size + (max_dot_size - min_dot_size) * (
                (value - np.nanmin(sample_values))
                / (np.nanmax(sample_values) - np.nanmin(sample_values))
            )
        handles.append(ax.scatter([], [], s=size, facecolor="white", edgecolor="black", linewidth=0.35))
        labels.append(str(value))

    ax.legend(
        handles,
        labels,
        title="Samples",
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.98),
        fontsize=LEGEND_FS,
        title_fontsize=LEGEND_FS,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if save_path:
        save_panel(fig, save_path, project_root)
    return fig, ax, plot_df


def plot_auprc_prevalence_and_flag(
    df: pd.DataFrame,
    cancer: str,
    project_root: Path,
    max_labels: int = 80,
    min_points: int = 30,
    min_score: float = 2.0,
    figsize=(3.3, 2),
    cmap=None,
    point_size: float = 20,
    alpha: float = 0.9,
    ring_size: float = 35,
    ring_lw: float = 0.5,
    annotate: bool = True,
    save_path=None,
):
    if cmap is None:
        raise ValueError("Pass cmap explicitly.")

    def _robust_z(x: np.ndarray) -> np.ndarray:
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        if (not np.isfinite(mad)) or mad < 1e-12:
            sd = np.nanstd(x)
            return (x - np.nanmean(x)) / (sd + 1e-12)
        return (x - med) / (1.4826 * mad)

    plot_df = df.sort_values("prevalence", ascending=False).copy()
    if len(plot_df) < min_points:
        return pd.DataFrame()
    y = plot_df["auprc"].astype(float).to_numpy()
    prev = plot_df["prevalence"].astype(float).to_numpy()
    x = np.arange(len(plot_df), dtype=float)
    ok = np.isfinite(y) & np.isfinite(prev)
    if ok.sum() < min_points:
        return pd.DataFrame()

    z_auprc = np.full_like(y, np.nan, dtype=float)
    z_prev = np.full_like(prev, np.nan, dtype=float)
    z_auprc[ok] = _robust_z(y[ok])
    z_prev[ok] = _robust_z(prev[ok])
    score = z_auprc - z_prev
    cand = ok & np.isfinite(score)
    if min_score is not None:
        cand &= score >= float(min_score)
    idx = np.where(cand)[0]
    if idx.size > 0:
        idx = idx[np.argsort(score[idx])[::-1]][:max_labels]

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    sc = ax.scatter(x[ok], y[ok], c=prev[ok], cmap=cmap, s=point_size, alpha=alpha, edgecolors="black", linewidths=0.25, zorder=2)
    if idx.size > 0:
        ax.scatter(x[idx], y[idx], facecolors="none", edgecolors="black", s=ring_size, linewidths=ring_lw, zorder=3)
        if annotate:
            texts = [ax.text(x[i], y[i], str(plot_df.index[i]), fontsize=ANNOTATION_FS - 1, ha="left", va="bottom") for i in idx]
            try:
                from adjustText import adjust_text

                adjust_text(texts, x=x[ok], y=y[ok], ax=ax, arrowprops=dict(arrowstyle="-", color="0.4", lw=0.5))
            except Exception:
                pass
    ax.set_xlabel("Rank by prevalence", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("AUPRC", fontsize=AXIS_LABEL_FS)
    ax.set_title(cancer, fontsize=TITLE_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS, length=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.margins(x=0.01)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.06)
    cbar.set_label("Prevalence", fontsize=COLORBAR_LABEL_FS, rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS, length=0)
    cbar.outline.set_linewidth(0.4)
    fig.tight_layout()
    if save_path:
        save_panel(fig, save_path, project_root)
    if idx.size == 0:
        return pd.DataFrame()
    return (
        plot_df.iloc[idx]
        .assign(score=score[idx], z_auprc=z_auprc[idx], z_prevalence=z_prev[idx], prevalence_rank=idx.astype(int))
        .reset_index()
        .rename(columns={"index": "gene"})
    )
