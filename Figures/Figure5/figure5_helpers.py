from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from itertools import combinations
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, precision_recall_curve
import seaborn as sns

from constants import (
    TITLE_FS,
    AXIS_LABEL_FS,
    TICK_FS,
    LEGEND_FS,
    ANNOTATION_FS,
    COLORBAR_LABEL_FS,
    COLORBAR_TICK_FS,
    ORANGE,
    BEIGE,
    GREEN,
    TEAL,
    PURPLE,
    GOLD,
    make_linear_cmap,
)


CMAP_BEIGE_PURPLE = make_linear_cmap([BEIGE, "#c2ab42", PURPLE], "figure5_beige_purple")
CMAP_BEIGE_TEAL_PURPLE = make_linear_cmap([BEIGE, TEAL, PURPLE], "figure5_beige_teal_purple")
CATEGORICAL_COLORS = [ORANGE, TEAL, PURPLE, GREEN, "#a84059", GOLD]
FOLD_COLOR = ORANGE
POSTER_COLORS = ["#a84059", "#e0cc9b", "#9571ab", "#8076ab", "#4a4270", "#e98d7d", "#cc8711", "#82a899"]
POSTER_CMAPS = {
    "balanced": make_linear_cmap([POSTER_COLORS[0], POSTER_COLORS[1], POSTER_COLORS[7]], "figure5_balanced"),
    "teal_rose": make_linear_cmap([POSTER_COLORS[7], POSTER_COLORS[1], POSTER_COLORS[0]], "figure5_teal_rose"),
    "beige_purple": make_linear_cmap([POSTER_COLORS[1], POSTER_COLORS[2], POSTER_COLORS[4]], "figure5_beige_purple2"),
    "warm": make_linear_cmap([POSTER_COLORS[5], POSTER_COLORS[0], POSTER_COLORS[6]], "figure5_warm"),
    "purple_mono": make_linear_cmap([POSTER_COLORS[3], POSTER_COLORS[2], POSTER_COLORS[4]], "figure5_purple_mono"),
}


@dataclass(frozen=True)
class Figure5Paths:
    project_root: Path

    @property
    def result_root(self) -> Path:
        return self.project_root / "Runs" / "SingleCell" / "output"

    @property
    def plot_input_root(self) -> Path:
        return self.result_root / "plot_inputs"

    @property
    def runs_root(self) -> Path:
        return self.result_root / "runs"

    @property
    def figure_root(self) -> Path:
        return self.project_root / "Figures" / "Figure5" / "output"

    @property
    def embedding_root(self) -> Path:
        return self.figure_root / "Embeddings"


def setup_figure5_style() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig, save_path: Path | None, *, tight_layout: bool = True) -> Path | None:
    if tight_layout:
        fig.tight_layout()
    if save_path is None:
        plt.show()
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return save_path


def _save_clustergrid(grid, save_path: Path | None) -> Path | None:
    if save_path is None:
        plt.show()
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        grid.savefig(save_path, bbox_inches="tight")
        plt.close(grid.fig)
    return save_path


def _palette_values(n: int, palette: str = "husl"):
    if n <= 0:
        return []
    if n <= 10:
        return sns.color_palette("tab10", n_colors=n)
    if n <= 20:
        return sns.color_palette("tab20", n_colors=n)
    return sns.color_palette(palette, n_colors=n)


def normalized_auprc(auprc, prevalence):
    """(AUPRC - prevalence) / (1 - prevalence), as for TCGA and the external cohorts."""
    return (auprc - prevalence) / (1 - prevalence)


def _auc_trapezoid(y, x) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def plot_auprc_boxplot(run_dir: Path, title: str, out_dir: Path, *, top_n: int = 12, figsize=(3, 2.8), point_size: int = 5) -> Path:
    metric_col = "average_precision"
    metrics = pd.read_csv(run_dir / "metric_summary_per_gene_folds.csv")
    top_genes = (
        metrics.groupby("gene", as_index=False)[metric_col].mean()
        .sort_values(metric_col, ascending=False)
        .head(top_n)["gene"].astype(str).tolist()
    )
    plot_df = metrics[metrics["gene"].astype(str).isin(top_genes)].copy()
    order = (
        plot_df.groupby("gene")[metric_col].mean()
        .sort_values(ascending=False).index.astype(str).tolist()
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=plot_df, x="gene", y=metric_col, order=order, color=BEIGE, fliersize=0, linewidth=1.0, ax=ax)
    sns.stripplot(data=plot_df, x="gene", y=metric_col, order=order, color=FOLD_COLOR, dodge=False, size=point_size, alpha=0.6, ax=ax)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_xlabel("", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("AUPRC", fontsize=AXIS_LABEL_FS)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", rotation_mode="anchor", fontsize=TICK_FS)
    ax.grid(axis="y", alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, out_dir / f"auprc_boxplot_{run_dir.name}.pdf")


def _load_oof_matrix(run_dir: Path, file_name: str) -> pd.DataFrame:
    df = pd.read_csv(run_dir / file_name, index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def _load_oof_labels_and_probabilities(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = _load_oof_matrix(run_dir, "oof_true_labels.csv").astype(int)
    probabilities = _load_oof_matrix(run_dir, "oof_probabilities.csv").astype(float)
    common = labels.index.intersection(probabilities.index)
    labels = labels.loc[common]
    probabilities = probabilities.loc[common, labels.columns]
    return labels, probabilities


def plot_adamson_10x005_pr_curves(
    run_dir: Path,
    *,
    title: str = "Adamson/Weissman 10X005",
    figsize: tuple[float, float] = (3.3, 2.7),
    save_path: Path | str | None = None,
) -> Path | None:
    labels, probabilities = _load_oof_labels_and_probabilities(run_dir)

    fig, ax = plt.subplots(figsize=figsize)
    colors = dict(zip(labels.columns, CATEGORICAL_COLORS))
    for gene in labels.columns:
        y_true = labels[gene]
        y_score = probabilities[gene]
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        auprc = average_precision_score(y_true, y_score)
        baseline = float(y_true.mean())
        color = colors[gene]
        ax.plot(
            recall,
            precision,
            color=color,
            lw=1.8,
            label=f"{gene}: AUPRC {auprc:.3f}",
        )
        ax.axhline(baseline, color=color, lw=0.9, ls="--", alpha=0.45)

    ax.set_title(f"{title} PR curves", fontsize=TITLE_FS)
    ax.set_xlabel("Recall", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("Precision", fontsize=AXIS_LABEL_FS)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=LEGEND_FS - 1, loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, None if save_path is None else Path(save_path))


def _genotype_code(frame: pd.DataFrame, genes: list[str]) -> pd.Series:
    return frame[genes].astype(int).astype(str).agg("".join, axis=1)


def _genotype_display_label(code: str, genes: list[str]) -> str:
    active = [gene for bit, gene in zip(code, genes) if bit == "1"]
    if not active:
        return "Control"
    return "+".join(active)


def _genotype_order(genes: list[str]) -> list[str]:
    codes = [format(i, f"0{len(genes)}b") for i in range(2 ** len(genes))]
    return sorted(codes, key=lambda code: (code.count("1"), code))


def plot_adamson_10x005_genotype_confusion(
    run_dir: Path,
    *,
    threshold: float = 0.5,
    title: str = "Adamson/Weissman 10X005",
    figsize: tuple[float, float] = (5.2, 4.5),
    save_path: Path | str | None = None,
) -> Path | None:
    labels, probabilities = _load_oof_labels_and_probabilities(run_dir)
    genes = list(labels.columns)
    predictions = (probabilities >= threshold).astype(int)

    true_code = _genotype_code(labels, genes)
    pred_code = _genotype_code(predictions, genes)
    order = _genotype_order(genes)
    counts = pd.crosstab(true_code, pred_code).reindex(index=order, columns=order, fill_value=0)
    row_sums = counts.sum(axis=1).replace(0, np.nan)
    normalized = counts.div(row_sums, axis=0).fillna(0.0)

    axis_labels = [_genotype_display_label(code, genes) for code in order]
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        normalized,
        cmap=CMAP_BEIGE_TEAL_PURPLE,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".0%",
        linewidths=0.45,
        linecolor="white",
        annot_kws={"fontsize": ANNOTATION_FS},  # controls cell annotation font size
        xticklabels=axis_labels,
        yticklabels=axis_labels,
        cbar_kws={"label": "Row-normalized fraction", "shrink": 0.85},
        ax=ax,
    )
    ax.set_title(title, fontsize=TITLE_FS)
    ax.set_xlabel("Predicted genotype", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("True genotype", fontsize=AXIS_LABEL_FS, labelpad=-12)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS)
    cbar.set_label("Row-normalized fraction", fontsize=COLORBAR_LABEL_FS, rotation=270, labelpad=9)
    return _save(fig, None if save_path is None else Path(save_path))


def plot_zhao_pr_curves(pr_points_csv: Path, out_dir: Path, *, figsize=(3.0, 2.3)) -> list[Path]:
    pr = pd.read_csv(pr_points_csv)
    outputs = []
    for target in sorted(pr["target"].astype(str).unique()):
        target_df = pr[pr["target"].astype(str).eq(target)]
        fig, ax = plt.subplots(figsize=figsize)
        for i, (fold, fold_df) in enumerate(target_df.groupby("fold")):
            fold_df = fold_df.sort_values("recall")
            fold_auprc = _auc_trapezoid(fold_df["precision"].values, fold_df["recall"].values)
            ax.plot(
                fold_df["recall"],
                fold_df["precision"],
                lw=1.6,
                color=CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)],
                alpha=0.85,
                label=f"Fold {fold}: {fold_auprc:.2f}",
            )
        baseline = float(target_df["baseline_prevalence"].iloc[0])
        ax.axhline(baseline, ls="--", color=PURPLE, lw=1.1, alpha=0.8, label=f"Baseline: {baseline:.2f}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall", fontsize=AXIS_LABEL_FS)
        ax.set_ylabel("Precision", fontsize=AXIS_LABEL_FS)
        ax.set_title(f"Zhao PR: {target}", fontsize=TITLE_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.grid(alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        target_lower = str(target).lower()
        if "panobinostat" in target_lower:
            legend_loc = "lower left"
        elif "etoposide" in target_lower:
            legend_loc = "upper right"
        else:
            legend_loc = "best"
        ax.legend(
            loc=legend_loc,
            title="Fold AUPRC",
            frameon=True,
            framealpha=0.4,
            facecolor="white",
            fontsize=LEGEND_FS - 1,
            title_fontsize=LEGEND_FS,
        )
        outputs.append(_save(fig, out_dir / f"zhao_pr_curve_{target}.pdf"))
    return outputs


def plot_tian_top10_heatmap(metrics_csv: Path, out_dir: Path, *, top_n: int = 10, figsize=(2.6, 2.8)) -> Path:
    df = pd.read_csv(metrics_csv)
    metric = "average_precision"
    grouped = df.groupby(["perturbation", "direction"], as_index=False)[metric].mean()
    top = (
        grouped.groupby("perturbation", as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=False)
        .head(top_n)["perturbation"].astype(str).tolist()
    )
    mat = (
        grouped[grouped["perturbation"].astype(str).isin(top)]
        .pivot(index="perturbation", columns="direction", values=metric)
        .reindex(top)
    )
    mat.columns = [f"Guide\n{str(label).replace('_to_', ' -> ')}" for label in mat.columns]
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        mat,
        annot=True,
        fmt=".2f",
        cmap=CMAP_BEIGE_TEAL_PURPLE,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "AUPRC", "shrink": 0.9},
        ax=ax,
    )
    ax.set_title("Tian", fontsize=TITLE_FS)
    ax.set_xlabel("Direction", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("Perturbation", fontsize=AXIS_LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS)
    cbar.set_label("AUPRC", fontsize=COLORBAR_LABEL_FS, rotation=270, labelpad=7)
    return _save(fig, out_dir / "tian_top10_perturbation_heatmap.pdf")


def plot_frangieh_scatter(metrics_csv: Path, out_dir: Path, *, top_labels: int = 4, figsize=(3.4, 2.7)) -> Path:
    df = pd.read_csv(metrics_csv).rename(columns={"average_precision": "auprc"})
    df["normalized_auprc"] = normalized_auprc(df["auprc"], df["prevalence"])
    summary = (
        df.groupby("gene", as_index=False)
        .agg(
            prevalence_mean=("prevalence", "mean"),
            auprc_mean=("auprc", "mean"),
            normalized_auprc_mean=("normalized_auprc", "mean"),
        )
        .sort_values("auprc_mean", ascending=False)
    )
    label_genes = set(summary.head(top_labels)["gene"].tolist())

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        summary["prevalence_mean"],
        summary["auprc_mean"],
        c=summary["normalized_auprc_mean"],
        cmap=CMAP_BEIGE_PURPLE,
        s=48,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
    )
    xmax = max(0.001, float(summary["prevalence_mean"].max()) * 1.05)
    ax.plot([0, xmax], [0, xmax], "--", color="gray", lw=1, label="baseline")

    for _, row in summary.iterrows():
        if row["gene"] in label_genes:
            ax.annotate(
                str(row["gene"]),
                (row["prevalence_mean"], row["auprc_mean"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=ANNOTATION_FS,
            )

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Normalized AUPRC", fontsize=COLORBAR_LABEL_FS, rotation=270, labelpad=7)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS)

    ax.set_title("Frangieh", fontsize=TITLE_FS)
    ax.set_xlabel("Mean prevalence", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("Mean AUPRC", fontsize=AXIS_LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, fontsize=LEGEND_FS)
    return _save(fig, out_dir / "frangieh_prevalence_vs_auprc_scatter.pdf")


def plot_ccle_tmb_scatter(
    ccle_dir: Path,
    out_dir: Path,
    *,
    level: str = "cell",
    booster: str = "xgboost",
    pointsize: int = 30,
    figsize=(3.1, 2.7),
) -> Path:
    if level not in {"cell", "model", "cell-line"}:
        raise ValueError("level must be 'cell', 'model', or 'cell-line'")
    level_key = "model" if level == "cell-line" else level
    level_label = "cell-line" if level_key == "model" else "cell"
    csv_path = ccle_dir / f"{level_key}_level_{booster}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    df = pd.read_csv(csv_path)
    true_col = "log_model_TMB"
    pred_col = "pred_log_model_TMB"
    d = df[[true_col, pred_col]].dropna().copy()
    pearson_r, pearson_p = pearsonr(d[true_col], d[pred_col])
    spearman_r, spearman_p = spearmanr(d[true_col], d[pred_col])

    def _fmt_p(p):
        p = float(p)
        return f"< {np.finfo(float).tiny:.1e}" if p == 0.0 else f"= {p:.2e}"

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(
        d[true_col],
        d[pred_col],
        c="#a84059",
        s=pointsize,
        alpha=0.75,
        edgecolors="white",
        linewidths=0.35,
    )
    ax.set_title(f"CCLE log TMB ({level_label}-level)", fontsize=TITLE_FS)
    ax.set_xlabel("True log model TMB", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("Predicted log model TMB", fontsize=AXIS_LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    legend_handles = [
        Line2D([], [], linestyle="none", label=f"Pearson r = {pearson_r:.2f}\np {_fmt_p(pearson_p)}"),
        Line2D([], [], linestyle="none", label=f"Spearman r = {spearman_r:.2f}\np {_fmt_p(spearman_p)}"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        fontsize=LEGEND_FS,
        handlelength=0,
        handletextpad=0.2,
        borderpad=0.25,
        labelspacing=0.35,
    )
    return _save(fig, out_dir / f"ccle_log_tmb_scatter_{level}_{booster}.pdf")


def plot_metric_bars(
    summary_csv: Path,
    folds_csv: Path,
    out_dir: Path,
    *,
    output_prefix: str,
    metric_col: str,
    fold_metric_col: str,
    metric_label: str,
    top_n: int = 12,
    rank_mode: str = "metric",
    figsize=(2.8, 2.82),
) -> Path:
    summary = pd.read_csv(summary_csv)
    folds = pd.read_csv(folds_csv)
    d = summary[["gene", metric_col, "oof_prevalence"]].copy()
    if rank_mode == "normalized":
        d["plot_value"] = normalized_auprc(d[metric_col], d["oof_prevalence"])
    else:
        d["plot_value"] = d[metric_col]
    d = d.dropna(subset=["plot_value", "oof_prevalence"]).sort_values("plot_value", ascending=False).head(top_n)
    order = d["gene"].astype(str).tolist()

    fold_df = folds[folds["gene"].astype(str).isin(order)].copy()
    if rank_mode == "normalized":
        fold_df["plot_metric"] = normalized_auprc(fold_df[fold_metric_col], fold_df["prevalence"])
    else:
        fold_df["plot_metric"] = fold_df[fold_metric_col]

    cmap = POSTER_CMAPS["teal_rose"]
    norm = Normalize(vmin=float(d["oof_prevalence"].min()), vmax=float(d["oof_prevalence"].max()))
    colors = cmap(norm(d["oof_prevalence"].to_numpy()))

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(order), dtype=float)
    ax.bar(x, d["plot_value"], color=colors, edgecolor="white", linewidth=0.6, width=0.78, zorder=2)
    rng = np.random.default_rng(42)
    for gene, gdf in fold_df.groupby("gene"):
        xi = order.index(str(gene))
        jitter = rng.uniform(-0.18, 0.18, size=len(gdf))
        ax.scatter(np.full(len(gdf), xi) + jitter, gdf["plot_metric"], s=18, color="#876186", alpha=0.9, edgecolors="white", linewidths=0.35, zorder=3)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.015)
    cbar.set_label("Prevalence", fontsize=COLORBAR_LABEL_FS, rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS)

    ylabel = metric_label if rank_mode == "metric" else f"Normalized {metric_label}"
    title_label = "Frangieh baseline per gene" if rank_mode == "metric" else f"Frangieh baseline (normalized {metric_label})"
    ax.set_title(title_label, fontsize=TITLE_FS)
    ax.set_xlabel("", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FS)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    for label in ax.get_xticklabels():
        label.set_rotation(60)
        label.set_rotation_mode("anchor")
        label.set_horizontalalignment("right")
        label.set_fontsize(TICK_FS - 1)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.grid(axis="y", alpha=0.22, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    suffix = rank_mode if rank_mode == "normalized" else "metric"
    return _save(fig, out_dir / f"{output_prefix}_{metric_label.lower().replace(' ', '_')}_{suffix}_teal_rose_bar.pdf")


def plot_ccle_multitask_gene_bars(
    metrics_csv: Path,
    out_dir: Path,
    *,
    metric_col: str,
    metric_label: str,
    top_n: int = 20,
    rank_mode: str = "metric",
    figsize=(5.5, 2.74),
) -> Path:
    df = pd.read_csv(metrics_csv)
    d = df[["gene", "fold", "prevalence", metric_col]].copy()
    if rank_mode == "normalized":
        d["plot_metric"] = normalized_auprc(d[metric_col], d["prevalence"])
    else:
        d["plot_metric"] = d[metric_col]
    d = d.dropna(subset=["plot_metric", "prevalence"])
    summary = (
        d.groupby("gene", as_index=False)
        .agg(plot_value=("plot_metric", "mean"), prevalence_mean=("prevalence", "mean"))
        .sort_values("plot_value", ascending=False)
        .head(top_n)
    )
    order = summary["gene"].astype(str).tolist()
    fold_df = d[d["gene"].astype(str).isin(order)].copy()

    cmap = POSTER_CMAPS["teal_rose"]
    norm = Normalize(vmin=float(summary["prevalence_mean"].min()), vmax=float(summary["prevalence_mean"].max()))
    colors = cmap(norm(summary["prevalence_mean"].to_numpy()))
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(order), dtype=float)
    ax.bar(x, summary["plot_value"], color=colors, edgecolor="white", linewidth=0.6, width=0.78, zorder=2)
    rng = np.random.default_rng(42)
    for gene, gdf in fold_df.groupby("gene"):
        xi = order.index(str(gene))
        ax.scatter(np.full(len(gdf), xi) + rng.uniform(-0.18, 0.18, size=len(gdf)), gdf["plot_metric"], s=18, color=POSTER_COLORS[0], alpha=0.9, edgecolors="white", linewidths=0.35, zorder=3)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.015)
    cbar.set_label("Prevalence", fontsize=COLORBAR_LABEL_FS, rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS)
    ylabel = metric_label if rank_mode == "metric" else f"Normalized {metric_label}"
    title_label = "CCLE multitask per gene" if rank_mode == "metric" else f"CCLE multitask (normalized {metric_label})"
    ax.set_title(title_label, fontsize=TITLE_FS)
    ax.set_xlabel("", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FS)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    for label in ax.get_xticklabels():
        label.set_rotation(60)
        label.set_rotation_mode("anchor")
        label.set_horizontalalignment("right")
        label.set_fontsize(TICK_FS - 1)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.grid(axis="y", alpha=0.22, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return _save(fig, out_dir / f"ccle_multitask_per_gene_{metric_col}_{rank_mode}_teal_rose_bar.pdf")


def load_embedding_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping = pd.read_csv(run_dir / "oof_cell_folds.csv")
    embeddings = pd.read_csv(run_dir / "oof_sample_embeddings.csv", index_col=0)
    metrics = pd.read_csv(run_dir / "metric_summary_per_gene.csv")
    fold_metrics = pd.read_csv(run_dir / "metric_summary_per_gene_folds.csv")
    mapping = mapping.set_index("sample_id", drop=False)
    common = mapping.index.intersection(embeddings.index)
    return mapping.loc[common].copy(), embeddings.loc[common].copy(), metrics, fold_metrics


def _available_head_weight_folds(run_dir: Path) -> list[int]:
    folds = []
    for path in run_dir.glob("fold_head_weights_*.csv"):
        try:
            folds.append(int(path.stem.rsplit("_", 1)[-1]))
        except ValueError:
            continue
    return sorted(folds)


def _read_head_weight_matrix(run_dir: Path, fold: int, *, include_bias: bool = False) -> pd.DataFrame:
    path = run_dir / f"fold_head_weights_{fold}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "gene" not in df.columns:
        raise ValueError(f"{path} must contain a 'gene' column.")
    weight_cols = [c for c in df.columns if str(c).startswith("w_")]
    if include_bias and "bias" in df.columns:
        weight_cols = ["bias"] + weight_cols
    if not weight_cols:
        raise ValueError(f"{path} does not contain head-weight columns named like 'w_0'.")
    matrix = df.set_index("gene")[weight_cols].apply(pd.to_numeric, errors="coerce")
    matrix.index = matrix.index.astype(str)
    return matrix.dropna(axis=0, how="any")


def head_weight_distance_summary(
    run_dir: Path,
    *,
    folds: list[int] | None = None,
    metric: str = "euclidean",
    linkage_method: str = "average",
    n_clusters: int = 15,
    include_bias: bool = False,
) -> dict[str, object]:
    folds = folds or _available_head_weight_folds(run_dir)
    if not folds:
        raise FileNotFoundError(f"No fold_head_weights_*.csv files found under {run_dir}.")

    weights_by_fold = {
        int(fold): _read_head_weight_matrix(run_dir, int(fold), include_bias=include_bias)
        for fold in folds
    }
    common = sorted(set.intersection(*(set(w.index) for w in weights_by_fold.values())))
    if len(common) < 2:
        raise ValueError("At least two shared target genes are required to summarize head-weight distances.")

    dist_by_fold = {}
    for fold, weights in weights_by_fold.items():
        x = weights.loc[common].to_numpy(dtype=float)
        condensed = pdist(x, metric=metric)
        condensed = np.nan_to_num(condensed, nan=1.0, posinf=1.0, neginf=1.0)
        dist_by_fold[fold] = pd.DataFrame(
            squareform(condensed),
            index=common,
            columns=common,
        )

    stack = np.stack([dist_by_fold[f].to_numpy() for f in sorted(dist_by_fold)])
    mean_dist = pd.DataFrame(stack.mean(axis=0), index=common, columns=common)
    mean_dist = (mean_dist + mean_dist.T) / 2
    np.fill_diagonal(mean_dist.values, 0)

    z = linkage(squareform(mean_dist.to_numpy(), checks=False), method=linkage_method)
    order = mean_dist.index[leaves_list(z)].astype(str).tolist()
    clusters = pd.Series(
        fcluster(z, t=n_clusters, criterion="maxclust"),
        index=mean_dist.index.astype(str),
        name="head_weight_cluster",
    )
    return {
        "folds": sorted(dist_by_fold),
        "metric": metric,
        "mean_dist": mean_dist,
        "dist_by_fold": dist_by_fold,
        "linkage": z,
        "order": order,
        "clusters": clusters,
    }


def build_head_weight_projection_tables(
    run_dir: Path,
    fold_metrics: pd.DataFrame,
    summary: dict[str, object],
    *,
    folds: list[int] | None = None,
    metric: str = "euclidean",
    n_neighbors: int = 3,
    min_dist: float = 0.3,
    leiden_resolution: float = 1.0,
    random_state: int = 42,
    include_bias: bool = False,
) -> tuple[dict[int, pd.DataFrame], str]:
    try:
        import umap  # type: ignore
    except ImportError:
        raise ImportError(
            "Target head-weight projection panels require umap-learn so the saved plots are true UMAPs."
        )
    try:
        import scanpy as sc  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Target head-weight Leiden co-clustering requires scanpy "
            "(sc.pp.neighbors and sc.tl.leiden)."
        ) from exc
    method_used = "UMAP"

    common = pd.Index(summary["mean_dist"].index).astype(str)  # type: ignore[index]
    folds = folds or list(summary["folds"])  # type: ignore[arg-type]

    metrics = fold_metrics.copy()
    if "gene" in metrics.columns:
        metrics["gene"] = metrics["gene"].astype(str)
    elif "gene_or_perturbation" in metrics.columns:
        metrics = metrics.rename(columns={"gene_or_perturbation": "gene"})
        metrics["gene"] = metrics["gene"].astype(str)
    else:
        metrics = metrics.reset_index().rename(columns={"index": "gene"})
        metrics["gene"] = metrics["gene"].astype(str)

    projections = {}
    for fold in sorted(map(int, folds)):
        weights = _read_head_weight_matrix(run_dir, fold, include_bias=include_bias).reindex(common).dropna()
        if weights.shape[0] < 3:
            raise ValueError(f"Fold {fold} has fewer than three shared target weight vectors.")
        x = weights.to_numpy(dtype=float)
        graph_n_neighbors = min(n_neighbors, weights.shape[0] - 1)
        reducer = umap.UMAP(
            n_neighbors=graph_n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        coords = reducer.fit_transform(x)
        adata = sc.AnnData(X=x)
        sc.pp.neighbors(
            adata,
            n_neighbors=graph_n_neighbors,
            metric=metric,
            random_state=random_state,
        )
        sc.tl.leiden(adata, resolution=leiden_resolution, random_state=random_state)

        projection = pd.DataFrame(
            {
                "gene": weights.index.astype(str),
                "fold": fold,
                "dim1": coords[:, 0],
                "dim2": coords[:, 1],
                "head_weight_leiden": adata.obs["leiden"].astype(str).to_numpy(),
                "weight_norm": np.linalg.norm(x, axis=1),
            },
            index=weights.index.astype(str),
        )
        fold_metric = metrics[metrics["fold"].eq(fold)].drop(columns="fold", errors="ignore")
        projection = projection.join(fold_metric.set_index("gene"), on="gene", how="left")
        projections[fold] = projection
    return projections, method_used


def plot_head_weight_projection_per_fold(
    projection_tables: dict[int, pd.DataFrame],
    save_path: Path | None,
    *,
    method_used: str,
    color_mode: str = "cluster",
    label_top_n: int | None = 0,
    continuous_cmap=None,
) -> Path | None:
    folds = sorted(projection_tables)
    fig, axes = plt.subplots(1, len(folds), figsize=(10, 2.15))
    if len(folds) == 1:
        axes = [axes]

    if color_mode == "leiden":
        color_mode = "cluster"

    all_df = pd.concat(projection_tables.values(), ignore_index=True)
    if color_mode == "cluster":
        cluster_order = sorted(
            all_df["head_weight_leiden"].dropna().astype(str).unique(),
            key=lambda x: int(x) if x.isdigit() else x,
        )
        palette = _palette_values(len(cluster_order), palette="husl")
        color_lookup = dict(zip(cluster_order, palette))
        cbar_mappable = None
    else:
        col_lookup = {
            "auprc": ("average_precision", "AUPRC"),
            "prevalence": ("prevalence", "Prevalence"),
            "weight_norm": ("weight_norm", "Weight norm"),
        }
        if color_mode not in col_lookup:
            raise ValueError("color_mode must be one of 'leiden', 'cluster', 'auprc', 'prevalence', or 'weight_norm'.")
        value_col, colorbar_label = col_lookup[color_mode]
        norm = Normalize(vmin=float(all_df[value_col].min()), vmax=float(all_df[value_col].max()))
        cbar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=continuous_cmap or CMAP_BEIGE_PURPLE)

    for ax_idx, (ax, fold) in enumerate(zip(axes, folds)):
        d = projection_tables[fold].copy()
        if color_mode == "cluster":
            colors = d["head_weight_leiden"].astype(str).map(color_lookup)
            ax.scatter(d["dim1"], d["dim2"], s=18, c=colors, alpha=0.86, linewidth=0)
        else:
            ax.scatter(
                d["dim1"],
                d["dim2"],
                s=20,
                c=d[value_col],
                cmap=continuous_cmap or CMAP_BEIGE_PURPLE,
                norm=norm,
                alpha=0.9,
                linewidth=0.2,
                edgecolor="white",
            )
        if label_top_n is None:
            label_df = d.copy()
        elif label_top_n and "average_precision" in d.columns:
            label_df = d.nlargest(label_top_n, "average_precision")
        else:
            label_df = pd.DataFrame()
        if not label_df.empty:
            try:
                from adjustText import adjust_text  # type: ignore
            except ImportError as exc:
                raise ImportError("Labeled target head-weight UMAPs require adjustText.") from exc
            texts = []
            for _, row in label_df.iterrows():
                texts.append(ax.text(
                    row["dim1"],
                    row["dim2"],
                    str(row["gene"]),
                    fontsize=max(TICK_FS - 4, 4),
                    color="#202020",
                    alpha=0.92,
                    path_effects=[patheffects.withStroke(linewidth=1.0, foreground="white", alpha=0.88)],
                ))
            adjust_text(
                texts,
                x=d["dim1"].to_numpy(),
                y=d["dim2"].to_numpy(),
                ax=ax,
                expand=(1.02, 1.05),
                force_text=(0.08, 0.12),
                force_static=(0.03, 0.05),
                force_pull=(0.04, 0.04),
                lim=80,
                min_arrow_len=12,
                arrowprops={"arrowstyle": "-", "color": "#707070", "lw": 0.2, "alpha": 0.32},
            )
        ax.set_title(f"Fold {fold}", fontsize=TITLE_FS)
        ax.set_xlabel(f"{method_used}1", fontsize=AXIS_LABEL_FS)
        if ax_idx == 0:
            ax.set_ylabel(f"{method_used}2", fontsize=AXIS_LABEL_FS)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.grid(True, color="#D9D9D9", linewidth=0.35, alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    if color_mode == "cluster":
        legend_ncol = 2 if len(cluster_order) > 16 else 1
        fig.subplots_adjust(left=0.05, right=0.83, bottom=0.18, top=0.84, wspace=0.24)
        handles = [
            Line2D([], [], marker="o", linestyle="none", markersize=4, color=color_lookup[c], label=c)
            for c in cluster_order
        ]
        axes[-1].legend(
            handles=handles,
            title="Leiden cluster",
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=max(LEGEND_FS - 2, 5),
            title_fontsize=LEGEND_FS,
            ncol=legend_ncol,
            columnspacing=0.65,
            handletextpad=0.25,
        )
    elif cbar_mappable is not None:
        fig.subplots_adjust(left=0.05, right=0.88, bottom=0.18, top=0.84, wspace=0.24)
        cbar_ax = fig.add_axes([0.9, 0.26, 0.012, 0.48])
        cbar = fig.colorbar(cbar_mappable, cax=cbar_ax)
        cbar.set_label(colorbar_label, fontsize=COLORBAR_LABEL_FS)
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS)

    return _save(fig, save_path, tight_layout=False)


def head_weight_leiden_coclustering_matrix(
    projection_tables: dict[int, pd.DataFrame],
    *,
    cluster_col: str = "head_weight_leiden",
) -> pd.DataFrame:
    targets = sorted(set().union(*[set(d["gene"].astype(str)) for d in projection_tables.values()]))
    co = pd.DataFrame(0, index=targets, columns=targets, dtype=int)
    for projection in projection_tables.values():
        if cluster_col not in projection.columns:
            raise ValueError(f"Projection table is missing '{cluster_col}'.")
        target_to_cluster = projection.set_index(projection["gene"].astype(str))[cluster_col].astype(str)
        for t1, t2 in combinations(target_to_cluster.index, 2):
            if target_to_cluster[t1] == target_to_cluster[t2]:
                co.loc[t1, t2] += 1
                co.loc[t2, t1] += 1
        for target in target_to_cluster.index:
            co.loc[target, target] += 1
    return co


def plot_head_weight_leiden_coclustering_matrix(
    co: pd.DataFrame,
    save_path: Path | None,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (8, 8),
) -> Path | None:
    do_cluster = co.shape[0] > 1 and co.shape[1] > 1

    grid = sns.clustermap(
        co,
        row_cluster=do_cluster,
        col_cluster=do_cluster,
        cmap=CMAP_BEIGE_TEAL_PURPLE,
        figsize=figsize,
        cbar_kws={"shrink": 0.4},
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=(0.08, 0.08),
        colors_ratio=(0.015, 0.015),
    )

    ax = grid.ax_heatmap
    ax.set_xlabel("Target", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("Target", fontsize=AXIS_LABEL_FS)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    row_order = grid.dendrogram_row.reordered_ind if do_cluster else range(len(co.index))
    col_order = grid.dendrogram_col.reordered_ind if do_cluster else range(len(co.columns))
    ax.set_yticklabels([co.index[i] for i in row_order], fontsize=5, rotation=0)
    ax.set_xticklabels([co.columns[i] for i in col_order], fontsize=5, rotation=90)
    ax.tick_params(axis="x", pad=-1)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")
    for label in ax.get_yticklabels():
        label.set_horizontalalignment("left")

    for dend_ax in [grid.ax_row_dendrogram, grid.ax_col_dendrogram]:
        for coll in dend_ax.collections:
            coll.set_linewidth(0.5)

    grid.fig.subplots_adjust(left=0.03, right=0.84, bottom=0.22, top=0.94)
    if title:
        grid.fig.suptitle(title, fontsize=TITLE_FS, y=0.995)

    cbar = grid.ax_heatmap.collections[0].colorbar
    cbar.set_label(
        "# folds co-clustered by Leiden",
        fontsize=COLORBAR_LABEL_FS,
        rotation=270,
        labelpad=10,
    )
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS)
    return _save_clustergrid(grid, save_path)


def centroid_distance_summary(
    mapping: pd.DataFrame,
    embeddings: pd.DataFrame,
    *,
    metric: str = "euclidean",
    linkage_method: str = "average",
    n_clusters: int = 15,
) -> dict[str, object]:
    folds = sorted(mapping["fold"].dropna().unique())
    dist_by_fold = {}
    for fold in folds:
        idx = mapping.index[mapping["fold"].eq(fold)].intersection(embeddings.index)
        x = embeddings.loc[idx]
        meta = mapping.loc[idx]
        centroids = x.groupby(meta["gene_or_perturbation"].astype(str)).mean()
        dist_by_fold[fold] = pd.DataFrame(
            squareform(pdist(centroids.to_numpy(), metric=metric)),
            index=centroids.index.astype(str),
            columns=centroids.index.astype(str),
        )
    common = sorted(set.intersection(*(set(d.index) for d in dist_by_fold.values())))
    stack = np.stack([dist_by_fold[f].loc[common, common].to_numpy() for f in folds])
    mean_dist = pd.DataFrame(stack.mean(axis=0), index=common, columns=common)
    mean_dist = (mean_dist + mean_dist.T) / 2
    np.fill_diagonal(mean_dist.values, 0)
    z = linkage(squareform(mean_dist.to_numpy(), checks=False), method=linkage_method)
    order = mean_dist.index[leaves_list(z)].astype(str).tolist()
    clusters = pd.Series(fcluster(z, t=n_clusters, criterion="maxclust"), index=mean_dist.index.astype(str), name="dendrogram_cluster")
    return {"folds": folds, "mean_dist": mean_dist, "linkage": z, "order": order, "clusters": clusters}


def plot_centroid_dendrogram(
    summary: dict[str, object],
    save_path: Path | None,
    *,
    figsize: tuple[float, float] | None = None,
) -> Path | None:
    mean_dist: pd.DataFrame = summary["mean_dist"]  # type: ignore[assignment]
    z = summary["linkage"]
    clusters: pd.Series = summary["clusters"]  # type: ignore[assignment]

    # Scale the figure modestly with the number of perturbations so dense label sets
    # (e.g. K562, ~155 targets ~= 14 in) stay legible; RPE1 (~69 targets) keeps 9x9.
    if figsize is None:
        side = max(9.0, 0.09 * len(mean_dist))
        figsize = (side, side)

    palette = _palette_values(clusters.nunique(), palette="husl")
    cluster_to_color = {
        cluster_id: palette[i]
        for i, cluster_id in enumerate(sorted(clusters.unique()))
    }

    row_colors = clusters.reindex(mean_dist.index).map(cluster_to_color)

    grid = sns.clustermap(
        mean_dist,
        row_linkage=z,
        col_linkage=z,
        row_colors=row_colors,
        col_colors=row_colors,
        cmap=CMAP_BEIGE_PURPLE,
        figsize=figsize,
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=(0.08, 0.08),
        colors_ratio=(0.015, 0.015),
        cbar_kws={"shrink": 0.4},
    )

    ax = grid.ax_heatmap

    ax.set_xlabel("Perturbation", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("Perturbation", fontsize=AXIS_LABEL_FS)

    # Put y tick labels on the right so they are not hidden by row dendrogram/colors.
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    ax.tick_params(axis="x", labelsize=5, rotation=90, pad=1)
    ax.tick_params(axis="y", labelsize=5, rotation=0, pad=2)

    # Preserve all labels after clustering order.
    row_order = grid.dendrogram_row.reordered_ind
    col_order = grid.dendrogram_col.reordered_ind

    ax.set_yticklabels(
        [mean_dist.index[i] for i in row_order],
        fontsize=5,
        rotation=0,
    )
    ax.set_xticklabels(
        [mean_dist.columns[i] for i in col_order],
        fontsize=5,
        rotation=90,
    )

    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")

    for label in ax.get_yticklabels():
        label.set_horizontalalignment("left")

    # Thinner dendrogram lines.
    for dend_ax in [grid.ax_row_dendrogram, grid.ax_col_dendrogram]:
        for coll in dend_ax.collections:
            coll.set_linewidth(0.5)

    # Give right-side labels and legend room.
    grid.fig.subplots_adjust(
        left=0.03,
        right=0.78,
        bottom=0.24,
        top=0.96,
    )

    # subplots_adjust drags the colorbar back to the cramped top-left corner, where
    # its rotated label is clipped by the figure edge. Pin it as a short bar in the
    # free right margin, just below the "Dendrogram cut" legend, AFTER the adjust.
    grid.cax.set_position([0.87, 0.44, 0.02, 0.14])

    handles = [
        Patch(facecolor=cluster_to_color[c], edgecolor="black", label=f"Cluster {c}")
        for c in sorted(cluster_to_color)
    ]

    grid.ax_heatmap.legend(
        handles=handles,
        title="Dendrogram cut",
        loc="upper left",
        bbox_to_anchor=(1.22, 1.0),
        frameon=True,
        fontsize=max(TICK_FS - 2, 6),
        title_fontsize=AXIS_LABEL_FS,
    )

    cbar = grid.ax_heatmap.collections[0].colorbar
    cbar.set_label(
        "Euclidean distance",
        fontsize=COLORBAR_LABEL_FS,
        rotation=270,
        labelpad=10,
    )
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS)

    return _save_clustergrid(grid, save_path)


def select_perturbations_from_clusters(summary: dict[str, object], *, excluded_clusters: tuple[int, ...] = (9,)) -> list[str]:
    order: list[str] = summary["order"]  # type: ignore[assignment]
    clusters: pd.Series = summary["clusters"]  # type: ignore[assignment]
    return [p for p in order if int(clusters.loc[p]) not in set(excluded_clusters)]


def build_fold_projection_tables(
    mapping: pd.DataFrame,
    embeddings: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    *,
    method: str = "scanpy_umap",
    max_points_per_fold: int | None = None,
    random_state: int = 42,
) -> tuple[dict[int, object], str]:
    if method not in {"scanpy_umap", "umap"}:
        raise ValueError("Figure 5 embedding panels use the legacy Scanpy UMAP/Leiden path; set method='scanpy_umap'.")
    try:
        import scanpy as sc  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Figure 5 embedding panels require scanpy to match the legacy notebook "
            "(sc.pp.neighbors, sc.tl.umap, sc.tl.leiden). Install scanpy in this environment."
        ) from exc

    rng = np.random.default_rng(random_state)
    adata_by_fold = {}
    for fold in sorted(mapping["fold"].dropna().unique()):
        idx = mapping.index[mapping["fold"].eq(fold)].intersection(embeddings.index)
        if max_points_per_fold is not None and len(idx) > max_points_per_fold:
            idx = pd.Index(rng.choice(idx.to_numpy(), size=max_points_per_fold, replace=False))
        adata = sc.AnnData(
            X=embeddings.loc[idx].to_numpy(dtype=np.float32),
            obs=mapping.loc[idx].copy(),
        )
        adata.obs["gene_or_perturbation"] = adata.obs["gene_or_perturbation"].astype(str)
        if fold_metrics is not None:
            fold_m = fold_metrics[fold_metrics["fold"].eq(fold)].drop(columns="fold", errors="ignore")
            if "gene" in fold_m.columns:
                fold_m = fold_m.copy()
                fold_m["gene"] = fold_m["gene"].astype(str)
                fold_m = fold_m.set_index("gene")
            elif "gene_or_perturbation" in fold_m.columns:
                fold_m = fold_m.copy()
                fold_m["gene_or_perturbation"] = fold_m["gene_or_perturbation"].astype(str)
                fold_m = fold_m.set_index("gene_or_perturbation")
            else:
                fold_m.index = fold_m.index.astype(str)
            adata.obs = adata.obs.join(fold_m, on="gene_or_perturbation", how="left")
        adata.obs["gene_or_perturbation"] = adata.obs["gene_or_perturbation"].astype(str).astype("category")
        sc.pp.neighbors(adata, use_rep="X")
        sc.tl.umap(adata)
        sc.tl.leiden(adata)
        adata_by_fold[int(fold)] = adata
    return adata_by_fold, "umap"


def _projection_obs(projection) -> pd.DataFrame:
    obs = projection.obs.copy()
    coords = projection.obsm["X_umap"]
    obs["dim1"] = coords[:, 0]
    obs["dim2"] = coords[:, 1]
    return obs


def _selected_perturbation_order(projection_tables: dict[int, object], selected_perturbations: list[str] | None) -> list[str]:
    folds = sorted(projection_tables)
    return [
        str(p)
        for p in (selected_perturbations or [])
        if any(_projection_obs(projection_tables[f])["gene_or_perturbation"].astype(str).eq(str(p)).any() for f in folds)
    ]


def _selected_perturbation_colors(selected_order: list[str]) -> dict[str, object]:
    base_colors = list(plt.cm.tab20.colors)
    return {pert: base_colors[i % len(base_colors)] for i, pert in enumerate(selected_order)}


def plot_projection_per_fold(
    projection_tables: dict[int, object],
    save_path: Path | None,
    *,
    method_used: str,
    color_mode: str,
    selected_perturbations: list[str] | None = None,
) -> Path | None:
    import scanpy as sc  # type: ignore

    folds = sorted(projection_tables)
    fig, axes = plt.subplots(1, len(folds), figsize=(3 * len(folds), 3))
    if len(folds) == 1:
        axes = [axes]
    selected = set(selected_perturbations or [])
    selected_order = _selected_perturbation_order(projection_tables, selected_perturbations)

    for ax, fold in zip(axes, folds):
        projection = projection_tables[fold]
        if color_mode == "perturbation":
            ad = projection.copy()
            col = "gene_or_perturbation"
            highlight_col = "perturbation_highlight"
            ad.obs[col] = ad.obs[col].astype(str)
            ad.obs[highlight_col] = ad.obs[col].where(ad.obs[col].isin(selected), "other")
            present = [x for x in selected_order if str(x) in set(ad.obs[highlight_col])]
            ad.obs[highlight_col] = pd.Categorical(
                ad.obs[highlight_col],
                categories=["other"] + [str(x) for x in present],
            )
            ad.uns.pop(f"{col}_colors", None)
            ad.uns.pop(f"{highlight_col}_colors", None)
            sc.pl.umap(
                ad,
                color=highlight_col,
                ax=ax,
                show=False,
                legend_loc=None,
                frameon=False,
                title=f"Fold {fold}",
                palette=["lightgrey"] + list(plt.cm.tab20.colors) * 10,
            )
        elif color_mode == "leiden":
            sc.pl.umap(
                projection,
                color="leiden",
                ax=ax,
                show=False,
                legend_loc=None,
                frameon=False,
                title=f"Fold {fold}",
            )
        elif color_mode == "auprc":
            sc.pl.umap(
                projection,
                color="average_precision",
                ax=ax,
                show=False,
                legend_loc=None,
                frameon=False,
                title=f"Fold {fold}",
                colorbar_loc=None if fold != folds[-1] else "right",
            )
        else:
            raise ValueError("color_mode must be one of 'perturbation', 'leiden', or 'auprc'.")
        ax.set_title(f"Fold {fold}", fontsize=TITLE_FS)
        ax.set_axis_off()
    return _save(fig, save_path)


def plot_selected_umap_perturbation_legend(
    projection_tables: dict[int, object],
    save_path: Path | None,
    *,
    selected_perturbations: list[str],
    ncol: int = 16,
) -> Path | None:
    perturbations = _selected_perturbation_order(projection_tables, selected_perturbations)
    colors = _selected_perturbation_colors(perturbations)
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=4.2,
            markerfacecolor=colors[pert],
            markeredgecolor="none",
            label=pert,
        )
        for pert in perturbations
    ]
    rows = max(1, math.ceil(len(perturbations) / ncol))
    fig_width = max(8, min(20, ncol * 1.05))
    fig_height = max(2.1, min(8.0, 0.31 * rows + 0.9))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        frameon=False,
        ncol=ncol,
        title="Selected perturbation",
        fontsize=max(LEGEND_FS - 2, 5),
        title_fontsize=LEGEND_FS,
        handletextpad=0.25,
        columnspacing=0.75,
        borderaxespad=0,
    )
    return _save(fig, save_path)


def perturbation_coclustering_matrix(projection_tables: dict[int, object]) -> pd.DataFrame:
    perts = sorted(set().union(*[set(_projection_obs(d)["gene_or_perturbation"].astype(str)) for d in projection_tables.values()]))
    co = pd.DataFrame(0, index=perts, columns=perts, dtype=int)
    for projection in projection_tables.values():
        d = _projection_obs(projection)
        leiden_col = "leiden" if "leiden" in d.columns else "cluster"
        ct = pd.crosstab(d["gene_or_perturbation"].astype(str), d[leiden_col].astype(str))
        pert_to_cluster = ct.idxmax(axis=1)
        for p1, p2 in combinations(pert_to_cluster.index, 2):
            if pert_to_cluster[p1] == pert_to_cluster[p2]:
                co.loc[p1, p2] += 1
                co.loc[p2, p1] += 1
        for p in pert_to_cluster.index:
            co.loc[p, p] += 1
    return co


def _perturbation_annotation_colors(perturbations, clusters: pd.Series, selected_perturbations: list[str] | None = None) -> pd.DataFrame:
    perturbations = pd.Index(perturbations).astype(str)
    cluster_labels = clusters.reindex(perturbations)
    ordered_clusters = sorted(cluster_labels.dropna().unique())
    palette = _palette_values(len(ordered_clusters), palette="husl")
    cluster_to_color = {c: palette[i] for i, c in enumerate(ordered_clusters)}
    selected = set(map(str, selected_perturbations or []))
    selected_colors = pd.Series(perturbations.isin(selected), index=perturbations).map({True: "#D55E00", False: "#D9D9D9"})
    return pd.DataFrame(
        {
            "dendrogram cluster": cluster_labels.map(cluster_to_color).fillna("#D9D9D9"),
            "selected": selected_colors,
        },
        index=perturbations,
    )


def plot_coclustering_matrix(
    co: pd.DataFrame,
    summary: dict[str, object],
    save_path: Path | None,
    *,
    selected_perturbations: list[str] | None = None,
) -> Path | None:
    clusters: pd.Series = summary["clusters"]  # type: ignore[assignment]

    shared = [p for p in co.index if p in clusters.index]
    co = co.loc[shared, shared]

    co_colors = _perturbation_annotation_colors(
        co.index,
        clusters,
        selected_perturbations,
    )

    do_cluster = co.shape[0] > 1 and co.shape[1] > 1

    grid = sns.clustermap(
        co,
        row_cluster=do_cluster,
        col_cluster=do_cluster,
        row_colors=co_colors,
        col_colors=co_colors,
        cmap=CMAP_BEIGE_TEAL_PURPLE,
        figsize=(9, 9),
        cbar_kws={"shrink": 0.4},

        # Force all perturbations to be shown
        xticklabels=True,
        yticklabels=True,

        # Make dendrograms smaller:
        # first value = row dendrogram width
        # second value = column dendrogram height
        dendrogram_ratio=(0.08, 0.08),

        # Make row_colors / col_colors thinner
        colors_ratio=(0.015, 0.015),
    )

    ax = grid.ax_heatmap

    ax.set_xlabel("Perturbation", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("Perturbation", fontsize=AXIS_LABEL_FS)

    # Put y labels on the right so they are not hidden behind row dendrogram/colors
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    ax.tick_params(axis="x", labelsize=5, rotation=90, pad=-1)

    # Ensure every label is present after clustering reordering
    row_order = grid.dendrogram_row.reordered_ind if do_cluster else range(len(co.index))
    col_order = grid.dendrogram_col.reordered_ind if do_cluster else range(len(co.columns))

    ax.set_yticklabels([co.index[i] for i in row_order], fontsize=5)
    ax.set_xticklabels([co.columns[i] for i in col_order], fontsize=5, rotation=90)

    # Make dendrogram lines thinner
    for dend_ax in [grid.ax_row_dendrogram, grid.ax_col_dendrogram]:
        for coll in dend_ax.collections:
            coll.set_linewidth(0.5)

    # Give tick labels more room
    grid.fig.subplots_adjust(left=0.03, right=0.86, bottom=0.22, top=0.96)

    cbar = grid.ax_heatmap.collections[0].colorbar
    cbar.set_label(
        "# folds co-clustered",
        fontsize=COLORBAR_LABEL_FS,
        rotation=270,
        labelpad=10,
    )
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS)

    return _save_clustergrid(grid, save_path)

def plot_selected_cluster_heatmap(
    projection_tables: dict[int, object],
    selected_perturbations: list[str],
    summary: dict[str, object],
    save_path: Path | None,
    *,
    fold: int = 1,
    selected_leiden_clusters: list[int | str] | None = None,
    min_cells: int = 0,
    figsize=(5, 2.8),
) -> Path | None:
    clusters: pd.Series = summary["clusters"]  # type: ignore[assignment]
    d = _projection_obs(projection_tables[fold])
    leiden_col = "leiden" if "leiden" in d.columns else "cluster"
    obs = d[["gene_or_perturbation", leiden_col]].copy()
    obs["gene_or_perturbation"] = obs["gene_or_perturbation"].astype(str)
    obs[leiden_col] = obs[leiden_col].astype(str)
    # Normalize each perturbation over all Leiden clusters before selecting rows
    # for display, so values retain their all-cell interpretation.
    ct_all = pd.crosstab(obs[leiden_col], obs["gene_or_perturbation"])
    keep_perts_all = ct_all.sum(axis=0)[ct_all.sum(axis=0) >= min_cells].index
    ct_all = ct_all.loc[:, keep_perts_all]
    if ct_all.shape[0] == 0 or ct_all.shape[1] == 0:
        raise ValueError(f"Fold {fold}: no Leiden clusters or perturbations left after filtering.")
    ct_norm_all = ct_all.div(ct_all.sum(axis=0), axis=1).fillna(0)

    if selected_leiden_clusters is not None:
        selected_leiden_clusters = [str(x) for x in selected_leiden_clusters]
        present_leiden_clusters = [cluster for cluster in selected_leiden_clusters if cluster in ct_all.index]
        if not present_leiden_clusters:
            raise ValueError(f"Fold {fold}: none of the selected Leiden clusters are present.")
        ct_norm_display = ct_norm_all.loc[present_leiden_clusters]
    else:
        ct_norm_display = ct_norm_all

    selected = [str(p) for p in selected_perturbations]
    keep_perts_show = [p for p in selected if p in ct_norm_all.columns and p in clusters.index]
    if len(keep_perts_show) == 0:
        raise ValueError(f"Fold {fold}: none of the selected perturbations are present after filtering.")
    mat = ct_norm_display.loc[:, keep_perts_show]
    n_perts = mat.shape[1]
    n_leiden = mat.shape[0]
    grid = sns.clustermap(
        mat,
        row_cluster=n_leiden > 1,
        col_cluster=n_perts > 1,
        cmap=CMAP_BEIGE_TEAL_PURPLE,
        vmin=0,
        vmax=ct_norm_all.to_numpy().max(),
        figsize=figsize,
        dendrogram_ratio=(0.06, 0.22),
        cbar_pos=(-0.03, 0.22, 0.025, 0.16),
        col_colors=None,
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"shrink": 0.6},
    )
    grid.ax_heatmap.set_xlabel("Selected perturbation", fontsize=AXIS_LABEL_FS, labelpad=2)
    grid.ax_heatmap.set_ylabel("Selected Leiden cluster", fontsize=AXIS_LABEL_FS)
    grid.ax_heatmap.tick_params(axis="x", labelsize=max(TICK_FS - 2, 5), pad=-2)
    grid.ax_heatmap.tick_params(axis="y", labelsize=TICK_FS, rotation=0, pad=1)
    cbar = grid.ax_heatmap.collections[0].colorbar
    cbar.ax.set_title("Cell fraction", fontsize=COLORBAR_LABEL_FS, pad=6)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS)
    return _save_clustergrid(grid, save_path)


def summarize_cluster_purity_vs_auprc_one_fold(
    projection_tables: dict[int, object],
    *,
    fold: int = 1,
    selected_leiden_clusters: list[int | str] | None = None,
    perturbation_col: str = "gene_or_perturbation",
    cluster_col: str = "leiden",
    auprc_col: str = "average_precision",
    min_cells_per_perturbation: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Summarize whether purer Leiden clusters correspond to higher AUPRC.

    Purity is computed per Leiden cluster as the fraction of cells in that
    cluster assigned to a given perturbation. Each perturbation is represented
    by the Leiden cluster where it reaches its highest purity.
    """
    if fold not in projection_tables:
        raise KeyError(f"Fold {fold} is not available in projection_tables.")

    obs = _projection_obs(projection_tables[fold])
    required = {perturbation_col, cluster_col, auprc_col}
    missing = sorted(required.difference(obs.columns))
    if missing:
        raise ValueError(f"Fold {fold}: missing required columns: {missing}")

    obs = obs.dropna(subset=[perturbation_col, cluster_col, auprc_col]).copy()
    obs[perturbation_col] = obs[perturbation_col].astype(str)
    obs[cluster_col] = obs[cluster_col].astype(str)

    if selected_leiden_clusters is not None:
        selected = {str(x) for x in selected_leiden_clusters}
        obs = obs[obs[cluster_col].isin(selected)].copy()

    if obs.empty:
        raise ValueError(f"Fold {fold}: no cells left after filtering.")

    counts = (
        obs.groupby([cluster_col, perturbation_col], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    cluster_totals = (
        obs.groupby(cluster_col, observed=False)
        .size()
        .rename("cluster_total")
        .reset_index()
    )
    perturbation_totals = (
        obs.groupby(perturbation_col, observed=False)
        .size()
        .rename("perturbation_total")
        .reset_index()
    )
    auprc = (
        obs.groupby(perturbation_col, observed=False, as_index=False)[auprc_col]
        .mean()
        .rename(columns={auprc_col: "auprc"})
    )

    long_df = (
        counts
        .merge(cluster_totals, on=cluster_col, how="left")
        .merge(perturbation_totals, on=perturbation_col, how="left")
        .merge(auprc, on=perturbation_col, how="left")
    )
    long_df["cluster_purity"] = long_df["n_cells"] / long_df["cluster_total"]

    summary = (
        long_df.sort_values(["cluster_purity", "n_cells"], ascending=[False, False])
        .groupby(perturbation_col, observed=False, as_index=False)
        .first()
        .rename(columns={
            cluster_col: "best_leiden_cluster",
            "cluster_purity": "best_cluster_purity",
        })
    )
    summary = summary[summary["perturbation_total"] >= min_cells_per_perturbation].copy()
    summary["fold"] = fold

    if len(summary) >= 2:
        spear = spearmanr(summary["best_cluster_purity"], summary["auprc"])
        pear = pearsonr(summary["best_cluster_purity"], summary["auprc"])
        spearman_rho = float(spear.statistic)
        spearman_p = float(spear.pvalue)
        pearson_r = float(pear.statistic)
        pearson_p = float(pear.pvalue)
    else:
        spearman_rho = spearman_p = pearson_r = pearson_p = float("nan")

    stats = {
        "fold": float(fold),
        "n_perturbations": float(len(summary)),
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
    }
    return long_df, summary, stats


def plot_cluster_purity_vs_auprc_one_fold(
    summary: pd.DataFrame,
    stats: dict[str, float],
    save_path: Path | None,
    *,
    title: str = "Cluster purity vs predictivity",
    label_top_n: int = 10,
    figsize=(3.1, 2.55),
) -> Path | None:
    plot_df = summary.dropna(subset=["best_cluster_purity", "auprc"]).copy()
    if plot_df.empty:
        raise ValueError("No perturbations available for plotting.")

    fig, ax = plt.subplots(figsize=figsize)
    size_values = plot_df["perturbation_total"].astype(float)
    sizes = 18 + 75 * (
        (size_values - size_values.min()) /
        (size_values.max() - size_values.min() + 1e-9)
    )

    cluster_order = sorted(plot_df["best_leiden_cluster"].astype(str).unique(), key=lambda x: int(x) if x.isdigit() else x)
    cluster_colors = {
        cluster: color
        for cluster, color in zip(
            cluster_order,
            _palette_values(len(cluster_order), palette="husl"),
        )
    }
    point_colors = plot_df["best_leiden_cluster"].astype(str).map(cluster_colors)

    ax.scatter(
        plot_df["best_cluster_purity"],
        plot_df["auprc"],
        s=sizes,
        c=point_colors,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.35,
    )

    label_df = plot_df.sort_values("auprc", ascending=False).head(label_top_n)
    for _, row in label_df.iterrows():
        ax.text(
            row["best_cluster_purity"] + 0.006,
            row["auprc"] + 0.006,
            str(row["gene_or_perturbation"]),
            fontsize=max(TICK_FS - 2, 5),
            ha="left",
            va="center",
        )

    spearman_p = stats.get("spearman_p", float("nan"))
    if np.isfinite(spearman_p):
        p_label = f"p={spearman_p:.1e}" if spearman_p < 1e-3 else f"p={spearman_p:.3f}"
    else:
        p_label = "p=NA"

    ax.text(
        0.97,
        0.05,
        f"Spearman rho={stats.get('spearman_rho', float('nan')):.2f}\n{p_label}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=max(TICK_FS - 1, 6),
    )
    ax.set_xlabel("Best cluster purity", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("AUPRC", fontsize=AXIS_LABEL_FS)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(axis="both", alpha=0.18, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=4,
            markerfacecolor=cluster_colors[cluster],
            markeredgecolor="black",
            markeredgewidth=0.35,
            label=cluster,
        )
        for cluster in cluster_order
    ]
    ax.legend(
        handles=legend_handles,
        title="Best Leiden",
        frameon=False,
        fontsize=max(TICK_FS - 2, 5),
        title_fontsize=max(TICK_FS - 1, 6),
        bbox_to_anchor=(1.12, 1.0),
        loc="upper left",
        borderaxespad=0,
        handletextpad=0.3,
        labelspacing=0.25,
    )

    return _save(fig, save_path)


def plot_leiden_composition_heatmap_for_fold(
    projection_tables: dict[int, object],
    summary: dict[str, object],
    selected_perturbations: list[str],
    save_path: Path | None,
    *,
    fold: int,
    min_cells: int = 0,
) -> Path | None:
    clusters: pd.Series = summary["clusters"]  # type: ignore[assignment]

    d = _projection_obs(projection_tables[fold])
    leiden_col = "leiden" if "leiden" in d.columns else "cluster"

    obs = d[["gene_or_perturbation", leiden_col]].copy()
    obs["gene_or_perturbation"] = obs["gene_or_perturbation"].astype(str)
    obs[leiden_col] = obs[leiden_col].astype(str)

    ct = pd.crosstab(obs[leiden_col], obs["gene_or_perturbation"])

    keep_perts = ct.sum(axis=0)[ct.sum(axis=0) >= min_cells].index
    keep_perts = [p for p in keep_perts if p in clusters.index]

    if len(keep_perts) == 0:
        raise ValueError(f"Fold {fold}: no perturbations left after filtering.")

    ct = ct.loc[:, keep_perts]

    if ct.shape[0] == 0 or ct.shape[1] == 0:
        raise ValueError(f"Fold {fold}: empty Leiden x perturbation matrix.")

    ct_norm = ct.div(ct.sum(axis=0), axis=1).fillna(0)

    col_colors = _perturbation_annotation_colors(
        ct_norm.columns,
        clusters,
        selected_perturbations,
    )

    n_perts = ct_norm.shape[1]
    n_clusters = ct_norm.shape[0]

    fig_width = max(7, 0.13 * n_perts)
    fig_height = max(4, 0.18 * n_clusters)

    grid = sns.clustermap(
        ct_norm,
        row_cluster=n_clusters > 1,
        col_cluster=n_perts > 1,
        col_colors=col_colors,
        cmap=CMAP_BEIGE_TEAL_PURPLE,
        vmin=0,
        vmax=max(float(ct_norm.to_numpy().max()), 1e-12),
        figsize=(fig_width, fig_height),
        dendrogram_ratio=(0.04, 0.05),
        colors_ratio=(0.04, 0.04),
        xticklabels=True,
        yticklabels=True,
        cbar_pos=None,
    )

    ax = grid.ax_heatmap

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    ax.tick_params(axis="x", labelsize=5, rotation=90, pad=1)
    ax.tick_params(axis="y", labelsize=6, rotation=0, pad=2)

    row_order = (
        grid.dendrogram_row.reordered_ind
        if n_clusters > 1
        else range(len(ct_norm.index))
    )
    col_order = (
        grid.dendrogram_col.reordered_ind
        if n_perts > 1
        else range(len(ct_norm.columns))
    )

    ax.set_yticklabels(
        [ct_norm.index[i] for i in row_order],
        fontsize=6,
        rotation=0,
    )
    ax.set_xticklabels(
        [ct_norm.columns[i] for i in col_order],
        fontsize=5,
        rotation=90,
    )

    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation_mode("anchor")

    for label in ax.get_yticklabels():
        label.set_horizontalalignment("left")

    ax.set_xlabel("Perturbation", fontsize=AXIS_LABEL_FS - 1)
    ax.set_ylabel("Leiden cluster", fontsize=AXIS_LABEL_FS - 1)

    for dend_ax in [grid.ax_row_dendrogram, grid.ax_col_dendrogram]:
        for coll in dend_ax.collections:
            coll.set_linewidth(0.5)

    grid.fig.subplots_adjust(
        left=0.08,
        right=0.86,
        bottom=0.30,
        top=0.90,
    )

    grid.fig.suptitle(
        f"Fold {fold}: perturbation distribution across Leiden clusters",
        fontsize=TITLE_FS,
        y=0.98,
    )

    # Create a stable, manually positioned colorbar.
    # Position = [left, bottom, width, height] in figure coordinates.
    cbar_ax = grid.fig.add_axes([0.025, 0.30, 0.018, 0.15])
    mappable = grid.ax_heatmap.collections[0]
    cbar = grid.fig.colorbar(mappable, cax=cbar_ax)

    cbar.set_label(
        "Cells fraction",
        fontsize=COLORBAR_LABEL_FS,
        rotation=270,
        labelpad=10,
    )
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS)

    return _save_clustergrid(grid, save_path)
