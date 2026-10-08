from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator

from constants import (
    ANNOTATION_FS,
    AXIS_LABEL_FS,
    BEIGE,
    COLORBAR_LABEL_FS,
    COLORBAR_TICK_FS,
    LEGEND_FS,
    ORANGE,
    PURPLE,
    TEAL,
    TICK_FS,
    TITLE_FS,
    CMAP_BEIGE_PURPLE,
)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_panel(fig, save_path: str | Path) -> None:
    save_path = Path(save_path)
    _ensure_parent(save_path)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"saved: {save_path}")


def save_table(df: pd.DataFrame, save_path: str | Path, **kwargs) -> None:
    save_path = Path(save_path)
    _ensure_parent(save_path)
    df.to_csv(save_path, **kwargs)
    print(f"saved: {save_path}")


def gene_symbol(value) -> str:
    return str(value).split("|")[-1]


def beeswarm_pairs_from_parquet(df: pd.DataFrame) -> list[str]:
    """Return SHAP feature columns that have matching expression columns."""
    return [
        col for col in df.columns
        if col != "sample_id" and not col.startswith("x_") and f"x_{col}" in df.columns
    ]


def load_beeswarm_table(results_root: str | Path, cancer: str, target_gene: str) -> pd.DataFrame:
    """Per-sample SHAP and expression of the top features of one target, one column pair per feature (x_ = expression)."""
    path = Path(results_root) / cancer / "shap" / target_gene / "sample_shap_top_features.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing SHAP sample table: {path}")
    long = pd.read_csv(path)
    features = long["feature"].drop_duplicates().tolist()  # rows are grouped by feature in SHAP rank order
    shap = long.pivot(index="sample_id", columns="feature", values="shap_value")[features]
    expression = long.pivot(index="sample_id", columns="feature", values="expression_value")[features].add_prefix("x_")
    df = pd.concat([shap, expression], axis=1).reset_index()
    df.columns.name = None
    if "sample_id" not in df.columns:
        raise ValueError(f"{path} is missing sample_id")
    if not beeswarm_pairs_from_parquet(df):
        raise ValueError(f"{path} has no SHAP/expression feature pairs")
    return df


def load_shap_feature_summaries(results_root: str | Path, exclude_all: bool = True) -> pd.DataFrame:
    """Load the per-target SHAP feature summaries of all cohorts into one standardized DataFrame."""
    results_root = Path(results_root)
    rows = []
    for shap_path in sorted(results_root.glob("*/shap/*/feature_summary.csv")):
        cohort = shap_path.parts[-4]
        if exclude_all and cohort.lower() == "all":
            continue
        df = pd.read_csv(shap_path).rename(columns={"target": "target_gene"})
        df["cohort"] = cohort
        required = {"cohort", "target_gene", "feature", "mean_abs_shap", "mean_shap"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{shap_path} is missing columns: {sorted(missing)}")
        df = df.loc[:, ["cohort", "target_gene", "feature", "mean_abs_shap", "mean_shap"]].copy()
        df["cohort"] = cohort
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["cohort", "target_gene", "feature", "feature_symbol", "mean_abs_shap", "mean_shap"])
    out = pd.concat(rows, ignore_index=True)
    out["feature_symbol"] = out["feature"].map(gene_symbol)
    out["target_gene"] = out["target_gene"].astype(str)
    out["mean_abs_shap"] = pd.to_numeric(out["mean_abs_shap"], errors="coerce")
    out["mean_shap"] = pd.to_numeric(out["mean_shap"], errors="coerce")
    return out.dropna(subset=["mean_abs_shap", "mean_shap"])


def load_mutation_metrics(results_root: str | Path, exclude_all: bool = True) -> pd.DataFrame:
    """Load per-cohort mutation-prediction summaries used for Figure 2 joins."""
    results_root = Path(results_root)
    rows = []
    for summary_path in sorted(results_root.glob("*/cv/metrics.csv")):
        cohort = summary_path.parts[-3]
        if exclude_all and cohort.lower() == "all":
            continue
        df = pd.read_csv(summary_path, index_col=0)
        required = {"auprc", "prevalence", "normalized_auprc"}
        if not required.issubset(df.columns):
            continue
        out = df.loc[:, ["auprc", "prevalence", "normalized_auprc"]].copy()
        out.insert(0, "target_gene", out.index.astype(str))
        out.insert(0, "cohort", cohort)
        metadata_path = results_root / cohort / "cv" / "run_metadata.json"
        samples = np.nan
        n_targets = np.nan
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
                samples = metadata.get("n_samples", np.nan)
                n_targets = metadata.get("n_targets", np.nan)
            except Exception:
                pass
        out["samples"] = samples
        out["n_targets"] = n_targets
        rows.append(out.reset_index(drop=True))
    if not rows:
        return pd.DataFrame(columns=["cohort", "target_gene", "auprc", "prevalence", "normalized_auprc"])
    return pd.concat(rows, ignore_index=True)


def shap_matrix_for_cohort(shap_long: pd.DataFrame, cohort: str) -> pd.DataFrame:
    """Return a legacy-style SHAP matrix: rows=features, columns=target genes."""
    sub = shap_long.loc[shap_long["cohort"].eq(cohort)].copy()
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(
        index="feature_symbol",
        columns="target_gene",
        values="mean_shap",
        aggfunc="mean",
    )


def plot_accuracy_vs_n_features(
    shap_df: pd.DataFrame,
    acc: pd.Series,
    *,
    cmap=None,
    prevalence: pd.Series | None = None,
    cancer: str = "",
    min_features: int = 1,
    threshold: float | None = None,
    annotate_thresh: float = 0.45,
    max_labels: int = 250,
    figsize: tuple = (3, 2.4),
    point_size: float = 30,
    save_path=None,
) -> pd.DataFrame:
    """Legacy Figure2 scatter: mutation-gene AUPRC versus number of SHAP features."""
    if cmap is None:
        raise ValueError("Pass cmap explicitly.")

    if threshold is None:
        n_features = (~shap_df.isna()).sum(axis=0)
        xlab = "Number of SHAP genes"
    else:
        n_features = (shap_df.abs() > float(threshold)).sum(axis=0)
        xlab = "Number of expression features"

    df = pd.DataFrame({"score": acc, "n_features": n_features}).dropna()
    df = df[df["n_features"] >= min_features].copy()

    cvals, norm, cm = None, None, None
    if prevalence is not None:
        df["prevalence"] = prevalence.reindex(df.index).astype(float)
        cvals = df["prevalence"].to_numpy(dtype=float)
        finite = np.isfinite(cvals)
        if finite.any():
            norm = Normalize(vmin=float(np.nanmin(cvals)), vmax=float(np.nanmax(cvals)))
            cm = cmap

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        df["n_features"].to_numpy(),
        df["score"].to_numpy(),
        c=cvals if cm is not None else None,
        cmap=cm,
        norm=norm,
        s=point_size,
        alpha=0.75,
        edgecolors="black",
        linewidths=0.35,
        zorder=2,
    )
    ax.set_xlabel(xlab, fontsize=AXIS_LABEL_FS, labelpad=6)
    ax.set_ylabel("AUPRC", fontsize=AXIS_LABEL_FS, labelpad=6)
    if cancer:
        ax.set_title(cancer, fontsize=TITLE_FS, pad=8)
    ax.tick_params(axis="both", which="major", length=0, labelsize=TICK_FS, pad=6)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if cm is not None and norm is not None:
        cbar = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
        cbar.set_label("Prevalence", fontsize=COLORBAR_LABEL_FS, rotation=270, labelpad=10)
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS, length=0)
        cbar.outline.set_visible(False)

    label_df = (
        df[df["score"] >= float(annotate_thresh)]
        .sort_values("score", ascending=False)
        .head(max_labels)
    )
    if not label_df.empty:
        texts = [
            ax.text(float(row["n_features"]), float(row["score"]), str(gene), fontsize=ANNOTATION_FS, zorder=3)
            for gene, row in label_df.iterrows()
        ]
        try:
            from adjustText import adjust_text

            adjust_text(
                texts,
                ax=ax,
                expand_points=(1.15, 1.15),
                expand_text=(1.15, 1.15),
                arrowprops=dict(arrowstyle="-", lw=0.5, color="gray", alpha=0.8),
            )
        except Exception:
            pass

    fig.tight_layout()
    if save_path:
        save_panel(fig, save_path)
    return df


def plot_beeswarm_from_table(
    beeswarm_df: pd.DataFrame,
    *,
    cancer: str,
    target_gene: str,
    max_display: int = 20,
    cmap=None,
    figsize: tuple[float, float] | None = None,
    point_size: float = 14,
    alpha: float = 0.85,
    title: str | None = None,
    prefer_shap: bool = True,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    """Styled SHAP beeswarm using saved per-sample SHAP and expression values."""
    feature_names = beeswarm_pairs_from_parquet(beeswarm_df)[:max_display]
    if not feature_names:
        raise ValueError("No beeswarm feature pairs available.")

    if prefer_shap:
        try:
            import shap

            shap_values = beeswarm_df.loc[:, feature_names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            expression_df = pd.DataFrame(
                {
                    feature: pd.to_numeric(beeswarm_df[f"x_{feature}"], errors="coerce").to_numpy(dtype=float)
                    for feature in feature_names
                }
            )
            plot_size = figsize if figsize is not None else (3.0, 0.2 * len(feature_names) + 0.8)
            shap.summary_plot(
                shap_values,
                features=expression_df,
                feature_names=feature_names,
                max_display=len(feature_names),
                show=False,
                cmap=cmap,
                plot_size=plot_size,
            )
            fig = plt.gcf()
            ax = plt.gca()
            for collection in ax.collections:
                try:
                    collection.set_sizes([point_size])
                except Exception:
                    pass
            ax.set_xlabel("SHAP value", fontsize=AXIS_LABEL_FS, labelpad=0)
            ax.set_ylabel("")
            ax.tick_params(axis="y", which="major", labelsize=TICK_FS, pad=-10)
            ax.tick_params(axis="x", which="major", labelsize=TICK_FS)
            ax.set_title(title if title is not None else f"{cancer} - {target_gene} SHAP", fontsize=TITLE_FS)
            if len(fig.axes) > 1:
                cbar_ax = fig.axes[-1]
                cbar_ax.set_ylabel("Gene expression", fontsize=COLORBAR_LABEL_FS, rotation=270, labelpad=6)
                cbar_ax.tick_params(labelsize=COLORBAR_TICK_FS)
            fig.tight_layout()
            if save_path is not None:
                save_panel(fig, save_path)
            else:
                plt.show()
            return _beeswarm_plot_table(beeswarm_df, feature_names)
        except ImportError:
            pass

    return _plot_beeswarm_fallback(
        beeswarm_df=beeswarm_df,
        feature_names=feature_names,
        cancer=cancer,
        target_gene=target_gene,
        cmap=cmap,
        figsize=figsize,
        point_size=point_size,
        alpha=alpha,
        title=title,
        save_path=save_path,
    )


def _beeswarm_plot_table(beeswarm_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for rank, feature in enumerate(feature_names, start=1):
        shap_values = pd.to_numeric(beeswarm_df[feature], errors="coerce")
        expression_values = pd.to_numeric(beeswarm_df[f"x_{feature}"], errors="coerce")
        valid = shap_values.notna() & expression_values.notna()
        if not valid.any():
            continue
        rows.append(
            pd.DataFrame(
                {
                    "sample_id": beeswarm_df.loc[valid, "sample_id"].astype(str).to_numpy(),
                    "feature": feature,
                    "rank": rank,
                    "shap_value": shap_values.loc[valid].to_numpy(dtype=float),
                    "expression": expression_values.loc[valid].to_numpy(dtype=float),
                }
            )
        )

    plot_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if plot_df.empty:
        raise ValueError("No valid beeswarm values after numeric conversion.")
    return plot_df


def _plot_beeswarm_fallback(
    *,
    beeswarm_df: pd.DataFrame,
    feature_names: list[str],
    cancer: str,
    target_gene: str,
    cmap=None,
    figsize: tuple[float, float] | None = None,
    point_size: float = 14,
    alpha: float = 0.85,
    title: str | None = None,
    save_path: str | Path | None = None,
) -> pd.DataFrame:
    plot_df = _beeswarm_plot_table(beeswarm_df, feature_names)
    feature_order = feature_names[:]
    y_positions = {feature: len(feature_order) - i for i, feature in enumerate(feature_order)}
    expr_low, expr_high = np.nanpercentile(plot_df["expression"], [1, 99])
    if not np.isfinite(expr_low) or not np.isfinite(expr_high) or expr_low == expr_high:
        expr_low = float(np.nanmin(plot_df["expression"]))
        expr_high = float(np.nanmax(plot_df["expression"]))
    norm = Normalize(vmin=expr_low, vmax=expr_high)
    color_cmap = cmap if cmap is not None else CMAP_BEIGE_PURPLE

    if figsize is None:
        figsize = (3.0, 0.2 * len(feature_order) + 0.8)
    fig, ax = plt.subplots(figsize=figsize)
    rng = np.random.default_rng(0)

    for feature in feature_order:
        sub = plot_df.loc[plot_df["feature"].eq(feature)]
        if sub.empty:
            continue
        y = y_positions[feature] + rng.normal(0, 0.055, size=sub.shape[0])
        ax.scatter(
            sub["shap_value"],
            y,
            c=sub["expression"],
            cmap=color_cmap,
            norm=norm,
            s=point_size,
            alpha=alpha,
            linewidths=0,
            rasterized=True,
        )

    ax.axvline(0, color="#777777", linewidth=0.6, zorder=0)
    ax.set_yticks([y_positions[feature] for feature in feature_order])
    ax.set_yticklabels(feature_order, fontsize=TICK_FS)
    ax.set_ylim(0.5, len(feature_order) + 0.5)
    ax.set_xlabel("SHAP value", fontsize=AXIS_LABEL_FS, labelpad=0)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS, pad=-10)
    ax.set_title(title if title is not None else f"{cancer} - {target_gene} SHAP", fontsize=TITLE_FS)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=color_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.03, fraction=0.08)
    cbar.set_label("Gene expression", fontsize=COLORBAR_LABEL_FS, rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS)

    fig.tight_layout()
    if save_path is not None:
        save_panel(fig, save_path)
    else:
        plt.show()
    return plot_df


def build_target_top_shap(
    shap_long: pd.DataFrame,
    target_gene: str,
    *,
    top_n_per_cancer: int = 50,
) -> pd.DataFrame:
    """Keep top SHAP features per cancer for one mutation target."""
    df = shap_long.loc[shap_long["target_gene"].eq(target_gene)].copy()
    if df.empty:
        return pd.DataFrame(columns=["cohort", "target_gene", "feature_symbol", "mean_abs_shap", "mean_shap"])
    df = (
        df.sort_values(["cohort", "mean_abs_shap"], ascending=[True, False])
        .groupby("cohort", group_keys=False)
        .head(top_n_per_cancer)
        .reset_index(drop=True)
    )
    return df


def recurrent_feature_table(
    target_top: pd.DataFrame,
    *,
    min_cancers: int = 5,
    max_features: int = 40,
) -> pd.DataFrame:
    if target_top.empty:
        return pd.DataFrame(columns=["feature_symbol", "n_cancers", "mean_abs_shap", "mean_shap"])
    summary = (
        target_top.groupby("feature_symbol", as_index=False)
        .agg(
            n_cancers=("cohort", "nunique"),
            mean_abs_shap=("mean_abs_shap", "mean"),
            mean_shap=("mean_shap", "mean"),
        )
    )
    summary = summary.loc[summary["n_cancers"] >= min_cancers].copy()
    return (
        summary.sort_values(["n_cancers", "mean_abs_shap", "feature_symbol"], ascending=[False, False, True], kind="mergesort")
        .head(max_features)
        .reset_index(drop=True)
    )


def truncated_cmap(cmap_name: str = "magma_r", minval: float = 0.1, maxval: float = 0.9, n: int = 256):
    base = plt.get_cmap(cmap_name)
    return LinearSegmentedColormap.from_list(
        f"trunc({cmap_name},{minval:.2f},{maxval:.2f})",
        base(np.linspace(minval, maxval, n)),
    )


def plot_target_feature_cancer_dotplot(
    target_top: pd.DataFrame,
    *,
    target_gene: str,
    min_cancers: int = 5,
    max_features: int = 40,
    cmap=None,
    figsize=(4.5, 3.0),
    dot_size: float = 90,
    cancer_order_by: str = "interaction_count",
    save_path=None,
) -> pd.DataFrame:
    """Legacy-style dotplot of recurrent expression features for one mutation target across cancers."""
    features = recurrent_feature_table(target_top, min_cancers=min_cancers, max_features=max_features)
    if features.empty:
        raise ValueError(f"No {target_gene} features passed min_cancers={min_cancers}.")
    if cmap is None:
        raise ValueError("Pass cmap explicitly. For the legacy TP53 panel use truncated_cmap('magma_r', 0.1, 0.9).")

    plot_df = target_top.loc[target_top["feature_symbol"].isin(features["feature_symbol"])].copy()
    plot_df = (
        plot_df.groupby(["cohort", "feature_symbol"], as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "mean"), mean_shap=("mean_shap", "mean"))
    )

    feature_order = features["feature_symbol"].tolist()
    if cancer_order_by == "interaction_count":
        cancer_order = (
            plot_df.groupby("cohort")["feature_symbol"]
            .nunique()
            .sort_values(ascending=False, kind="mergesort")
            .index.tolist()
        )
    elif cancer_order_by == "alphabetical":
        cancer_order = sorted(plot_df["cohort"].unique())
    else:
        raise ValueError("cancer_order_by must be 'interaction_count' or 'alphabetical'")
    xmap = {cancer: i for i, cancer in enumerate(cancer_order)}
    ymap = {feature: i for i, feature in enumerate(feature_order[::-1])}

    colors = plot_df["mean_shap"].to_numpy(float)
    vmin = float(np.nanmin(colors))
    vmax = float(np.nanmax(colors))
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax) if vmin < 0 < vmax else Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    sc = ax.scatter(
        plot_df["cohort"].map(xmap),
        plot_df["feature_symbol"].map(ymap),
        c=colors,
        s=dot_size,
        cmap=cmap,
        norm=norm,
        edgecolors="black",
        linewidths=0.35,
        alpha=0.9,
        zorder=3,
    )
    ax.set_xticks(range(len(cancer_order)))
    ax.set_xticklabels(cancer_order, rotation=90, fontsize=TICK_FS)
    ax.set_yticks(range(len(feature_order)))
    ax.set_yticklabels(feature_order[::-1], fontsize=TICK_FS)
    ax.set_xlim(-0.6, len(cancer_order) - 0.4)
    ax.set_ylim(-0.6, len(feature_order) - 0.4)
    ax.set_axisbelow(True)
    ax.grid(which="major", color="0.88", linestyle="-", linewidth=0.6)
    ax.set_title(target_gene, fontsize=TITLE_FS, pad=3)
    ax.tick_params(axis="both", length=0)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.055)
    cbar.set_label("Mean signed SHAP", fontsize=COLORBAR_LABEL_FS, rotation=270, labelpad=10)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS, length=0)

    if save_path:
        save_panel(fig, save_path)
    return (
        plot_df.merge(features, on="feature_symbol", how="left", suffixes=("", "_feature"))
        .sort_values(["n_cancers", "mean_abs_shap_feature", "feature_symbol", "cohort"], ascending=[False, False, True, True])
        .reset_index(drop=True)
    )


def summarize_directional_recurrence(target_top: pd.DataFrame, *, min_cancers: int = 6, max_genes: int = 20) -> pd.DataFrame:
    if target_top.empty:
        return pd.DataFrame(columns=["feature_symbol", "n_pos", "n_neg", "n_total", "mean_abs_shap"])
    summary = (
        target_top.groupby("feature_symbol", as_index=False)
        .agg(
            n_pos=("mean_shap", lambda s: int((s > 0).sum())),
            n_neg=("mean_shap", lambda s: int((s < 0).sum())),
            mean_abs_shap=("mean_abs_shap", "mean"),
        )
    )
    summary["n_total"] = summary["n_pos"] + summary["n_neg"]
    summary = summary.loc[summary["n_total"] >= min_cancers].copy()
    return (
        summary.sort_values(["n_total", "mean_abs_shap", "feature_symbol"], ascending=[False, False, True], kind="mergesort")
        .head(max_genes)
        .reset_index(drop=True)
    )


def plot_directional_recurrence(
    target_top: pd.DataFrame,
    *,
    target_gene: str = "TP53",
    min_cancers: int = 6,
    max_genes: int = 20,
    figsize=(3.5, 3.0),
    save_path=None,
) -> pd.DataFrame:
    summary = summarize_directional_recurrence(target_top, min_cancers=min_cancers, max_genes=max_genes)
    if summary.empty:
        raise ValueError("No recurrent directional features passed the filter.")

    y = np.arange(summary.shape[0])
    neg = -summary["n_neg"].to_numpy(float)
    pos = summary["n_pos"].to_numpy(float)

    fig, ax = plt.subplots(figsize=figsize)
    for yy in y:
        ax.axhline(yy, color="black", lw=0.6, alpha=0.08, zorder=0)
    ax.barh(y, neg, color="#d15472", alpha=0.85, height=0.65, label="Negative mean SHAP", zorder=2)
    ax.barh(y, pos, color=TEAL, alpha=0.85, height=0.65, label="Positive mean SHAP", zorder=2)
    ax.axvline(0, color="black", lw=1.2, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(summary["feature_symbol"], fontsize=TICK_FS)
    ax.invert_yaxis()
    ax.set_xlabel("Number of cancers", fontsize=AXIS_LABEL_FS)
    ax.set_title(f"{target_gene} recurrent SHAP direction", fontsize=TITLE_FS, pad=4)
    ax.tick_params(axis="both", length=0, labelsize=TICK_FS)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    limit = max(float(summary["n_pos"].max()), float(summary["n_neg"].max())) + 1.5
    ax.set_xlim(-limit, limit)

    for i, row in summary.iterrows():
        if row["n_neg"] > 0:
            ax.text(-row["n_neg"] - 0.25, i, str(int(row["n_neg"])), ha="right", va="center", fontsize=ANNOTATION_FS)
        if row["n_pos"] > 0:
            ax.text(row["n_pos"] + 0.25, i, str(int(row["n_pos"])), ha="left", va="center", fontsize=ANNOTATION_FS)

    fig.tight_layout()
    if save_path:
        save_panel(fig, save_path)
    return summary


def pathway_records(shap_long: pd.DataFrame, rows: list[tuple[str, str]], genes: list[str]) -> pd.DataFrame:
    """Return one row per requested cancer-target-feature combination."""
    records = []
    for cohort, target in rows:
        sub = shap_long.loc[(shap_long["cohort"].eq(cohort)) & (shap_long["target_gene"].eq(target))]
        for gene in genes:
            hit = sub.loc[sub["feature_symbol"].eq(gene)]
            if hit.empty:
                records.append({"cohort": cohort, "target_gene": target, "feature_symbol": gene, "mean_abs_shap": np.nan, "mean_shap": np.nan})
            else:
                top = hit.sort_values("mean_abs_shap", ascending=False).iloc[0]
                records.append(
                    {
                        "cohort": cohort,
                        "target_gene": target,
                        "feature_symbol": gene,
                        "mean_abs_shap": float(top["mean_abs_shap"]),
                        "mean_shap": float(top["mean_shap"]),
                    }
                )
    return pd.DataFrame(records)


def pathway_long_table(shap_long: pd.DataFrame, *, rows: list[tuple[str, str]] | None = None, genes: list[str] | None = None) -> pd.DataFrame:
    """Convert clean SHAP long output to the legacy pathway plotting schema."""
    df = pd.DataFrame(
        {
            "cancer": shap_long["cohort"].astype(str),
            "target": shap_long["target_gene"].astype(str),
            "feature": shap_long["feature_symbol"].astype(str),
            "shap": pd.to_numeric(shap_long["mean_shap"], errors="coerce"),
            "mean_abs_shap": pd.to_numeric(shap_long["mean_abs_shap"], errors="coerce"),
        }
    )
    if rows is not None:
        row_set = set(rows)
        df = df.loc[df.apply(lambda r: (r["cancer"], r["target"]) in row_set, axis=1)].copy()
    if genes is not None:
        df = df.loc[df["feature"].isin(genes)].copy()
    return df


def top_features_for_rows(
    shap_long: pd.DataFrame,
    rows: list[tuple[str, str]],
    *,
    top_n: int = 8,
) -> list[str]:
    keys = pd.MultiIndex.from_tuples(rows, names=["cohort", "target_gene"])
    sub = shap_long.set_index(["cohort", "target_gene"])
    sub = sub.loc[sub.index.intersection(keys)].reset_index()
    if sub.empty:
        return []
    return (
        sub.groupby("feature_symbol", as_index=False)
        .agg(max_abs_shap=("mean_abs_shap", "max"), n_pairs=("target_gene", "size"))
        .sort_values(["max_abs_shap", "n_pairs"], ascending=False)
        .head(top_n)["feature_symbol"]
        .tolist()
    )


def plot_pathway_dotplot(
    shap_long: pd.DataFrame,
    *,
    rows: list[tuple[str, str]],
    genes: list[str],
    block_color: str,
    title: str = "",
    figsize=(3.6, 2.8),
    global_max: float | None = None,
    save_path=None,
) -> pd.DataFrame:
    """Curated pathway dotplot: rows are cancer-target pairs and columns are SHAP features."""
    plot_df = pathway_records(shap_long, rows, genes)
    values = plot_df["mean_abs_shap"].to_numpy(float)
    finite = np.isfinite(values)
    global_max = float(np.nanmax(values[finite])) if global_max is None and finite.any() else (global_max or 1.0)
    global_max = max(global_max, 1e-12)

    row_labels = [f"{target}\n{cohort}" for cohort, target in rows]
    xmap = {gene: i for i, gene in enumerate(genes)}
    ymap = {label: i for i, label in enumerate(row_labels[::-1])}
    plot_df["row_label"] = [f"{target}\n{cohort}" for cohort, target in zip(plot_df["cohort"], plot_df["target_gene"])]

    fill_cmap = LinearSegmentedColormap.from_list("pathway_fill", [BEIGE, block_color], N=256)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.add_patch(
        FancyBboxPatch(
            (-0.62, -0.62),
            len(genes) + 0.24,
            len(rows) + 0.24,
            boxstyle="round,pad=0.03,rounding_size=0.06",
            facecolor="#fafafa",
            edgecolor="#dddddd",
            linewidth=0.8,
            zorder=0,
        )
    )
    for x in range(len(genes)):
        ax.axvline(x, color="#ececec", lw=0.55, zorder=1)
    for y in range(len(rows)):
        ax.axhline(y, color="#ececec", lw=0.55, zorder=1)

    sub = plot_df.dropna(subset=["mean_abs_shap"]).copy()
    if not sub.empty:
        xs = sub["feature_symbol"].map(xmap)
        ys = sub["row_label"].map(ymap)
        norm_vals = sub["mean_abs_shap"].to_numpy(float) / global_max
        sizes = 18 + 165 * np.sqrt(norm_vals)
        ax.scatter(xs, ys, s=sizes, c=norm_vals, cmap=fill_cmap, vmin=0, vmax=1, edgecolors="#bbbbbb", linewidths=0.45, zorder=3)

    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=45, ha="left", rotation_mode="anchor", fontsize=TICK_FS)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(row_labels[::-1], fontsize=TICK_FS)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_xlim(-0.55, len(genes) - 0.45)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=TITLE_FS, loc="left", pad=0)

    legend_vals = [global_max * f for f in (0.1, 0.5, 1.0)]
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=np.sqrt(18 + 165 * np.sqrt(v / global_max)),
            markerfacecolor=fill_cmap(v / global_max),
            markeredgecolor="#bbbbbb",
            markeredgewidth=0.45,
            label=f"{v:.2f}",
        )
        for v in legend_vals
    ]
    ax.legend(handles=handles, title="Mean |SHAP|", loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=LEGEND_FS, title_fontsize=LEGEND_FS, frameon=False)

    if save_path:
        save_panel(fig, save_path)
    return plot_df


def plot_collapsed_pathway_dotplot(
    shap_long: pd.DataFrame,
    *,
    targets: list[str],
    genes: list[str],
    block_color: str,
    title: str = "",
    figsize=(3.4, 1.8),
    legend_loc: str = "upper left",
    legend_bbox: tuple = (1.01, 1.45),
    save_path=None,
) -> pd.DataFrame:
    """Collapsed target-feature dotplot across cancers."""
    sub = shap_long.loc[shap_long["target_gene"].isin(targets) & shap_long["feature_symbol"].isin(genes)].copy()
    agg = (
        sub.groupby(["target_gene", "feature_symbol"], as_index=False)
        .agg(n_cancers=("cohort", "nunique"), mean_abs_shap=("mean_abs_shap", "mean"))
    )
    max_cancers = max(int(agg["n_cancers"].max()) if not agg.empty else 1, 1)
    max_shap = max(float(agg["mean_abs_shap"].max()) if not agg.empty else 1.0, 1e-12)
    fill_cmap = LinearSegmentedColormap.from_list("pathway_collapsed_fill", [BEIGE, block_color], N=256)

    xmap = {gene: i for i, gene in enumerate(genes)}
    ymap = {target: i for i, target in enumerate(targets[::-1])}
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.add_patch(FancyBboxPatch((-0.62, -0.62), len(genes) + 0.24, len(targets) + 0.24, boxstyle="round,pad=0.03,rounding_size=0.06", facecolor="#fafafa", edgecolor="#dddddd", linewidth=0.8, zorder=0))
    for x in range(len(genes)):
        ax.axvline(x, color="#ececec", lw=0.55, zorder=1)
    for y in range(len(targets)):
        ax.axhline(y, color="#ececec", lw=0.55, zorder=1)
    if not agg.empty:
        norm_vals = agg["mean_abs_shap"].to_numpy(float) / max_shap
        sizes = 12 + 185 * (agg["n_cancers"].to_numpy(float) / max_cancers) ** 1.25
        ax.scatter(agg["feature_symbol"].map(xmap), agg["target_gene"].map(ymap), s=sizes, c=norm_vals, cmap=fill_cmap, vmin=0, vmax=1, edgecolors="#bbbbbb", linewidths=0.45, zorder=3)
    ax.set_xticks(range(len(genes)))
    ax.set_xticklabels(genes, rotation=45, ha="left", rotation_mode="anchor", fontsize=TICK_FS)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=5)
    ax.set_yticks(range(len(targets)))
    ax.set_yticklabels(targets[::-1], fontsize=TICK_FS)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_xlim(-0.55, len(genes) - 0.45)
    ax.set_ylim(-0.55, len(targets) - 0.45)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=TITLE_FS, loc="left", pad=0)

    def _size(n):
        return 12 + 185 * (n / max_cancers) ** 1.25

    legend_ns = sorted({1, max(1, max_cancers // 2), max_cancers})
    size_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=np.sqrt(_size(n)),
            markerfacecolor="0.75",
            markeredgecolor="#bbbbbb",
            markeredgewidth=0.5,
            label=str(n),
        )
        for n in legend_ns
    ]
    leg = ax.legend(
        handles=size_handles,
        title="# cancers",
        loc=legend_loc,
        bbox_to_anchor=legend_bbox,
        fontsize=LEGEND_FS,
        title_fontsize=LEGEND_FS,
        frameon=False,
        borderaxespad=0,
        labelspacing=0.8,
        handletextpad=0.4,
    )

    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors

    sm = mcm.ScalarMappable(cmap=fill_cmap, norm=mcolors.Normalize(vmin=0, vmax=max_shap))
    sm.set_array([])
    fig.canvas.draw()
    leg_bbox_fig = leg.get_window_extent().transformed(fig.transFigure.inverted())
    ax_bbox = ax.get_position()
    cbar_x = leg_bbox_fig.x0 + (leg_bbox_fig.width - 0.012) / 2
    cbar_y0 = ax_bbox.y0
    cbar_y1 = leg_bbox_fig.y0 - 0.02
    if (cbar_y1 - cbar_y0) > 0.08:
        cax = fig.add_axes([cbar_x, cbar_y0, 0.012, cbar_y1 - cbar_y0])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("Mean |SHAP|", fontsize=COLORBAR_LABEL_FS, rotation=270, labelpad=10)
        cb.ax.tick_params(labelsize=COLORBAR_TICK_FS)

    if save_path:
        save_panel(fig, save_path)
    return agg


def plot_pathway_network(
    df: pd.DataFrame,
    *,
    target_order: list,
    candidate_features: list,
    top_features_per_target: int = 6,
    figsize: tuple = (4, 3.5),
    title: str = "",
    legend_loc: str = "lower left",
    legend_bbox: tuple = (0.15, 0.02),
    save_path=None,
    show: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Legacy three-layer pathway network: cancer -> target -> SHAP feature."""
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors

    present = [feature for feature in candidate_features if feature in df["feature"].values]
    plot_df = df[df["target"].isin(target_order) & df["feature"].isin(present)].copy()
    if plot_df.empty:
        raise ValueError("No pathway interactions available for the requested network.")

    tf_edges = (
        plot_df.groupby(["target", "feature"])
        .agg(weight=("shap", lambda s: s.abs().sum()), count=("cancer", "nunique"))
        .reset_index()
        .sort_values(["target", "weight"], ascending=[True, False])
    )
    tf_edges = tf_edges.groupby("target").head(top_features_per_target).reset_index(drop=True)

    t_order = [target for target in target_order if target in tf_edges["target"].values]
    selected_pairs = tf_edges[["target", "feature"]].drop_duplicates()
    plot_df = plot_df.merge(selected_pairs, on=["target", "feature"], how="inner")
    ct_edges = (
        plot_df.groupby(["cancer", "target"])
        .agg(weight=("shap", lambda s: s.abs().sum()), count=("feature", "nunique"))
        .reset_index()
    )
    tf_edges = (
        plot_df.groupby(["target", "feature"])
        .agg(weight=("shap", lambda s: s.abs().sum()), count=("cancer", "nunique"))
        .reset_index()
        .sort_values(["target", "weight"], ascending=[True, False])
    )
    ct_edges = ct_edges[ct_edges["target"].isin(t_order)].copy()
    cancers = sorted(ct_edges["cancer"].unique())

    x_c, x_t, x_f = 0.10, 0.46, 0.84

    def _space(items, top=0.88, bottom=0.12):
        if len(items) == 1:
            return {items[0]: 0.5}
        step = (top - bottom) / max(len(items) - 1, 1)
        return {item: top - i * step for i, item in enumerate(items)}

    c_pos = _space(cancers)
    t_pos = _space(t_order)

    feat_order = []
    for target in t_order:
        feat_order.extend(
            tf_edges.loc[tf_edges["target"] == target]
            .sort_values("weight", ascending=False)["feature"]
            .tolist()
        )
    feat_order = list(dict.fromkeys(feat_order))
    f_pos = _space(feat_order, top=0.94, bottom=0.06)

    all_w = pd.concat([ct_edges["weight"], tf_edges["weight"]])
    e_norm = mcolors.Normalize(vmin=float(all_w.min()), vmax=float(all_w.max()))
    e_cmap = CMAP_BEIGE_PURPLE
    ct_edges["recurrence_type"] = "selected features in cancer-target"
    tf_edges["recurrence_type"] = "cancer contexts for target-feature"
    cnt_max = max(
        int(ct_edges["count"].max()) if not ct_edges.empty else 1,
        int(tf_edges["count"].max()) if not tf_edges.empty else 1,
        1,
    )

    def _edge_width(count):
        """Map recurrence to line width with recurrence=1 visible but thin."""
        count = np.asarray(count, dtype=float)
        if cnt_max <= 1:
            return np.full_like(count, 1.2, dtype=float)
        return 1.2 + 2.8 * (count - 1.0) / (cnt_max - 1.0)

    ct_edges["edge_width"] = _edge_width(ct_edges["count"])
    tf_edges["edge_width"] = _edge_width(tf_edges["count"])

    def _scale(series, lo=100, hi=500):
        if series.empty:
            return {}
        mn, mx = series.min(), series.max()
        if mn == mx:
            return {key: (lo + hi) / 2 for key in series.index}
        return ((series - mn) / (mx - mn) * (hi - lo) + lo).to_dict()

    c_sz = _scale(ct_edges.groupby("cancer")["count"].sum())
    t_sz = _scale(
        pd.concat(
            [
                ct_edges.groupby("target")["count"].sum(),
                tf_edges.groupby("target")["count"].sum(),
            ]
        )
        .groupby(level=0)
        .sum()
    )
    f_sz = _scale(tf_edges.groupby("feature")["count"].sum())

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for _, row in ct_edges.iterrows():
        ax.plot(
            [x_c, x_t],
            [c_pos[row["cancer"]], t_pos[row["target"]]],
            lw=row["edge_width"],
            color=e_cmap(e_norm(row["weight"])),
            alpha=0.6,
            solid_capstyle="round",
            zorder=1,
        )

    for _, row in tf_edges.iterrows():
        ax.plot(
            [x_t, x_f],
            [t_pos[row["target"]], f_pos[row["feature"]]],
            lw=row["edge_width"],
            color=e_cmap(e_norm(row["weight"])),
            alpha=0.6,
            solid_capstyle="round",
            zorder=1,
        )

    for node in cancers:
        ax.scatter(x_c, c_pos[node], s=c_sz.get(node, 180), color=TEAL, edgecolors="white", linewidths=0.5, zorder=4)
        ax.text(x_c - 0.02, c_pos[node], node, ha="right", va="center", fontsize=TICK_FS, color="0.25", zorder=5)

    for node in t_order:
        ax.scatter(x_t, t_pos[node], s=t_sz.get(node, 280), color=PURPLE, edgecolors="white", linewidths=0.5, zorder=4)
        ax.text(x_t, t_pos[node] + 0.03, node, ha="center", va="bottom", fontsize=ANNOTATION_FS, color="0.15", zorder=5)

    for node in feat_order:
        ax.scatter(x_f, f_pos[node], s=f_sz.get(node, 120), color=ORANGE, edgecolors="white", linewidths=0.5, zorder=4)
        ax.text(x_f + 0.03, f_pos[node], node, ha="left", va="center", fontsize=TICK_FS, color="0.25", zorder=5)

    for x, label in [(x_c, "Cancer"), (x_t, "Target"), (x_f, "Feature")]:
        ax.text(x, 1.02, label, ha="center", va="top", fontsize=AXIS_LABEL_FS, color="0.15", zorder=5)
    if title:
        ax.set_title(title, fontsize=TITLE_FS, loc="center", pad=15)

    sm = mcm.ScalarMappable(cmap=e_cmap, norm=e_norm)
    sm.set_array([])
    cax = fig.add_axes([0.93, 0.32, 0.012, 0.30])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Summed |SHAP|", fontsize=COLORBAR_LABEL_FS, rotation=270, labelpad=10)
    cb.ax.tick_params(labelsize=COLORBAR_TICK_FS)

    leg_counts = sorted({1, max(1, int(round(cnt_max / 2))), int(cnt_max)})
    width_handles = [
        Line2D([0], [0], color="0.45", lw=float(_edge_width([value])[0]), label=str(value))
        for value in leg_counts
    ]
    leg_w = ax.legend(
        handles=width_handles,
        title="Recurrence",
        loc=legend_loc,
        bbox_to_anchor=legend_bbox,
        fontsize=LEGEND_FS,
        title_fontsize=LEGEND_FS,
        frameon=False,
    )
    ax.add_artist(leg_w)

    fig.subplots_adjust(left=0.02, right=0.90, top=0.90, bottom=0.05)
    if save_path:
        save_panel(fig, save_path)
    elif show:
        plt.show()
    return ct_edges, tf_edges
