"""Data-loading and plotting helpers for Figure3.ipynb.

Figure 3 shows that model prediction scores reflect mutation severity:
  WT < Silent < Missense < High-impact (panel A)
  Silent-mutated genes score higher than WT within the same sample (panels B/C + permutation)
  Missense scores correlate with PolyPhen / SIFT deleteriousness (panels D/E)
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from scipy.stats import wilcoxon, kruskal

from constants import TICK_FS, AXIS_LABEL_FS, ANNOTATION_FS, ORANGE, TEAL, PURPLE

# ---------------------------------------------------------------------------
# Mutation-effect category definitions
# ---------------------------------------------------------------------------

HIGH_IMPACT_EFFECTS = frozenset({
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Translation_Start_Site",
    "Nonstop_Mutation",
    "stop_gained",
    "frameshift_variant",
    "start_lost",
    "initiator_codon_variant",
    "stop_lost",
})
MISSENSE_EFFECTS = frozenset({"Missense_Mutation", "missense_variant"})
SILENT_EFFECTS = frozenset({"Silent", "synonymous_variant", "stop_retained_variant"})
SPLICE_RELATED_EFFECTS = frozenset({
    "Splice_Site",
    "splice_acceptor_variant",
    "splice_donor_variant",
})
OTHER_MUTATION_EFFECTS = frozenset({
    "3'UTR", "Intron", "5'UTR", "RNA", "3'Flank", "5'Flank",
    "In_Frame_Del", "In_Frame_Ins",
    "3_prime_UTR_variant", "5_prime_UTR_variant", "intron_variant",
    "upstream_gene_variant", "downstream_gene_variant",
    "non_coding_transcript_exon_variant", "non_coding_transcript_variant",
    "NMD_transcript_variant", "mature_miRNA_variant",
    "inframe_deletion", "inframe_insertion", "protein_altering_variant",
}) | SPLICE_RELATED_EFFECTS

# Figure 3-specific palette (separate from shared project palette)
MUT_COLOR_WT          = "#b993a2"
MUT_COLOR_SILENT      = "#cfb766"
MUT_COLOR_MISSENSE    = "#FDB462"
MUT_COLOR_HIGH_IMPACT = "#da768e"

MUT_EFFECT_ORDER = ("no_mutation", "silent", "missense", "high_impact")
MUT_EFFECT_PALETTE = {
    "no_mutation": MUT_COLOR_WT,
    "silent":      MUT_COLOR_SILENT,
    "missense":    MUT_COLOR_MISSENSE,
    "high_impact": MUT_COLOR_HIGH_IMPACT,
}
MUT_EFFECT_LABELS = {
    "no_mutation": "WT",
    "silent":      "Silent",
    "missense":    "Missense",
    "high_impact": "High impact",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def normalize_sample4(values: pd.Series) -> pd.Series:
    """Normalize TCGA barcodes to sample IDs, e.g. TCGA-XX-XXXX-01.

    GDC files often use aliquot-like sample IDs with a letter suffix in the
    fourth field, such as TCGA-XX-XXXX-01A. Prediction outputs use the two-digit
    sample type, TCGA-XX-XXXX-01, so trim the fourth field to its first two
    characters.
    """
    def _one(value) -> str:
        parts = str(value).split("-")
        if len(parts) >= 4 and parts[0] == "TCGA":
            return "-".join([parts[0], parts[1], parts[2], parts[3][:2]])
        return str(value)

    return values.map(_one) if isinstance(values, pd.Series) else pd.Series(values).map(_one)


def mutation_effect_rank(value) -> int:
    """Severity rank for MC3 MAF and GDC Sequence Ontology effect strings."""
    if pd.isna(value):
        return 0
    terms = [term.strip() for term in re.split(r"[;&,]", str(value)) if term.strip()]
    ranks = []
    for term in terms:
        if term in HIGH_IMPACT_EFFECTS:
            ranks.append(4)
        elif term in MISSENSE_EFFECTS:
            ranks.append(3)
        elif term in SILENT_EFFECTS:
            ranks.append(2)
        elif term in OTHER_MUTATION_EFFECTS:
            ranks.append(1)
        else:
            ranks.append(1)
    return max(ranks) if ranks else 0


def mutation_effect_label(value) -> str | float:
    return {4: "high_impact", 3: "missense", 2: "silent", 1: "other_mutation"}.get(
        mutation_effect_rank(value),
        np.nan,
    )


def load_full_mutation_maf(path: str | Path, *, pass_only: bool = True) -> pd.DataFrame:
    """Load MC3 MAF/Xena variants or GDC Mutect2 and add a normalized sample4 key.

    Handles both the standard MAF column names (Tumor_Sample_Barcode / Hugo_Symbol /
    Variant_Classification), the MC3 Xena column names (sample / gene / effect), and
    the GDC Mutect2 column names (Sample_ID / gene / effect / filter).
    """
    maf_df = pd.read_csv(path, sep="\t", low_memory=False)
    if pass_only and "filter" in maf_df.columns:
        maf_df = maf_df.loc[maf_df["filter"].astype(str).eq("PASS")].copy()

    maf_df = maf_df.rename(
        columns={
            "gene": "Hugo_Symbol",
            "effect": "Variant_Classification",
            "chrom": "Chromosome",
            "start": "Start_Position",
            "end": "End_Position",
            "ref": "Reference_Allele",
            "alt": "Tumor_Seq_Allele2",
            "Sample_ID": "Tumor_Sample_Barcode",
        }
    )
    if "Tumor_Sample_Barcode" not in maf_df.columns:
        if "sample" in maf_df.columns:
            maf_df = maf_df.rename(columns={"sample": "Tumor_Sample_Barcode"})
        else:
            raise KeyError("Could not find a sample barcode column in mutation file.")
    if "Hugo_Symbol" not in maf_df.columns or "Variant_Classification" not in maf_df.columns:
        raise KeyError("Mutation file must contain gene/effect columns.")

    barcode_col = "Tumor_Sample_Barcode"
    maf_df["sample4"] = normalize_sample4(maf_df[barcode_col])
    maf_df["Hugo_Symbol"] = maf_df["Hugo_Symbol"].astype(str)
    maf_df["Variant_Classification"] = maf_df["Variant_Classification"].astype(str)
    maf_df["mutation_effect_rank"] = maf_df["Variant_Classification"].map(mutation_effect_rank).astype(int)
    maf_df["mutation_effect_class"] = maf_df["Variant_Classification"].map(mutation_effect_label)
    return maf_df


def load_legacy_mc3_event_maf(
    event_dir: str | Path,
    cancer_types: list[str] | tuple[str, ...],
    *,
    file_template: str = "{cancer}_mc3.txt.gz",
    pass_only: bool = True,
) -> pd.DataFrame:
    """Load and concatenate per-cohort MC3 event files used by the Clean workflow."""
    event_dir = Path(event_dir)
    frames = []
    missing = []
    for cancer in cancer_types:
        cancer = str(cancer).upper()
        path = event_dir / file_template.format(cancer=cancer)
        if not path.exists():
            missing.append(str(path))
            continue
        frames.append(
            load_full_mutation_maf(path, pass_only=pass_only).assign(
                Cancer=cancer,
                source_file=path.name,
            )
        )
    if missing:
        raise FileNotFoundError(f"Missing legacy MC3 event files: {missing}")
    if not frames:
        raise FileNotFoundError(f"No legacy MC3 event files found under {event_dir}.")
    return pd.concat(frames, axis=0, ignore_index=True)


def load_predictions(
    results_dir: str | Path,
    skip_cancers: tuple[str, ...] = ("All", "UVM"),
) -> dict[str, pd.DataFrame]:
    """Load per-cancer k-fold predicted mutation probabilities."""
    results_dir = Path(results_dir)
    skip_lower = {s.lower() for s in skip_cancers}
    predictions = {}
    for cancer_dir in sorted(results_dir.iterdir()):
        if not cancer_dir.is_dir() or cancer_dir.name.lower() in skip_lower:
            continue
        prob_path = cancer_dir / "cv" / "oof_probabilities.csv"
        if prob_path.exists():
            predictions[cancer_dir.name] = pd.read_csv(prob_path, index_col=0)
    return predictions


def load_auprc_metrics(
    results_dir: str | Path,
    skip_cancers: tuple[str, ...] = ("All", "UVM"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load AUPRC, prevalence, and normalised AUPRC tables across all cancers."""
    results_dir = Path(results_dir)
    skip_lower = {s.lower() for s in skip_cancers}
    auprc, prevalence, normalized_auprc = {}, {}, {}
    for cancer_dir in sorted(results_dir.iterdir()):
        if not cancer_dir.is_dir() or cancer_dir.name.lower() in skip_lower:
            continue
        summary_path = cancer_dir / "cv" / "metrics.csv"
        if not summary_path.exists():
            continue
        summary_df = pd.read_csv(summary_path, index_col=0)
        cancer = cancer_dir.name
        auprc[cancer] = summary_df["auprc"]
        prevalence[cancer] = summary_df["prevalence"]
        normalized_auprc[cancer] = summary_df["normalized_auprc"]
    return pd.DataFrame(auprc), pd.DataFrame(prevalence), pd.DataFrame(normalized_auprc)


def build_pred_long(
    predictions: dict[str, pd.DataFrame],
    normalized_auprc_df: pd.DataFrame,
    normalized_auprc_threshold: float = 0.15,
) -> pd.DataFrame:
    """Filter to well-predicted genes and melt predictions to long form."""
    cancer_frames = {}
    for cancer, pred_df in predictions.items():
        if cancer not in normalized_auprc_df.columns:
            continue
        selected_genes = normalized_auprc_df.loc[
            normalized_auprc_df[cancer] > normalized_auprc_threshold, cancer
        ].sort_values(ascending=False).index
        cancer_frames[cancer] = pred_df.loc[:, pred_df.columns.intersection(selected_genes)].copy()
    combined = pd.concat(cancer_frames, names=["Cancer", "sample"]).reset_index()
    return combined.melt(id_vars=["Cancer", "sample"], var_name="gene", value_name="pred_prob")

# ---------------------------------------------------------------------------
# MAF preparation and mutation-effect assignment
# ---------------------------------------------------------------------------

def keep_preferred_transcripts(maf_df: pd.DataFrame) -> pd.DataFrame:
    """Prefer PICK transcripts; fall back to CANONICAL; else keep all."""
    truthy = {"1", "TRUE", "True", "YES", "Yes", "yes"}
    if "PICK" in maf_df.columns:
        mask = maf_df["PICK"].astype(str).isin(truthy)
        if mask.any():
            return maf_df.loc[mask].copy()
    if "CANONICAL" in maf_df.columns:
        mask = maf_df["CANONICAL"].astype(str).isin(truthy)
        if mask.any():
            return maf_df.loc[mask].copy()
    return maf_df


def prepare_maf_for_join(maf_df: pd.DataFrame) -> pd.DataFrame:
    """Normalise join columns and keep preferred transcript rows."""
    maf_df = maf_df.copy()
    if "sample4" not in maf_df.columns:
        barcode_col = (
            "Tumor_Sample_Barcode"
            if "Tumor_Sample_Barcode" in maf_df.columns
            else "sample"
            if "sample" in maf_df.columns
            else "Sample_ID"
        )
        maf_df["sample4"] = normalize_sample4(maf_df[barcode_col])
    if "Hugo_Symbol" not in maf_df.columns and "gene" in maf_df.columns:
        maf_df = maf_df.rename(columns={"gene": "Hugo_Symbol"})
    if "Variant_Classification" not in maf_df.columns and "effect" in maf_df.columns:
        maf_df = maf_df.rename(columns={"effect": "Variant_Classification"})
    maf_df["Hugo_Symbol"] = maf_df["Hugo_Symbol"].astype(str)
    maf_df["Variant_Classification"] = maf_df["Variant_Classification"].astype(str)
    if "mutation_effect_rank" not in maf_df.columns:
        maf_df["mutation_effect_rank"] = maf_df["Variant_Classification"].map(mutation_effect_rank).astype(int)
    if "mutation_effect_class" not in maf_df.columns:
        maf_df["mutation_effect_class"] = maf_df["Variant_Classification"].map(mutation_effect_label)
    return keep_preferred_transcripts(maf_df)


def prediction_join_frame(pred_long_df: pd.DataFrame) -> pd.DataFrame:
    """Return a join-ready frame keyed by (sample4, Hugo_Symbol)."""
    return pred_long_df.assign(
        sample4=normalize_sample4(pred_long_df["sample"]),
        Hugo_Symbol=pred_long_df["gene"].astype(str),
    )[["Cancer", "sample4", "Hugo_Symbol", "pred_prob"]]


def assign_mutation_effects(
    pred_long_df: pd.DataFrame,
    full_mutations_maf_df: pd.DataFrame,
    *,
    keep_other_as_separate: bool = True,
    drop_other_mutations: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Assign the highest-priority mutation effect to each (sample, gene) prediction.

    Priority: high_impact > missense > silent > other_mutation > no_mutation (WT).
    """
    if keep_other_as_separate and drop_other_mutations:
        raise ValueError("Set at most one of keep_other_as_separate or drop_other_mutations.")

    pred_df = pred_long_df.dropna(subset=["pred_prob"]).copy()
    pred_df["sample4"] = normalize_sample4(pred_df["sample"])
    pred_df["Hugo_Symbol"] = pred_df["gene"].astype(str)

    maf_df = prepare_maf_for_join(full_mutations_maf_df)
    rank_to_label = {4: "high_impact", 3: "missense", 2: "silent", 1: "other_mutation"}

    join_cols = ["sample4", "Hugo_Symbol"]
    maf_df["effect_rank"] = maf_df["mutation_effect_rank"].astype(int)
    max_rank_df = maf_df.groupby(join_cols, as_index=False)["effect_rank"].max()
    max_rank_df["mut_effect"] = max_rank_df["effect_rank"].map(rank_to_label)

    labels_to_keep = ["high_impact", "missense", "silent"]
    if keep_other_as_separate or drop_other_mutations:
        labels_to_keep.append("other_mutation")
    max_rank_df = max_rank_df.loc[
        max_rank_df["mut_effect"].isin(labels_to_keep), join_cols + ["mut_effect"]
    ].copy()

    out = pred_df[join_cols + ["Cancer", "sample", "gene", "pred_prob"]].merge(
        max_rank_df, on=join_cols, how="left"
    )
    out["mut_effect"] = out["mut_effect"].fillna("no_mutation")

    if drop_other_mutations:
        out = out.loc[out["mut_effect"] != "other_mutation"].copy()

    if verbose:
        print(out["mut_effect"].value_counts(dropna=False))
    return out


def deduplicate_variant_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop coordinate-level duplicate variant rows when available."""
    dedup_cols = [
        col for col in [
            "sample4", "Hugo_Symbol", "Chromosome",
            "Start_Position", "Reference_Allele", "Tumor_Seq_Allele2",
        ]
        if col in df.columns
    ]
    return df.drop_duplicates(subset=dedup_cols) if len(dedup_cols) >= 3 else df


def build_silent_only_df(
    full_mutations_maf_df: pd.DataFrame,
    pred_long_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build silent mutation rows for (sample, gene) pairs with no coding mutation."""
    maf_df = prepare_maf_for_join(full_mutations_maf_df)
    join_cols = ["sample4", "Hugo_Symbol"]

    silent_pairs = maf_df.loc[maf_df["mutation_effect_class"].eq("silent"), join_cols].drop_duplicates()
    coding_pairs = (
        maf_df.loc[maf_df["mutation_effect_class"].isin(["high_impact", "missense"]), join_cols]
        .drop_duplicates().assign(_has_coding=True)
    )
    silent_only_pairs = (
        silent_pairs.merge(coding_pairs, on=join_cols, how="left")
        .loc[lambda d: d["_has_coding"].isna(), join_cols].drop_duplicates()
    )
    silent_only_maf = (
        maf_df.merge(silent_only_pairs, on=join_cols, how="inner")
        .loc[lambda d: d["mutation_effect_class"].eq("silent")].copy()
    )
    silent_only_maf = deduplicate_variant_rows(silent_only_maf)
    out = silent_only_maf.merge(prediction_join_frame(pred_long_df), on=join_cols, how="left")
    n_before = len(out)
    out = out.loc[out["pred_prob"].notna()].copy()
    print(f"Silent-only (sample, gene) pairs : {len(silent_only_pairs):,}")
    print(f"Silent mutation rows              : {n_before:,}")
    print(f"  with predicted probability      : {len(out):,}")
    return out


def build_missense_only_df(
    full_mutations_maf_df: pd.DataFrame,
    pred_long_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build missense mutation rows for (sample, gene) pairs with no high-impact mutation."""
    maf_df = prepare_maf_for_join(full_mutations_maf_df)
    join_cols = ["sample4", "Hugo_Symbol"]

    missense_pairs = maf_df.loc[maf_df["mutation_effect_class"].eq("missense"), join_cols].drop_duplicates()
    hi_pairs = (
        maf_df.loc[maf_df["mutation_effect_class"].eq("high_impact"), join_cols]
        .drop_duplicates().assign(_has_high_impact=True)
    )
    missense_only_pairs = (
        missense_pairs.merge(hi_pairs, on=join_cols, how="left")
        .loc[lambda d: d["_has_high_impact"].isna(), join_cols].drop_duplicates()
    )
    missense_only_maf = (
        maf_df.merge(missense_only_pairs, on=join_cols, how="inner")
        .loc[lambda d: d["mutation_effect_class"].eq("missense")].copy()
    )
    missense_only_maf = deduplicate_variant_rows(missense_only_maf)
    out = missense_only_maf.merge(prediction_join_frame(pred_long_df), on=join_cols, how="left")
    n_before = len(out)
    out = out.loc[out["pred_prob"].notna()].copy()
    print(f"Missense-only (sample, gene) pairs : {len(missense_only_pairs):,}")
    print(f"Missense mutation rows             : {n_before:,}")
    print(f"  with predicted probability       : {len(out):,}")
    return out

# ---------------------------------------------------------------------------
# Internal helpers shared by all paired-panel functions
# ---------------------------------------------------------------------------

def _format_p(value: float) -> str:
    if pd.isna(value):
        return "nan"
    min_positive_float = np.nextafter(0, 1)
    return f"< {min_positive_float:.2e}" if value == 0 else f"{value:.2e}"


def _paired_wilcoxon(delta_vals: np.ndarray) -> float:
    if len(delta_vals) == 0:
        return np.nan
    if np.allclose(delta_vals, 0):
        return 1.0
    return float(wilcoxon(delta_vals, zero_method="wilcox", alternative="two-sided").pvalue)


def _per_cancer_wilcoxon(
    paired_reset: pd.DataFrame,
    cancer_col: str,
    min_pairs: int,
) -> tuple[pd.DataFrame, list[str], list[np.ndarray]]:
    rows = []
    for cancer, sub in paired_reset.groupby(cancer_col, sort=False):
        delta = sub["delta"].to_numpy(dtype=float)
        if len(delta) < min_pairs:
            continue
        rows.append({
            "Cancer": cancer,
            "n_pairs": len(delta),
            "mean_delta": float(np.mean(delta)),
            "median_delta": float(np.median(delta)),
            "wilcoxon_p": _paired_wilcoxon(delta),
        })
    if not rows:
        raise ValueError(f"No cancers have >= {min_pairs} valid pairs.")
    stats = pd.DataFrame(rows).sort_values("median_delta").reset_index(drop=True)
    cancer_order = stats["Cancer"].tolist()
    delta_by_cancer = [
        paired_reset.loc[paired_reset[cancer_col] == c, "delta"].to_numpy(dtype=float)
        for c in cancer_order
    ]
    return stats, cancer_order, delta_by_cancer


def _render_paired_panel(
    a_vals: np.ndarray,
    b_vals: np.ndarray,
    label_a: str,
    label_b: str,
    delta_vals: np.ndarray,
    pooled_p: float,
    per_cancer_stats: pd.DataFrame,
    delta_by_cancer: list[np.ndarray],
    cancer_order: list[str],
    *,
    ylabel_left: str,
    xlabel_right: str,
    color_a: str,
    color_b: str,
    delta_color: str,
    figsize: tuple[float, float],
    save_path: str | Path | None,
) -> plt.Figure:
    pooled_n = len(delta_vals)
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": [1.0, 1.45]},
    )

    box = ax1.boxplot(
        [a_vals, b_vals], positions=[1, 2], widths=0.5,
        patch_artist=True, showfliers=False,
        medianprops={"linewidth": 1.4, "color": "black"},
    )
    for patch, color in zip(box["boxes"], [color_a, color_b]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    rng = np.random.default_rng(0)
    line_idx = rng.choice(pooled_n, size=min(pooled_n, 2000), replace=False)
    for idx in line_idx:
        ax1.plot([1, 2], [a_vals[idx], b_vals[idx]], alpha=0.08, linewidth=0.6, color="grey")
    ax1.scatter(np.full(pooled_n, 1), a_vals, s=8, alpha=0.22, color=color_a)
    ax1.scatter(np.full(pooled_n, 2), b_vals, s=8, alpha=0.22, color=color_b)

    ax1.set_xticks([1, 2])
    ax1.set_xticklabels([label_a, label_b], fontsize=TICK_FS)
    ax1.set_ylabel(ylabel_left, fontsize=AXIS_LABEL_FS)
    ax1.tick_params(axis="y", labelsize=TICK_FS, length=0)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.text(
        0.05, 0.98,
        f"n = {pooled_n}\npaired p = {_format_p(pooled_p)}\nmean delta = {delta_vals.mean():.4f}\nmedian delta = {np.median(delta_vals):.4f}",
        transform=ax1.transAxes, va="top", ha="left", fontsize=ANNOTATION_FS,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75, "linewidth": 0},
    )

    box2 = ax2.boxplot(delta_by_cancer, vert=False, labels=cancer_order, showfliers=False, patch_artist=True)
    for patch in box2["boxes"]:
        patch.set_facecolor(delta_color)
        patch.set_alpha(0.5)
    ax2.axvline(0, linestyle="--", linewidth=1, color="black")
    ax2.set_xlabel(xlabel_right, fontsize=AXIS_LABEL_FS)
    ax2.tick_params(axis="x", labelsize=TICK_FS, length=0)
    ax2.tick_params(axis="y", labelsize=TICK_FS, length=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    x0, x1 = ax2.get_xlim()
    xr = x1 - x0
    ax2.set_xlim(x0, x1 + 0.38 * xr)
    text_x = x1 + 0.02 * xr
    stats_map = per_cancer_stats.set_index("Cancer")
    for y_pos, cancer in enumerate(cancer_order, start=1):
        row = stats_map.loc[cancer]
        ax2.text(
            text_x, y_pos,
            f"n={int(row['n_pairs'])}, p={row['wilcoxon_p']:.1e}",
            va="center", ha="left", fontsize=ANNOTATION_FS,
        )

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return fig

# ---------------------------------------------------------------------------
# Panel A — Violin plot by mutation-effect category
# ---------------------------------------------------------------------------

def plot_pred_by_mut_effect(
    mutation_effects_df: pd.DataFrame,
    *,
    order: tuple[str, ...] = MUT_EFFECT_ORDER,
    label_map: dict[str, str] | None = None,
    palette: dict[str, str] | None = None,
    figsize: tuple[float, float] = (3.6, 3.1),
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    label_map = MUT_EFFECT_LABELS if label_map is None else label_map
    palette   = MUT_EFFECT_PALETTE if palette   is None else palette

    plot_df = mutation_effects_df.loc[mutation_effects_df["mut_effect"].isin(order)].copy()
    groups = [
        plot_df.loc[plot_df["mut_effect"] == g, "pred_prob"].dropna().to_numpy(dtype=float)
        for g in order
    ]
    groups = [g for g in groups if len(g) > 0]
    p_value = kruskal(*groups).pvalue if len(groups) >= 2 else np.nan

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        data=plot_df, x="mut_effect", y="pred_prob",
        order=list(order), palette=palette,
        inner="box", cut=0, linewidth=0.8, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Predicted probability", fontsize=AXIS_LABEL_FS)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label_map.get(v, v) for v in order], fontsize=TICK_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if not pd.isna(p_value):
        ax.text(
            0.04, 0.98,
            f"Kruskal-Wallis\np = {_format_p(float(p_value))}",
            transform=ax.transAxes, va="top", ha="left", fontsize=ANNOTATION_FS,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75, "linewidth": 0},
        )

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax

# ---------------------------------------------------------------------------
# Panels B/C — Gene × cancer paired comparison
# ---------------------------------------------------------------------------

def plot_paired_gc_panel(
    df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    *,
    gene_col: str = "Hugo_Symbol",
    cancer_col: str = "Cancer",
    pred_col: str = "pred_prob",
    min_per_group: int = 3,
    min_pairs_per_cancer: int = 8,
    label_a: str | None = None,
    label_b: str | None = None,
    color_a: str = TEAL,
    color_b: str = ORANGE,
    delta_color: str = PURPLE,
    figsize: tuple[float, float] = (5, 3.2),
    save_path: str | Path | None = None,
) -> dict:
    """Paired comparison at the (gene, cancer) level: mean pred_prob per stratum."""
    required = {gene_col, cancer_col, group_col, pred_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    x = df.loc[df[group_col].isin([group_a, group_b]) & df[pred_col].notna()].copy()
    counts = x.groupby([gene_col, cancer_col, group_col])[pred_col].size().unstack(group_col, fill_value=0)
    valid  = (counts.get(group_a, 0) >= min_per_group) & (counts.get(group_b, 0) >= min_per_group)
    means  = x.groupby([gene_col, cancer_col, group_col])[pred_col].mean().unstack(group_col)
    paired_df = means.loc[valid, [group_a, group_b]].dropna().copy()

    if paired_df.empty:
        raise ValueError(f"No ({gene_col}, {cancer_col}) pairs with >= {min_per_group} rows in both groups.")

    paired_df["delta"] = paired_df[group_b] - paired_df[group_a]
    a_vals     = paired_df[group_a].to_numpy(dtype=float)
    b_vals     = paired_df[group_b].to_numpy(dtype=float)
    delta_vals = paired_df["delta"].to_numpy(dtype=float)
    pooled_p   = _paired_wilcoxon(delta_vals)

    print(
        f"Gene x cancer  {group_b} vs {group_a}: n={len(paired_df)}  "
        f"p={_format_p(pooled_p)}  mean delta={delta_vals.mean():.4f}  median delta={np.median(delta_vals):.4f}"
    )

    per_cancer_stats, cancer_order, delta_by_cancer = _per_cancer_wilcoxon(
        paired_df.reset_index(), cancer_col, min_pairs_per_cancer
    )
    fig = _render_paired_panel(
        a_vals, b_vals,
        label_a if label_a is not None else group_a.replace("_", " ").title(),
        label_b if label_b is not None else group_b.replace("_", " ").title(),
        delta_vals, pooled_p,
        per_cancer_stats, delta_by_cancer, cancer_order,
        ylabel_left="Mean prediction per (gene, cancer)",
        xlabel_right=f"delta = {(label_b or group_b).replace('_', ' ').title()} - {(label_a or group_a).replace('_', ' ').title()}",
        color_a=color_a, color_b=color_b, delta_color=delta_color,
        figsize=figsize, save_path=save_path,
    )
    return {"paired": paired_df, "pooled_p": pooled_p, "pooled_n": len(paired_df), "per_cancer_stats": per_cancer_stats, "fig": fig}

# ---------------------------------------------------------------------------
# Panels B/C — Sample-level paired comparison
# ---------------------------------------------------------------------------

def plot_paired_sample_panel(
    df: pd.DataFrame,
    group_a: str,
    group_b: str,
    *,
    group_col: str = "mut_effect",
    sample_col: str | None = None,
    cancer_col: str = "Cancer",
    pred_col: str = "pred_prob",
    min_genes_per_group: int = 3,
    min_samples_per_cancer: int = 8,
    label_a: str | None = None,
    label_b: str | None = None,
    color_a: str = TEAL,
    color_b: str = ORANGE,
    delta_color: str = PURPLE,
    figsize: tuple[float, float] = (5, 3.2),
    save_path: str | Path | None = None,
) -> dict:
    """Paired comparison at the sample level: mean pred_prob across genes per sample."""
    if sample_col is None:
        sample_col = next(
            (col for col in ("Tumor_Sample_Barcode", "sample", "sample4") if col in df.columns), None
        )
        if sample_col is None:
            raise ValueError("Cannot find a sample column. Pass sample_col explicitly.")

    required = {sample_col, cancer_col, group_col, pred_col}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    x = df.loc[df[group_col].isin([group_a, group_b]) & df[pred_col].notna()].copy()
    counts = x.groupby([sample_col, cancer_col, group_col])[pred_col].size().unstack(group_col, fill_value=0)
    valid  = (counts.get(group_a, 0) >= min_genes_per_group) & (counts.get(group_b, 0) >= min_genes_per_group)
    means  = x.groupby([sample_col, cancer_col, group_col])[pred_col].mean().unstack(group_col)
    paired_df = means.loc[valid, [group_a, group_b]].dropna().copy()

    if paired_df.empty:
        raise ValueError(f"No ({sample_col}, {cancer_col}) pairs with >= {min_genes_per_group} rows in both groups.")

    paired_df["delta"] = paired_df[group_b] - paired_df[group_a]
    a_vals     = paired_df[group_a].to_numpy(dtype=float)
    b_vals     = paired_df[group_b].to_numpy(dtype=float)
    delta_vals = paired_df["delta"].to_numpy(dtype=float)
    pooled_p   = _paired_wilcoxon(delta_vals)

    print(
        f"Sample-level  {group_b} vs {group_a}: n={len(paired_df)}  "
        f"p={_format_p(pooled_p)}  mean delta={delta_vals.mean():.4f}  median delta={np.median(delta_vals):.4f}"
    )

    per_cancer_stats, cancer_order, delta_by_cancer = _per_cancer_wilcoxon(
        paired_df.reset_index(), cancer_col, min_samples_per_cancer
    )
    fig = _render_paired_panel(
        a_vals, b_vals,
        label_a if label_a is not None else group_a.replace("_", " ").title(),
        label_b if label_b is not None else group_b.replace("_", " ").title(),
        delta_vals, pooled_p,
        per_cancer_stats, delta_by_cancer, cancer_order,
        ylabel_left="Mean prediction per sample",
        xlabel_right=f"delta = {(label_b or group_b).replace('_', ' ').title()} - {(label_a or group_a).replace('_', ' ').title()}",
        color_a=color_a, color_b=color_b, delta_color=delta_color,
        figsize=figsize, save_path=save_path,
    )
    return {"paired": paired_df, "pooled_p": pooled_p, "pooled_n": len(paired_df), "per_cancer_stats": per_cancer_stats, "fig": fig}

# ---------------------------------------------------------------------------
# Panels D/E — PolyPhen / SIFT score parsing
# ---------------------------------------------------------------------------

def parse_polyphen(value) -> float:
    """Extract numeric PolyPhen-2 score from MAF string like probably_damaging(0.998)."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text == "." or text.lower().startswith("unknown"):
        return np.nan
    match = re.search(r"\(([\d.]+)\)", text)
    return float(match.group(1)) if match else np.nan


def parse_sift(value) -> float:
    """Extract numeric SIFT score from MAF string like deleterious(0.01). Lower = more deleterious."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text == ".":
        return np.nan
    match = re.search(r"\(([\d.]+)\)", text)
    return float(match.group(1)) if match else np.nan


def plot_score_split_panel(
    df: pd.DataFrame,
    score_col: str,
    score_name: str,
    direction: str,
    *,
    cutoff: float | None = None,
    sample_col: str = "Tumor_Sample_Barcode",
    cancer_col: str = "Cancer",
    pred_col: str = "pred_prob",
    min_per_group: int = 3,
    min_samples_per_cancer: int = 8,
    color_low: str = TEAL,
    color_high: str = ORANGE,
    delta_color: str = PURPLE,
    figsize: tuple[float, float] = (6.2, 3.6),
    save_path: str | Path | None = None,
) -> dict:
    """Split missense mutations by score threshold and compare predictions per sample.

    direction='high_minus_low': PolyPhen (higher = more damaging, expect higher pred_prob).
    direction='low_minus_high': SIFT     (lower = more deleterious, expect higher pred_prob).
    """
    if direction not in {"high_minus_low", "low_minus_high"}:
        raise ValueError("direction must be 'high_minus_low' or 'low_minus_high'.")
    if sample_col not in df.columns:
        sample_col = next((col for col in ("sample4", "sample") if col in df.columns), sample_col)

    x = df.dropna(subset=[sample_col, cancer_col, pred_col, score_col]).copy()
    score_cutoff = float(x[score_col].median()) if cutoff is None else float(cutoff)
    x["_score_group"] = np.where(x[score_col] >= score_cutoff, "high", "low")

    counts = x.groupby([sample_col, cancer_col, "_score_group"])[pred_col].size().unstack("_score_group", fill_value=0)
    valid  = (counts.get("low", 0) >= min_per_group) & (counts.get("high", 0) >= min_per_group)
    means  = x.groupby([sample_col, cancer_col, "_score_group"])[pred_col].mean().unstack("_score_group")
    paired_df = means.loc[valid, ["low", "high"]].dropna().copy()

    if paired_df.empty:
        raise ValueError(f"No sample pairs with >= {min_per_group} rows in both score halves.")

    if direction == "high_minus_low":
        paired_df["delta"] = paired_df["high"] - paired_df["low"]
        a_vals, b_vals = paired_df["low"].to_numpy(dtype=float), paired_df["high"].to_numpy(dtype=float)
        label_a, label_b = "Low", "High"
        xlabel_right = f"delta = High - Low [{score_name}]"
    else:
        paired_df["delta"] = paired_df["low"] - paired_df["high"]
        a_vals, b_vals = paired_df["high"].to_numpy(dtype=float), paired_df["low"].to_numpy(dtype=float)
        label_a, label_b = "High", "Low"
        xlabel_right = f"delta = Low - High [{score_name}]"

    delta_vals = paired_df["delta"].to_numpy(dtype=float)
    pooled_p   = _paired_wilcoxon(delta_vals)
    cutoff_src = "median" if cutoff is None else "user-defined"
    print(
        f"{score_name} ({direction}): n={len(paired_df)}  p={_format_p(pooled_p)}  "
        f"cutoff={score_cutoff:.3f} ({cutoff_src})  mean delta={delta_vals.mean():.4f}  "
        f"median delta={np.median(delta_vals):.4f}"
    )

    per_cancer_stats, cancer_order, delta_by_cancer = _per_cancer_wilcoxon(
        paired_df.reset_index(), cancer_col, min_samples_per_cancer
    )
    fig = _render_paired_panel(
        a_vals, b_vals, label_a, label_b,
        delta_vals, pooled_p,
        per_cancer_stats, delta_by_cancer, cancer_order,
        ylabel_left=f"Mean prediction per sample\n[{score_name} {label_a.lower()} vs {label_b.lower()}]",
        xlabel_right=xlabel_right,
        color_a=color_low, color_b=color_high, delta_color=delta_color,
        figsize=figsize, save_path=save_path,
    )
    return {
        "paired": paired_df, "pooled_p": pooled_p, "pooled_n": len(paired_df),
        "per_cancer_stats": per_cancer_stats, "cutoff": score_cutoff, "fig": fig,
    }

# ---------------------------------------------------------------------------
# Permutation test — WT/silent gene × cancer delta
# ---------------------------------------------------------------------------

def compute_paired_gc(
    df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    *,
    gene_col: str = "Hugo_Symbol",
    cancer_col: str = "Cancer",
    pred_col: str = "pred_prob",
    min_per_group: int = 3,
    valid_pairs: pd.Index | None = None,
) -> dict:
    """Compute gene-by-cancer paired deltas without plotting (used in permutation loop)."""
    x = df.loc[df[group_col].isin([group_a, group_b]) & df[pred_col].notna()]
    means = x.groupby([gene_col, cancer_col, group_col])[pred_col].mean().unstack(group_col)
    counts = x.groupby([gene_col, cancer_col, group_col])[pred_col].size().unstack(group_col, fill_value=0)
    valid = (counts.get(group_a, 0) >= min_per_group) & (counts.get(group_b, 0) >= min_per_group)
    if valid_pairs is not None:
        valid = valid & valid.index.isin(valid_pairs)
    paired_df = means.loc[valid, [group_a, group_b]].dropna().copy()

    if paired_df.empty:
        return {"paired": paired_df, "pooled_p": np.nan, "pooled_n": 0}

    paired_df["delta"] = paired_df[group_b] - paired_df[group_a]
    delta_vals = paired_df["delta"].to_numpy(dtype=float)
    return {"paired": paired_df, "pooled_p": _paired_wilcoxon(delta_vals), "pooled_n": len(paired_df)}


def permute_wt_silent_within_sample(
    df: pd.DataFrame,
    *,
    group_col: str = "mut_effect",
    group_a: str = "no_mutation",
    group_b: str = "silent",
    out_col: str = "mut_effect_perm",
    sample_col: str | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Shuffle WT/silent labels within each sample (within-sample null)."""
    if rng is None:
        rng = np.random.default_rng()
    if sample_col is None:
        sample_col = next(
            (col for col in ("Tumor_Sample_Barcode", "sample", "sample4") if col in df.columns), None
        )
        if sample_col is None:
            raise ValueError("Cannot find a sample column. Pass sample_col explicitly.")

    out = df.copy()
    out[out_col] = out[group_col].to_numpy(dtype=object)
    mask = out[group_col].isin([group_a, group_b])
    for _, idx in out.loc[mask].groupby(sample_col).groups.items():
        labels = out.loc[idx, out_col].to_numpy(dtype=object).copy()
        rng.shuffle(labels)
        out.loc[idx, out_col] = labels
    return out


def _single_permutation(
    df: pd.DataFrame,
    valid_pairs: pd.Index,
    seed: int,
    i: int,
    sample_col: str,
) -> dict:
    """One permutation iteration — module-level so joblib can pickle it."""
    rng = np.random.default_rng(seed)
    perm_df = permute_wt_silent_within_sample(df, sample_col=sample_col, rng=rng)
    perm_gc = compute_paired_gc(
        perm_df, group_col="mut_effect_perm",
        group_a="no_mutation", group_b="silent", valid_pairs=valid_pairs,
    )
    deltas = perm_gc["paired"]["delta"]
    return {
        "perm": i,
        "deltas": deltas.rename(i),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(deltas.median()),
        "n_pairs": len(deltas),
        "n_upward": int((deltas > 0).sum()),
        "frac_upward": float((deltas > 0).mean()),
        "wilcoxon_p": perm_gc["pooled_p"],
    }


def run_gc_permutations(
    df: pd.DataFrame,
    n_perm: int = 200,
    seed: int = 1,
    sample_col: str | None = None,
    n_jobs: int = 1,
) -> dict:
    """Run repeated WT/silent within-sample permutations for gene-by-cancer deltas.

    Returns real results alongside per-permutation summaries and per-(gene,cancer) stats.
    n_perm=1000 is the manuscript setting; use n_perm=50 for quick validation.

    n_jobs: number of parallel workers passed to joblib.Parallel.
      1  = sequential (default, no joblib dependency).
      -1 = use all available cores.
      If joblib freezes in a Jupyter notebook, pass prefer='threads' by editing the call below,
      or keep n_jobs=1.
    """
    if sample_col is None:
        sample_col = next(
            (col for col in ("Tumor_Sample_Barcode", "sample", "sample4") if col in df.columns), None
        )
        if sample_col is None:
            raise ValueError("Cannot find a sample column. Pass sample_col explicitly.")

    rng = np.random.default_rng(seed)
    real_gc = compute_paired_gc(df, group_col="mut_effect", group_a="no_mutation", group_b="silent", min_per_group=3)
    valid_pairs = real_gc["paired"].index
    seeds = rng.integers(0, 10_000_000, size=n_perm)

    if n_jobs == 1:
        results = [
            _single_permutation(df, valid_pairs, int(s), i, sample_col)
            for i, s in enumerate(seeds)
        ]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(
            delayed(_single_permutation)(df, valid_pairs, int(s), i, sample_col)
            for i, s in enumerate(seeds)
        )

    global_rows    = [{k: v for k, v in r.items() if k != "deltas"} for r in results]
    perm_delta_cols = [r["deltas"] for r in results]

    perm_summary  = pd.DataFrame(global_rows)
    perm_delta_mat = pd.concat(perm_delta_cols, axis=1)
    real_delta = real_gc["paired"]["delta"]
    perm_mean  = perm_delta_mat.mean(axis=1)
    perm_std   = perm_delta_mat.std(axis=1)
    z_score    = (real_delta - perm_mean) / perm_std.replace(0, np.nan)
    valid_n    = perm_delta_mat.notna().sum(axis=1)
    emp_p      = (perm_delta_mat.ge(real_delta, axis=0).sum(axis=1) + 1) / (valid_n + 1)

    gc_stats = pd.DataFrame({
        "real_delta": real_delta, "perm_mean": perm_mean, "perm_std": perm_std,
        "gap": real_delta - perm_mean, "z_score": z_score,
        "emp_p": emp_p, "n_perm_valid": valid_n,
    }).reset_index()

    return {
        "real_gc": real_gc, "perm_summary": perm_summary,
        "gc_stats": gc_stats, "perm_delta_mat": perm_delta_mat,
    }


def plot_gc_permutation_histogram(
    perm_test: dict,
    *,
    stat_col: str = "mean_delta",
    real_value: float | None = None,
    alternative: str = "greater",
    bins: int = 30,
    figsize: tuple[float, float] = (3.5, 3.1),
    null_color: str = TEAL,
    real_color: str = MUT_COLOR_HIGH_IMPACT,
    mean_color: str = "grey",
    max_xticks: int = 6,
    save_path: str | Path | None = None,
) -> tuple[plt.Figure, plt.Axes, float]:
    """Plot the WT/silent permutation null distribution against the observed statistic."""
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'.")

    perm_values = perm_test["perm_summary"][stat_col].dropna().astype(float)
    if perm_values.empty:
        raise ValueError("No permutation values to plot.")

    if real_value is None:
        real_delta = perm_test["real_gc"]["paired"]["delta"].dropna().astype(float)
        lookup = {
            "median_delta": float(real_delta.median()),
            "frac_upward": float((real_delta > 0).mean()),
            "n_upward": float((real_delta > 0).sum()),
            "wilcoxon_p": float(perm_test["real_gc"]["pooled_p"]),
        }
        real_value = lookup.get(stat_col, float(real_delta.mean()))

    if stat_col == "wilcoxon_p" and alternative == "greater":
        alternative = "less"

    n_perm = len(perm_values)
    if alternative == "greater":
        p_value = float(((perm_values >= real_value).sum() + 1) / (n_perm + 1))
    elif alternative == "less":
        p_value = float(((perm_values <= real_value).sum() + 1) / (n_perm + 1))
    else:
        center   = float(perm_values.mean())
        null_ext = perm_values.sub(center).abs()
        obs_ext  = abs(real_value - center)
        p_value  = float(((null_ext >= obs_ext).sum() + 1) / (n_perm + 1))

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(perm_values, bins=bins, color=null_color, alpha=0.72, edgecolor="white", linewidth=0.7)
    ax.axvline(real_value, color=real_color, linewidth=2.0, label="Observed")
    ax.axvline(float(perm_values.mean()), color=mean_color, linewidth=1.0, linestyle="--", alpha=0.65, label="Null mean")

    p_text = "< 1e-300" if p_value < 1e-300 else f"< {p_value:.1e}"
    ax.text(
        0.9, 0.95,
        f"observed = {real_value:.4f}\np {p_text}",
        transform=ax.transAxes, ha="right", va="top", fontsize=ANNOTATION_FS,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.78, "linewidth": 0},
    )
    ax.set_xlabel(stat_col.replace("_", " "), fontsize=AXIS_LABEL_FS, labelpad=10)
    ax.set_ylabel("Permutations", fontsize=AXIS_LABEL_FS)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=max_xticks))
    ax.tick_params(axis="x", labelsize=TICK_FS, length=0)
    ax.tick_params(axis="y", labelsize=TICK_FS, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E3DDCF", linewidth=0.7, alpha=0.7)
    ax.legend(frameon=False, fontsize=TICK_FS, bbox_to_anchor=(0.54, 0.83))
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax, p_value
