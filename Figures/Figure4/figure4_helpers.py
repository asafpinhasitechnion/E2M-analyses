"""Helpers for the Figure 4 notebook.

The notebook owns the analysis parameters. This module owns only reusable
I/O, plotting, statistics, and source-data export functions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import json
import logging
import re
from functools import lru_cache

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D


# Manuscript style rcParams, replicated here so this helper
# stays standalone while matching the active Figure 4 panels: transparent backgrounds,
# embedded fonts, and only the left/bottom axis spines (no full box around UMAPs).
mpl.rcParams["figure.facecolor"] = "none"
mpl.rcParams["axes.facecolor"] = "none"
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)


POSTER_COLORS = [
    "#aa4b53", "#8f6bb1", "#e0cc9b", "#7aa08f", "#d9826b", "#566b9f",
    "#c06c9b", "#6a8f55", "#c9a34e", "#5e8bb8", "#9b6b43", "#7f7f7f",
]
RESPONSE_PALETTE = {"R": "#558771", "NR": "#aa4b53"}
FULL_RESPONSE_PALETTE = {"CR": "#558771", "PR": "#82a899", "SD": "#e0cc9b", "PD": "#aa4b53"}
RESPONSE_ORDER = ["R", "NR"]
FULL_RESPONSE_ORDER = ["CR", "PR", "SD", "PD"]

# Compact labels for plot titles so per-cluster panels stay short, e.g. "Morrison BOR by emb cluster".
SPACE_SHORT = {"embedding": "emb", "expression": "expr", "mutation_pca": "mut"}
CLINICAL_VAR_SHORT = {"Full_response": "BOR", "Response": "response", "OS_time": "OS", "PFS_time": "PFS"}


def _short_space(space: str) -> str:
    return SPACE_SHORT.get(str(space), str(space))


def _short_var(var: str) -> str:
    return CLINICAL_VAR_SHORT.get(str(var), str(var))

TITLE_FS = 10
AXIS_LABEL_FS = 9
TICK_FS = 8
LEGEND_FS = 8
ANNOTATION_FS = 8
COLORBAR_LABEL_FS = 8
COLORBAR_TICK_FS = 7
LEGEND_FRAME_ALPHA = 0.4
COMPACT_LEGEND_FS = LEGEND_FS - 0.5


def _truncated_magma():
    # Shared external AUPRC tile colormap:
    # reversed magma truncated to [0, 0.8] so high values render dark, low values warm.
    from matplotlib.colors import ListedColormap
    return ListedColormap(plt.get_cmap("magma_r")(np.linspace(0, 0.8, 256)), name="truncated_magma")


CMAP_TRUNCATED_MAGMA = _truncated_magma()


def build_label_palette(labels, colors=POSTER_COLORS) -> dict:
    labels = pd.Series(labels).dropna().astype(str)
    unique = list(dict.fromkeys(labels.tolist()))
    return {label: colors[i % len(colors)] for i, label in enumerate(unique)}


@dataclass(frozen=True)
class Figure4Config:
    project_root: Path
    tcga_results_root: Path
    tcga_data_dir: Path
    tcga_config_path: Path
    tcga_mutation_label_dir: Path
    tcga_clinical_path: Path
    external_results_root: Path
    external_config_path: Path
    figure_dir: Path
    source_data_dir: Path

    @classmethod
    def from_project_root(cls, project_root: str | Path) -> "Figure4Config":
        root = Path(project_root).resolve()
        tcga_run = root / "Runs" / "TCGA"
        figure_dir = root / "Figures" / "Figure4" / "output"
        return cls(
            project_root=root,
            tcga_results_root=tcga_run / "output" / "cpm",
            tcga_data_dir=tcga_run / "data",
            tcga_config_path=tcga_run / "config" / "e2m.yaml",
            tcga_mutation_label_dir=tcga_run / "data" / "mutations",
            tcga_clinical_path=root / "data" / "GDC-PANCAN.TCGA_phenotype.tsv.gz",
            external_results_root=root / "Runs" / "External" / "output",
            external_config_path=root / "Runs" / "External" / "config" / "e2m.yaml",
            figure_dir=figure_dir,
            source_data_dir=figure_dir / "source_data",
        )

    def ensure_inputs(self) -> None:
        missing = [
            path for path in (
                self.tcga_results_root,
                self.tcga_mutation_label_dir,
                self.tcga_clinical_path,
                self.external_results_root,
            )
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError("Missing Figure 4 input paths: " + ", ".join(map(str, missing)))

    def ensure_outputs(self) -> None:
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.source_data_dir.mkdir(parents=True, exist_ok=True)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"name": name, "path": str(getattr(self, name))}
            for name in (
                "project_root", "tcga_results_root", "tcga_data_dir", "tcga_config_path", "tcga_mutation_label_dir",
                "tcga_clinical_path", "external_results_root", "external_config_path", "figure_dir", "source_data_dir",
            )
        ])


def save_source_table(df: pd.DataFrame, config: Figure4Config, name: str) -> Path:
    config.ensure_outputs()
    path = config.source_data_dir / name
    df.to_csv(path, index=False)
    print(f"saved {path}")
    return path


def print_and_save(df: pd.DataFrame, config: Figure4Config, name: str, *, head: int = 20) -> pd.DataFrame:
    save_source_table(df, config, name)
    print((df.head(head) if len(df) > head else df).to_string(index=False))
    if len(df) > head:
        print(f"... {len(df)} rows total")
    return df


def projection_dependency_status() -> pd.DataFrame:
    rows = []
    for module_name, purpose in (
        ("umap", "UMAP projection"),
        ("igraph", "Leiden graph"),
        ("leidenalg", "Leiden clustering"),
        ("lifelines", "Kaplan-Meier/log-rank plotting"),
    ):
        try:
            __import__(module_name)
            available = True
        except ImportError:
            available = False
        rows.append({"module": module_name, "purpose": purpose, "available": available})
    return pd.DataFrame(rows)


def add_bh_fdr(
    df: pd.DataFrame,
    *,
    p_col: str = "p_value",
    fdr_col: str = "p_fdr",
    group_cols: str | list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out[fdr_col] = np.nan
    if group_cols is not None:
        group_cols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
        for _, idx in out.groupby(group_cols, dropna=False).groups.items():
            adjusted = add_bh_fdr(out.loc[idx], p_col=p_col, fdr_col=fdr_col)
            out.loc[idx, fdr_col] = adjusted[fdr_col]
        return out
    p = pd.to_numeric(out[p_col], errors="coerce") if p_col in out.columns else pd.Series(dtype=float)
    valid = p.notna() & np.isfinite(p)
    if not valid.any():
        return out
    ordered = p.loc[valid].sort_values().index
    ranked = p.loc[ordered].astype(float).to_numpy()
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out.loc[ordered, fdr_col] = np.clip(adjusted, 0, 1)
    return out


def tcga_sample_id(sample_id: str) -> str:
    value = str(sample_id)
    parts = value.split("-")
    if len(parts) >= 4 and parts[0] == "TCGA":
        parts[3] = parts[3][:2]
        return "-".join(parts[:4])
    return value


def slug(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return text.strip("_") or "na"


def _read_indexed_frame(path: Path, index_name: str = "sample") -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    if index_name not in frame.columns:
        index_name = frame.columns[0]
    frame = frame.set_index(index_name)
    frame.index = frame.index.astype(str)
    return frame


def load_tcga_sample_embeddings(config: Figure4Config, cancer: str) -> pd.DataFrame:
    path = config.tcga_results_root / cancer / "embeddings.csv"
    emb = pd.read_csv(path, index_col=0)
    emb.index = pd.Index([tcga_sample_id(x) for x in emb.index.astype(str)], name="sample")
    emb = emb[~emb.index.duplicated(keep="first")]
    return emb.apply(pd.to_numeric, errors="coerce")


def load_tcga_expression(config: Figure4Config, cancer: str) -> pd.DataFrame:
    """Expression matrix the TCGA model was trained on (E2M with the Runs/TCGA config, cached in the data folder)."""
    import e2m

    data = e2m.Dataset.from_tcga([cancer], data_dir=config.tcga_data_dir, config=config.tcga_config_path, with_tmb=False)
    expression = data.expression
    expression.index = pd.Index([tcga_sample_id(x) for x in expression.index.astype(str)], name="sample")
    return expression[~expression.index.duplicated(keep="first")]


@lru_cache(maxsize=1)
def _load_tcga_clinical_table(path: str) -> pd.DataFrame:
    clinical = pd.read_csv(path, sep="\t", low_memory=False)
    clinical["sample"] = clinical["sample"].astype(str).map(tcga_sample_id)
    return clinical.drop_duplicates("sample").set_index("sample")


def load_tcga_bundle(config: Figure4Config, cancer: str) -> dict:
    cancer = cancer.upper()
    embeddings = load_tcga_sample_embeddings(config, cancer)
    expression = load_tcga_expression(config, cancer)
    common = embeddings.index.intersection(expression.index)
    clinical = _load_tcga_clinical_table(str(config.tcga_clinical_path)).reindex(common)
    return {
        "cancer": cancer,
        "samples": common,
        "clinical": clinical,
        "expression": expression.loc[common],
        "embeddings": embeddings.loc[common],
    }


def load_tcga_bundles(config: Figure4Config, *, cancers: Iterable[str]) -> dict[str, dict]:
    config.ensure_inputs()
    return {cancer.upper(): load_tcga_bundle(config, cancer) for cancer in cancers}


def _tcga_prediction_target_genes(config: Figure4Config, cancer: str) -> list[str]:
    path = config.tcga_results_root / cancer / "cv" / "oof_probabilities.csv"
    header = pd.read_csv(path, nrows=0).columns.tolist()
    return header[1:]


def _tcga_mutation_label_path(config: Figure4Config, cancer: str) -> Path:
    path = config.tcga_mutation_label_dir / f"{cancer}_mc3_gene_level.txt.gz"
    if not path.exists():
        raise FileNotFoundError(f"{cancer}: missing MC3 gene-level mutation label file {path}")
    return path


def load_tcga_mutation_labels(
    config: Figure4Config,
    cancer: str,
    *,
    samples: Iterable[str] | None = None,
    target_genes: Iterable[str] | None = None,
) -> pd.DataFrame:
    cancer = cancer.upper()
    labels = pd.read_csv(_tcga_mutation_label_path(config, cancer), sep="\t")
    labels = labels.rename(columns={labels.columns[0]: "gene"}).set_index("gene").T
    labels.index = pd.Index([tcga_sample_id(x) for x in labels.index.astype(str)], name="sample")
    labels = labels.groupby(labels.index).max()
    labels = labels.T.groupby(labels.T.index).max().T
    labels = (labels.apply(pd.to_numeric, errors="coerce").fillna(0) > 0).astype(np.int8)
    if target_genes is not None:
        genes = [str(g) for g in target_genes if str(g) in labels.columns]
        labels = labels.loc[:, genes]
    if samples is not None:
        sample_index = pd.Index([tcga_sample_id(x) for x in samples], name="sample")
        labels = labels.reindex(sample_index).fillna(0).astype(np.int8)
    return labels


def run_umap_cluster(
    X: pd.DataFrame,
    *,
    n_neighbors: int,
    min_dist: float,
    leiden_resolution: float,
    metric: str = "euclidean",
    scale: bool = True,
    cluster_method: str = "leiden",
    n_clusters: int = 4,
    random_state: int = 0,
) -> dict:
    import umap
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    values = X.to_numpy(dtype=float)
    if scale:
        values = StandardScaler().fit_transform(values)
    n_neighbors = max(2, min(int(n_neighbors), values.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=float(min_dist),
        metric=metric,
        random_state=int(random_state),
    )
    coords_arr = reducer.fit_transform(values)
    if cluster_method == "kmeans":
        labels = KMeans(n_clusters=int(n_clusters), n_init=20, random_state=int(random_state)).fit_predict(values)
    elif cluster_method == "leiden":
        import igraph as ig
        import leidenalg
        graph = reducer.graph_.tocsr()
        rows, cols = graph.nonzero()
        g = ig.Graph(n=values.shape[0], edges=list(zip(rows.tolist(), cols.tolist())), directed=False)
        g.es["weight"] = graph.data.tolist()
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=g.es["weight"],
            resolution_parameter=float(leiden_resolution),
            seed=int(random_state),
        )
        labels = np.asarray(part.membership, dtype=int)
    else:
        raise ValueError("cluster_method must be 'leiden' or 'kmeans'.")
    return {
        "coords": pd.DataFrame(coords_arr, index=X.index, columns=["UMAP1", "UMAP2"]),
        "labels": pd.Series(labels, index=X.index, name="cluster").astype(str),
    }


def compute_mutation_pca_umap(
    mutation_labels: pd.DataFrame,
    *,
    n_components: int,
    n_neighbors: int,
    min_dist: float,
    leiden_resolution: float,
    cluster_method: str,
    n_clusters: int,
    random_state: int,
) -> dict:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    values = StandardScaler().fit_transform(mutation_labels.to_numpy(dtype=float))
    n = min(int(n_components), values.shape[0], values.shape[1])
    pca = PCA(n_components=n, random_state=int(random_state))
    pcs = pd.DataFrame(pca.fit_transform(values), index=mutation_labels.index, columns=[f"PC{i+1}" for i in range(n)])
    out = run_umap_cluster(
        pcs,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        leiden_resolution=leiden_resolution,
        metric="euclidean",
        scale=False,
        cluster_method=cluster_method,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    out["pca"] = pcs
    out["explained_variance_ratio"] = pca.explained_variance_ratio_
    return out


def recompute_tcga_expression_embedding_spaces(
    bundles: dict[str, dict],
    *,
    params: dict[str, dict],
    random_state: int = 0,
) -> dict[str, dict]:
    """Recompute expression and model-embedding UMAP/Leiden spaces from saved matrices."""
    for cancer, bundle in bundles.items():
        if "expression" not in bundle or "embeddings" not in bundle:
            raise KeyError(
                f"{cancer}: expression/embedding matrices are required to recompute clusters. "
                "Load the TCGA bundle with load_tcga_bundle()."
            )
        p = params[cancer]
        bundle["expression_umap"] = run_umap_cluster(
            bundle["expression"],
            n_neighbors=int(p["expr_nn"]),
            min_dist=float(p.get("expr_min_dist", 0.3)),
            leiden_resolution=float(p["expr_leiden_res"]),
            metric=p.get("expr_metric", "cosine"),
            scale=bool(p.get("expr_scale", True)),
            cluster_method=p.get("expr_cluster_method", "leiden"),
            n_clusters=int(p.get("expr_clusters", 4)),
            random_state=random_state,
        )
        bundle["embedding_umap"] = run_umap_cluster(
            bundle["embeddings"],
            n_neighbors=int(p["emb_nn"]),
            min_dist=float(p.get("emb_min_dist", 0.3)),
            leiden_resolution=float(p["emb_leiden_res"]),
            metric=p.get("emb_metric", "euclidean"),
            scale=bool(p.get("emb_scale", True)),
            cluster_method=p.get("emb_cluster_method", "leiden"),
            n_clusters=int(p.get("emb_clusters", 4)),
            random_state=random_state,
        )
    return bundles


def add_tcga_mutation_spaces(
    config: Figure4Config,
    bundles: dict[str, dict],
    *,
    params: dict[str, dict],
    random_state: int = 0,
) -> dict[str, dict]:
    for cancer, bundle in bundles.items():
        p = params[cancer]
        labels = load_tcga_mutation_labels(
            config,
            cancer,
            samples=bundle["samples"],
            target_genes=_tcga_prediction_target_genes(config, cancer),
        )
        labels = labels.loc[:, labels.sum(axis=0) > 0]
        if labels.shape[1] < 2:
            raise ValueError(f"{cancer}: mutation matrix has fewer than two nonzero genes.")
        bundle["mutation_labels"] = labels
        bundle["mutation_pca_umap"] = compute_mutation_pca_umap(
            labels,
            n_components=int(p.get("mut_pca_components", 256)),
            n_neighbors=int(p["mut_nn"]),
            min_dist=float(p.get("mut_min_dist", 0.3)),
            leiden_resolution=float(p["mut_leiden_res"]),
            cluster_method=p.get("mut_cluster_method", "leiden"),
            n_clusters=int(p.get("mut_clusters", 4)),
            random_state=random_state,
        )
    return bundles


def summarize_tcga_bundles(bundles: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for cancer, bundle in sorted(bundles.items()):
        for space, key in (("embedding", "embedding_umap"), ("expression", "expression_umap"), ("mutation_pca", "mutation_pca_umap")):
            if key not in bundle:
                continue
            labels = bundle[key]["labels"].astype(str)
            counts = labels.value_counts().sort_index()
            rows.append({
                "cohort": cancer,
                "space": space,
                "n_samples": int(len(labels)),
                "n_clusters": int(counts.shape[0]),
                "cluster_sizes": "; ".join(f"{k}:{int(v)}" for k, v in counts.items()),
            })
    return pd.DataFrame(rows)


def _pick_column(frame: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    for col in candidates:
        if col in frame.columns:
            return frame[col]
    return pd.Series(np.nan, index=frame.index)


def derive_os_columns(clinical: pd.DataFrame) -> pd.DataFrame:
    out = clinical.copy()
    if "OS_time" in out.columns and "OS_event" in out.columns:
        return out
    death = pd.to_numeric(_pick_column(out, ("demographic.days_to_death", "days_to_death.demographic", "days_to_death")), errors="coerce")
    follow = pd.to_numeric(_pick_column(out, ("diagnoses.days_to_last_follow_up", "days_to_last_follow_up.diagnoses", "days_to_last_follow_up")), errors="coerce")
    out["OS_time"] = death.where(~death.isna(), follow)
    status = _pick_column(out, ("demographic.vital_status", "vital_status.demographic", "vital_status")).astype("string").str.lower()
    event = pd.Series(np.nan, index=out.index, dtype=object)
    event[status.str.contains("dead", na=False)] = 1
    event[status.str.contains("alive", na=False)] = 0
    out["OS_event"] = pd.to_numeric(event, errors="coerce")
    return out


def _prepare_survival_groups(
    clinical: pd.DataFrame,
    groups,
    *,
    time_col: str,
    event_col: str,
    min_group_size: int,
) -> tuple[pd.DataFrame, pd.Index]:
    groups = pd.Series(groups, name="group").astype(str)
    if time_col not in clinical.columns or event_col not in clinical.columns:
        raise KeyError(f"Missing survival columns in clinical: {time_col!r}, {event_col!r}")
    surv = clinical[[time_col, event_col]].copy()
    surv[time_col] = pd.to_numeric(surv[time_col], errors="coerce")
    surv[event_col] = pd.to_numeric(surv[event_col], errors="coerce")
    surv = surv.dropna()
    surv = surv[surv[time_col] > 0].rename(columns={time_col: "time", event_col: "event"})
    surv["event"] = surv["event"].astype(bool)
    df = surv.join(groups, how="inner").dropna(subset=["group"])
    counts = df["group"].value_counts()
    keep = counts[counts >= min_group_size].index
    df = df[df["group"].isin(keep)].copy()
    if df["group"].nunique() < 2:
        raise ValueError(f"Need >=2 groups with >= {min_group_size} samples (got {dict(counts)}).")
    return df, keep


def _logrank_pvalue_basic(df: pd.DataFrame, sorted_groups: list[str]) -> float:
    from scipy.stats import chi2
    event_times = np.sort(df.loc[df["event"], "time"].unique())
    k = len(sorted_groups)
    observed = np.zeros(k)
    expected = np.zeros(k)
    variance = np.zeros((k, k))
    for t in event_times:
        at_risk = np.array([np.sum((df["group"] == g) & (df["time"] >= t)) for g in sorted_groups], dtype=float)
        events = np.array([np.sum((df["group"] == g) & (df["time"] == t) & df["event"]) for g in sorted_groups], dtype=float)
        n = at_risk.sum()
        d = events.sum()
        if n <= 1 or d == 0:
            continue
        frac = at_risk / n
        observed += events
        expected += d * frac
        variance += d * (n - d) / (n - 1.0) * (np.diag(frac) - np.outer(frac, frac))
    diff = observed - expected
    stat = float(diff @ np.linalg.pinv(variance) @ diff)
    return float(chi2.sf(stat, max(k - 1, 1)))


def _style_legend_frame(legend, *, alpha: float = LEGEND_FRAME_ALPHA) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("black")
    frame.set_linewidth(0.5)
    frame.set_alpha(alpha)


def _compact_cluster_name(group_name: str) -> str:
    name = str(group_name).strip()
    if not name:
        return ""
    lower = name.lower()
    if lower == "cluster":
        return "Clus."
    name = re.sub(r"\b[Cc]luster\b", "Clus.", name)
    name = re.sub(r"\b[Ll]eiden\b", "clus.", name)
    return name


def _km_group_label(group_name: str, group, n: int) -> str:
    prefix = _compact_cluster_name(group_name)
    label = f"{prefix} {group}".strip()
    return f"{label} (n={int(n)})"


def plot_umap_categorical(
    coords: pd.DataFrame,
    labels,
    *,
    title: str,
    save_path: str | Path | None,
    palette: dict | None = None,
    show_legend: bool = False,
    legend_title: str = "",
    figsize: tuple = (1.9, 1.9),
    point_size: float = 9,
) -> tuple[plt.Figure, plt.Axes, dict]:
    # Keep UMAP styling stable across the clean Figure 4 panels.
    raw_labels = pd.Series(labels, index=coords.index)
    missing = raw_labels.isna()
    labels = raw_labels.astype("string")
    palette = {str(k): v for k, v in palette.items()} if palette is not None else build_label_palette(labels.loc[~missing])
    colors = labels.astype(str).map(palette)
    colors.loc[missing] = "#BDBDBD"
    colors = colors.fillna("#BDBDBD")
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(coords["UMAP1"], coords["UMAP2"], c=colors, s=point_size, edgecolors="none", alpha=0.95)
    ax.set_title(title, fontsize=TITLE_FS - 1)
    ax.set_xlabel("UMAP1", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("UMAP2", fontsize=AXIS_LABEL_FS)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)
    if show_legend:
        handles = [Line2D([0], [0], marker="o", linestyle="", markersize=3, color=c, label=l) for l, c in palette.items()]
        legend = ax.legend(
            handles=handles,
            title=legend_title,
            fontsize=COMPACT_LEGEND_FS,
            title_fontsize=COMPACT_LEGEND_FS,
            frameon=True,
            framealpha=LEGEND_FRAME_ALPHA,
            facecolor="white",
            loc="best",
            handletextpad=0.3,
            borderpad=0.25,
            labelspacing=0.25,
            borderaxespad=0.25,
        )
        _style_legend_frame(legend)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    return fig, ax, palette


def _km_step_points(time, event) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=bool)
    event_times = np.sort(np.unique(time[event]))
    x, y = [0.0], [1.0]
    censor_x, censor_y = [], []
    survival = 1.0
    for t in event_times:
        at_risk = np.sum(time >= t)
        events_at_t = np.sum((time == t) & event)
        if at_risk > 0:
            survival *= 1.0 - events_at_t / at_risk
        x.extend([float(t), float(t)])
        y.extend([y[-1], survival])
    last_time = float(np.max(time)) if len(time) else 0.0
    if x[-1] < last_time:
        x.append(last_time)
        y.append(y[-1])
    for t in np.sort(np.unique(time[~event])):
        idx = np.searchsorted(np.asarray(x), t, side="right") - 1
        censor_x.append(float(t))
        censor_y.append(float(y[max(idx, 0)]))
    return np.asarray(x), np.asarray(y), np.asarray(censor_x), np.asarray(censor_y)


def plot_km_curves(
    clinical: pd.DataFrame,
    groups,
    *,
    time_col: str,
    event_col: str,
    title: str,
    save_path: str | Path | None,
    group_name: str = "Cluster",
    palette: dict | None = None,
    min_group_size: int = 3,
    figsize: tuple = (2.7, 2.3),
    censor_markers: bool = True,
    max_time: float | None = None,
    time_label: str = "Days",
    p_adjusted: float | None = None,
    adjusted_label: str = "p(FDR)",
    p_in_legend: bool = False,
) -> dict:
    df, keep = _prepare_survival_groups(clinical, groups, time_col=time_col, event_col=event_col, min_group_size=min_group_size)
    sorted_groups = sorted(df["group"].unique(), key=str)
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import multivariate_logrank_test
        p_value = float(multivariate_logrank_test(df["time"], df["group"], df["event"]).p_value)
        lifelines_available = True
    except ImportError:
        p_value = _logrank_pvalue_basic(df, sorted_groups)
        lifelines_available = False
    palette = {str(k): v for k, v in palette.items()} if palette is not None else build_label_palette(sorted_groups)
    fig, ax = plt.subplots(figsize=figsize)
    for group in sorted_groups:
        mask = df["group"] == group
        label = _km_group_label(group_name, group, int(mask.sum()))
        if lifelines_available:
            kmf = KaplanMeierFitter()
            kmf.fit(df.loc[mask, "time"], event_observed=df.loc[mask, "event"], label=label)
            kmf.plot(ax=ax, ci_show=False, show_censors=censor_markers, censor_styles={"ms": 7}, color=palette.get(str(group)))
        else:
            x, y, cx, cy = _km_step_points(df.loc[mask, "time"], df.loc[mask, "event"])
            ax.step(x, y, where="post", label=label, color=palette.get(str(group)))
            if censor_markers and len(cx):
                ax.plot(cx, cy, linestyle="", marker="+", ms=7, color=palette.get(str(group)))
    if max_time is not None:
        ax.set_xlim(0, max_time)
    ax.set_ylim(0, 1.02)
    # p-value label (log-rank, or FDR-adjusted when provided).
    if p_adjusted is not None and np.isfinite(p_adjusted):
        p_label = f"{adjusted_label}={p_adjusted:.2e}"
    else:
        p_text = f"{p_value:.2e}" if np.isfinite(p_value) else "NA"
        p_label = f"log-rank p={p_text}"
    if p_in_legend:
        # TCGA preference: p-value sits in the legend title; plot title stays clean.
        ax.set_title(title, fontsize=TITLE_FS - 1)
        legend = ax.legend(
            title=p_label,
            fontsize=COMPACT_LEGEND_FS,
            title_fontsize=COMPACT_LEGEND_FS,
            frameon=True,
            framealpha=LEGEND_FRAME_ALPHA,
            facecolor="white",
            handlelength=0.9,
            handletextpad=0.35,
            borderpad=0.25,
            labelspacing=0.25,
            borderaxespad=0.25,
        )
    else:
        # External convention: p-value lives in the title; legend stays simple.
        ax.set_title(f"{title}\n{p_label}", fontsize=TITLE_FS - 1)
        legend = ax.legend(
            fontsize=COMPACT_LEGEND_FS,
            frameon=True,
            framealpha=LEGEND_FRAME_ALPHA,
            facecolor="white",
            handlelength=0.9,
            handletextpad=0.35,
            borderpad=0.25,
            labelspacing=0.25,
            borderaxespad=0.25,
        )
    ax.set_xlabel(time_label, fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("Survival probability", fontsize=AXIS_LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    _style_legend_frame(legend)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    return {"p_value": p_value, "n_samples": int(len(df)), "n_groups": int(df["group"].nunique()), "kept_groups": list(keep)}


def _tcga_representation_items(bundles: dict[str, dict]) -> dict[str, dict[str, dict]]:
    items = {}
    for cancer, bundle in bundles.items():
        clinical = derive_os_columns(bundle["clinical"])
        reps = {
            "expression": {**bundle["expression_umap"], "clinical_rep": clinical},
            "embedding": {**bundle["embedding_umap"], "clinical_rep": clinical},
        }
        if "mutation_pca_umap" in bundle:
            reps["mutation_pca"] = {**bundle["mutation_pca_umap"], "clinical_rep": clinical}
        items[cancer] = reps
    return items


def compute_tcga_km_summary(
    items: dict[str, dict[str, dict]],
    *,
    km_min_group_size: int,
    max_time_by_cancer: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows = []
    max_time_by_cancer = max_time_by_cancer or {}
    for cancer, spaces in items.items():
        for space, rep in spaces.items():
            labels = rep["labels"].astype(str)
            clinical = rep["clinical_rep"].reindex(labels.index)
            try:
                df, keep = _prepare_survival_groups(clinical, labels, time_col="OS_time", event_col="OS_event", min_group_size=km_min_group_size)
                sorted_groups = sorted(df["group"].unique(), key=str)
                try:
                    from lifelines.statistics import multivariate_logrank_test
                    p_value = float(multivariate_logrank_test(df["time"], df["group"], df["event"]).p_value)
                except ImportError:
                    p_value = _logrank_pvalue_basic(df, sorted_groups)
                status = "computed"
            except Exception as exc:
                df, keep, p_value = pd.DataFrame(), [], np.nan
                status = f"skipped: {exc}"
            rows.append({
                "cohort": cancer,
                "space": space,
                "km_p_value": p_value,
                "km_n_samples": int(len(df)),
                "km_n_groups": int(df["group"].nunique()) if not df.empty else 0,
                "kept_groups": ";".join(map(str, keep)),
                "max_time": max_time_by_cancer.get(cancer),
                "km_status": status,
            })
    summary = pd.DataFrame(rows)
    return add_bh_fdr(summary, p_col="km_p_value", fdr_col="km_p_fdr", group_cols="space")


def plot_tcga_representations_direct(
    items: dict[str, dict[str, dict]],
    km_summary: pd.DataFrame,
    *,
    output_dir: str | Path,
    km_min_group_size: int,
    umap_figsize: tuple,
    km_figsize: tuple,
    show_umap_legend: bool,
    max_time_by_cancer: dict[str, float] | None = None,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    umap_dir = output_dir / "umaps"
    km_dir = output_dir / "km_plots"
    cross_dir = output_dir / "cross_cluster_umaps"
    source_dir = output_dir / "source_data"
    for path in (umap_dir, km_dir, cross_dir, source_dir):
        path.mkdir(parents=True, exist_ok=True)
    space_labels = {"expression": "Expression", "embedding": "Embedding", "mutation_pca": "Mutation PCA"}
    short_space_labels = {"expression": "expr", "embedding": "emb", "mutation_pca": "mut"}
    group_names = {"expression": "expr Leiden", "embedding": "emb Leiden", "mutation_pca": "mut Leiden"}
    cross_pairs = (("embedding", "expression"), ("expression", "embedding"), ("mutation_pca", "embedding"), ("embedding", "mutation_pca"))
    max_time_by_cancer = max_time_by_cancer or {}
    p_lookup = {(r["cohort"], r["space"]): r for _, r in km_summary.iterrows()} if not km_summary.empty else {}
    normalized = {}
    rows = []
    for cancer, spaces in items.items():
        normalized[cancer] = {}
        for space, rep in spaces.items():
            coords = rep["coords"].copy()
            labels = rep["labels"].astype(str)
            title = f"{cancer} - {space_labels.get(space, space)}"
            prefix = f"{cancer.lower()}_{space}"
            _, _, palette = plot_umap_categorical(
                coords, labels, title=title, save_path=umap_dir / f"{prefix}_umap.pdf",
                show_legend=show_umap_legend, figsize=umap_figsize,
            )
            normalized[cancer][space] = {**rep, "palette": palette}
            source = coords.copy()
            source["cluster"] = labels
            source["sample"] = source.index.astype(str)
            source["cohort"] = cancer
            source["space"] = space
            source.to_csv(source_dir / f"{prefix}_umap_source.csv", index=False)
            km_row = p_lookup.get((cancer, space), {})
            p_fdr = pd.to_numeric(pd.Series([km_row.get("km_p_fdr")]), errors="coerce").iloc[0]
            try:
                stats = plot_km_curves(
                    rep["clinical_rep"], labels, time_col="OS_time", event_col="OS_event",
                    title=title, group_name=group_names.get(space, "Cluster"), palette=palette,
                    min_group_size=km_min_group_size, figsize=km_figsize,
                    max_time=max_time_by_cancer.get(cancer), time_label="Days",
                    p_adjusted=float(p_fdr) if np.isfinite(p_fdr) else None,
                    p_in_legend=True,
                    save_path=km_dir / f"{prefix}_km.pdf",
                )
                km_status = "saved"
            except Exception as exc:
                stats = {}
                km_status = f"skipped: {exc}"
            rows.append({
                "cohort": cancer, "space": space, "plot_type": "primary",
                "n_samples": int(len(coords)), "n_clusters": int(labels.nunique()),
                "umap_file": f"umaps/{prefix}_umap.pdf",
                "km_file": f"km_plots/{prefix}_km.pdf",
                "source_file": f"source_data/{prefix}_umap_source.csv",
                "umap_status": "saved", "km_status": km_status,
                "km_p_value": km_row.get("km_p_value", stats.get("p_value", pd.NA)),
                "km_p_fdr": km_row.get("km_p_fdr", pd.NA),
                "km_n_samples": km_row.get("km_n_samples", stats.get("n_samples", pd.NA)),
                "km_n_groups": km_row.get("km_n_groups", stats.get("n_groups", pd.NA)),
            })
    for cancer, spaces in normalized.items():
        for base_space, color_space in cross_pairs:
            if base_space not in spaces or color_space not in spaces:
                continue
            base = spaces[base_space]
            color = spaces[color_space]
            common = base["coords"].index.intersection(color["labels"].index)
            if common.empty:
                continue
            base_short = short_space_labels.get(base_space, slug(base_space))
            color_short = short_space_labels.get(color_space, slug(color_space))
            prefix = f"{cancer.lower()}_{base_short}_by_{color_short}"
            plot_umap_categorical(
                base["coords"].loc[common],
                color["labels"].loc[common],
                title=f"{cancer} - {base_short} by {color_short} clusters",
                palette=color["palette"],
                save_path=cross_dir / f"{prefix}_umap.pdf",
                show_legend=show_umap_legend,
                figsize=umap_figsize,
            )
            rows.append({
                "cohort": cancer, "space": f"{base_short}_by_{color_short}",
                "plot_type": "cross_umap", "n_samples": int(len(common)),
                "n_clusters": int(color["labels"].loc[common].nunique()),
                "umap_file": f"cross_cluster_umaps/{prefix}_umap.pdf",
                "km_file": pd.NA, "source_file": pd.NA,
                "umap_status": "saved", "km_status": pd.NA,
                "km_p_value": pd.NA, "km_p_fdr": pd.NA,
                "km_n_samples": pd.NA, "km_n_groups": pd.NA,
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "clinical_relevance_plot_summary.csv", index=False)
    return summary


def run_tcga_part(
    config: Figure4Config,
    *,
    cancers: Iterable[str] | None,
    params: dict[str, dict],
    plot: bool,
    include_mutation: bool,
    km_min_group_size: int,
    umap_figsize: tuple,
    km_figsize: tuple,
    show_umap_legend: bool,
    max_time_by_cancer: dict[str, float] | None,
) -> dict[str, pd.DataFrame | dict]:
    config.ensure_outputs()
    bundles = load_tcga_bundles(config, cancers=list(params) if cancers is None else cancers)
    bundles = recompute_tcga_expression_embedding_spaces(bundles, params=params)
    if include_mutation:
        bundles = add_tcga_mutation_spaces(config, bundles, params=params)
    bundle_summary = summarize_tcga_bundles(bundles)
    save_source_table(bundle_summary, config, "figure4_tcga_bundle_cluster_summary.csv")
    items = _tcga_representation_items(bundles)
    km_summary = compute_tcga_km_summary(items, km_min_group_size=km_min_group_size, max_time_by_cancer=max_time_by_cancer)
    save_source_table(km_summary, config, "figure4_tcga_km_summary_precomputed.csv")
    plot_summary = pd.DataFrame()
    if plot:
        plot_summary = plot_tcga_representations_direct(
            items, km_summary, output_dir=config.figure_dir / "tcga_embeddings",
            km_min_group_size=km_min_group_size, umap_figsize=umap_figsize,
            km_figsize=km_figsize, show_umap_legend=show_umap_legend,
            max_time_by_cancer=max_time_by_cancer,
        )
        save_source_table(plot_summary, config, "figure4_tcga_km_summary.csv")
    return {"bundles": bundles, "bundle_summary": bundle_summary, "km_summary": km_summary, "plot_summary": plot_summary}


EXTERNAL_COHORT_LABELS = {
    "CPTAC_CMI": "CPTAC/CMI", "GIDE": "Gide", "HUGO": "Hugo", "IMMUNOPOG": "IMMUNOPOG",
    "LIU": "Liu", "METABRIC": "METABRIC", "MORRISON": "Morrison", "RIAZ": "Riaz", "VAN_ALLEN": "Van Allen",
}


def external_cohort_methods(config: Figure4Config) -> dict[str, str]:
    """Each cohort's main integration method: the first of its methods in the External run's config."""
    cohorts = yaml.safe_load(config.external_config_path.read_text(encoding="utf-8"))["cohorts"]
    return {cohort: settings["methods"][0] for cohort, settings in cohorts.items()}


def available_external_cohorts(config: Figure4Config) -> list[str]:
    return sorted(c for c in external_cohort_methods(config) if (external_bundle_path(config, c) / "manifest.json").exists())


def external_bundle_path(config: Figure4Config, cohort: str) -> Path:
    return config.external_results_root / cohort / external_cohort_methods(config)[cohort]


def load_external_bundle(result_dir: str | Path) -> dict:
    result_dir = Path(result_dir)
    with open(result_dir / "manifest.json", "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    probabilities = _read_indexed_frame(result_dir / "probabilities.csv.gz")
    site_metrics_path = result_dir / "site_metrics.csv"
    return {
        "result_dir": result_dir,
        "manifest": manifest,
        "clinical": _read_indexed_frame(result_dir / "clinical.csv.gz"),
        "embeddings_external": _read_indexed_frame(result_dir / "embeddings_external.csv.gz"),
        "embeddings_tcga": _read_indexed_frame(result_dir / "embeddings_tcga.csv.gz"),
        "predictions": (probabilities >= 0.5).astype(int),
        "probabilities": probabilities,
        "all_metrics": pd.read_csv(result_dir / "metrics.csv").rename(columns={"target": "gene"}),
        "site_metrics": pd.read_csv(site_metrics_path).rename(columns={"target": "gene"}) if site_metrics_path.exists() else None,
    }


def load_external_bundles(config: Figure4Config, cohorts: Iterable[str] | None = None) -> dict[str, dict]:
    config.ensure_inputs()
    selected = available_external_cohorts(config) if cohorts is None else [c.upper() for c in cohorts]
    return {cohort: load_external_bundle(external_bundle_path(config, cohort)) for cohort in selected}


def summarize_external_bundles(bundles: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for cohort, bundle in sorted(bundles.items()):
        manifest = bundle["manifest"]
        metrics_summary = manifest.get("metrics") or {}
        rows.append({
            "cohort": cohort,
            "label": EXTERNAL_COHORT_LABELS.get(cohort, cohort),
            "method": manifest.get("method"),
            "n_external_samples": int(bundle["embeddings_external"].shape[0]),
            "n_embedding_features": int(bundle["embeddings_external"].shape[1]),
            "n_profiled_samples": manifest["external"]["profiled_samples"],
            "n_metrics_rows": int(0 if bundle["all_metrics"] is None else len(bundle["all_metrics"])),
            "n_evaluable": metrics_summary.get("n_evaluable"),
            "median_normalized_auprc": metrics_summary.get("median_normalized_auprc"),
            "n_train_samples": manifest["tcga"]["samples"],
            "n_train_targets": manifest["targets"]["n"],
        })
    return pd.DataFrame(rows)


def top_external_metrics(bundles: dict[str, dict], *, top_n: int, min_positive: int, score_col: str) -> pd.DataFrame:
    rows = []
    for cohort, bundle in sorted(bundles.items()):
        metrics = bundle["all_metrics"]
        if metrics is None or score_col not in metrics.columns:
            continue
        sub = metrics.copy()
        if "evaluable" in sub.columns:
            sub = sub[sub["evaluable"].astype(bool)]
        if "n_positive" in sub.columns:
            sub = sub[pd.to_numeric(sub["n_positive"], errors="coerce") >= min_positive]
        for rank, (_, row) in enumerate(sub.sort_values(score_col, ascending=False).head(top_n).iterrows(), start=1):
            rows.append({
                "cohort": cohort, "rank": rank, "gene": row.get("gene"),
                "n_samples": row.get("n_samples"), "n_positive": row.get("n_positive"),
                "prevalence": row.get("prevalence"), "auprc": row.get("auprc"),
                "normalized_auprc": row.get("normalized_auprc"), "roc_auc": row.get("roc_auc"),
            })
    return pd.DataFrame(rows)


def top_cptac_site_metrics(bundle: dict, *, top_n: int, min_site_samples: int, min_positive: int, score_col: str) -> pd.DataFrame:
    metrics = bundle.get("site_metrics")
    if metrics is None or score_col not in metrics.columns:
        return pd.DataFrame()
    sub = metrics.copy()
    if "evaluable" in sub.columns:
        sub = sub[sub["evaluable"].astype(bool)]
    sub = sub[pd.to_numeric(sub["site_n_samples"], errors="coerce") >= min_site_samples]
    sub = sub[pd.to_numeric(sub["n_positive"], errors="coerce") >= min_positive]
    rows = []
    for site, site_df in sub.groupby("site", sort=True):
        for rank, (_, row) in enumerate(site_df.sort_values(score_col, ascending=False).head(top_n).iterrows(), start=1):
            rows.append({
                "site": site, "rank": rank, "gene": row.get("gene"),
                "site_n_samples": row.get("site_n_samples"), "n_positive": row.get("n_positive"),
                "prevalence": row.get("prevalence"), "auprc": row.get("auprc"),
                "normalized_auprc": row.get("normalized_auprc"), "roc_auc": row.get("roc_auc"),
            })
    return pd.DataFrame(rows)


def summarize_external_prediction_part(
    config: Figure4Config,
    *,
    top_n: int,
    min_positive: int,
    cptac_min_site_samples: int,
    cptac_min_positive: int,
    score_col: str = "normalized_auprc",
) -> dict[str, pd.DataFrame]:
    bundles = load_external_bundles(config)
    overview = summarize_external_bundles(bundles)
    top_metrics = top_external_metrics(bundles, top_n=top_n, min_positive=min_positive, score_col=score_col)
    cptac_site_top = top_cptac_site_metrics(
        bundles["CPTAC_CMI"], top_n=top_n, min_site_samples=cptac_min_site_samples,
        min_positive=cptac_min_positive, score_col=score_col,
    ) if "CPTAC_CMI" in bundles else pd.DataFrame()
    save_source_table(overview, config, "figure4_external_bundle_overview.csv")
    save_source_table(top_metrics, config, "figure4_external_top_gene_metrics.csv")
    if not cptac_site_top.empty:
        save_source_table(cptac_site_top, config, "figure4_cptac_site_top_gene_metrics.csv")
    return {"bundles": bundles, "overview": overview, "top_metrics": top_metrics, "cptac_site_top": cptac_site_top}


# ---------------------------------------------------------------------------
# External integration UMAPs (TCGA background vs external foreground)
# ---------------------------------------------------------------------------

def _cohort_prefix(cohort: str) -> str:
    """Figure-file prefix for a cohort (CPTAC_CMI files use the short 'cptac' stem)."""
    return "cptac" if cohort.upper() == "CPTAC_CMI" else cohort.lower()


def plot_integration_umap(
    coords: pd.DataFrame,
    *,
    title: str,
    save_path: str | Path | None = None,
    label_col: str = "dataset",
    background_label: str = "TCGA",
    background_color: str = "#D8D8D8",
    foreground_palette: dict | None = None,
    figsize: tuple = (3, 3),
    background_point_size: float = 6.0,
    foreground_point_size: float = 10.0,
    background_alpha: float = 0.45,
    foreground_alpha: float = 0.95,
    show_legend: bool = True,
    legend_title: str = "",
) -> dict:
    """Two-layer integration scatter: TCGA background first, external foreground on top."""
    required = {"UMAP1", "UMAP2", label_col}
    missing = required - set(coords.columns)
    if missing:
        raise KeyError(f"plot_integration_umap missing columns: {sorted(missing)}")

    bg = coords.loc[coords[label_col].astype(str) == str(background_label)]
    fg = coords.loc[coords[label_col].astype(str) != str(background_label)].copy()
    fg[label_col] = fg[label_col].astype(str)
    fg_labels = list(dict.fromkeys(fg[label_col].tolist()))

    if foreground_palette is None:
        foreground_palette = build_label_palette(fg_labels)
    else:
        foreground_palette = {str(k): v for k, v in foreground_palette.items()}

    fig, ax = plt.subplots(figsize=figsize)
    if not bg.empty:
        ax.scatter(
            bg["UMAP1"], bg["UMAP2"], c=background_color, s=background_point_size,
            edgecolors="none", alpha=background_alpha, label=str(background_label),
        )
    for label in fg_labels:
        m = fg[label_col] == label
        ax.scatter(
            fg.loc[m, "UMAP1"], fg.loc[m, "UMAP2"], c=foreground_palette.get(label, "#000000"),
            s=foreground_point_size, edgecolors="none", alpha=foreground_alpha, label=label,
        )
    ax.set_title(title, fontsize=TITLE_FS - 1)
    ax.set_xlabel("UMAP1", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("UMAP2", fontsize=AXIS_LABEL_FS)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)
    if show_legend:
        legend = ax.legend(
            title=legend_title,
            fontsize=COMPACT_LEGEND_FS,
            title_fontsize=COMPACT_LEGEND_FS,
            frameon=True,
            framealpha=LEGEND_FRAME_ALPHA,
            facecolor="white",
            loc="best",
            markerscale=0.65,
            handletextpad=0.3,
            borderpad=0.25,
            labelspacing=0.25,
            borderaxespad=0.25,
        )
        _style_legend_frame(legend)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    plt.show()
    return {"n_background": int(len(bg)), "n_foreground": int(len(fg)), "foreground_palette": foreground_palette}


def plot_external_integration(
    config: Figure4Config,
    cohort: str,
    *,
    save: bool = True,
    show_legend: bool = True,
) -> dict:
    """Render before/after-normalization integration UMAPs for one external cohort.

    Reads the precomputed coordinates saved in the bundle
    (`umap_{before,after}.csv`) -- no UMAP is recomputed here.
    CPTAC/CMI foreground is colored by sequencing batch; other cohorts by dataset.
    """
    cohort = cohort.upper()
    bundle_dir = external_bundle_path(config, cohort)
    out_dir = config.figure_dir / cohort.lower()
    prefix = _cohort_prefix(cohort)
    label = EXTERNAL_COHORT_LABELS.get(cohort, cohort)
    color_col = "batch" if cohort == "CPTAC_CMI" else "dataset"

    results: dict[str, dict | None] = {}
    for stage in ("before", "after"):
        path = bundle_dir / f"umap_{stage}.csv"
        if not path.exists():
            results[stage] = None
            continue
        coords = _read_indexed_frame(path)
        col = color_col if color_col in coords.columns else "dataset"
        legend_title = "Batch" if col == "batch" else ""
        results[stage] = plot_integration_umap(
            coords, title=f"{label} - {stage} normalization", label_col=col,
            legend_title=legend_title, show_legend=show_legend,
            save_path=(out_dir / f"{prefix}_integration_{stage}.pdf") if save else None,
        )
    return results


def plot_all_external_integration(
    config: Figure4Config,
    cohorts: Iterable[str] | None = None,
    *,
    save: bool = True,
    show_legend: bool = True,
) -> dict[str, dict]:
    """Render integration UMAPs for every available external cohort (or a subset)."""
    config.ensure_inputs()
    selected = available_external_cohorts(config) if cohorts is None else [c.upper() for c in cohorts]
    out = {}
    for cohort in selected:
        if not (external_bundle_path(config, cohort) / "umap_before.csv").exists():
            continue
        out[cohort] = plot_external_integration(config, cohort, save=save, show_legend=show_legend)
    return out


# ---------------------------------------------------------------------------
# External AUPRC tile-line plots (cohort-level + CPTAC site-stratified)
# ---------------------------------------------------------------------------

def _draw_auprc_tile_row(ax, ranked, *, gene_col, score_col, cmap, vmin, vmax, title):
    values = ranked[score_col].astype(float).to_numpy()[None, :]
    n_tiles = values.shape[1]
    ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xlim(-0.5, n_tiles - 0.5)
    ax.set_ylim(0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    for x in np.arange(-0.5, n_tiles + 0.5, 1.0):
        ax.axvline(x, color="white", linewidth=0.55)
    ax.axhline(-0.5, color="white", linewidth=0.55)
    ax.axhline(0.5, color="white", linewidth=0.55)
    denom = float(vmax - vmin) if vmax > vmin else 1.0
    for j, (_, row) in enumerate(ranked.iterrows()):
        scaled = float(np.clip((float(row[score_col]) - vmin) / denom, 0.0, 1.0))
        r, g, b, _ = cmap(scaled)
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        text_color = "black" if luminance > 0.48 else "white"
        ax.text(j, 0, str(row[gene_col]), ha="center", va="center", fontsize=ANNOTATION_FS - 3.0, color=text_color)
    ax.set_title(title, fontsize=TICK_FS, pad=2, loc="center")
    high_value = float(ranked.iloc[0][score_col])
    low_value = float(ranked.iloc[-1][score_col])
    ax.text(-0.65, 0, f"{high_value:.2f}", ha="right", va="center", fontsize=TICK_FS, clip_on=False)
    ax.text(n_tiles - 0.5 + 0.15, 0, f"{low_value:.2f}", ha="left", va="center", fontsize=TICK_FS, clip_on=False)


def plot_auprc_tile_line(
    metrics: pd.DataFrame,
    *,
    title: str,
    save_path: str | Path | None = None,
    top_n: int = 10,
    gene_col: str = "gene",
    score_col: str = "normalized_auprc",
    min_positive: int | None = None,
    cmap=CMAP_TRUNCATED_MAGMA,
    vmin: float = 0.0,
    vmax: float = 1.0,
    figsize: tuple | None = None,
) -> pd.DataFrame:
    """Single-row tile plot of the top-N genes by `score_col` for one cohort/site."""
    ranked = metrics.loc[metrics[score_col].notna()].copy()
    if "evaluable" in ranked.columns:
        ranked = ranked.loc[ranked["evaluable"].astype(bool)]
    if min_positive is not None and "n_positive" in ranked.columns:
        ranked = ranked.loc[ranked["n_positive"].astype(int) >= int(min_positive)]
    ranked = ranked.sort_values(score_col, ascending=False).head(int(top_n)).copy()
    if ranked.empty:
        raise ValueError(f"No evaluable {score_col} values for {title}.")
    n_tiles = len(ranked)
    if figsize is None:
        figsize = (0.285 * n_tiles + 0.20, 0.36)
    cmap_obj = cmap.copy() if hasattr(cmap, "copy") else plt.get_cmap(cmap).copy()
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.set_position([0.0, 0.26, 1.0, 0.38])
    _draw_auprc_tile_row(ax, ranked, gene_col=gene_col, score_col=score_col, cmap=cmap_obj, vmin=vmin, vmax=vmax, title=title)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.01)
    plt.show()
    return ranked


def plot_auprc_tile_legend(
    *,
    save_path: str | Path | None = None,
    label: str = "Normalized AUPRC",
    cmap=CMAP_TRUNCATED_MAGMA,
    vmin: float = 0.0,
    vmax: float = 1.0,
    figsize: tuple = (0.34, 1.55),
) -> Path | None:
    """Standalone 0-to-1 color legend matching the AUPRC tile plots."""
    cmap_obj = cmap.copy() if hasattr(cmap, "copy") else plt.get_cmap(cmap).copy()
    norm = plt.Normalize(vmin=float(vmin), vmax=float(vmax))
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj), cax=ax,
        orientation="vertical", ticks=[float(vmin), float(vmax)],
    )
    cbar.set_label(label, fontsize=COLORBAR_LABEL_FS, labelpad=3)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FS, length=2, pad=1)
    cbar.outline.set_linewidth(0.5)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.01)
    plt.show()
    return save_path


def plot_site_auprc_tile_lines(
    site_metrics: pd.DataFrame,
    *,
    save_path: str | Path | None = None,
    sites: list[str] | None = None,
    site_label_map: dict[str, str] | None = None,
    top_n: int = 10,
    min_site_samples: int = 20,
    min_positive: int | None = None,
    gene_col: str = "gene",
    score_col: str = "normalized_auprc",
    cmap=CMAP_TRUNCATED_MAGMA,
    vmin: float = 0.0,
    vmax: float = 1.0,
    row_height: float = 0.55,
) -> dict:
    """Stacked tile rows, one per primary site, for site-stratified external metrics."""
    df = site_metrics.copy()
    if "evaluable" in df.columns:
        df = df.loc[df["evaluable"].astype(bool)]
    df = df.loc[df[score_col].notna()]
    df = df.loc[df["site_n_samples"].astype(int) >= int(min_site_samples)]
    if min_positive is not None and "n_positive" in df.columns:
        df = df.loc[df["n_positive"].astype(int) >= int(min_positive)]

    if sites is None:
        site_sizes = df.groupby("site")["site_n_samples"].first().sort_values(ascending=False)
        sites = list(site_sizes.index)
    label_map = site_label_map or {}

    ranked_per_site: dict[str, pd.DataFrame] = {}
    for site in sites:
        sub = df.loc[df["site"] == site].sort_values(score_col, ascending=False).head(int(top_n))
        if not sub.empty:
            ranked_per_site[site] = sub
    if not ranked_per_site:
        raise ValueError("No sites met the threshold for AUPRC tile lines.")

    n_rows = len(ranked_per_site)
    n_tiles_max = max(len(r) for r in ranked_per_site.values())
    width = 0.285 * n_tiles_max + 1.2
    fig, axes = plt.subplots(n_rows, 1, figsize=(width, row_height * n_rows), dpi=300)
    if n_rows == 1:
        axes = [axes]
    cmap_obj = cmap.copy() if hasattr(cmap, "copy") else plt.get_cmap(cmap).copy()
    for ax, (site, ranked) in zip(axes, ranked_per_site.items()):
        n_samples = int(ranked["site_n_samples"].iloc[0])
        display = label_map.get(site, site)
        _draw_auprc_tile_row(ax, ranked, gene_col=gene_col, score_col=score_col, cmap=cmap_obj,
                             vmin=vmin, vmax=vmax, title=f"{display} (n = {n_samples})")
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.02)
    plt.show()
    return ranked_per_site


def plot_external_auprc_tiles(
    part: dict,
    config: Figure4Config,
    *,
    top_n: int = 10,
    min_positive: int = 5,
    score_col: str = "normalized_auprc",
    cohorts: Iterable[str] | None = None,
    plot_legend: bool = True,
) -> pd.DataFrame:
    """Cohort-level AUPRC tile plots for every external bundle with mutation metrics.

    Accepts the dict returned by `summarize_external_prediction_part` (uses its
    already-loaded `bundles`). Saves one tile PDF per cohort under
    `Figures/Figure4/output/<cohort>/` plus a shared 0-1 legend.
    """
    bundles = part["bundles"] if "bundles" in part else part
    selected = sorted(bundles) if cohorts is None else [c.upper() for c in cohorts]
    frames = []
    for cohort in selected:
        bundle = bundles.get(cohort)
        if bundle is None or bundle.get("all_metrics") is None:
            continue
        metrics = bundle["all_metrics"]
        if score_col not in metrics.columns:
            continue
        prefix = _cohort_prefix(cohort)
        out_dir = config.figure_dir / cohort.lower()
        try:
            ranked = plot_auprc_tile_line(
                metrics, title=EXTERNAL_COHORT_LABELS.get(cohort, cohort),
                save_path=out_dir / f"{prefix}_top{top_n}_{score_col}_gene_tiles.pdf",
                top_n=top_n, min_positive=min_positive, score_col=score_col,
            )
        except ValueError:
            continue
        frames.append(ranked.assign(cohort=cohort))
    if plot_legend:
        plot_auprc_tile_legend(save_path=config.figure_dir / "auprc_tile_legend_0_1.pdf")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_cptac_site_auprc_tiles(
    part: dict,
    config: Figure4Config,
    *,
    top_n: int = 10,
    min_site_samples: int = 40,
    min_positive: int = 10,
    score_col: str = "normalized_auprc",
) -> dict:
    """Stacked site-stratified AUPRC tile plot for CPTAC/CMI (all sites in one figure)."""
    bundles = part["bundles"] if "bundles" in part else part
    bundle = bundles.get("CPTAC_CMI")
    if bundle is None or bundle.get("site_metrics") is None:
        return {}
    out_dir = config.figure_dir / "cptac_cmi"
    return plot_site_auprc_tile_lines(
        bundle["site_metrics"],
        save_path=out_dir / f"cptac_site_top{top_n}_{score_col}_gene_tiles.pdf",
        top_n=top_n, min_site_samples=min_site_samples, min_positive=min_positive,
        score_col=score_col, site_label_map=CPTAC_SITE_LABEL_MAP,
    )


# Display-label overrides for long CPTAC/CMI primary-site names; file slugs derive from these.
CPTAC_SITE_LABEL_MAP = {
    "Bronchus and lung": "Lung",
    "Uterus, NOS": "Uterus",
    "Prostate gland": "Prostate",
    "Hematopoietic and reticuloendothelial systems": "Hematopoietic",
    "Other and unspecified female genital organs": "Female genital",
}


def _cptac_site_label(site: str) -> str:
    return CPTAC_SITE_LABEL_MAP.get(site, site)


def _cptac_site_slug(site: str) -> str:
    return _cptac_site_label(site).lower().replace(" ", "_").replace(",", "")


def plot_cptac_per_site_auprc_tiles(
    part: dict,
    config: Figure4Config,
    *,
    top_n: int = 10,
    min_site_samples: int = 40,
    min_positive: int = 10,
    score_col: str = "normalized_auprc",
) -> dict:
    """One AUPRC tile PDF per CPTAC/CMI primary site.

    Writes one tile file per CPTAC/CMI primary site as
    `cptac_<slug>_top{N}_<score>_gene_tiles.pdf`
    under `Figures/Figure4/output/cptac_cmi/`. Display labels use `CPTAC_SITE_LABEL_MAP`;
    file slugs are the lower-cased, space/comma-stripped display label.
    """
    bundles = part["bundles"] if "bundles" in part else part
    bundle = bundles.get("CPTAC_CMI")
    if bundle is None or bundle.get("site_metrics") is None:
        return {}
    site_metrics = bundle["site_metrics"]
    out_dir = config.figure_dir / "cptac_cmi"
    tile_sites = (
        site_metrics.groupby("site")["site_n_samples"].first()
        .loc[lambda s: s >= int(min_site_samples)]
        .sort_values(ascending=False)
        .index.tolist()
    )
    results: dict[str, pd.DataFrame] = {}
    for site in tile_sites:
        display_name = _cptac_site_label(site)
        slug = _cptac_site_slug(site)
        site_df = site_metrics[site_metrics["site"] == site]
        n = int(site_df["site_n_samples"].iloc[0])
        try:
            results[site] = plot_auprc_tile_line(
                site_df, title=f"{display_name} (n = {n})",
                save_path=out_dir / f"cptac_{slug}_top{top_n}_{score_col}_gene_tiles.pdf",
                top_n=top_n, min_positive=min_positive, score_col=score_col,
            )
        except ValueError:
            continue
    return results


CPTAC_SITE_COL = "cases.primary_site"


def run_cptac_per_site_clinical(
    config: Figure4Config,
    bundle: dict | None = None,
    *,
    site_params_default: dict | None = None,
    site_params: dict | None = None,
    site_min_samples: int = 120,
    km_min_group_size: int = 10,
    min_os_samples: int = 20,
    expr_pca_components: int = 256,
    random_state: int = 0,
    plot: bool = True,
) -> dict:
    """Per-primary-site embedding/expression UMAP+Leiden and OS Kaplan-Meier for CPTAC/CMI.

    For every site with at least `site_min_samples` samples this recomputes embedding and
    expression-PCA UMAP+Leiden, saves cluster and cross-cluster UMAPs, and runs an OS log-rank
    test per representation (skipped when fewer than `min_os_samples` OS-usable samples).
    OS p-values are BH-FDR corrected within representation space and the KM panels are redrawn
    with the adjusted value. PDFs -> Figures/Figure4/output/cptac_cmi/cptac_<slug>_*.pdf.
    """
    if bundle is None:
        bundle = load_external_bundle(external_bundle_path(config, "CPTAC_CMI"))
    clinical = bundle["clinical"]
    emb = bundle["embeddings_external"]
    expression = load_external_expression(bundle)
    out_dir = config.figure_dir / "cptac_cmi"
    out_dir.mkdir(parents=True, exist_ok=True)

    site_params_default = site_params_default or {
        "emb_nn": 8, "emb_min_dist": 0.3, "emb_leiden_res": 0.35,
        "expr_nn": 8, "expr_min_dist": 0.3, "expr_leiden_res": 0.35,
    }
    site_params = site_params or {}

    site_counts = clinical[CPTAC_SITE_COL].value_counts()
    selected_sites = site_counts[site_counts >= int(site_min_samples)].index.tolist()

    os_t = pd.to_numeric(clinical["OS_time"], errors="coerce")
    os_e = pd.to_numeric(clinical["OS_event"], errors="coerce")
    os_usable = os_t.notna() & os_e.notna() & (os_t > 0)

    km_rows: list[dict] = []
    km_jobs: list[dict] = []
    for site in selected_sites:
        display_name = _cptac_site_label(site)
        slug = _cptac_site_slug(site)
        p = {**site_params_default, **site_params.get(site, {})}
        site_idx = clinical.index[clinical[CPTAC_SITE_COL] == site].intersection(emb.index)
        n = len(site_idx)
        if n < 3:
            continue
        site_clinical = clinical.loc[site_idx]
        n_os = int(os_usable.loc[site_idx].sum())

        emb_proj = run_umap_cluster(
            emb.loc[site_idx], n_neighbors=p["emb_nn"], min_dist=p["emb_min_dist"],
            leiden_resolution=p["emb_leiden_res"], random_state=random_state,
        )
        site_pcs = expression_pca_components(
            expression.loc[site_idx], n_components=min(int(expr_pca_components), n - 1), random_state=random_state,
        )
        expr_proj = run_umap_cluster(
            site_pcs, n_neighbors=p["expr_nn"], min_dist=p["expr_min_dist"],
            leiden_resolution=p["expr_leiden_res"], random_state=random_state, scale=False,
        )

        if plot:
            plot_umap_categorical(emb_proj["coords"], emb_proj["labels"], title=f"{display_name} - Embedding (n = {n})", save_path=out_dir / f"cptac_{slug}_embedding_umap_clusters.pdf")
            plot_umap_categorical(expr_proj["coords"], expr_proj["labels"], title=f"{display_name} - Expression (n = {n})", save_path=out_dir / f"cptac_{slug}_expression_umap_clusters.pdf")
            plot_umap_categorical(emb_proj["coords"], expr_proj["labels"].loc[emb_proj["coords"].index], title=f"{display_name} - emb by expr clusters", save_path=out_dir / f"cptac_{slug}_emb_by_expr_clusters.pdf")
            plot_umap_categorical(expr_proj["coords"], emb_proj["labels"].loc[expr_proj["coords"].index], title=f"{display_name} - expr by emb clusters", save_path=out_dir / f"cptac_{slug}_expr_by_emb_clusters.pdf")

        for space, proj in (("embedding", emb_proj), ("expression", expr_proj)):
            save_path = out_dir / f"cptac_{slug}_{space}_os_km.pdf"
            if n_os >= int(min_os_samples):
                try:
                    stats = plot_km_curves(
                        site_clinical, proj["labels"], time_col="OS_time", event_col="OS_event",
                        title=f"CPTAC {display_name} OS - {space}", time_label="Days",
                        min_group_size=km_min_group_size,
                        save_path=save_path if plot else None,
                    )
                    km_jobs.append({"site": display_name, "space": space, "clinical": site_clinical, "labels": proj["labels"], "save_path": save_path})
                except Exception as exc:
                    stats = {"p_value": np.nan, "n_samples": n_os, "n_groups": 0, "status": f"skipped: {exc}"}
            else:
                stats = {"p_value": np.nan, "n_samples": n_os, "n_groups": 0, "status": f"skipped: only {n_os} OS-usable"}
            km_rows.append({"site": display_name, "space": space, "n_site": n, "n_os": n_os,
                            "p_value": stats.get("p_value"), "n_groups": stats.get("n_groups"), "status": stats.get("status", "saved")})

    km_summary = add_bh_fdr(pd.DataFrame(km_rows), p_col="p_value", fdr_col="p_fdr", group_cols="space") if km_rows else pd.DataFrame()

    # Redraw saved KM panels with the space-specific FDR value in the title.
    if plot and not km_summary.empty:
        for job in km_jobs:
            match = (km_summary["site"] == job["site"]) & (km_summary["space"] == job["space"])
            p_fdr = pd.to_numeric(km_summary.loc[match, "p_fdr"], errors="coerce")
            if p_fdr.empty or not np.isfinite(p_fdr.iloc[0]):
                continue
            try:
                plot_km_curves(
                    job["clinical"], job["labels"], time_col="OS_time", event_col="OS_event",
                    title=f"CPTAC {job['site']} OS - {job['space']}", time_label="Days",
                    min_group_size=km_min_group_size, p_adjusted=float(p_fdr.iloc[0]), adjusted_label="p(FDR)",
                    save_path=job["save_path"],
                )
            except Exception:
                pass

    if not km_summary.empty:
        save_source_table(km_summary, config, "figure4_cptac_site_os_km_summary.csv")
        km_summary.to_csv(out_dir / "cptac_site_os_km_summary.csv", index=False)
    return {"km_summary": km_summary, "selected_sites": [_cptac_site_label(s) for s in selected_sites]}


def load_external_expression(bundle: dict) -> pd.DataFrame:
    return _read_indexed_frame(bundle["result_dir"] / "expression_external.csv.gz")


def expression_pca_components(expression: pd.DataFrame, *, n_components: int, random_state: int) -> pd.DataFrame:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    values = StandardScaler().fit_transform(expression.to_numpy(dtype=float))
    n = min(int(n_components), values.shape[0], values.shape[1])
    pca = PCA(n_components=n, random_state=int(random_state))
    return pd.DataFrame(pca.fit_transform(values), index=expression.index, columns=[f"PC{i+1}" for i in range(n)])


def compute_external_spaces(bundle: dict, *, params: dict, random_state: int) -> dict[str, pd.DataFrame | pd.Series]:
    emb = bundle["embeddings_external"]
    expression = load_external_expression(bundle).loc[emb.index]
    emb_params = params["embedding"]
    expr_params = params["expression"]
    emb_proj = run_umap_cluster(emb, random_state=random_state, **emb_params)
    expr_pcs = expression_pca_components(expression, n_components=emb.shape[1], random_state=random_state)
    expr_proj = run_umap_cluster(expr_pcs, random_state=random_state, scale=False, **expr_params)
    return {
        "embedding_coords": emb_proj["coords"],
        "embedding_clusters": emb_proj["labels"],
        "expression_coords": expr_proj["coords"],
        "expression_clusters": expr_proj["labels"],
        "expression_pcs": expr_pcs,
    }


def chisq_test(cluster_labels: pd.Series, values: pd.Series) -> dict:
    from scipy.stats import chi2_contingency
    sub = pd.DataFrame({"c": cluster_labels, "v": values}).dropna()
    sub["c"] = sub["c"].astype(str)
    sub["v"] = sub["v"].astype(str)
    table = pd.crosstab(sub["c"], sub["v"])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"chi2": np.nan, "p_value": np.nan, "dof": np.nan, "table": table}
    chi2, p, dof, _ = chi2_contingency(table.to_numpy())
    return {"chi2": float(chi2), "p_value": float(p), "dof": int(dof), "table": table}


def kruskal_test(cluster_labels: pd.Series, values: pd.Series) -> dict:
    from scipy.stats import kruskal
    sub = pd.DataFrame({"c": cluster_labels.astype(str), "v": pd.to_numeric(values, errors="coerce")}).dropna()
    groups = [g["v"].to_numpy() for _, g in sub.groupby("c") if len(g) > 0]
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        return {"H": np.nan, "p_value": np.nan, "n_groups": len(groups)}
    H, p = kruskal(*groups)
    return {"H": float(H), "p_value": float(p), "n_groups": len(groups)}


def plot_stacked_proportions(
    df: pd.DataFrame,
    *,
    cluster_col: str,
    value_col: str,
    title: str,
    save_path: str | Path | None,
    value_order: list[str] | None = None,
    palette: dict | None = None,
    legend_title: str = "",
    p_value: float | None = None,
    figsize: tuple = (2.5, 2.3),
    legend_outside: bool = False,
) -> pd.DataFrame:
    sub = df[[cluster_col, value_col]].dropna().copy()
    sub[cluster_col] = sub[cluster_col].astype(str)
    sub[value_col] = sub[value_col].astype(str)
    prop = pd.crosstab(sub[cluster_col], sub[value_col], normalize="index", dropna=False)
    cluster_order = sorted(prop.index.tolist(), key=lambda x: (len(str(x)), str(x)))
    value_order = value_order or list(prop.columns)
    prop = prop.reindex(index=cluster_order, columns=value_order).fillna(0.0)
    palette = {str(k): v for k, v in palette.items()} if palette else build_label_palette(value_order)
    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(prop))
    x = np.arange(len(prop.index))
    for col in prop.columns:
        vals = prop[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, width=0.82, color=palette.get(str(col), "#BDBDBD"), edgecolor="none", label=str(col))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(prop.index)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Cluster", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("Proportion", fontsize=AXIS_LABEL_FS)
    title_text = title + (f"\nchi-square p={p_value:.2e}" if p_value is not None and np.isfinite(p_value) else "")
    ax.set_title(title_text, fontsize=TITLE_FS - 1)
    legend_kwargs = {
        "title": legend_title,
        "fontsize": COMPACT_LEGEND_FS,
        "title_fontsize": COMPACT_LEGEND_FS,
        "frameon": True,
        "framealpha": LEGEND_FRAME_ALPHA,
        "handlelength": 0.9,
        "handletextpad": 0.35,
        "borderpad": 0.25,
        "labelspacing": 0.25,
        "borderaxespad": 0.25,
    }
    if legend_outside:
        legend_kwargs.update({"loc": "center left", "bbox_to_anchor": (1.01, 0.5)})
    legend = ax.legend(**legend_kwargs)
    _style_legend_frame(legend)
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return prop.reset_index().rename(columns={cluster_col: "cluster"})


def plot_box_by_cluster(
    df: pd.DataFrame,
    *,
    cluster_col: str,
    value_col: str,
    ylabel: str,
    title: str,
    save_path: str | Path | None,
    p_value: float | None = None,
    log_y: bool = False,
    figsize: tuple = (2.5, 2.3),
    random_seed: int = 0,
) -> pd.DataFrame:
    sub = df[[cluster_col, value_col]].copy()
    sub[cluster_col] = sub[cluster_col].astype(str)
    sub[value_col] = pd.to_numeric(sub[value_col], errors="coerce")
    sub = sub.dropna(subset=[value_col])
    if log_y:
        sub = sub[sub[value_col] > 0]
    order = sorted(sub[cluster_col].unique(), key=lambda x: (len(str(x)), str(x)))
    grouped = [sub.loc[sub[cluster_col] == c, value_col].to_numpy(dtype=float) for c in order]
    fig, ax = plt.subplots(figsize=figsize)
    box = ax.boxplot(grouped, positions=np.arange(len(order)), widths=0.58, patch_artist=True, showfliers=False)
    for i, patch in enumerate(box["boxes"]):
        patch.set_facecolor(POSTER_COLORS[i % len(POSTER_COLORS)])
        patch.set_alpha(0.85)
    rng = np.random.default_rng(random_seed)
    for i, values in enumerate(grouped):
        ax.scatter(np.full(len(values), i) + rng.uniform(-0.18, 0.18, size=len(values)), values, color="black", alpha=0.28, s=4, edgecolors="none")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlabel("Cluster", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FS)
    ax.set_title(title + (f"\nKW p={p_value:.2e}" if p_value is not None and np.isfinite(p_value) else ""), fontsize=TITLE_FS - 1)
    if log_y:
        ax.set_yscale("log")
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
    return sub


def _cluster_summary(cohort: str, spaces: dict[str, pd.DataFrame | pd.Series]) -> pd.DataFrame:
    rows = []
    for space, key in (("embedding", "embedding_clusters"), ("expression", "expression_clusters")):
        clusters = spaces[key].astype(str)
        counts = clusters.value_counts().sort_index()
        rows.append({
            "cohort": cohort, "space": space, "n_samples": int(clusters.shape[0]),
            "n_clusters": int(counts.shape[0]),
            "cluster_sizes": "; ".join(f"{k}:{int(v)}" for k, v in counts.items()),
        })
    return pd.DataFrame(rows)


def save_external_space_source(config: Figure4Config, cohort: str, spaces: dict[str, pd.DataFrame | pd.Series]) -> None:
    for space in ("embedding", "expression"):
        coords = spaces[f"{space}_coords"].copy()
        clusters = spaces[f"{space}_clusters"].astype(str).rename("cluster")
        out = coords.join(clusters)
        out.insert(0, "sample", out.index.astype(str))
        save_source_table(out.reset_index(drop=True), config, f"figure4_{cohort.lower()}_{space}_umap_clusters.csv")


def run_external_clinical_cohort(
    config: Figure4Config,
    cohort: str,
    *,
    params: dict,
    endpoints: Iterable[tuple[str, str, str]] | None = None,
    response_cols: Iterable[str] = ("Response", "Full_response"),
    duration_cols: Iterable[str] = (),
    plot: bool = True,
) -> dict[str, pd.DataFrame | dict]:
    config.ensure_outputs()
    cohort = cohort.upper()
    bundle = load_external_bundle(external_bundle_path(config, cohort))
    clinical = bundle["clinical"]
    spaces = compute_external_spaces(bundle, params=params, random_state=int(params.get("random_state", 0)))
    label = EXTERNAL_COHORT_LABELS.get(cohort, cohort)
    out_dir = config.figure_dir / cohort.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    save_external_space_source(config, cohort, spaces)
    cluster_summary = _cluster_summary(cohort, spaces)
    save_source_table(cluster_summary, config, f"figure4_{cohort.lower()}_cluster_summary.csv")
    if plot:
        plot_umap_categorical(spaces["embedding_coords"], spaces["embedding_clusters"], title=f"{label} - Embedding (n = {len(spaces['embedding_coords'])})", save_path=out_dir / f"{cohort.lower()}_embedding_umap_clusters.pdf")
        plot_umap_categorical(spaces["expression_coords"], spaces["expression_clusters"], title=f"{label} - Expression (n = {len(spaces['expression_coords'])})", save_path=out_dir / f"{cohort.lower()}_expression_umap_clusters.pdf")
        plot_umap_categorical(spaces["embedding_coords"], spaces["expression_clusters"].loc[spaces["embedding_coords"].index], title=f"{label} - emb by expr clusters", save_path=out_dir / f"{cohort.lower()}_emb_by_expr_clusters.pdf")
        plot_umap_categorical(spaces["expression_coords"], spaces["embedding_clusters"].loc[spaces["expression_coords"].index], title=f"{label} - expr by emb clusters", save_path=out_dir / f"{cohort.lower()}_expr_by_emb_clusters.pdf")
    if endpoints is None:
        endpoints = [(n, t, e) for n, t, e in (("OS", "OS_time", "OS_event"), ("PFS", "PFS_time", "PFS_event")) if t in clinical.columns and e in clinical.columns]
    km_rows = []
    for endpoint, time_col, event_col in endpoints:
        for space in ("embedding", "expression"):
            clusters = spaces[f"{space}_clusters"]
            try:
                stats = plot_km_curves(
                    clinical, clusters, time_col=time_col, event_col=event_col,
                    title=f"{label} {endpoint} - {space}", time_label="Days",
                    min_group_size=int(params.get("km_min_group_size", 10)),
                    save_path=(out_dir / f"{cohort.lower()}_{space}_{endpoint.lower()}_km.pdf") if plot else None,
                )
            except Exception as exc:
                stats = {"p_value": np.nan, "n_samples": 0, "n_groups": 0, "kept_groups": [], "error": str(exc)}
            km_rows.append({"cohort": cohort, "space": space, "endpoint": endpoint, **stats})
    km_summary = pd.DataFrame(km_rows)
    if not km_summary.empty:
        save_source_table(km_summary, config, f"figure4_{cohort.lower()}_km_summary.csv")
    morrison_layout = cohort == "MORRISON"
    response_rows, response_prop_frames = [], []
    for col in response_cols:
        if col not in clinical.columns:
            continue
        for space in ("embedding", "expression"):
            clusters = spaces[f"{space}_clusters"]
            stats = chisq_test(clusters, clinical.loc[clusters.index, col])
            response_rows.append({"cohort": cohort, "space": space, "variable": col, "chi2": stats.get("chi2"), "p_value": stats.get("p_value"), "dof": stats.get("dof")})
            if plot:
                value_order = RESPONSE_ORDER if col == "Response" else FULL_RESPONSE_ORDER if col == "Full_response" else None
                palette = RESPONSE_PALETTE if col == "Response" else FULL_RESPONSE_PALETTE if col == "Full_response" else None
                prop = plot_stacked_proportions(
                    pd.DataFrame({"cluster": clusters.astype(str), col: clinical.loc[clusters.index, col]}),
                    cluster_col="cluster",
                    value_col=col,
                    title=(
                        f"{label} {_short_var(col)}"
                        if morrison_layout
                        else f"{label} {_short_var(col)} by {_short_space(space)} cluster"
                    ),
                    value_order=value_order,
                    palette=palette,
                    legend_title="" if morrison_layout else col,
                    p_value=stats.get("p_value"),
                    figsize=(2.5, 2.3),
                    legend_outside=morrison_layout,
                    save_path=out_dir / f"{cohort.lower()}_{space}_{col.lower()}_proportions.pdf",
                )
                prop.insert(0, "variable", col)
                prop.insert(0, "space", space)
                prop.insert(0, "cohort", cohort)
                response_prop_frames.append(prop)
        if plot:
            coords = spaces["embedding_coords"]
            palette = RESPONSE_PALETTE if col == "Response" else FULL_RESPONSE_PALETTE if col == "Full_response" else None
            plot_umap_categorical(coords, clinical.loc[coords.index, col], title=f"{label} - Embedding ({_short_var(col)})", palette=palette, show_legend=True, legend_title=col, save_path=out_dir / f"{cohort.lower()}_embedding_umap_{col.lower()}.pdf")
    response_summary = pd.DataFrame(response_rows)
    if not response_summary.empty:
        save_source_table(response_summary, config, f"figure4_{cohort.lower()}_response_summary.csv")
    response_proportions = pd.concat(response_prop_frames, ignore_index=True) if response_prop_frames else pd.DataFrame()
    if not response_proportions.empty:
        save_source_table(response_proportions, config, f"figure4_{cohort.lower()}_response_proportions.csv")
    duration_rows, duration_value_frames = [], []
    for col in duration_cols:
        if col not in clinical.columns:
            continue
        for space in ("embedding", "expression"):
            clusters = spaces[f"{space}_clusters"]
            stats = kruskal_test(clusters, clinical.loc[clusters.index, col])
            duration_rows.append({"cohort": cohort, "space": space, "variable": col, **stats})
            values = pd.DataFrame({"cohort": cohort, "space": space, "variable": col, "sample": clusters.index.astype(str), "cluster": clusters.astype(str).to_numpy(), "value": pd.to_numeric(clinical.loc[clusters.index, col], errors="coerce").to_numpy()}).dropna(subset=["value"])
            duration_value_frames.append(values)
            if plot:
                plot_box_by_cluster(
                    pd.DataFrame({"cluster": clusters.astype(str), col: clinical.loc[clusters.index, col]}),
                    cluster_col="cluster",
                    value_col=col,
                    ylabel=col,
                    title=(
                        f"{label} {_short_var(col)}"
                        if morrison_layout
                        else f"{label} {_short_var(col)} by {_short_space(space)} cluster"
                    ),
                    p_value=stats.get("p_value"),
                    log_y=True,
                    save_path=out_dir / f"{cohort.lower()}_{space}_{col.lower()}_boxplot.pdf",
                )
    duration_summary = pd.DataFrame(duration_rows)
    if not duration_summary.empty:
        save_source_table(duration_summary, config, f"figure4_{cohort.lower()}_duration_summary.csv")
    duration_values = pd.concat(duration_value_frames, ignore_index=True) if duration_value_frames else pd.DataFrame()
    if not duration_values.empty:
        save_source_table(duration_values, config, f"figure4_{cohort.lower()}_duration_values.csv")
    return {
        "bundle": bundle, "spaces": spaces, "cluster_summary": cluster_summary,
        "km_summary": km_summary, "response_summary": response_summary,
        "response_proportions": response_proportions,
        "duration_summary": duration_summary, "duration_values": duration_values,
    }


# ---------------------------------------------------------------------------
# 3b. Clustering-parameter robustness sweep — cluster-morphology visualization
# ---------------------------------------------------------------------------

# Figure-4 accent palette (mirrors figures/constants.py) so the sweep panels
# match the rest of the figure without importing the notebook-side constants.
GREEN = "#558771"
ORANGE = "#DD8D6E"
PURPLE = "#885784"
GRAY = "#555555"


def _sweep_cell_pvalue(clinical, labels, metric: str, min_group_size: int) -> float:
    """Clinical-association p-value for one clustering, matching the sweep's stat
    path. ``metric`` is e.g. 'km:OS', 'km:PFS', 'chisq:Response', 'kruskal:OS_time'."""
    kind, col = metric.split(":")
    try:
        if kind == "km":
            df, _ = _prepare_survival_groups(
                clinical, labels, time_col=f"{col}_time", event_col=f"{col}_event",
                min_group_size=int(min_group_size),
            )
            from lifelines.statistics import multivariate_logrank_test
            return float(multivariate_logrank_test(df["time"], df["group"], df["event"]).p_value)
        if kind == "chisq":
            return float(chisq_test(labels, clinical.loc[labels.index, col]).get("p_value", np.nan))
        if kind == "kruskal":
            return float(kruskal_test(labels, clinical.loc[labels.index, col]).get("p_value", np.nan))
    except Exception:
        return float("nan")
    return float("nan")


def plot_cluster_sweep_grid(
    source: pd.DataFrame,
    clinical: pd.DataFrame,
    *,
    center_nn: int,
    center_res: float,
    min_dist: float,
    scale: bool,
    metric: str,
    title: str,
    min_group_size: int = 10,
    sig_alpha: float = 0.05,
    nn_offsets: tuple = (-2, -1, 0, 1, 2),
    res_offsets: tuple = (-0.10, -0.05, 0.0, 0.05, 0.10),
    random_state: int = 0,
    panel_size: float = 1.65,
    point_size: float = 5.0,
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Grid of UMAP panels (rows = n_neighbors, cols = leiden_resolution) around a
    clustering center, colored by Leiden cluster (Figure-4 palette), with the
    ``metric`` p annotated per cell (green if p < ``sig_alpha``) and the center cell
    boxed. Each panel is an independent UMAP+Leiden fit, mirroring the sweep."""
    nns = sorted({max(2, int(center_nn) + d) for d in nn_offsets})
    ress = sorted({round(float(center_res) + d, 2) for d in res_offsets if round(float(center_res) + d, 2) > 0})
    n_rows, n_cols = len(nns), len(ress)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(panel_size * n_cols, panel_size * n_rows), squeeze=False)
    sig_c, nonsig_c, center_c = GREEN, GRAY, PURPLE  # constants.py palette
    for i, nn in enumerate(nns):
        for j, res in enumerate(ress):
            ax = axes[i][j]
            try:
                out = run_umap_cluster(
                    source, n_neighbors=int(nn), min_dist=float(min_dist),
                    leiden_resolution=float(res), random_state=int(random_state), scale=scale,
                )
                coords, labels = out["coords"], out["labels"]
                palette = build_label_palette(labels)
                colors = labels.astype(str).map(palette).fillna("#BDBDBD")
                ax.scatter(coords["UMAP1"], coords["UMAP2"], c=list(colors), s=point_size, edgecolors="none", alpha=0.95)
                p = _sweep_cell_pvalue(clinical, labels, metric, min_group_size)
                is_sig = np.isfinite(p) and p < sig_alpha
                p_txt = f"p={p:.2g}" if np.isfinite(p) else "p=NA"
                ax.set_title(f"nn={nn}, res={res}\nk={labels.nunique()}  {p_txt}",
                             fontsize=TICK_FS - 1, color=sig_c if is_sig else nonsig_c)
            except Exception:
                ax.set_title(f"nn={nn}, res={res}\n(failed)", fontsize=TICK_FS - 1, color=nonsig_c)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(length=0)
            is_center = int(nn) == int(center_nn) and abs(float(res) - float(center_res)) < 1e-9
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(center_c if is_center else "#D8D8D8")
                spine.set_linewidth(2.0 if is_center else 0.6)
    fig.suptitle(title, fontsize=TITLE_FS, y=1.0)
    fig.text(0.5, -0.006, "leiden_resolution  →", ha="center", fontsize=AXIS_LABEL_FS)
    fig.text(-0.004, 0.5, "n_neighbors  →", va="center", rotation=90, fontsize=AXIS_LABEL_FS)
    fig.tight_layout(rect=[0.012, 0.012, 1, 0.975])
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, axes


def plot_robustness_summary(
    master: pd.DataFrame,
    *,
    sig_alpha: float = 0.05,
    title: str = "Clustering-parameter robustness (embedding vs expression)",
    save_path: str | Path | None = None,
    show: bool = True,
):
    """Dumbbell plot of frac_sig for the learned-embedding vs expression-PCA space,
    one row per (unit, metric), sorted by embedding robustness."""
    piv = master.pivot_table(index=["unit", "metric"], columns="space", values="frac_sig").reset_index()
    if "embedding" not in piv.columns:
        piv["embedding"] = np.nan
    if "expression" not in piv.columns:
        piv["expression"] = np.nan
    piv["label"] = piv["unit"].astype(str) + "  " + piv["metric"].astype(str)
    piv = piv.sort_values(["embedding", "expression"], ascending=True, na_position="first").reset_index(drop=True)
    emb = piv["embedding"].to_numpy(dtype=float)
    expr = piv["expression"].to_numpy(dtype=float)
    y = np.arange(len(piv))
    fig, ax = plt.subplots(figsize=(4.3, max(2.2, 0.26 * len(piv))))
    for yi, e, x in zip(y, emb, expr):
        lo = np.nanmin([e, x]) if np.isfinite(e) or np.isfinite(x) else np.nan
        hi = np.nanmax([e, x]) if np.isfinite(e) or np.isfinite(x) else np.nan
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [yi, yi], color="#C9C9C9", lw=1.0, zorder=1)
    ax.scatter(expr, y, color=ORANGE, s=24, zorder=2, label="expression-PCA", edgecolors="none")
    ax.scatter(emb, y, color=GREEN, s=24, zorder=3, label="learned embedding", edgecolors="none")
    ax.set_yticks(y)
    ax.set_yticklabels(piv["label"], fontsize=TICK_FS - 1)
    ax.set_xlabel(f"fraction of grid with p < {sig_alpha:g}  (frac_sig)", fontsize=AXIS_LABEL_FS)
    ax.set_xlim(-0.02, 1.02)
    ax.tick_params(labelsize=TICK_FS)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.legend(fontsize=LEGEND_FS, frameon=False, loc="lower right")
    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax
