"""Bring TCGA and an external cohort onto the same scale on their shared genes, before E2M.

Methods (TCGA is always one batch):
- none: each side is log-transformed on its own, sample by sample (CPM + log1p for counts); no cohort-level step.
- combat: ComBat (inmoose pycombat_norm) on the log values of TCGA and the cohort together.
- rank: each gene is ranked across samples within each side separately and scaled to (0, 1].
- sample_rank: each sample's genes are ranked within that sample and scaled to (0, 1]; sample by sample, no cohort-level step.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

METHODS = ("none", "combat", "rank", "sample_rank")
SCALES = ("counts", "tpm", "log")


def to_log(df: pd.DataFrame, scale: str, cpm: bool) -> pd.DataFrame:
    """Per-sample log values: counts -> log1p(CPM) (or log1p(counts)), tpm -> log1p(tpm), log -> unchanged."""
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {SCALES}")
    if scale == "log":
        return df
    values = df.clip(lower=0)
    if scale == "counts" and cpm:
        values = values.div(values.sum(axis=1), axis=0) * 1e6
    return np.log1p(values)


def _batches(tcga: pd.DataFrame, external: pd.DataFrame, external_batches: pd.Series | None) -> pd.Series:
    ext = external_batches.reindex(external.index).fillna("External") if external_batches is not None else pd.Series("External", index=external.index)
    return pd.concat([pd.Series("TCGA", index=tcga.index), ext.astype(str)]).rename("batch")


def _combat(combined: pd.DataFrame, batch: pd.Series) -> pd.DataFrame:
    from inmoose.pycombat import pycombat_norm

    keep = combined.var(axis=0, ddof=0) > 0
    for name in batch.unique():
        rows = batch.index[batch == name]
        if len(rows) >= 2:
            keep &= combined.loc[rows].var(axis=0, ddof=0) > 0
    combined = combined.loc[:, keep]
    corrected = pycombat_norm(counts=combined.T.to_numpy(dtype=np.float64), batch=batch.to_numpy())
    return pd.DataFrame(np.asarray(corrected).T, index=combined.index, columns=combined.columns)


def _rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=0, method="average") / max(len(df), 1)


def integrate(
    tcga: pd.DataFrame,
    external: pd.DataFrame,
    method: str,
    *,
    tcga_scale: str,
    external_scale: str,
    cpm: bool,
    external_batches: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """TCGA and external model inputs on their shared genes (genes a method cannot use are dropped)."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    shared = tcga.columns.intersection(external.columns).sort_values()
    tcga, external = tcga[shared].fillna(0.0), external[shared].fillna(0.0)
    batch = _batches(tcga, external, external_batches)

    tcga, external = to_log(tcga, tcga_scale, cpm), to_log(external, external_scale, cpm)
    if method == "none":
        return tcga, external
    if method == "rank":
        return _rank(tcga), _rank(external)
    if method == "sample_rank":
        return _rank(tcga.T).T, _rank(external.T).T
    combined = _combat(pd.concat([tcga, external]), batch)
    return combined.loc[tcga.index], combined.loc[external.index]


def umap_coordinates(tcga: pd.DataFrame, external: pd.DataFrame, external_batches: pd.Series | None, *, label: str,
                     max_features: int = 5000, n_neighbors: int = 30, min_dist: float = 0.25, random_state: int = 42) -> pd.DataFrame:
    """UMAP of TCGA and external samples on their most variable shared genes (standardized, euclidean)."""
    import umap
    from sklearn.preprocessing import StandardScaler

    shared = tcga.columns.intersection(external.columns)
    matrix = pd.concat([tcga[shared], external[shared]]).fillna(0.0)
    matrix = matrix[matrix.var(axis=0).sort_values(ascending=False).index[:max_features]]
    coords = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, metric="euclidean",
                       random_state=random_state).fit_transform(StandardScaler().fit_transform(matrix))
    out = pd.DataFrame(coords, index=matrix.index, columns=["UMAP1", "UMAP2"])
    out["dataset"] = ["TCGA"] * len(tcga) + [label] * len(external)
    out["batch"] = _batches(tcga, external, external_batches).replace("External", label).to_numpy()
    return out
