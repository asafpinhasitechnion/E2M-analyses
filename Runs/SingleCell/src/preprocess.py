"""Expression preprocessing shared by every single-cell analysis (scPerturb datasets and CCLE)."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import scanpy as sc

import config

# HGNC table, as for the External cohorts (2022-01 monthly archive)
HGNC_URL = "https://storage.googleapis.com/public-download-files/hgnc/archive/archive/monthly/tsv/hgnc_complete_set_2022-01-01.txt"


def protein_coding(var: pd.DataFrame) -> np.ndarray:
    """HGNC protein-coding genes, by Ensembl ID where var has one (ensembl_id), else by symbol (var_names)."""
    path = config.DATA_DIR / Path(HGNC_URL).name
    if not path.exists():
        urlretrieve(HGNC_URL, path)
    hgnc = pd.read_csv(path, sep="\t", usecols=["symbol", "locus_group", "ensembl_gene_id"], dtype=str)
    coding = hgnc[hgnc["locus_group"] == "protein-coding gene"]
    keep = var.index.astype(str).isin(coding["symbol"])
    if "ensembl_id" in var.columns:
        keep |= var["ensembl_id"].astype(str).str.split(".").str[0].isin(coding["ensembl_gene_id"].dropna())
    return np.asarray(keep)


def preprocess(adata):
    """normalize_total and log1p on all genes (raw counts), then keep the features: genes detected in at least
    config.MIN_CELLS_PER_FEATURE cells, and protein-coding if config.PROTEIN_CODING_ONLY."""
    keep = np.asarray((adata.X > 0).sum(axis=0)).ravel() >= config.MIN_CELLS_PER_FEATURE
    if config.PROTEIN_CODING_ONLY:
        keep &= protein_coding(adata.var)
    sc.pp.normalize_total(adata, target_sum=config.PREPROCESS_TARGET_SUM)
    sc.pp.log1p(adata)
    print(f"  [preprocess] kept {int(keep.sum())} of {adata.n_vars} genes as features")
    return adata[:, keep].copy()
