"""Per-cell perturbation labels where scPerturb's `perturbation` column is not the cell's actual perturbation."""

from __future__ import annotations

import numpy as np
import pandas as pd

FRANGIEH_CONTROL_GUIDES = ("NO_SITE", "ONE_NON-GENE")  # guide names of the non-targeting and non-gene-site controls


def mcfarland_trametinib(adata):
    """McFarland: scPerturb labels every cell of the trametinib time course "Trametinib", but the course was
    cell-hashed: obs hash_tag is Tram_<time> (treated), DMSO_<time> / Untreated_48hr (vehicle), multiplet or unknown.
    Vehicle cells become controls, multiplet and unknown cells are left out; cells without a hash tag (the unhashed
    24 h experiment) stay trametinib-treated."""
    obs = adata.obs
    tag = obs["hash_tag"].astype(str)  # missing tags read as "nan"
    hashed = obs["perturbation"].astype(str).eq("Trametinib") & tag.ne("nan")
    label = obs["perturbation"].astype(str)
    label[hashed & (tag.str.startswith("DMSO_") | tag.str.startswith("Untreated"))] = "control"
    drop = hashed & ~tag.str.startswith("Tram_") & ~tag.str.startswith("DMSO_") & ~tag.str.startswith("Untreated")
    adata = adata[~drop.to_numpy()].copy()
    adata.obs["perturbation"] = label[~drop].astype("category")
    print(f"  [labels] McFarland trametinib: {int((hashed & label.eq('control')).sum())} vehicle cells -> control, "
          f"{int(drop.sum())} multiplet/unknown cells left out")
    return adata


def frangieh_single_guide(adata):
    """Frangieh: cells with exactly one guide, labelled with its target gene, as in the authors' analysis (their
    sgRNA column, set for cells with one guide only; their linear model has one guide per cell). Control guides,
    NO_SITE (non-targeting) and ONE_NON-GENE_SITE (intergenic) in Supplementary Table 1, give "control". scPerturb's
    `perturbation` instead labels cells with several guides by the first gene, and cells with no guide, or a control
    guide next to a targeting one, "control"; those cells are left out."""
    guide = adata.obs["sgRNA"].astype(str)
    keep = guide.ne("nan").to_numpy()
    gene = guide[keep].str.rsplit("_", n=1).str[0]
    old = adata.obs["perturbation"].astype(str).to_numpy()[keep]
    adata = adata[keep].copy()
    adata.obs["perturbation"] = pd.Categorical(np.where(gene.str.startswith(FRANGIEH_CONTROL_GUIDES), "control", gene))
    print(f"  [labels] Frangieh single-guide cells: kept {int(keep.sum())} of {len(keep)} "
          f"({int((adata.obs['perturbation'] == 'control').sum())} controls), "
          f"{int((old != adata.obs['perturbation'].astype(str).to_numpy()).sum())} labels changed")
    return adata
