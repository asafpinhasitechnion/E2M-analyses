"""Alteration-level labels (for example KRAS G12C) from MAF-style mutation events."""

from __future__ import annotations

import numpy as np
import pandas as pd

# label: (gene, description)
ALTERATIONS = {
    "EGFR_classic_activating": ("EGFR", "EGFR exon 19 deletions or L858R."),
    "EGFR_exon20ins": ("EGFR", "EGFR exon 20 insertions."),
    "KRAS_G12C": ("KRAS", "KRAS G12C only."),
    "BRAF_V600E": ("BRAF", "BRAF V600E strict label."),
    "BRAF_V600_any": ("BRAF", "Any BRAF V600 substitution."),
    "PIK3CA_hotspot_core": ("PIK3CA", "PIK3CA E542, E545, or H1047."),
    "PIK3CA_hotspot_extended": ("PIK3CA", "Core PIK3CA hotspots plus N345, C420, or Q546."),
    "ESR1_LBD": ("ESR1", "ESR1 ligand-binding-domain endocrine-resistance mutations."),
    "IDH1_R132": ("IDH1", "Any IDH1 R132 alteration."),
    "IDH2_R140_R172": ("IDH2", "Any IDH2 R140 or R172 alteration."),
    "ERBB2_TKD_or_exon20": ("ERBB2", "ERBB2 tyrosine-kinase-domain/exon 20 activating alterations."),
    "ERBB2_activating_extended": ("ERBB2", "ERBB2 TKD/exon 20 alterations plus S310."),
    "MET_ex14": ("MET", "MET exon-14-skipping-like splice alterations."),
    "KRAS_hotspot_broad": ("KRAS", "KRAS G12, G13, Q61, or A146."),
}

# Event columns joined into the text the labels are matched against
TEXT_COLUMNS = ["Variant_Classification", "HGVSp_Short", "HGVSp", "Protein_Change", "Amino_Acid_Change", "HGVSc"]


def alteration_labels(events: pd.DataFrame, sample_ids) -> pd.DataFrame:
    """1 if the sample has an event matching the label, else 0. `events` has MAF columns: Tumor_Sample_Barcode,
    Hugo_Symbol, Start_Position and some of TEXT_COLUMNS."""
    labels = pd.DataFrame(0, index=pd.Index(sample_ids, name="sample"), columns=list(ALTERATIONS), dtype=np.int8)
    events = events[events["Tumor_Sample_Barcode"].isin(labels.index)]
    if events.empty:
        return labels
    text = events[[c for c in TEXT_COLUMNS if c in events]].fillna("").astype(str).agg(" ".join, axis=1).str.upper()
    gene = events["Hugo_Symbol"].astype(str).str.upper()
    start = pd.to_numeric(events["Start_Position"], errors="coerce")
    for label, mask in _masks(gene, text, start).items():
        labels.loc[events.loc[mask, "Tumor_Sample_Barcode"].unique(), label] = 1
    return labels


def _any(text: pd.Series, tokens: list[str]) -> pd.Series:
    return text.str.contains("|".join(tokens), regex=True, na=False)


def _masks(gene: pd.Series, text: pd.Series, start: pd.Series) -> dict[str, pd.Series]:
    inframe_ins = _any(text, ["IN_FRAME_INS", "INS", "DUP"]) & ~_any(text, ["DELINS", "DEL"])
    deletion = _any(text, ["IN_FRAME_DEL", "DEL", "DELINS"])

    egfr = gene.eq("EGFR")
    egfr_ex19del = egfr & deletion & (_any(text, ["E746", "L747", "T751", "S752", "P753"]) | start.between(55242400, 55242560))
    egfr_l858r = egfr & _any(text, ["L858R", "LEU858ARG"])
    egfr_ex20ins = egfr & inframe_ins & (
        _any(text, ["A763", "Y764", "V765", "S768", "V769", "D770", "N771", "P772", "H773", "V774"])
        | start.between(55248900, 55249180)
    )

    kras = gene.eq("KRAS")
    braf = gene.eq("BRAF")
    pik3ca = gene.eq("PIK3CA")
    erbb2 = gene.eq("ERBB2")
    erbb2_tkd = erbb2 & (
        _any(text, ["Y772", "G776", "D769", "V777", "L755", "ALA775", "TYR772", "GLY776", "ASP769", "VAL777", "LEU755"])
        | (inframe_ins & _any(text, ["Y772", "A775", "G776", "V777", "D769", "P780"]))
    )
    met_ex14 = gene.eq("MET") & text.str.contains("SPLICE", regex=False, na=False) & (
        _any(text, ["X100", "X101", "X102", "X103"]) | start.between(116412035, 116412055)
    )

    return {
        "EGFR_classic_activating": egfr_ex19del | egfr_l858r,
        "EGFR_exon20ins": egfr_ex20ins,
        "KRAS_G12C": kras & _any(text, ["G12C", "GLY12CYS"]),
        "BRAF_V600E": braf & _any(text, ["V600E", "VAL600GLU"]),
        "BRAF_V600_any": braf & _any(text, ["V600", "VAL600"]),
        "PIK3CA_hotspot_core": pik3ca & _any(text, ["E542", "E545", "H1047", "GLU542", "GLU545", "HIS1047"]),
        "PIK3CA_hotspot_extended": pik3ca & _any(text, ["E542", "E545", "H1047", "N345", "C420", "Q546", "GLU542", "GLU545", "HIS1047", "ASN345", "CYS420", "GLN546"]),
        "ESR1_LBD": gene.eq("ESR1") & _any(text, ["Y537", "D538", "E380Q", "S463P", "L536", "TYR537", "ASP538", "GLU380GLN", "SER463PRO", "LEU536"]),
        "IDH1_R132": gene.eq("IDH1") & _any(text, ["R132", "ARG132"]),
        "IDH2_R140_R172": gene.eq("IDH2") & _any(text, ["R140", "R172", "ARG140", "ARG172"]),
        "ERBB2_TKD_or_exon20": erbb2_tkd,
        "ERBB2_activating_extended": erbb2_tkd | (erbb2 & _any(text, ["S310", "SER310"])),
        "MET_ex14": met_ex14,
        "KRAS_hotspot_broad": kras & _any(text, ["G12", "G13", "Q61", "A146", "GLY12", "GLY13", "GLN61", "ALA146"]),
    }
