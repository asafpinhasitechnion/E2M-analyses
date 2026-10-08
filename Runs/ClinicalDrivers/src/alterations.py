from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlterationDefinition:
    label: str
    genes: tuple[str, ...]
    category: str
    description: str


ALTERATION_DEFINITIONS: tuple[AlterationDefinition, ...] = (
    AlterationDefinition("EGFR_classic_activating", ("EGFR",), "hotspot_snv_indel", "EGFR exon 19 deletions or L858R."),
    AlterationDefinition("EGFR_exon20ins", ("EGFR",), "hotspot_snv_indel", "EGFR exon 20 insertions."),
    AlterationDefinition("KRAS_G12C", ("KRAS",), "hotspot_snv_indel", "KRAS G12C only."),
    AlterationDefinition("BRAF_V600E", ("BRAF",), "hotspot_snv_indel", "BRAF V600E strict label."),
    AlterationDefinition("BRAF_V600_any", ("BRAF",), "hotspot_snv_indel", "Any BRAF V600 substitution."),
    AlterationDefinition("PIK3CA_hotspot_core", ("PIK3CA",), "hotspot_snv_indel", "PIK3CA E542, E545, or H1047."),
    AlterationDefinition("PIK3CA_hotspot_extended", ("PIK3CA",), "hotspot_snv_indel", "Core PIK3CA hotspots plus N345, C420, or Q546."),
    AlterationDefinition("ESR1_LBD", ("ESR1",), "hotspot_snv_indel", "ESR1 ligand-binding-domain endocrine-resistance mutations."),
    AlterationDefinition("IDH1_R132", ("IDH1",), "hotspot_snv_indel", "Any IDH1 R132 alteration."),
    AlterationDefinition("IDH2_R140_R172", ("IDH2",), "hotspot_snv_indel", "Any IDH2 R140 or R172 alteration."),
    AlterationDefinition("ERBB2_TKD_or_exon20", ("ERBB2",), "hotspot_snv_indel", "ERBB2 tyrosine-kinase-domain/exon 20 activating alterations."),
    AlterationDefinition("ERBB2_activating_extended", ("ERBB2",), "hotspot_snv_indel", "ERBB2 TKD/exon 20 alterations plus S310."),
    AlterationDefinition("FGFR3_susceptible_point", ("FGFR3",), "hotspot_snv_indel", "FGFR3 susceptible point mutations."),
    AlterationDefinition("MET_ex14", ("MET",), "splice_event", "MET exon-14-skipping-like splice alterations."),
    AlterationDefinition("KRAS_hotspot_broad", ("KRAS",), "biologic_hotspot", "KRAS G12, G13, Q61, or A146."),
)

ALTERATION_LABELS: tuple[str, ...] = tuple(defn.label for defn in ALTERATION_DEFINITIONS)


def alteration_definition_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "label": defn.label,
                "genes": ";".join(defn.genes),
                "category": defn.category,
                "description": defn.description,
            }
            for defn in ALTERATION_DEFINITIONS
        ]
    )


def alteration_labels_for_primary_targets(primary_targets: Iterable[str]) -> tuple[str, ...]:
    primary = {str(gene).upper() for gene in primary_targets}
    return tuple(defn.label for defn in ALTERATION_DEFINITIONS if primary.intersection(defn.genes))


def build_alteration_matrix(events: pd.DataFrame, sample_ids: Iterable[str]) -> pd.DataFrame:
    sample_index = pd.Index([str(sample) for sample in sample_ids], name="sample")
    labels = pd.DataFrame(0, index=sample_index, columns=ALTERATION_LABELS, dtype=np.int8)
    if events.empty:
        return labels

    frame = _standardize_events(events)
    if frame.empty:
        return labels

    frame = frame[frame["sample"].isin(sample_index)]
    if frame.empty:
        return labels

    for label, mask in _alteration_masks(frame).items():
        positive_samples = frame.loc[mask, "sample"].dropna().astype(str).unique()
        if len(positive_samples):
            labels.loc[labels.index.intersection(positive_samples), label] = 1
    return labels


def _standardize_events(events: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        "sample": _first_column(events, ["sample", "Tumor_Sample_Barcode", "SAMPLE_ID"]),
        "gene": _first_column(events, ["gene", "Hugo_Symbol"]),
        "variant": _first_column(events, ["effect", "Variant_Classification", "variant"]),
        "protein": _first_column(events, ["HGVSp_Short", "Amino_Acid_Change", "Protein_Change", "HGVSp"]),
        "protein_long": _first_column(events, ["HGVSp", "Amino_Acid_Change", "Protein_Change", "HGVSp_Short"]),
        "coding": _first_column(events, ["HGVSc", "coding", "cDNA_Change"]),
        "chromosome": _first_column(events, ["chr", "Chromosome"]),
        "start": _first_column(events, ["start", "Start_Position"]),
        "end": _first_column(events, ["end", "End_Position"]),
    }
    if not colmap["sample"] or not colmap["gene"]:
        return pd.DataFrame()

    out = pd.DataFrame(
        {
            "sample": events[colmap["sample"]].astype(str),
            "gene": events[colmap["gene"]].astype(str).str.upper(),
            "variant": _optional_series(events, colmap["variant"]),
            "protein": _optional_series(events, colmap["protein"]),
            "protein_long": _optional_series(events, colmap["protein_long"]),
            "coding": _optional_series(events, colmap["coding"]),
            "chromosome": _optional_series(events, colmap["chromosome"]),
            "start": pd.to_numeric(_optional_series(events, colmap["start"]), errors="coerce"),
            "end": pd.to_numeric(_optional_series(events, colmap["end"]), errors="coerce"),
        }
    )
    out["text"] = (
        out[["variant", "protein", "protein_long", "coding"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.upper()
    )
    return out.dropna(subset=["sample", "gene"])


def _first_column(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in frame.columns:
            return col
    return None


def _optional_series(frame: pd.DataFrame, col: str | None) -> pd.Series:
    if col and col in frame.columns:
        return frame[col].astype(str)
    return pd.Series("", index=frame.index, dtype=str)


def _contains_any(text: pd.Series, tokens: Iterable[str]) -> pd.Series:
    pattern = "|".join(tokens)
    return text.str.contains(pattern, regex=True, na=False)


def _is_inframe_insertion(frame: pd.DataFrame) -> pd.Series:
    text = frame["text"]
    return text.str.contains("IN_FRAME_INS|INS|DUP", regex=True, na=False) & ~text.str.contains("DELINS|DEL", regex=True, na=False)


def _is_deletion_like(frame: pd.DataFrame) -> pd.Series:
    return frame["text"].str.contains("IN_FRAME_DEL|DEL|DELINS", regex=True, na=False)


def _coord_between(frame: pd.DataFrame, start: int, end: int) -> pd.Series:
    pos = frame["start"]
    return pos.notna() & pos.between(start, end)


def _alteration_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    text = frame["text"]
    gene = frame["gene"]

    egfr = gene.eq("EGFR")
    egfr_ex19del = egfr & _is_deletion_like(frame) & (
        _contains_any(text, ["E746", "L747", "T751", "S752", "P753"])
        | _coord_between(frame, 55242400, 55242560)
    )
    egfr_l858r = egfr & _contains_any(text, ["L858R", "LEU858ARG"])
    egfr_ex20ins = egfr & _is_inframe_insertion(frame) & (
        _contains_any(text, ["A763", "Y764", "V765", "S768", "V769", "D770", "N771", "P772", "H773", "V774"])
        | _coord_between(frame, 55248900, 55249180)
    )

    kras = gene.eq("KRAS")
    braf = gene.eq("BRAF")
    pik3ca = gene.eq("PIK3CA")
    esr1 = gene.eq("ESR1")
    erbb2 = gene.eq("ERBB2")
    fgfr3 = gene.eq("FGFR3")
    met = gene.eq("MET")

    erbb2_tkd = erbb2 & (
        _contains_any(text, ["Y772", "G776", "D769", "V777", "L755", "ALA775", "TYR772", "GLY776", "ASP769", "VAL777", "LEU755"])
        | (_is_inframe_insertion(frame) & _contains_any(text, ["Y772", "A775", "G776", "V777", "D769", "P780"]))
    )
    met_ex14 = met & text.str.contains("SPLICE", regex=False, na=False) & (
        _contains_any(text, ["X100", "X101", "X102", "X103"])
        | _coord_between(frame, 116412035, 116412055)
    )

    masks = {
        "EGFR_classic_activating": egfr_ex19del | egfr_l858r,
        "EGFR_exon20ins": egfr_ex20ins,
        "KRAS_G12C": kras & _contains_any(text, ["G12C", "GLY12CYS"]),
        "BRAF_V600E": braf & _contains_any(text, ["V600E", "VAL600GLU"]),
        "BRAF_V600_any": braf & _contains_any(text, ["V600", "VAL600"]),
        "PIK3CA_hotspot_core": pik3ca & _contains_any(text, ["E542", "E545", "H1047", "GLU542", "GLU545", "HIS1047"]),
        "PIK3CA_hotspot_extended": pik3ca & _contains_any(text, ["E542", "E545", "H1047", "N345", "C420", "Q546", "GLU542", "GLU545", "HIS1047", "ASN345", "CYS420", "GLN546"]),
        "ESR1_LBD": esr1 & _contains_any(text, ["Y537", "D538", "E380Q", "S463P", "L536", "TYR537", "ASP538", "GLU380GLN", "SER463PRO", "LEU536"]),
        "IDH1_R132": gene.eq("IDH1") & _contains_any(text, ["R132", "ARG132"]),
        "IDH2_R140_R172": gene.eq("IDH2") & _contains_any(text, ["R140", "R172", "ARG140", "ARG172"]),
        "ERBB2_TKD_or_exon20": erbb2_tkd,
        "ERBB2_activating_extended": erbb2_tkd | (erbb2 & _contains_any(text, ["S310", "SER310"])),
        "FGFR3_susceptible_point": fgfr3 & _contains_any(text, ["S249", "R248", "Y373", "G370", "K650", "SER249", "ARG248", "TYR373", "GLY370", "LYS650"]),
        "MET_ex14": met_ex14,
        "KRAS_hotspot_broad": kras & _contains_any(text, ["G12", "G13", "Q61", "A146", "GLY12", "GLY13", "GLN61", "ALA146"]),
    }
    return masks
