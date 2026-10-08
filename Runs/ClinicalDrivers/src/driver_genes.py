"""Clinically focused driver-gene definitions for external validation QC."""

# Candidate driver genes per cancer type -- these
# are the clinically relevant point/indel drivers for each tissue.
CANCER_DRIVER_GENES = {
    "PAAD": ["KRAS", "TP53", "SMAD4", "CDKN2A"],
    "LUAD": ["EGFR", "KRAS", "TP53", "KEAP1", "STK11", "MET", "ERBB2", "BRAF"],
    "BRCA": ["TP53", "PIK3CA", "GATA3", "MAP3K1", "CDH1", "PTEN", "BRCA1", "BRCA2", "ESR1"],
    "COLO": ["APC", "KRAS", "TP53", "PIK3CA", "BRAF", "SMAD4", "NRAS"],
    "GLIOMA": ["IDH1", "IDH2", "TP53", "ATRX", "CIC", "FUBP1", "EGFR", "PTEN"],
    "CHOL": ["IDH1", "IDH2", "KRAS", "TP53", "BRAF", "ARID1A", "SMAD4"],
}

# Flat driver list used by the standardizer to keep only driver-gene mutation
# labels; derived from CANCER_DRIVER_GENES so there is a single source of truth.
TARGET_DRIVER_GENES = sorted({gene for genes in CANCER_DRIVER_GENES.values() for gene in genes})

NONSILENT_VARIANT_CLASSES = {
    "Missense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Nonsense_Mutation",
    "Nonstop_Mutation",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Translation_Start_Site",
    "Splice_Site",
    "Splice_Region",
    "Start_Codon_Del",
    "Start_Codon_Ins",
    "Start_Codon_SNP",
}

HOTSPOT_PATTERNS = {
    "BRAF": ["V600"],
    "EGFR": ["L858", "E746", "L747", "T790", "G719", "L861", "S768", "exon 19"],
    "ERBB2": ["S310", "Y772", "V777", "L755", "D769", "G776"],
    "FGFR3": ["S249", "R248", "Y373", "G370", "K650"],
    "IDH1": ["R132"],
    "IDH2": ["R140", "R172"],
    "KRAS": ["G12", "G13", "Q61", "A146"],
    "NRAS": ["G12", "G13", "Q61"],
    "HRAS": ["G12", "G13", "Q61"],
    "PIK3CA": ["E542", "E545", "H1047"],
}
