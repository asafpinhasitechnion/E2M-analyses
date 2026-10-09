"""Driver genes per cancer type and the coding mutation classes."""

from e2m.config import load_config

# Candidate driver genes per cancer type: the clinically relevant point/indel drivers for each tissue.
CANCER_DRIVER_GENES = {
    "PAAD": ["KRAS", "TP53", "SMAD4", "CDKN2A"],
    "LUAD": ["EGFR", "KRAS", "TP53", "KEAP1", "STK11", "MET", "ERBB2", "BRAF"],
    "BRCA": ["TP53", "PIK3CA", "GATA3", "MAP3K1", "CDH1", "PTEN", "BRCA1", "BRCA2", "ESR1"],
    "COLO": ["APC", "KRAS", "TP53", "PIK3CA", "BRAF", "SMAD4", "NRAS"],
    "GLIOMA": ["IDH1", "IDH2", "TP53", "ATRX", "CIC", "FUBP1", "EGFR", "PTEN"],
    "CHOL": ["IDH1", "IDH2", "KRAS", "TP53", "BRAF", "ARID1A", "SMAD4"],
}

DRIVER_GENES = sorted({gene for genes in CANCER_DRIVER_GENES.values() for gene in genes})

# The coding MAF classes used for the TCGA labels and TMB (E2M config), as in Runs/External
CODING_VARIANT_CLASSES = set(load_config()["tmb"]["coding_variant_classes"])
