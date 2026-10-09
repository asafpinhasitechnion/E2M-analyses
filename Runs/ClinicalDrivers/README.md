# Clinical driver alterations

This run trains XGBoost classifiers on TCGA expression to predict clinically relevant driver mutations, either per gene or per specific alteration (for example KRAS G12C), and tests them on external cohorts from cBioPortal and GEO. Before training, TCGA and each external cohort are integrated with ComBat. The summary table `output/summary/clinical_driver_results.csv` is used for Supplementary Table 3c (`Tables/supplementary_tables.py`, sheet `S3c_driver_validation`). No script under `Figures/` reads this run's outputs.

## Running

Run all commands from `Runs/ClinicalDrivers`:

    python src/prepare_cohorts.py     # download each cohort and write it in one format to data/standardized
    python src/run_drivers.py         # train on TCGA and test on each cohort, gene and alteration modes

Both take `--only` to run some of the sources (`prepare_cohorts.py`: cBioPortal study IDs or GEO accessions) or cohorts (`run_drivers.py`: cohort keys of `config/config.yaml`).

Configuration:
- `config/config.yaml`: data paths, XGBoost parameters, target-selection thresholds, and one entry per cohort: source (cBioPortal study or GEO series; default the cohort key), driver list, matching TCGA cancer type(s), TCGA expression file (`counts` or `tpm`), whether to apply log1p to each side, and an optional sample filter.
- `src/driver_genes.py`: the driver genes of each cancer type (`CANCER_DRIVER_GENES`) and the coding variant classes (from the E2M config).
- `src/alterations.py`: the alteration-level labels.

All data are downloaded by the code. TCGA data are loaded with E2M (`e2m.Dataset.from_tcga`, settings from `../TCGA/config/e2m.yaml`) from `../TCGA/data`, and missing files are downloaded there:
- `expression/TCGA-{CANCER}.star_{tpm|counts}.tsv.gz`: Xena GDC hub STAR expression. TPM for PAAD, LUAD, LGG, GBM, BRCA, COAD, READ and CHOL; counts for PAAD (PRINCE).
- `annotation/gencode.v36.annotation.gtf.gene.probemap` and `annotation/gencode.v36.annotation.gtf.gz`: gene symbols and protein-coding genes.
- `mutations/{CANCER}_mc3_gene_level.txt.gz`: gene-level MC3 labels.
- `mutation_events/{CANCER}_mc3.txt.gz`: MC3 mutation events (Xena TCGA hub), for the alteration-level labels.

Main packages: pandas, numpy, pyyaml, requests, scikit-learn, xgboost, inmoose (`pycombat_norm`, as in Runs/External).

## Data

### Cohorts

| Cohort key | Source | Expression type | Driver list | TCGA cancers | TCGA measure |
|---|---|---|---|---|---|
| paad_iatlas_prince_2022 | cBioPortal `paad_iatlas_prince_2022` (PRINCE PDAC trial) | log2 upper-quartile normalized counts | PAAD | PAAD | counts |
| paad_qcmg_uq_2016 | cBioPortal `paad_qcmg_uq_2016` (QCMG / Bailey 2016) | RNA-seq V2 RSEM | PAAD | PAAD | tpm |
| luad_cas_2020 | cBioPortal `luad_cas_2020` | RNA-seq FPKM | LUAD | LUAD | tpm |
| luad_oncosg_2020 | cBioPortal `luad_oncosg_2020` | RNA-seq V2 RSEM z-scores | LUAD | LUAD | tpm |
| difg_glass | cBioPortal `difg_glass` (GLASS Consortium) | RNA-seq TPM | GLIOMA | LGG, GBM | tpm |
| metabric | cBioPortal `brca_metabric` | Illumina microarray | BRCA | BRCA | tpm |
| pog570_bcgsc_2020__BRCA | cBioPortal `pog570_bcgsc_2020`, ANALYSIS_COHORT = BRCA | RNA-seq RPKM | BRCA | BRCA | tpm |
| pog570_bcgsc_2020__COLO | same study, ANALYSIS_COHORT = COLO | RNA-seq RPKM | COLO | COAD, READ | tpm |
| pog570_bcgsc_2020__LUAD | same study, ONCOTREE_CODE = LUAD | RNA-seq RPKM | LUAD | LUAD | tpm |
| pog570_bcgsc_2020__PAAD | same study, ONCOTREE_CODE = PAAD | RNA-seq RPKM | PAAD | PAAD | tpm |
| pog570_bcgsc_2020__CHOL | same study, ANALYSIS_COHORT = CHOL | RNA-seq RPKM | CHOL | CHOL | tpm |
| GSE39582 | GEO GSE39582 (Marisa / CIT), colon cancer | GPL570 Affymetrix microarray | COLO | COAD, READ | tpm |
| GSE31210 | GEO GSE31210 (Okayama), lung adenocarcinoma | GPL570 Affymetrix microarray | LUAD | LUAD | tpm |

Driver genes per cancer type (`CANCER_DRIVER_GENES`):
- PAAD: KRAS, TP53, SMAD4, CDKN2A
- LUAD: EGFR, KRAS, TP53, KEAP1, STK11, MET, ERBB2, BRAF
- BRCA: TP53, PIK3CA, GATA3, MAP3K1, CDH1, PTEN, BRCA1, BRCA2, ESR1
- COLO: APC, KRAS, TP53, PIK3CA, BRAF, SMAD4, NRAS
- GLIOMA: IDH1, IDH2, TP53, ATRX, CIC, FUBP1, EGFR, PTEN
- CHOL: IDH1, IDH2, KRAS, TP53, BRAF, ARID1A, SMAD4

Only point mutations and small indels are modeled, no fusions or copy-number changes.

### cBioPortal studies

- Files are downloaded from the cBioPortal DataHub GitHub repository (`public/{study}`) at a fixed commit (`a766d41`, 2026-05-13; `DATAHUB_COMMIT` in `src/prepare_cohorts.py`). The next commit lost the Git LFS file of the paad_qcmg_uq_2016 mutations.
- Expression: the `MRNA_EXPRESSION` file with datatype `CONTINUOUS`; z-score files only when there is no other. Values are used as given; duplicate gene symbols are averaged.
- Mutations: the MAF file, coding classes only (E2M's: Missense, Nonsense, Nonstop, Frame_Shift_Del/Ins, In_Frame_Del/Ins, Splice_Site, Translation_Start_Site), driver genes only.
- Gene-level label: 1 if the sample has at least one such mutation in the gene, otherwise 0. Samples not in the study's sequenced list (`case_lists/cases_sequenced.txt`) get missing labels, for example 64 of 355 in GLASS. For targeted panels (METABRIC, 173 genes), genes outside the sample's panel are missing; ESR1 is not on the METABRIC panel.

### GEO series

- Downloads `{GSE}_series_matrix.txt.gz` from the NCBI GEO FTP and the GPL570 annotation (`GPL570.annot.gz`). Only tumour samples are kept.
- Expression: values used as given. Probes are mapped to the first gene symbol of the GPL570 annotation and averaged per gene.
- GSE39582: tumours are the samples with dataset other than "Non Tumoral". TP53, KRAS and BRAF are 1 for "M", 0 for "WT", and missing otherwise.
- GSE31210: tumours are "tissue: primary lung tumor" (the 20 normal lung samples are removed). From "gene alteration status": EGFR = "EGFR mutation +", KRAS = "KRAS mutation +", ALK = "ALK-fusion +", all other values 0. ALK is not in the LUAD driver list, so it is not modeled.
- GEO series have no per-variant records, so they have no alteration-level labels.

### Alteration-level labels (alterations.py)

The labels come from coding mutation events: the cBioPortal MAF for the external cohorts and the MC3 events for TCGA. For each event, the variant class, protein change and coding change are joined into one upper-case text, and the labels are matched against it (and, where given, the genomic start position):
- EGFR_classic_activating: EGFR exon 19 deletion (deletion at E746, L747, T751, S752 or P753, or start position 55242400-55242560) or L858R.
- EGFR_exon20ins: EGFR in-frame insertion/duplication at A763-V774, or start position 55248900-55249180.
- KRAS_G12C; KRAS_hotspot_broad: KRAS G12, G13, Q61 or A146.
- BRAF_V600E; BRAF_V600_any.
- PIK3CA_hotspot_core: E542, E545 or H1047. PIK3CA_hotspot_extended: these plus N345, C420 or Q546.
- ESR1_LBD: Y537, D538, E380Q, S463P or L536.
- IDH1_R132; IDH2_R140_R172.
- ERBB2_TKD_or_exon20: Y772, G776, D769, V777, L755 or A775, or an in-frame insertion at these residues or P780. ERBB2_activating_extended: these plus S310.
- MET_ex14: MET splice event with protein change X1000-X1039, or start position 116412035-116412055.

An alteration label is missing where its gene is missing. A cohort is tested on an alteration label only if its gene is in the cohort's driver list.

### Integration with TCGA

- TCGA (E2M): protein-coding genes; Xena values back to the linear scale (2^x - 1); counts (PRINCE) converted to CPM, TPM used as is. Every gene mutated in at least one TCGA sample is a gene label.
- POG570: samples are kept by the clinical column and values in `filter`.
- log1p (after clipping negative values to 0) is applied to each side as set in the config. TCGA: always. External: paad_qcmg_uq_2016, luad_cas_2020, difg_glass and the POG570 splits.
- Genes: only genes in both TCGA and the cohort; a gene is dropped if it has a non-finite value or zero variance inside any batch.
- ComBat (inmoose `pycombat_norm`, default settings) on TCGA and the cohort together. Batches are the TCGA cancer types (`TCGA_<CANCER>`, so LGG/GBM and COAD/READ are separate) and the cohort.

### Targets, model and evaluation

- A target (gene or alteration label) is selected if it has at least 3 TCGA positives (`min_tcga_positive`) and at least 5 external positives (`min_external_positive`). A target with the same external labels as an earlier target is skipped (for example BRAF_V600_any when every V600 is V600E).
- One XGBClassifier per target, trained on all TCGA samples of the matched cancer type(s): n_estimators 300, learning_rate 0.1, random_state 42, n_jobs -1, eval_metric logloss, scale_pos_weight = negatives/positives.
- Metrics on the external samples with a label: ROC AUC, AUPRC (average precision) and normalized AUPRC = (AUPRC - prevalence) / (1 - prevalence). A target is `evaluable` when these samples contain both classes.

## Outputs

`data/standardized/<source>/` (from prepare_cohorts.py): `expression.csv.gz` (samples x genes), `mutations_gene_level.csv.gz` (samples x driver genes), `clinical_sample.csv.gz`, and for cBioPortal studies `mutations_long.csv.gz` (the coding driver events).

`output/` (from run_drivers.py):
- `gene/<cohort>/` and `alteration/<cohort>/`: `targets.csv` (selection, counts and metrics per target), `probabilities.csv.gz` (external predictions), `manifest.json` (cohort settings, sample numbers, genes after integration, batch sizes).
- `summary/clinical_driver_results.csv`: both modes, with columns cohort_id, label, cancer, label_mode, target, alteration_definition, external_n_positive, external_n_evaluable, prevalence, roc_auc, auprc, normalized_auprc and evaluable. `Tables/supplementary_tables.py` keeps the rows with `evaluable` = True for Supplementary Table S3c.
