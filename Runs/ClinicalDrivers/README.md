# Clinical driver alterations

This run trains XGBoost classifiers on TCGA expression to predict clinically relevant driver mutations, either per gene or per specific alteration (for example KRAS G12C), and tests them on external cohorts from cBioPortal, GEO and METABRIC. Before training, TCGA and each external cohort are integrated with ComBat. The summary table `output/summary/clinical_driver_results.csv` is used for Supplementary Table 3c (`Tables/supplementary_tables.py`, sheet `S3c_driver_validation`). No script under `Figures/` reads this run's outputs.

## Running

Run all commands from `Runs/ClinicalDrivers`:

    python src/prepare_cbioportal.py --all     # download and standardize the cBioPortal studies
    python src/prepare_geo.py --all            # download and standardize the GEO series
    python src/run_drivers.py                  # transfer TCGA -> each cohort, gene and alteration modes

Options:
- `prepare_cbioportal.py`: `--study ID` (repeatable) instead of `--all`, `--force-download`, `--keep-raw` (by default raw downloads are deleted after standardization), `--summarize-only`, `--root`.
- `prepare_geo.py`: `--accession GSE...` (repeatable) instead of `--all`, `--keep-raw`, `--summarize-only`, `--root`.
- `run_drivers.py`: `--only COHORT ...` (cohort keys from `config/config.yaml`), `--output-dir` (default `output`), `--summarize-only` (only rebuilds the summary tables), `--config`.

Configuration:
- `config/datasets.json` lists the cBioPortal study IDs (`phase1_cbioportal`) and GEO accessions (`phase2_geo`) that `--all` processes.
- `config/config.yaml` holds the data paths, XGBoost parameters, target-selection thresholds and one entry per cohort: driver list, matching TCGA cancer type(s), which TCGA expression file to use (`counts` or `tpm`), whether to apply log1p to each side, and optional sample filters.
- `src/driver_genes.py` defines the candidate driver genes for each cancer type (`CANCER_DRIVER_GENES`).
- `src/alterations.py` defines the alteration-level labels.

TCGA inputs. Paths are relative to `Runs/ClinicalDrivers`; the first three are shared with Runs/TCGA and loaded with E2M (`e2m.Dataset.from_tcga`, settings from `../TCGA/config/e2m.yaml`):
- `../TCGA/data/expression/TCGA-{CANCER}.star_{tpm|counts}.tsv.gz`: per-cancer Xena GDC hub STAR expression. TPM files are needed for PAAD, LUAD, LGG, GBM, BRCA, COAD, READ and CHOL. The PAAD counts file is needed for the PRINCE cohort.
- `../TCGA/data/annotation/gencode.v36.annotation.gtf.gene.probemap` (Ensembl ID to gene symbol) and `../TCGA/data/annotation/gencode.v36.annotation.gtf.gz` (used to keep protein-coding genes).
- `../TCGA/data/mutations/{CANCER}_mc3_gene_level.txt.gz`: gene-level MC3 mutation matrices.
- `data/mc3.v0.2.8.PUBLIC.toil.xena.gz`: MC3 mutation events, used for the TCGA alteration-level labels. Not downloaded by the code. TODO (author): source URL.

Missing files of the first three kinds are downloaded into `Runs/TCGA/data/` by E2M, so this run can be started before Runs/TCGA.

Main packages: pandas, numpy, pyyaml, scikit-learn (metrics), xgboost, pycombat (`from pycombat import Combat`). The prepare scripts download through Python's `urllib` and need network access to cbioportal.org, github.com and ftp.ncbi.nlm.nih.gov.

## Data

### Cohorts

| Cohort key (config.yaml) | Source / accession | Expression type (config) | Download | Driver list | TCGA training cancers | TCGA measure |
|---|---|---|---|---|---|---|
| paad_iatlas_prince_2022 | cBioPortal `paad_iatlas_prince_2022` (PRINCE PDAC trial) | log2 upper-quartile normalized counts | automatic | PAAD | PAAD | counts |
| paad_qcmg_uq_2016 | cBioPortal `paad_qcmg_uq_2016` (QCMG / Bailey 2016) | RNA-seq V2 RSEM continuous | automatic | PAAD | PAAD | tpm |
| luad_cas_2020 | cBioPortal `luad_cas_2020` | RNA-seq FPKM | automatic | LUAD | LUAD | tpm |
| luad_oncosg_2020 | cBioPortal `luad_oncosg_2020` | RNA-seq V2 RSEM z-scores | automatic | LUAD | LUAD | tpm |
| difg_glass | cBioPortal `difg_glass` (GLASS Consortium) | RNA-seq TPM | automatic | GLIOMA | LGG, GBM | tpm |
| pog570_bcgsc_2020__BRCA | cBioPortal `pog570_bcgsc_2020`, samples with ANALYSIS_COHORT = BRCA | RNA-seq RPKM | automatic | BRCA | BRCA | tpm |
| pog570_bcgsc_2020__COLO | same study, ANALYSIS_COHORT = COLO | RNA-seq RPKM | automatic | COLO | COAD, READ | tpm |
| pog570_bcgsc_2020__LUAD | same study, ONCOTREE_CODE = LUAD | RNA-seq RPKM | automatic | LUAD | LUAD | tpm |
| pog570_bcgsc_2020__PAAD | same study, ONCOTREE_CODE = PAAD | RNA-seq RPKM | automatic | PAAD | PAAD | tpm |
| pog570_bcgsc_2020__CHOL | same study, ANALYSIS_COHORT = CHOL | RNA-seq RPKM | automatic | CHOL | CHOL | tpm |
| metabric | cBioPortal `brca_metabric` files, read from `data/external/METABRIC` | Illumina microarray | manual | BRCA | BRCA | tpm |
| GSE39582 | GEO GSE39582 (Marisa / CIT), colon cancer, tumor samples only | GPL570 Affymetrix microarray | automatic | COLO | COAD, READ | tpm |
| GSE31210 | GEO GSE31210 (Okayama), lung adenocarcinoma | GPL570 Affymetrix microarray | automatic | LUAD | LUAD | tpm |

Sample numbers are not stated in the code or config. `prepare_*` writes them to `data/standardized/*/qc_summary.csv`, and `run_drivers.py` writes them to each `manifest.json` (`external_shape`, `tcga_shape`). TODO (author): add n per cohort.

METABRIC download is manual. Put `data_mrna_illumina_microarray.txt`, `data_mutations.txt`, `data_clinical_sample.txt` and `data_clinical_patient.txt` from the cBioPortal `brca_metabric` study in `data/external/METABRIC`. TODO (author): give the exact download source/version.

Driver genes per cancer type (`CANCER_DRIVER_GENES` in `src/driver_genes.py`):
- PAAD: KRAS, TP53, SMAD4, CDKN2A
- LUAD: EGFR, KRAS, TP53, KEAP1, STK11, MET, ERBB2, BRAF
- BRCA: TP53, PIK3CA, GATA3, MAP3K1, CDH1, PTEN, BRCA1, BRCA2, ESR1
- COLO: APC, KRAS, TP53, PIK3CA, BRAF, SMAD4, NRAS
- GLIOMA: IDH1, IDH2, TP53, ATRX, CIC, FUBP1, EGFR, PTEN
- CHOL: IDH1, IDH2, KRAS, TP53, BRAF, ARID1A, SMAD4

The file describes these as the clinically relevant point/indel drivers for each tissue. No fusion or copy-number labels are modeled in this run.

### cBioPortal studies (prepare_cbioportal.py)

- Study metadata and molecular profiles come from the cBioPortal API (`https://www.cbioportal.org/api`). Study files are downloaded from the cBioPortal DataHub GitHub repository (`public/{study_id}`, master branch). The tarball list `DOWNLOAD_BASES` is empty, so the script always uses the GitHub fallback.
- Expression: the `MRNA_EXPRESSION` file with datatype `CONTINUOUS`; if there is none, any `MRNA_EXPRESSION` file is used. Files whose names contain "zscore" are used only when no other file exists. Values are read as given. Rows are gene symbols (`Hugo_Symbol`), and duplicate symbols are averaged.
- Mutations: the `MUTATION_EXTENDED` (MAF) file, filtered to the nonsilent classes listed in `NONSILENT_VARIANT_CLASSES` (Missense, Frame_Shift_Del/Ins, Nonsense, Nonstop, In_Frame_Del/Ins, Translation_Start_Site, Splice_Site, Splice_Region, Start_Codon_Del/Ins/SNP) and to the driver genes plus the genes in `HOTSPOT_PATTERNS`. If the MAF is missing or is a Git LFS pointer, mutations for the driver genes are fetched from the API using the `{study}_sequenced` sample list.
- Gene-level label: 1 if the sample has at least one such mutation in the gene, otherwise 0. The label matrix is indexed by the expression samples, so expression samples with no mutation record get 0.

### GEO series (prepare_geo.py)

- Downloads `{GSE}_series_matrix.txt.gz` from the NCBI GEO FTP and the GPL570 annotation (`GPL570.annot.gz`).
- Expression: the series-matrix values are used as given (no transform). Probes are mapped to the first gene symbol in the GPL570 "Gene symbol" column and averaged per gene.
- GSE39582 labels come from the sample characteristics. TP53, KRAS and BRAF are 1 for "M" and 0 for "WT", and other values are missing (not evaluable). `is_tumor` is 0 when dataset = "Non Tumoral". MMR (dMMR/pMMR) and CIMP (+/-) are also parsed into `driver_labels.csv.gz`, but they are not in the gene-level label matrix and are not modeled.
- GSE31210 labels come from "gene alteration status". EGFR = "EGFR mutation +", KRAS = "KRAS mutation +" and ALK = "ALK-fusion +", with all other values counted as 0. ALK is not in the LUAD driver list, so it is not modeled.
- GEO cohorts have no per-variant records, so they produce no alteration-level labels.

### Alteration-level labels (alterations.py)

The labels come from mutation events: the external cBioPortal `mutations_long.csv.gz`, the METABRIC MAF, and for TCGA the MC3 events file (barcodes are trimmed to the sample ID, e.g. TCGA-XX-XXXX-01A). For each event, the variant class, protein change and coding change are joined into one upper-case text string, and the labels are matched against that string (and, where given, the genomic start position):
- EGFR_classic_activating: EGFR exon 19 deletion-like event (DEL/DELINS text at E746, L747, T751, S752 or P753, or start position 55242400-55242560) or L858R.
- EGFR_exon20ins: EGFR in-frame insertion/duplication at residues A763-V774, or start position 55248900-55249180.
- KRAS_G12C; KRAS_hotspot_broad: KRAS G12, G13, Q61 or A146.
- BRAF_V600E; BRAF_V600_any.
- PIK3CA_hotspot_core: E542, E545 or H1047. PIK3CA_hotspot_extended: these plus N345, C420 or Q546.
- ESR1_LBD: Y537, D538, E380Q, S463P or L536.
- IDH1_R132; IDH2_R140_R172.
- ERBB2_TKD_or_exon20: Y772, G776, D769, V777, L755 or A775, or an in-frame insertion at these residues or P780. ERBB2_activating_extended: these plus S310.
- FGFR3_susceptible_point: S249, R248, Y373, G370 or K650.
- MET_ex14: MET splice event with protein change X1000-X1039 (text "X100"-"X103") or start position 116412035-116412055.

A cohort is tested on an alteration label only if one of the label's genes is in the cohort's driver list.

### Integration with TCGA (transfer.py)

- TCGA (E2M): protein-coding genes only, duplicate symbols summed. Xena values are converted back to the linear scale (2^x - 1, no CPM), and samples must be present in both expression and MC3 (one vial per tumour). Every gene mutated in at least one training sample is a label.
- External filters: samples must have both expression and mutation labels. For POG570, samples are kept by the clinical column and values in `filter`. For GSE39582 (`tumor_only: true`), only samples with `is_tumor` = 1 are kept.
- log1p (natural log, after clipping negative values to 0) is applied to each side when the config says so. TCGA: always. External: true for paad_qcmg_uq_2016, luad_cas_2020, difg_glass and the POG570 splits; false for PRINCE, luad_oncosg_2020, METABRIC and the GEO series.
- Genes: duplicate symbols are averaged and only genes shared by TCGA and the cohort are kept. A gene is dropped if it has any non-finite value, has zero variance over all samples, or has zero variance inside any batch of 2 or more samples.
- ComBat (pycombat `Combat().fit_transform` with default settings, no covariates) is fitted on TCGA and the cohort together. Batches are the TCGA cancer type (`TCGA_<CANCER>`, so LGG/GBM and COAD/READ are separate batches) and the external cohort key.

### Targets, model and evaluation

- A target (gene or alteration label) is selected if it has a label in TCGA and in the cohort, at least 3 TCGA positives (`min_tcga_positive`) and at least 5 external positives (`min_external_positive`). The reasons for skipping a target are written to `qc/target_selection.csv`.
- One XGBClassifier is trained per target on all TCGA samples of the matched cancer type(s), with n_estimators 300, learning_rate 0.1, random_state 42, n_jobs -1, eval_metric logloss and scale_pos_weight = negatives/positives. There is no TCGA cross-validation in this run.
- Prediction: probability on the external cohort; class = probability >= 0.5.
- Metrics, computed on external samples with a non-missing label: ROC AUC, AUPRC (average precision), normalized AUPRC = (AUPRC - prevalence) / (1 - prevalence), accuracy, precision, recall, F1, MCC and the confusion counts. A target is `evaluable` when its evaluable samples contain both classes.

## Outputs

`data/standardized/<study or GSE>/` (from the prepare scripts):
- `expression.csv.gz`, `mutations_gene_level.csv.gz`, `clinical_sample.csv.gz`, `target_gene_counts.csv`, `qc_summary.csv`.
- cBioPortal only: `mutations_long.csv.gz` (used for alteration labels), `mutations_hotspot_long.csv.gz`, `api_study_metadata.json`, `api_molecular_profiles.json`.
- GEO only: `driver_labels.csv.gz` (used for the GSE39582 tumor filter) and `annotation_term_hits.csv`.
- `data/standardized/phase1_qc_summary.csv` and `phase2_geo_qc_summary.csv` (QC summaries over all studies).

`output/` (from run_drivers.py):
- `xgboost_gene/` and `xgboost_alteration/`, with one folder per cohort containing `manifest.json`, `qc/target_selection.csv`, `qc/features.csv`, `qc/batch_counts.csv`, `predictions/predictions.csv.gz`, `predictions/probabilities.csv.gz` and `metrics/gene_metrics.csv`. Each mode folder also has `run_summary.csv`, and `xgboost_alteration/` has `alteration_label_definitions.csv`.
- `summary/xgboost_gene_target_metrics.csv` and `summary/xgboost_alteration_target_metrics.csv`: target selection merged with metrics, for all configured targets.
- `summary/clinical_driver_results.csv`: both modes combined, with columns cohort_id, label, cancer, model_type, label_mode, target, alteration_definition, external_n_positive, external_n_evaluable, prevalence, roc_auc, auprc, normalized_auprc and evaluable. `Tables/supplementary_tables.py` reads it, keeps the rows with `evaluable` = True and writes them as Supplementary Table S3c (`S3c_driver_validation`).
- `run_manifest.json`: completion time and the number of cohorts per mode.
