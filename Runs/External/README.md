# External cohorts

Trains E2M on TCGA and predicts mutations in eight external cohorts (CPTAC/CMI, ImmunoPOG, METABRIC, and the melanoma cohorts Hugo, Liu, Riaz, Van Allen, Morrison), for several ways of bringing TCGA and the cohort onto the same scale. Outputs are used by Figure 4 and Supplementary Table S3.

## Running

Install E2M as in `Runs/TCGA/README.md`, plus `inmoose`, `umap-learn`, `openpyxl` and `requests`. From `Runs/External`:

```
python src/run_external.py                                   # all cohorts, all their methods
python src/run_external.py --cohorts RIAZ --methods combat_seq
```

Settings are in `config/e2m.yaml`: per cohort the TCGA cancer types, the TCGA expression file, the scale of the cohort's expression, and the integration methods (the first is the main result). TCGA data, preprocessing (CPM) and the model come from `Runs/TCGA/config/e2m.yaml`; the melanoma cohorts mark the top 20 targets of the TCGA-SKCM run (`Runs/TCGA/output/cpm/SKCM/cv/metrics.csv`), so Runs/TCGA must be run first. `check_cohorts.ipynb` shows the processing of each cohort step by step.

## Method

- TCGA training data: `e2m.Dataset.from_tcga` (as in Runs/TCGA), linear counts or TPM. Targets: genes mutated in at least 3% and at least 10 of the TCGA training samples.
- Integration, on the genes shared with the cohort (`src/integration.py`, TCGA is one batch): `none` (each sample on its own: CPM + log1p, or log1p of TPM); `combat` / `combat_ref` (ComBat on the log values, jointly or with TCGA as the reference batch); `combat_seq` / `combat_seq_ref` (ComBat-seq on counts, then CPM + log1p); `rank` (each gene ranked across samples within each side).
- Model: an E2M model trained on all TCGA samples of the cohort's cancer types and applied to the cohort; one model per cohort and method.
- Labels: a sample is profiled if it (or its patient) has rows in the cohort's mutation data; it is 1 for a gene with a coding call and 0 otherwise. Coding calls are the 9 MAF classes of the TCGA labels (Missense, Nonsense, Frame_Shift_Del/Ins, In_Frame_Del/Ins, Splice_Site, Translation_Start_Site, Nonstop) or their equivalents in other annotations (Sequence Ontology: missense, stop gained/lost, start lost, frameshift, in-frame insertion/deletion, splice acceptor/donor, protein altering; Oncotator: Start_Codon_SNP/Del/Ins, Stop_Codon_Del). Samples that are not profiled keep their clinical data and have missing labels, which are left out of the metrics.
- Every RNA sample with clinical data is kept (for the embeddings and clinical analyses).

## Data

Files go under `data/external/<folder>/`. METABRIC, Liu, Riaz and Morrison are downloaded by the code, and so are the expression and mutations of ImmunoPOG and the expression of Hugo and Van Allen; the remaining files (CPTAC/CMI and the supplementary tables) are placed manually.

### CPTAC/CMI (`CMI_CPTAC/`)
- Source: NCI GDC, projects CPTAC-2, CPTAC-3, CMI-MBC, CMI-MPC (cart of 2026-02-15): STAR gene counts (`*.rna_seq.augmented_star_gene_counts.tsv`), masked somatic MAFs (`*.wxs.aliquot_ensemble_masked.maf.gz`), and the sample sheet, `sample.tsv` and `clinical.tsv` of the same cart.
- The counts were combined into `expression_counts.tsv.gz` (protein-coding genes by symbol, `unstranded` counts, one column per file UUID) and the MAFs into `mutations_merged.maf.gz` (with the source file name on each row).
- Processing: the sample sheet links each file to its sample(s); `sample.tsv` says whether a sample is tumour or normal. Normal RNA files are not used; RNA files of the same sample(s) are averaged. An RNA sample gets the mutations of the MAF(s) of the same tumour sample; without one, those of all the patient's MAFs if it is the patient's only RNA sample. MAFs without any call are not used. One row per patient: a sample with mutation data first, then a Primary tumour, then the first sample ID.
- Clinical, combined over a patient's rows: OS (dead: days to death, otherwise the longest follow-up) and recurrence (any progression or recurrence: earliest days to recurrence, otherwise the longest follow-up). Tumour type: TCGA-style code from primary site and primary diagnosis (`cancer`). Batch: GDC project.

### ImmunoPOG (`IMMUNOPOG/`)
- Source: BCGSC, www.bcgsc.ca/downloads/immunoPOG/ (`ImmunoPOG_expression_raw_counts_all.txt.gz`, `ImmunoPOG_coding_mutations.txt.gz`, downloaded), and Pender et al. 2021 (Clin Cancer Res 27:202) Supplementary Table 1 (`10780432ccr201163-sup-240190_4_supp_6604687_qh8c2m.xlsx`, clinical), placed manually.
- Processing: raw counts by HGNC symbol (duplicates averaged); 98 patients, all with RNA, WES and clinical data. Mutations by Sequence Ontology / SnpEff consequence (one patient has two tumour libraries, combined). Response: clinical benefit (DCB vs NCB). Tumour type as given in the table (`Cancer`).

### METABRIC (`METABRIC/`, downloaded)
- Source: cBioPortal study `brca_metabric`, from the cBioPortal datahub (github.com/cBioPortal/datahub, commit a022eca): `data_mrna_illumina_microarray.txt`, `data_mutations.txt`, `data_gene_panel_matrix.txt`, `data_clinical_sample.txt`, `data_clinical_patient.txt`, and the panel gene list `reference_data/gene_panels/data_gene_panel_metabric_173.txt`.
- Processing: Illumina microarray values (already log-scaled) by symbol (1,980 samples). Mutations from targeted sequencing of 173 genes: the samples in the gene panel matrix are profiled (1,904 with expression), and genes outside the panel have missing labels. OS from `OS_MONTHS` (converted to days) and `OS_STATUS`. Trained on TCGA-BRCA TPM.

### Hugo (`HUGO/`)
- Source: GEO GSE78220, NCBI-generated raw counts (`GSE78220_raw_counts_GRCh38.p13_NCBI.tsv.gz`, downloaded), and Hugo et al. 2016 supplementary Table S1 (`NIHMS765463-supplement-5.xlsx`: S1A/S1B clinical, S1D mutations), placed manually.
- Processing: GSM IDs mapped to patients; Entrez IDs mapped to symbols (mygene, cached in `entrez_gene_symbol_cache.csv`); mutations by Oncotator class (28 RNA samples, 27 patients). Pt27 has two RNA samples: Pt27A, the 1st biopsy (R upper arm, the location in S1A), gets Pt27's mutations; Pt27B, the 2nd biopsy (R upper back), has missing labels. Both get Pt27's clinical data.

### Liu (`LIU/`, downloaded)
- Source: the authors' repository github.com/vanallenlab/schadendorf-pd1 (commit 536010a): `rnaseq_rawcounts.txt` and `all_muts_12_1_2020_ref_alt_counts_added.maf` (from `data/addData.zip`), `Supplemental_Table_1_wAge.tsv` (= Supplementary Table 1).
- Processing: raw counts (121 patients with RNA); mutations by Oncotator class (all 144 patients profiled; the labels match the authors' `geneTumorMatrix.txt`); OS, PFS and response from the supplementary table (binary response: CR/PR/MR vs SD/PD).

### Riaz (`RIAZ/`, downloaded)
- Source: the authors' repository github.com/riazn/bms038_analysis (commit 137111c): `CountData.BMS038.txt`, `SampleTableCorrected.9.19.16.csv`, `pre_therapy_nonsynonmous_mutations.csv` (= Table S3), `bms038_clinical_data.csv`.
- Processing: the RNA samples of the authors' sample table (pre- and on-treatment; it leaves out the ".2" libraries and Pt68_On). The mutations are from the pre-treatment exome, so only pre-treatment RNA samples of exome patients get labels. OS, PFS and response per patient for all RNA samples (`OS_SOR` / `PFS_SOR` are censoring flags: event = 1 - flag).

### Van Allen (`VAN_ALLEN/`)
- Source: expression from the authors' repository github.com/vanallenlab/VanAllen_CTLA4_Science_RNASeq_TPM (commit ca459ce, `TPM_RSEM_VAScience2015.txt`, downloaded); Van Allen et al. 2015 supplementary Tables S1 (`tables1.mutation_list_all_patients.xlsx`, mutations) and S2 (`tables2.clinical_and_genome_characteristics_each_patient.xlsx`, clinical), placed manually.
- Processing: RSEM TPM (STAR, Gencode v19; 42 pre-treatment patients), Ensembl IDs mapped to symbols (mygene, cached in `ensembl_gene_symbol_cache.csv`); mutations by Oncotator class. Clinical from the exome sheet of S2, plus the transcriptome sheet for the 2 RNA patients without exome (Pat20, Pat91: kept with OS, no labels). Trained on TCGA-SKCM TPM.

### Morrison (`MORRISON/`, downloaded)
- Source: the authors' repository github.com/ParkerICI/MORRISON-1-public (commit a4ac597): ComBat batch-corrected logCPM (all samples), RNA, subject and WES metadata, WES variants.
- Processing: logCPM as given (442 RNA samples, 371 subjects; six cohorts combined by the authors, including Liu, Van Allen and CheckMate 038 = Riaz). Mutations by Sequence Ontology consequence per WES sample; an RNA sample gets the WES sample of the same subject and timepoint, otherwise the subject's WES if it is the subject's only RNA sample (209 labelled). OS/PFS have no event status; the repository gives months, but the Liu and Van Allen subjects are in days, so only the others are converted to days.

## Outputs

`output/<cohort>/<method>/`: `metrics.csv` (per target; `top_tcga_target` marks the melanoma top 20), `site_metrics.csv` (CPTAC/CMI and ImmunoPOG), `probabilities.csv.gz`, `labels.csv.gz`, `clinical.csv.gz`, `embeddings_tcga.csv.gz`, `embeddings_external.csv.gz`, `expression_external.csv.gz` (the integrated expression the model was applied to), `umap_before.csv` / `umap_after.csv`, `model/` (the trained E2M model), `manifest.json`. `output/runs_summary.csv` summarizes all runs.
