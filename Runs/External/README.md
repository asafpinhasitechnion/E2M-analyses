# External cohorts

This analysis trains E2M on TCGA and evaluates mutation predictions in eight external cohorts: CPTAC/CMI, ImmunoPOG, METABRIC, Hugo, Liu, Riaz, Van Allen, and Morrison. The results are used in Figure 4 and Supplementary Table S3.

## Getting started

**Run the TCGA analysis first.** Install the packages as in the top-level README (`requirements.txt`), and follow `Runs/TCGA/README.md` to generate the training data and cross-validation results.

From `Runs/External`, run:

```bash
python src/run_external.py                              # All cohorts and configured methods
python src/run_external.py --cohorts RIAZ --methods sample_rank
python src/run_external.py --normalization none         # TCGA counts without CPM
```

Cohort settings are in `config/e2m.yaml`, including the matching TCGA cancer types, expression inputs, data scale, and integration methods. 
All cohorts use `sample_rank` (genes ranked within each sample), except Morrison, which uses `rank` (each gene ranked across the samples of its own cohort), since its data are already batch-corrected logCPM across studies.

By default, the melanoma cohorts use the **top 20 TCGA-SKCM mutation targets**, ranked by normalized AUPRC in `Runs/TCGA/output/SKCM/cv/metrics.csv`. Use `--top-n` to change this number.

## How it works

1. **Train on TCGA.** The pipeline loads TCGA expression and mutation data with `e2m.Dataset.from_tcga` and trains a separate E2M model for each cohort and integration method, using the configured cancer types. Outside the melanoma cohorts, targets must be mutated in at least 3% and 10 TCGA training samples.
2. **Match expression data.** Only genes shared by TCGA and the external cohort are used. Integration methods include:
   - `none`: log1p of CPM, of counts (without CPM), or of TPM; cohorts already on a log scale are used as given; no cross-cohort correction.
   - `combat`: batch-correct log expression across TCGA and the external cohort using ComBat.
   - `rank`: rank each gene across samples separately in TCGA and the external cohort.
   - `sample_rank`: rank genes within each sample, independently of CPM.
3. **Predict and evaluate.** Coding mutations are defined using the TCGA MAF classes or equivalent Sequence Ontology/Oncotator annotations (including missense, nonsense, frameshift, in-frame, splice, and start/stop changes). A profiled sample is labeled 1 if a gene has a coding mutation and 0 otherwise. Samples without mutation profiling retain their RNA and clinical data but have missing labels and are excluded from mutation metrics.

## External data

Data are stored in `data/external/<folder>/`. Most public datasets are downloaded by the pipeline. **CPTAC/CMI data and the supplementary Excel tables for ImmunoPOG, Hugo, and Van Allen must be supplied manually.**

- **CPTAC/CMI:** Download STAR gene counts, masked somatic MAFs, the sample sheet, `sample.tsv`, and `clinical.tsv` from NCI GDC projects CPTAC-2, CPTAC-3, CMI-MBC, and CMI-MPC (original cart: 2026-02-15). Prepare `expression_counts.tsv.gz` from protein-coding, unstranded counts and `mutations_merged.maf.gz` from the MAFs. Results are grouped by primary site, with GDC project as batch.
- **ImmunoPOG:** Expression counts and coding mutations are downloaded from [BCGSC](https://www.bcgsc.ca/downloads/immunoPOG/). Add Pender et al. (2021) Supplementary Table 1, `10780432ccr201163-sup-240190_4_supp_6604687_qh8c2m.xlsx`, for clinical data. Response is defined as DCB versus NCB.
- **METABRIC:** Downloaded from the [cBioPortal datahub](https://github.com/cBioPortal/datahub) (`brca_metabric`, commit `a022eca`). Microarray expression; mutation data cover a 173-gene panel, so genes outside the panel have missing labels. Models train on TCGA-BRCA TPM.
- **Hugo:** RNA counts come from GEO [GSE78220](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE78220). Add Hugo et al. (2016) Supplementary Table S1, `NIHMS765463-supplement-5.xlsx`, for mutations and clinical data.
- **Liu:** Downloaded from the [authors' repository](https://github.com/vanallenlab/schadendorf-pd1) (commit `536010a`). Includes mutation profiling and OS, PFS, and response annotations from the study.
- **Riaz:** Downloaded from the [authors' repository](https://github.com/riazn/bms038_analysis) (commit `137111c`). Includes pre- and on-treatment RNA, but only pre-treatment RNA samples with matching exomes receive mutation labels. Clinical endpoints are retained for both timepoints.
- **Van Allen:** TPM expression is downloaded from the [authors' repository](https://github.com/vanallenlab/VanAllen_CTLA4_Science_RNASeq_TPM) (commit `ca459ce`). Add Van Allen et al. (2015) Supplementary Tables S1 (`tables1.mutation_list_all_patients.xlsx`) and S2 (`tables2.clinical_and_genome_characteristics_each_patient.xlsx`). Two patients without exomes (Pat20 and Pat91) had clinical data but no mutation labels. Models train on TCGA-SKCM TPM.
- **Morrison:** Downloaded from the [authors' repository](https://github.com/ParkerICI/MORRISON-1-public) (commit `a4ac597`), using the supplied batch-corrected logCPM. This combined dataset **overlaps with Liu, Riaz, and Van Allen**. OS/PFS times are available, but event status is not.

## Outputs

Each run writes to `output/<cohort>/<method>/`, or `output/counts/<cohort>/<method>/` when using `--normalization none`.

| File | Contents |
| --- | --- |
| `metrics.csv` | Mutation prediction metrics per target |
| `site_metrics.csv` | Per-site metrics (CPTAC/CMI by primary site, ImmunoPOG by tumour type, METABRIC by detailed cancer type; sites with ≥ 20 samples) |
| `probabilities.csv.gz`, `labels.csv.gz` | Predictions and observed mutation labels |
| `clinical.csv.gz` | Clinical data |
| `embeddings_tcga.csv.gz`, `embeddings_external.csv.gz` | Model embeddings |
| `expression_external.csv.gz` | Expression matrix passed to the model after integration |
| `umap_before.csv`, `umap_after.csv` | UMAP coordinates before and after integration |
| `model/` | Trained E2M model |
| `manifest.json` | Run settings and metadata |

`output/runs_summary.csv` summarizes all runs.