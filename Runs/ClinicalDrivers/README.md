# Clinical Driver Alterations

This analysis trains XGBoost models on TCGA gene expression to predict cancer driver mutations and tests them on external cBioPortal and GEO cohorts. Predictions cover both **driver genes** (for example, any coding KRAS mutation) and **specific alterations** (such as KRAS G12C or EGFR L858R).

The main integration method is `sample_rank`, which ranks genes within each sample and allows prediction sample-by-sample. For microarray cohorts, results are also compared with ComBat batch correction, which corrects the whole cohort jointly with TCGA, so samples cannot be predicted one by one.

## Running the analysis

From `Runs/ClinicalDrivers`:

```bash
python src/prepare_cohorts.py  # Download and standardize external data
python src/run_drivers.py      # Train TCGA models and evaluate external cohorts
```

Both scripts accept `--only` to run selected sources or cohorts. Raw downloads are removed after standardization unless `prepare_cohorts.py --keep-raw` is used.

**Configuration**

- `config/config.yaml`: Data paths, model settings, target thresholds, and cohort-specific inputs and filters.
- `src/driver_genes.py`: Driver genes by cancer type and eligible coding mutation classes.
- `src/alterations.py`: Definitions of specific clinically relevant alterations.

**Dependencies:** Install the packages as in the top-level README (`requirements.txt`).

## Datasets

| Cohort | Source | Expression | TCGA training reference |
|---|---|---|---|
| `paad_iatlas_prince_2022` | cBioPortal (PRINCE) | Log2 UQ counts | PAAD (counts) |
| `paad_qcmg_uq_2016` | cBioPortal (QCMG) | RSEM | PAAD (counts) |
| `luad_cas_2020` | cBioPortal | FPKM | LUAD (FPKM) |
| `luad_oncosg_2020` | cBioPortal | RSEM z-scores | LUAD (counts, z-scored) |
| `difg_glass` | cBioPortal (GLASS) | TPM | LGG, GBM |
| `metabric` | cBioPortal | Illumina microarray | BRCA |
| `pog570_bcgsc_2020__BRCA` | cBioPortal (POG570) | RPKM | BRCA (FPKM) |
| `pog570_bcgsc_2020__COLO` | cBioPortal (POG570) | RPKM | COAD, READ (FPKM) |
| `pog570_bcgsc_2020__LUAD` | cBioPortal (POG570) | RPKM | LUAD (FPKM) |
| `pog570_bcgsc_2020__PAAD` | cBioPortal (POG570) | RPKM | PAAD (FPKM) |
| `GSE39582` | GEO (colorectal) | Affymetrix GPL570 | COAD, READ |
| `GSE31210` | GEO (lung adenocarcinoma) | Affymetrix GPL570 | LUAD |

TCGA uses TPM unless another scale is listed. POG570 subsets are selected using `ANALYSIS_COHORT` (BRCA, COLO) or `ONCOTREE_CODE` (LUAD, PAAD). GLASS samples belonging to TCGA patients are excluded to avoid overlap with training data. Cancer-specific driver gene lists are defined in `src/driver_genes.py`.

### Data preparation

**TCGA:** Loaded with `e2m.Dataset.from_tcga` using `../TCGA/config/e2m.yaml`. Missing Xena GDC STAR expression, Gencode v36 annotations, MC3 gene-level labels, and MC3 mutation events are downloaded to `../TCGA/data/` (under `expression/`, `annotation/`, `mutations/`, and `mutation_events/`, respectively). Expression is restricted to protein-coding genes and converted from Xena's `log2(x+1)` scale. Counts are converted to CPM for PRINCE, QCMG, and OncoSG; other references use TPM or FPKM as configured.

**cBioPortal:** Data are downloaded from the [DataHub](https://github.com/cBioPortal/datahub) at commit `a766d41` (May 13, 2026). Duplicate gene symbols are averaged, and driver mutations come from MAF files. 

A gene is labeled 1 when a qualifying mutation is present and 0 when profiled without one. Missing sequencing data produce missing labels, including in targeted panels.

**GEO:** Series matrices and GPL570 annotations are downloaded from NCBI. Probes mapping to the same gene are averaged, and non-tumor samples are excluded. `GSE39582` provides TP53, KRAS, and BRAF labels (`M`/`WT`); `GSE31210` provides EGFR and KRAS status. ALK is not included because it is absent from the LUAD driver list. GEO supports **gene-level predictions only**.

## Mutation targets

**Gene-level:** Any qualifying coding mutation in a driver gene, including missense, nonsense, nonstop, frameshift, in-frame indels, splice-site, and translation-start variants. Fusions and copy-number changes are not included.

**Alteration-level:** Specific variants or groups of variants identified from cBioPortal MAFs or TCGA MC3 events:

| Gene | Alterations |
|---|---|
| EGFR | Activating exon 19 deletions / L858R; exon 20 insertions |
| KRAS | G12C; G12/G13/Q61/A146 hotspots |
| BRAF | V600E; all V600 |
| PIK3CA | E542/E545/H1047; extended set including N345/C420/Q546 |
| ESR1 | LBD mutations: Y537/D538/E380Q/S463P/L536 |
| IDH1 / IDH2 | IDH1 R132; IDH2 R140/R172 |
| ERBB2 | TKD/exon 20 alterations; extended set including S310 |
| MET | Exon 14 skipping-associated splice alterations |

Exact matching rules, including genomic positions where needed, are in `src/alterations.py`. Alteration labels are missing when the gene label is missing, and only genes in the cohort's driver list are tested.

## Expression integration and modeling

Only genes shared by TCGA and the external dataset are retained. Expression is transformed according to the cohort configuration: negative values are clipped to zero before `log1p`; TCGA is always transformed. The external side gets log1p only if it's still linear - for LUAD CAS, GLASS, GSE31210, and POG570. PRINCE, QCMG, METABRIC and GSE39582, are all logged. For OncoSG's gene-wise z-scores, TCGA is z-scored per gene as well. Genes with non-finite values or zero variance in any batch (each TCGA cancer type and the cohort) are removed.

Two integration methods are used:

- **`sample_rank` (main result):** Rank genes independently within each sample. No joint batch correction is needed, so individual samples can be predicted directly.
- **`combat` (microarray comparison):** Joint correction with `inmoose.pycombat_norm` (default settings), treating each TCGA cancer type and the external cohort as separate batches. Used for the microarray cohorts: METABRIC, GSE39582, and GSE31210.

Each cohort's `integrations` setting lists its methods, with the first as the main result.

Models are trained separately for each target using all matched TCGA samples. A target needs **at least 5 positive samples in both TCGA and the external cohort**. Targets whose external labels are identical to an earlier target are skipped.

The `XGBClassifier` uses 300 trees, learning rate 0.1, `random_state=42`, `n_jobs=-1`, `eval_metric="logloss"`, and `scale_pos_weight = n_negative / n_positive`. The configured device is `cuda` (GPU and CPU trees may differ slightly).

Evaluation includes **ROC AUC**, **AUPRC** (average precision), and **normalized AUPRC**, calculated as `(AUPRC - prevalence) / (1 - prevalence)`. Only labeled external samples contribute to evaluation. A target is marked `evaluable` when those samples contain both positive and negative cases.

## Outputs

**Standardized data** are saved under `data/standardized/<source>/`:

| File | Contents |
|---|---|
| `expression.csv.gz` | Sample-by-gene expression matrix |
| `mutations_gene_level.csv.gz` | Sample-by-driver-gene labels |
| `clinical_sample.csv.gz` | Sample annotations |
| `mutations_long.csv.gz` | Driver mutation events (cBioPortal only) |

**Model results** are saved under `output/`:

| Path | Contents |
|---|---|
| `integration_<method>/gene/<cohort>/` | Gene-level predictions and metrics |
| `integration_<method>/alteration/<cohort>/` | Alteration-level predictions and metrics |
| `integration_<method>/summary/clinical_driver_results.csv` | All cohorts evaluated with that method |
| `summary/integration_comparison.csv` / `.xlsx` | Sample-rank vs. ComBat comparison for microarray cohorts |
| `summary/clinical_driver_results.csv` | Main results using each cohort's selected integration |

Each cohort result folder includes `targets.csv` (target counts and metrics), `probabilities.csv.gz` (sample-level predictions), and `manifest.json` (settings, sample counts, retained genes, and batch sizes). The summary includes cohort, cancer type, target, alteration definition, label mode, sample counts, prevalence, performance metrics, evaluability, and, for the main summary, integration method.

**Supplementary Table S3c** is generated by `Tables/supplementary_tables.py` from the main summary, keeping only `evaluable = True` results.
