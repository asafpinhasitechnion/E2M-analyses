# Clinical Driver Alterations

This analysis trains XGBoost classifiers on TCGA gene expression to predict cancer driver mutations and evaluates their performance on independent cBioPortal and GEO cohorts.

Predictions are performed at two levels:
- **Gene-level:** Presence of a coding mutation in a driver gene (e.g., KRAS).
- **Alteration-level:** Presence of a specific clinically relevant alteration (e.g., KRAS G12C or EGFR L858R).

Two expression integration methods are compared: **ComBat**, applied jointly to TCGA and each external cohort, and **sample-wise gene ranking (`sample_rank`)**, which enables independent, single-sample predictions.

## 1. Running

Run from `Runs/ClinicalDrivers`:

```bash
python src/prepare_cohorts.py  # Download and standardize external cohorts
python src/run_drivers.py      # Train TCGA models and evaluate external cohorts
```

Both scripts support `--only` to select specific sources or cohorts; with `run_drivers.py --only`, only those cohorts' rows of the summary are replaced. By default, `prepare_cohorts.py` deletes raw downloads after standardization; use `--keep-raw` to retain them.

**Configuration files:**
- `config/config.yaml`: Data paths, XGBoost parameters, target thresholds, and cohort-specific settings (source, driver genes, TCGA reference, expression transformations, sample filters).
- `src/driver_genes.py`: Cancer-specific driver genes and qualifying coding variant classes.
- `src/alterations.py`: Clinically relevant alteration definitions.

**Dependencies:** `pandas`, `numpy`, `pyyaml`, `requests`, `scikit-learn`, `xgboost`, `inmoose`, and E2M.

## 2. Datasets

### External cohorts

| Cohort | Source | Expression | TCGA reference |
|---|---|---|---|
| `paad_iatlas_prince_2022` | cBioPortal, PRINCE | Log2 UQ counts | PAAD (counts) |
| `paad_qcmg_uq_2016` | cBioPortal, QCMG | RNA-seq RSEM | PAAD (counts) |
| `luad_cas_2020` | cBioPortal | RNA-seq FPKM | LUAD (FPKM) |
| `luad_oncosg_2020` | cBioPortal | RNA-seq RSEM z-scores | LUAD (counts, z-scored) |
| `difg_glass` | cBioPortal, GLASS | RNA-seq TPM | LGG, GBM |
| `metabric` | cBioPortal, METABRIC | Illumina microarray | BRCA |
| `pog570_bcgsc_2020__BRCA` | cBioPortal, POG570 | RNA-seq RPKM | BRCA (FPKM) |
| `pog570_bcgsc_2020__COLO` | cBioPortal, POG570 | RNA-seq RPKM | COAD, READ (FPKM) |
| `pog570_bcgsc_2020__LUAD` | cBioPortal, POG570 | RNA-seq RPKM | LUAD (FPKM) |
| `pog570_bcgsc_2020__PAAD` | cBioPortal, POG570 | RNA-seq RPKM | PAAD (FPKM) |
| `GSE39582` | GEO, colorectal cancer | Affymetrix GPL570 | COAD, READ |
| `GSE31210` | GEO, lung adenocarcinoma | Affymetrix GPL570 | LUAD |

TCGA expression uses TPM unless otherwise indicated. POG570 subsets are defined using `ANALYSIS_COHORT` (BRCA, COLO) or `ONCOTREE_CODE` (LUAD, PAAD). GLASS samples of TCGA patients are left out, as these patients are in the training data.

Driver genes are defined separately for PAAD, LUAD, BRCA, COLO, GLIOMA, and CHOL (not used by any cohort) in `src/driver_genes.py`.

### Data acquisition and processing

**TCGA:** Loaded using `e2m.Dataset.from_tcga` with configuration `../TCGA/config/e2m.yaml`. Missing files are downloaded to `../TCGA/data/`:

- `expression/TCGA-{CANCER}.star_{tpm|fpkm|counts}.tsv.gz`: Xena GDC STAR expression.
- `annotation/gencode.v36.annotation.gtf.gene.probemap` and `gencode.v36.annotation.gtf.gz`: Gene annotations.
- `mutations/{CANCER}_mc3_gene_level.txt.gz`: Gene-level mutation labels.
- `mutation_events/{CANCER}_mc3.txt.gz`: Mutation events for alteration-level labels.

Expression is restricted to protein-coding genes and converted from Xena's `log2(x+1)` scale to linear values. TCGA counts (PRINCE, QCMG, OncoSG) are converted to CPM; the other cohorts use TPM or FPKM as given.

**cBioPortal:** Study data are downloaded from the DataHub GitHub repository at a fixed commit (`a766d41`, 2026-05-13; `DATAHUB_COMMIT` in `src/prepare_cohorts.py`). Continuous mRNA expression is preferred over z-scores, and duplicate gene symbols are averaged. Coding mutations are extracted from MAF files using E2M variant classes and the configured driver genes.

Gene labels are 1 for mutated and 0 for wild type. Samples without sequencing data receive missing labels. For targeted panels, genes not covered by the panel are also missing (e.g., ESR1 in METABRIC). The next DataHub commit lost the Git LFS file of the QCMG mutations.

**GEO:** Series matrices and GPL570 annotations are downloaded from NCBI. Probes are mapped to gene symbols and averaged; non-tumor samples are removed.

- `GSE39582`: TP53, KRAS, and BRAF labels from `M`/`WT` annotations.
- `GSE31210`: EGFR and KRAS mutation labels from alteration-status annotations; ALK is excluded because it is not in the LUAD driver list.

GEO cohorts provide gene-level labels only.

## 3. Mutation targets

**Gene-level:** A sample is positive if it contains at least one qualifying coding mutation in the driver gene. Qualifying classes include missense, nonsense, nonstop, frameshift, in-frame indels, splice-site, and translation-start mutations. Fusions and copy-number changes are not modeled.

**Alteration-level:** Labels are derived from cBioPortal MAF or TCGA MC3 mutation events, using variant classes, protein/coding changes, and genomic positions where applicable.

| Gene | Alteration targets |
|---|---|
| EGFR | Classic activating (exon 19 deletions, L858R); exon 20 insertions |
| KRAS | G12C; broad hotspots (G12, G13, Q61, A146) |
| BRAF | V600E; any V600 |
| PIK3CA | Core hotspots (E542, E545, H1047); extended (+N345, C420, Q546) |
| ESR1 | LBD mutations (Y537, D538, E380Q, S463P, L536) |
| IDH1 / IDH2 | IDH1 R132; IDH2 R140/R172 |
| ERBB2 | TKD/exon 20 alterations; extended activating alterations (+S310) |
| MET | Exon 14 skipping-associated splice alterations |

Exact matching definitions, including genomic coordinates, are implemented in `src/alterations.py`. Alteration labels inherit missingness from their corresponding gene labels and are evaluated only for genes included in the cohort's driver list.

## 4. Expression integration

TCGA and each external cohort are processed using two approaches.

**Shared preprocessing:**
1. Retain genes present in both datasets.
2. Apply `log1p` after clipping negative values to zero, according to configuration. TCGA is always transformed; external datasets are transformed for LUAD CAS, GLASS, GSE31210, and POG570.
3. For OncoSG (z-scores per gene), z-score TCGA per gene as well.
4. Remove genes with missing or non-finite values or zero variance in any batch.

**ComBat:** Joint batch correction using `inmoose.pycombat_norm` with default settings. TCGA cancer types are separate batches (e.g., COAD and READ), alongside the external cohort.

**Sample rank:** Independently rank genes within each sample. This avoids joint batch correction and allows prediction on individual samples.

ComBat, the first method in `integrations` (config), is the main result in `output/`; sample rank is written to `output/integration_sample_rank/`.

## 5. Model training and evaluation

A target is selected if it has:
- At least **3 TCGA positives** (`min_tcga_positive`).
- At least **5 external positives** (`min_external_positive`).

Targets with identical external labels to an earlier target are skipped.

A separate `XGBClassifier` is trained for each target using all matched TCGA samples, with:

```python
n_estimators=300
learning_rate=0.1
random_state=42
n_jobs=-1
eval_metric="logloss"
scale_pos_weight=n_negative / n_positive
device="cuda"  # GPU trees differ slightly from CPU trees
```

Models are evaluated on external samples with available mutation labels using:

- **ROC AUC**
- **AUPRC** (average precision)
- **Normalized AUPRC:** `(AUPRC - prevalence) / (1 - prevalence)`

Targets are marked `evaluable` only when the external samples with available labels contain both classes.

## 6. Outputs

### Standardized data

Generated by `prepare_cohorts.py` under `data/standardized/<source>/`:

| File | Contents |
|---|---|
| `expression.csv.gz` | Samples × genes expression |
| `mutations_gene_level.csv.gz` | Samples × driver genes mutation labels |
| `clinical_sample.csv.gz` | Sample annotations |
| `mutations_long.csv.gz` | Coding driver mutation events (cBioPortal only) |

### Model results

Generated by `run_drivers.py` under `output/`:

| Path | Contents |
|---|---|
| `gene/<cohort>/` | Gene-level results |
| `alteration/<cohort>/` | Alteration-level results |
| `summary/integration_comparison.csv` / `.xlsx` | ComBat vs. sample-rank metrics and differences |
| `summary/clinical_driver_results.csv` | Consolidated results across cohorts and label modes |

Each cohort result directory includes:
- `targets.csv`: Target selection, counts, and metrics.
- `probabilities.csv.gz`: External sample predictions.
- `manifest.json`: Cohort settings, sample numbers, retained genes, and batch sizes.

`clinical_driver_results.csv` contains cohort identifiers, cancer type, label mode, target, alteration definition, positive/evaluable counts, prevalence, ROC AUC, AUPRC, normalized AUPRC, and evaluability.

**Supplementary Table S3c** is generated by `Tables/supplementary_tables.py` from the main (ComBat) result, retaining only rows with `evaluable = True`.