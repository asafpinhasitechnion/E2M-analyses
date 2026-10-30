# Single-cell analyses

Predicts each cell's perturbation (genetic or drug) from its expression on eight scPerturb datasets, and DepMap mutation status and TMB from CCLE single-cell expression. Results feed Figure 5 (`Figures/Figure5/Figure5.ipynb`) and Supplementary Table S4; dataset sizes are in Supplementary Table S5.

## Running

From `Runs/SingleCell`:

```
python src/run_all.py                          # download missing files, then run all steps
python src/run_all.py --only ccle              # run selected steps only
python src/run_all.py --skip ccle --no-download
python src/run_all.py --check-md5              # verify md5 of downloaded files
```

Steps (`src/steps/`), in order:

1. `ccle`: CCLE log-TMB regression (XGBoost) and mutation prediction for the 200 most frequently mutated genes (multitask network).
2. `perturbation_multitask`: multitask network on Adamson (3 files), McFarland, Replogle K562 and RPE1, Frangieh and Zhao.
3. `frangieh_per_gene`: one XGBoost classifier per perturbation on Frangieh.
4. `tian_guide_transfer`: Tian CRISPRa, train on one guide and test on the other.
5. `frangieh_conditions`: Frangieh metrics per immune condition (needs steps 2 and 3).
6. `zhao_pr_curves`: Zhao precision-recall curves for etoposide and panobinostat (needs step 2).
7. `plot_inputs`: copies the files used by Figure 5 and Table S4 to `output/plot_inputs/`.

Settings are in `config/config.yaml` (seed 42, 5 folds, XGBoost parameters, file names). `SINGLE_CELL_USE_GPU=0/1` forces XGBoost to CPU/GPU (default: GPU if available); `SINGLE_CELL_MULTITASK_KEYS=adamson,zhao,...` restricts step 2 to some datasets.

## Data

- scPerturb (Zenodo record 7041849): downloaded automatically to `data/` when missing or incomplete; interrupted downloads resume.
- CCLE: placed manually in `data/CCLE/` (the step is skipped otherwise): `CCLE_adata.h5ad` (Broad Single Cell Portal, SCP542; raw counts, `obs["Model_ID"]`) and `OmicsSomaticMutations.csv` (DepMap, matched by `ModelID`).

## Methods

Common to all datasets (`src/preprocess.py`):

- Input: raw counts in `adata.X`, kept sparse; normalized to 10,000 counts per cell and log1p-transformed (Scanpy).
- Features: HGNC protein-coding genes (matched by Ensembl ID where available, otherwise by symbol) detected in at least 3 cells. No HVG selection, batch correction or cell QC beyond the dataset-specific steps below.
- Labels: one binary output per perturbation with at least 300 cells. Each perturbed cell is positive for its perturbation and negative for the others; control cells are negative for all targets. Cells with other perturbations are removed.
- Metrics: AUPRC on held-out cells, and normalized AUPRC = (AUPRC - prevalence) / (1 - prevalence). A fold without positives for a target has no AUPRC.

Models:

- Multitask network (`src/models/multitask.py`, `src/models/presets.py`): hidden layers 512 and 256 with ReLU and dropout 0.3, one sigmoid output per target, BCE with per-target positive weights (neg/pos, clipped to 1-30), Adam (learning rate 1.5e-4, weight decay 8e-4), batch size 1024, up to 100 epochs, learning-rate reduction on plateau, gradient clipping 1.0, and early stopping (patience 6) on mean validation AUPRC over a random 20% of the training cells. No input scaling.
- XGBoost (`src/models/per_gene.py`): `n_estimators=300`, `learning_rate=0.1`, `tree_method=hist`, as for the bulk analyses.

Dataset-specific:

| Dataset | Processing | Cross-validation |
|---|---|---|
| Replogle K562, RPE1 | label column `gene` | 5-fold, cell level |
| Adamson 10X001, 10X005 (UPR), 10X010 | each file a separate run; constructs mapped to target genes with a fixed table (`src/datasets/adamson.py`), so a cell can be positive for several genes; dataset-specific non-targeting constructs are controls | 5-fold, cell level |
| McFarland | normal-quality cells only (`cell_quality`); in the hashed trametinib experiment, DMSO and untreated cells (by `hash_tag`) are controls and multiplet or unknown hashes are removed | 5-fold, grouped by cell line |
| Zhao | etoposide and panobinostat only; the other drugs were each given in a single sample, so they cannot be evaluated with sample-grouped CV | 5-fold, grouped by sample |
| Frangieh | single-guide cells only, as in the original analysis; immune condition (`perturbation_2`) used only to report metrics per condition. Per-gene XGBoost: one classifier per perturbation, with all other cells as negatives | multitask: 5-fold, cell level; per-gene: stratified 5-fold |
| Tian CRISPRa | genes with exactly two guides, each in at least 100 cells; for each gene, XGBoost trained on one guide and tested on the other, in both directions; control cells split into two halves for training and testing | by guide |
| CCLE | cell-line mutations and log1p(TMB) assigned to all cells of the line; coding mutations only (`src/datasets/ccle.py`); cells of lines without coding mutations removed; targets are the 200 genes mutated in the most lines (at least 20); TMB predictions also averaged per cell line | 5-fold, grouped by cell line |

## Outputs

- `output/runs/<run name>/`: one folder per run, with `run_metadata.json`, per-target metrics (pooled out-of-fold and per fold), out-of-fold predictions, labels and folds, and, for the multitask network, cell embeddings (last hidden layer) and head weights.
- `output/plot_inputs/`: the files used by Figure 5 and Table S4, one subfolder per dataset (list in `src/steps/plot_inputs.py`). Figure 5 also reads the Replogle embeddings and head weights directly from `output/runs/`.
