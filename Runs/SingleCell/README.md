# Single-cell perturbation datasets

This run trains models that predict, for each cell, which perturbation it received (genetic or drug) from its expression profile, on eight scPerturb datasets, and predicts DepMap mutation status and tumor mutational burden from CCLE single-cell expression. Results feed Figure 5 (`Figures/Figure5/Figure5.ipynb`, `figure5_helpers.py`) and Supplementary Table S4 (`Tables/supplementary_tables.py`).

## Running

From `Runs/SingleCell`:

```
python src/run_all.py                          # download missing files, then run all steps
python src/run_all.py --only ccle              # run selected steps only
python src/run_all.py --skip ccle --no-download
python src/run_all.py --check-md5              # verify md5 of downloaded files
```

Steps, in order (`src/steps/`):

1. `ccle` - CCLE log-TMB regression (XGBoost) and top-150 mutated-gene multitask model.
2. `perturbation_multitask` - multitask neural network CV on Adamson (3 files), McFarland, Replogle K562, Replogle RPE1, Frangieh, Zhao.
3. `frangieh_per_gene` - one XGBoost classifier per perturbation on Frangieh.
4. `tian_guide_transfer` - Tian CRISPRa, train on one guide and test on the other.
5. `frangieh_conditions` - Frangieh metrics per immune condition (needs steps 2 and 3).
6. `zhao_pr_curves` - Zhao PR curves for etoposide and panobinostat (needs step 2).
7. `plot_inputs` - copy the files used by Figure 5 and Table S4 into `output/plot_inputs`.

Environment variables:

- `SINGLE_CELL_USE_GPU=0/1` forces XGBoost to CPU/GPU (default: GPU if the installed XGBoost has CUDA). The neural network uses CUDA whenever `torch.cuda.is_available()`.
- `SINGLE_CELL_MULTITASK_KEYS=adamson,zhao,...` restricts step 2 to some datasets (keys as in `dataset_files` in the config).

Config: `config/config.yaml`

- `random_state: 42`, `cv_folds: 5`
- `multitask_preset: simple_wide_run` (defined in `src/models/presets.py`)
- `preprocess_target_sum: 10000` (normalize_total target before log1p, all datasets including CCLE)
- `xgboost`: `n_estimators: 300`, `learning_rate: 0.1`, `n_jobs: -1`
- `zenodo_record_id` and `dataset_files` (scPerturb file names), `ccle_files` (CCLE inputs)

Main packages imported: scanpy, numpy, pandas, scipy, scikit-learn, xgboost, torch, pyyaml. TODO: versions.

Models:

- Multitask network (`src/models/multitask.py`, preset `simple_wide_run`): fully connected, hidden layers 512 and 256 with ReLU and dropout 0.3, one sigmoid output per target, BCE loss with per-target positive weight (neg/pos, clipped to 1-30), Adam (lr 1.5e-4, weight decay 8e-4), batch 1024, up to 100 epochs, ReduceLROnPlateau on validation loss, gradient clipping 1.0. Early stopping (patience 6) on mean validation AUPRC, using a random 20% of the training cells of each fold. No input scaling.
- Per-gene XGBoost (`src/models/per_gene.py`): `XGBClassifier` with the config parameters plus `tree_method: hist`.
- Tian: scikit-learn `LogisticRegression` (saga, L2, C=1.0, max_iter 1000), one model per target gene.

## Data

All scPerturb files are read from `Runs/SingleCell/data/`. CCLE inputs are read from `Runs/SingleCell/data/CCLE/`.

Common to all scPerturb datasets:

- Download: automatic. `run_all.py` reads the Zenodo record metadata from `https://zenodo.org/api/records/7041849` and downloads each file in `dataset_files` that is missing or smaller than the expected size. If the metadata cannot be fetched it uses `https://zenodo.org/records/7041849/files/<file name>`. Downloads resume from a `.part` file.
- Matrix: `adata.X` as stored in the file (no layer is selected); raw integer counts (checked in Adamson 10X001, Tian and Zhao). The matrix stays sparse; the network makes one mini-batch dense at a time.
- Normalization: `sc.pp.normalize_total(target_sum=10000)` then `sc.pp.log1p`.
- No gene filtering, no HVG selection, no cell QC, no batch correction in this code. All genes in the file are used as features.
- Eligible targets: perturbation labels with at least `min_cells_per_gene` cells (mc300 or mc800), control labels excluded from the count. Cells with other (non-eligible) perturbations are dropped.
- Controls: cells whose label is in the control list are kept as negatives (all targets 0). Default control labels: `non-targeting`, `control`. No subsampling of controls.
- Labels: one binary output per eligible perturbation.
- CV: 5 folds. Cell-level `KFold` (shuffled, seed 42) unless a group column is given, in which case `GroupKFold` on that column. A fold without positives for a target has no AUPRC (missing, not 0). Normalized AUPRC = (AUPRC - prevalence) / (1 - prevalence), as for TCGA.

### Adamson (Adamson/Weissman 2016)

- Source: scPerturb, Zenodo record 7041849. Three files:
  - `AdamsonWeissman2016_GSM2406675_10X001.h5ad` (pilot single-gene UPR-TF screen)
  - `AdamsonWeissman2016_GSM2406677_10X005.h5ad` (combinatorial UPR-sensor screen)
  - `AdamsonWeissman2016_GSM2406681_10X010.h5ad` (large single-gene screen)
- Download: automatic.
- Processing:
  - Common normalization; label column `perturbation`.
  - Construct names are mapped to target gene symbols with a fixed table (`src/datasets/adamson.py`). A construct can map to several genes (for example `ATF6_PERK_IRE1_pMJ158` -> ATF6, EIF2AK3, ERN1), so a cell can be positive for more than one output. Outputs are gene symbols, not constructs.
  - Controls: `non-targeting`, `63(mod)_pBA580`, `62(mod)_pBA581`, `3x_neg_ctrl_pMJ144-1`, `3x_neg_ctrl_pMJ144-2`, `Gal4-4(mod)_pBA582`.
  - Minimum 300 cells per target gene (counted over all constructs mapping to it). Cells labeled `*` (mapped to no gene) are dropped.
  - Each file is a separate run. CV: cell-level KFold.
  - Final sizes (run_metadata.json):
    - 10X001: 5,752 cells (1,769 controls), 35,635 features, 7 targets.
    - 10X005: 14,676 cells (3,366 controls), 32,738 features, 3 targets.
    - 10X010: 61,184 cells (7,295 controls), 32,738 features, 80 targets.

### Replogle K562 and RPE1 (Replogle/Weissman 2022)

- Source: scPerturb, Zenodo record 7041849: `ReplogleWeissman2022_K562_essential.h5ad`, `ReplogleWeissman2022_rpe1.h5ad`.
- Download: automatic.
- Processing:
  - Common normalization. Label column chosen automatically as the first of `gene`, `perturbation`, `target_gene`, `perturbed_gene` present in `obs`; for both files this was `gene`.
  - Minimum 300 cells per perturbation. Controls: `non-targeting`, `control`.
  - CV: cell-level KFold.
  - Final sizes:
    - K562 essential: 75,767 cells (10,691 controls), 8,563 features, 155 targets.
    - RPE1: 55,545 cells (11,485 controls), 8,749 features, 69 targets.

### Frangieh (Frangieh/Izar 2021)

- Source: scPerturb, Zenodo record 7041849: `FrangiehIzar2021_RNA.h5ad`.
- Download: automatic.
- Processing:
  - Common normalization; label column `perturbation`; immune condition column `perturbation_2` (used only to report metrics per condition).
  - Minimum 800 cells per perturbation.
  - Multitask model: controls `non-targeting`, `control`; cell-level KFold. 136,396 cells (57,605 controls), 23,712 features, 79 targets.
  - Per-gene XGBoost: same 79 perturbations (`control` excluded from the count). For each perturbation, positives are its cells and negatives are the cells of the other 78 perturbations and the controls, the same 136,396 cells as the multitask model. CV: `StratifiedKFold` on cells (shuffled, seed 42), per gene.
  - `frangieh_conditions` recomputes metrics within each value of `perturbation_2`, overall and per fold, from the saved held-out predictions of both models.

### McFarland (McFarland/Tsherniak 2020)

- Source: scPerturb, Zenodo record 7041849: `McFarlandTsherniak2020.h5ad`.
- Download: automatic.
- Processing:
  - Common normalization; label column `perturbation`.
  - Minimum 300 cells per perturbation. Controls: `non-targeting`, `control`.
  - CV: GroupKFold on `obs["cell_line"]` (209 cell lines), so train and test folds share no cell line.
  - Final size: 182,875 cells (29,143 controls), 32,738 features, 17 targets.

### Zhao (Zhao/Sims 2021)

- Source: scPerturb, Zenodo record 7041849: `ZhaoSims2021.h5ad`.
- Download: automatic.
- Processing:
  - Common normalization; label column `perturbation`.
  - Minimum 300 cells per perturbation. Controls: `non-targeting`, `control`.
  - Targets: `etoposide` and `panobinostat` only. The other four drugs (Ana-12, Ispenisib, RO4929097, Tazemetostat) were each given in sample PW030 only, so CV grouped by sample cannot train and test them; their cells are removed.
  - CV: GroupKFold on `obs["sample"]` (10 samples).
  - Final size: 147,046 cells (88,313 controls), 60,725 features, 2 targets.
  - `zhao_pr_curves` computes per-fold precision-recall curves for `etoposide` (36,513 positive cells) and `panobinostat` (22,220 positive cells) from the held-out probabilities.

### Tian (Tian/Kampmann 2021, CRISPRa)

- Source: scPerturb, Zenodo record 7041849: `TianKampmann2021_CRISPRa.h5ad`.
- Download: automatic.
- Processing:
  - Common normalization; label column `perturbation`, guide column `guide_id`; control label `control`.
  - Targets: perturbations with exactly two guides and at least 100 cells per guide (mcg100): 30 targets.
  - Split by guide, not by cell: for each target, train on guide A and test on guide B, then the reverse (guides ordered alphabetically). Control cells are shuffled (seed 42 + direction) and split into two disjoint halves, one for training and one for testing.
  - One logistic regression per target: target cells vs control cells.

### CCLE single-cell cell lines with DepMap mutations

- Source:
  - Expression: `CCLE_adata.h5ad`. TODO: origin of this file (publication / accession / how it was built). `X`: raw integer counts, 56,982 cells x 30,314 genes, sparse.
  - Mutations: `OmicsSomaticMutations.csv` (DepMap). TODO: DepMap release (sequence-ontology `VariantInfo`, so 23Q4 or later; 1,929 models).
- Download: manual. Place both files in `Runs/SingleCell/data/CCLE/`. The step is skipped if either is missing. `obs` must contain `Model_ID` matching `ModelID` in the mutation file.
- Processing:
  - `sc.pp.filter_genes(min_cells=3)`, `sc.pp.normalize_total(target_sum=10000)`, `sc.pp.log1p`.
  - Mutations: coding rows only, by `VariantInfo`, as for TCGA: missense, nonsense, nonstop, frameshift, in-frame, start codon and splice site (MAF-like names up to DepMap 23Q2, which also list silent mutations; sequence-ontology terms from 23Q4). TMB of a model = number of such rows; the regression target is log1p(TMB), assigned to every cell of the model.
  - Cells whose model has no such mutation are removed.
  - Multitask targets: the 150 genes mutated (coding rows) in the largest number of the 202 models in the data, among genes mutated in at least 20 of them (ties by gene name). A cell is positive for a gene if its model carries a mutation in it.
  - No HVG selection or batch correction.
  - CV: 5-fold `KFold` over unique `Model_ID`, so all cells of a cell line are in the same fold. Used for both the XGBoost regressor and the multitask network (same `simple_wide_run` preset).
  - Final size: 54,759 cells from 202 models, 150 target genes. TODO: number of features after `filter_genes` (printed to the log only).

## Outputs

`output/runs/<run name>/`, one folder per model run:

- Multitask runs: `<file stem>_mc<min cells>_simple_wide_run[_cvgrp_<group column>]`. Adamson runs have the prefix `admason_`. Files: `run_metadata.json`, `metric_summary_per_gene.csv` (pooled out-of-fold), `metric_summary_per_gene_folds.csv`, `oof_probabilities.csv`, `oof_true_labels.csv`, `oof_cell_folds.csv`, `oof_sample_embeddings.csv` (last hidden layer), `fold_head_weights_<k>.csv`, `head_weights_per_fold.csv`, training histories. Frangieh also has `metric_summary_per_gene_by_condition*.csv`; Zhao also has the `*_selected_targets.csv` PR-curve files.
- `FrangiehIzar2021_RNA_per_gene_mc800_baseline/`: `predictions_<gene>.csv`, `fold_metrics_<gene>.csv`, summaries, `run_metadata.json`.
- `TianKampmann2021_CRISPRa_two_guide_per-gene_mcg100_simple_wide_run/`: `directional_metrics_per_gene.csv`, `directional_predictions_long.csv`, `directional_summary.csv`, `run_metadata.json`.
- `ccle_tmb_and_multitask/`: `log_tmb_fold_metrics.csv`, `cell_level_xgboost.csv`, `model_level_xgboost.csv` (cell predictions averaged per model), `meta_xgboost.csv`, `multitask_fold_metrics_150.csv`, `multitask_summary_150.csv`.

`output/plot_inputs/`: a copy of the files used downstream, one subfolder per dataset (`Adamson_10X001`, `Adamson_10X005`, `Adamson_10X010`, `McFarland`, `Replogle_K562`, `Replogle_RPE1`, `Frangieh`, `Frangieh_per_gene`, `Tian`, `Zhao`, `CCLE`), plus `_bundle_manifest.json` listing each file as ok or missing. The file list is in `src/steps/plot_inputs.py`.

Used by:

- `Figures/Figure5/Figure5.ipynb` (with `figure5_helpers.py`): reads `output/plot_inputs/`, and reads the two Replogle run folders directly from `output/runs/` (embeddings and head weights, which are not copied to `plot_inputs`).
- `Tables/supplementary_tables.py` (Table S4): reads `output/plot_inputs/` (Adamson 10X010 and 10X005, Frangieh, McFarland, Replogle K562/RPE1, Zhao, Frangieh_per_gene, Tian).
