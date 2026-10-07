# TCGA

Predicts gene mutation status and tumor mutational burden (TMB) from RNA expression in TCGA with E2M. For each cancer type, and for all cancer types together ("all"), it runs 5-fold cross-validation of the multitask neural network, trains a full-data model for sample embeddings and head weights, and runs SHAP on separate XGBoost models for the well-predicted targets. TMB is predicted pan-cancer with XGBoost regression. Outputs are used by Figures 1-4 and Supplementary Tables 1-3.

## Installation

Python 3.10. Install PyTorch with CUDA, then E2M (https://github.com/yizhak-lab-ccg/E2M) with its SHAP extra:

```
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
pip install "e2m[interpretation] @ git+https://github.com/yizhak-lab-ccg/E2M.git@b201b22"
```

E2M's `requirements-lock.txt` lists the exact package versions.

## Running

From `Runs/TCGA`:

```
python src/run_tcga.py                                  # all 32 cancer types and "all", CPM input
python src/run_tcga.py --normalization none             # log1p counts instead of CPM
python src/run_tcga.py --cohorts LUAD --tasks mutation  # one cancer type, one task
```

Options:
- `--cohorts`: cancer codes, and `all` for the pan-cancer run. Default: the 32 cancer types in the config, then `all`.
- `--tasks`: any of `mutation` (cross-validation, full-data model, embeddings, head weights), `shap` (needs the `mutation` outputs), `tmb`. Default: all three.
- `--normalization`: `cpm` (default) or `none`.
- `--config`: default `config/e2m.yaml`. `--data-dir`: default `data/`.

All settings are in `config/e2m.yaml`. A GPU is used when available; training is deterministic on a given machine and software stack, and close but not identical on other hardware.

## Data

Downloaded automatically into `data/` on first use:

- `data/expression/TCGA-{cancer}.star_counts.tsv.gz`: UCSC Xena GDC hub, STAR gene counts per cohort, stored as log2(count + 1).
- `data/mutations/{cancer}_mc3_gene_level.txt.gz`: Xena TCGA hub, MC3 gene-level binary mutation matrices (non-silent mutations).
- `data/mutation_events/{cancer}_mc3.txt.gz`: Xena TCGA hub, MC3 mutation events, for TMB.
- `data/annotation/gencode.v36.annotation.gtf.gz`: GENCODE v36 gene annotation.

## Processing

Expression (per cohort, in this order):
1. Xena values are converted back to counts: 2^x - 1, negative values set to 0.
2. CPM: each sample is divided by its total count over all genes in the file and multiplied by 1e6 (skipped with `--normalization none`).
3. Ensembl IDs (version removed) are mapped to gene symbols with GENCODE v36 `gene_name`; only protein-coding genes are kept.
4. Genes sharing a symbol are summed (mostly PAR_Y copies, which are all zero).
5. log1p.
6. Inside each model, every gene is standardized (StandardScaler) on that model's training samples.

Samples: expression and mutation labels are matched on the sample ID `TCGA-XX-XXXX-NN` (patient and sample type, as in MC3). When a sample has several vials in the expression file, the earliest vial letter is used. Samples without MC3 labels are not used. Sample types are not filtered (for example, SKCM metastases, type 06, are included).

Mutation targets: protein-coding genes mutated in at least 5% of the matched samples, then the 400 most frequent (ties broken by gene name). For "all", prevalence is computed over all cancer types together.

TMB: number of coding MC3 events per sample (Missense_Mutation, Frame_Shift_Del, Frame_Shift_Ins, Nonsense_Mutation, Nonstop_Mutation, In_Frame_Del, In_Frame_Ins, Translation_Start_Site, Splice_Site). Target: log2(TMB + 1). Samples without any coding event are dropped.

## Models

Multitask network: input genes -> 512 -> 256 -> one output per target. Each hidden layer is Linear, LayerNorm, GELU and Dropout (0.3). Adam, learning rate 5e-4, weight decay 3e-4, batch size 64, weighted binary cross-entropy with pos_weight = negatives/positives per target, gradient clipping 1.0, learning rate halved after 2 epochs without improvement. Up to 100 epochs with early stopping (patience 6) on the loss of a 15% validation split of the training samples. Seed 42.

Cross-validation: 5 folds, shuffled, seed 42; stratified by cancer type for "all". Metrics are computed on the pooled out-of-fold predictions (AUPRC, normalized AUPRC = (AUPRC - prevalence) / (1 - prevalence), ROC AUC, and threshold metrics at 0.5); per-fold metrics are also written.

Full-data model: the same network and early stopping, trained on all samples of the cohort. Used for sample embeddings (256-dimensional encoder output) and output-head weights.

SHAP: for each target with pooled normalized AUPRC > 0.05, a separate XGBoost classifier (300 trees, learning rate 0.1, scale_pos_weight = negatives/positives, seed 42) is trained on all samples of the cohort and explained with Tree SHAP on all samples. These models are used only for interpretation.

TMB: XGBoost regression (300 trees, learning rate 0.1), 5-fold cross-validation stratified by cancer type, on all 32 cancer types together; metrics overall and per cancer type.

## Outputs

`output/cpm/` (or `output/counts/` with `--normalization none`), one folder per cohort (`ACC` ... `UVM`, `all`):
- `preprocessing_manifest.json`: per cancer type, the number of expression, MC3 and matched samples, duplicate symbols summed, the final number of samples, genes and targets, and the SHA-256 of each source file.
- `cv/`: `oof_probabilities.csv` and `oof_predictions.csv` (samples x targets), `metrics.csv` (pooled, per target), `fold_metrics.csv`, `fold_assignments.csv`, `run_metadata.json` (settings, split, timing, device, package versions, E2M version and commit), `folds/fold_{k}/` (the model of each fold).
- `model/`: full-data model: `model.pt` (weights and scaler), `features.json` (input gene order), `targets.json` (target order), `feature_means.npy`, `model_metadata.json`, `training_history.csv`.
- `embeddings.csv` (samples x 256), `head_weights.csv` (targets x 256).
- `shap/{target}/`: `feature_summary.csv` (all genes ranked by mean |SHAP|), `sample_shap_top_features.csv.gz` (per-sample SHAP and expression of the top 20 genes), `metadata.json`.

`output/cpm/tmb/`: `oof_predictions.csv`, `summary.csv` (overall and per cancer type), `fold_metrics.csv`, `run_metadata.json`, `folds/` (the model of each fold).
