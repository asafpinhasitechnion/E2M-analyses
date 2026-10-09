# Expression to Mutation: analyses

Code for the analyses, figures and supplementary tables of the manuscript. Models are trained with E2M (https://github.com/yizhak-lab-ccg/E2M). Each run has its own README with its data sources, processing and outputs.

| Part | Folder | Feeds |
|---|---|---|
| TCGA | `Runs/TCGA` | Figures 1-3, Tables S1-S3 |
| External cohorts | `Runs/External` | Figure 4, Tables S2-S3 |
| Clinical drivers | `Runs/ClinicalDrivers` | Table S3c |
| Single-cell | `Runs/SingleCell` | Figure 5, Table S4 |

## Installation

Python 3.10, PyTorch with CUDA, E2M, and the packages used by the runs and figures:

```
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu126
pip install "e2m[interpretation] @ git+https://github.com/yizhak-lab-ccg/E2M.git@b201b22"
pip install inmoose==0.8.1 umap-learn==0.5.12 scanpy==1.11.5 openpyxl==3.1.5 mygene \
    matplotlib==3.10.9 seaborn==0.13.2 adjustText==1.4.0 lifelines==0.30.0 pingouin==0.6.1 statsmodels==0.14.6 \
    igraph==1.0.0 leidenalg==0.12.0 nbclient==0.11.0 nbformat==5.11.1 ipykernel==7.3.0
```

A GPU is used when available (the neural networks, and XGBoost in TCGA).

## Running

From this folder:

```
python run_all.py                                   # everything, TCGA counts with CPM
python run_all.py tcga external tables --normalization none   # the variant without CPM (log1p of counts)
python run_all.py external singlecell               # selected parts, in this order
```

The order is TCGA, then Figures 1-3; External, then Figure 4; Clinical drivers; Single-cell, then Figure 5; and last the supplementary tables. External and Clinical drivers read the TCGA data folder (`Runs/TCGA/data`). Figures alone: `python Figures/run_figures.py tcga external singlecell tables`.

Two run variants differ in how TCGA counts are scaled before the model: `--normalization cpm` (default; counts per million, then log1p) or `none` (log1p of the counts, no library-size step). TCGA and External write to `output/cpm/` or `output/counts/`, Figures 1-4 to `Figures/FigureN/output/<variant>/`, and the tables to `Tables/output/<variant>/<method>/`. Clinical drivers and single-cell (Figure 5) do not depend on the variant. Figure 4 and the tables are made for each External method in `--external-methods` (default `main sample_rank`): `main` uses each cohort's first method in `Runs/External/config/e2m.yaml`, `sample_rank` the within-sample gene ranks for every cohort. Each executed notebook is saved in its output folder.

## Data

Most data are downloaded by the code on first use into each run's `data/` folder. Files to place by hand:

- External: CPTAC/CMI, and the supplementary tables of Hugo, Van Allen and ImmunoPOG (`Runs/External/README.md`).
- Single-cell: `CCLE_adata.h5ad` and DepMap `OmicsSomaticMutations.csv` in `Runs/SingleCell/data/CCLE` (`Runs/SingleCell/README.md`).
