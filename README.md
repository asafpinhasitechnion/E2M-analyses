# Expression to Mutation: analyses

Code for the analyses, figures and supplementary tables of the manuscript. Models are trained with E2M (https://github.com/yizhak-lab-ccg/E2M). Each run has its own README with its data sources, processing and outputs.

| Part | Folder | Feeds |
|---|---|---|
| TCGA | `Runs/TCGA` | Figures 1-3, Tables S1-S3 |
| External cohorts | `Runs/External` | Figure 4, Tables S2-S3 |
| Clinical drivers | `Runs/ClinicalDrivers` | Table S3c |
| Single-cell | `Runs/SingleCell` | Figure 5, Table S4 |

## Installation

Python 3.10. `requirements.txt` has the exact versions of all packages (PyTorch for CUDA 12.6, E2M at commit `b201b22`; inmoose needs a C++ compiler):

```
python -m pip install -r requirements.txt
```

Or with Docker (the image holds the software; this folder is mounted with the code, data and outputs):

```
docker build -t e2m-analyses .
docker run --gpus all --rm -v "<this folder>":/work e2m-analyses python run_all.py
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

Two run variants differ in how TCGA counts are scaled before the model: `--normalization cpm` (default; counts per million, then log1p) or `none` (log1p of the counts, no library-size step). TCGA and External write to `output/` (CPM) or `output/counts/`, Figures 1-3 to `Figures/FigureN/output/`, Figure 4 to `Figures/Figure4/output/<method>/` and the tables to `Tables/output/<method>/` (each with `counts/` before the method for the variant without CPM). Clinical drivers and single-cell (Figure 5) do not depend on the variant. Figure 4 and the tables are made for each External method in `--external-methods` (default `main`): `main` uses each cohort's first method in `Runs/External/config/e2m.yaml`; another method (e.g. `combat`) is used for the cohorts that ran it, and the first method for the others. Each executed notebook is saved in its output folder.

## Data

Most data are downloaded by the code on first use into each run's `data/` folder. Files to place by hand:

- External: CPTAC/CMI, and the supplementary tables of Hugo, Van Allen and ImmunoPOG (`Runs/External/README.md`).
- Single-cell: `CCLE_adata.h5ad` and DepMap `OmicsSomaticMutations.csv` in `Runs/SingleCell/data/CCLE` (`Runs/SingleCell/README.md`).
