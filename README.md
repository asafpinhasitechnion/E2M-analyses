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

Or with Docker:

```
docker build -t e2m-analyses .
docker run --gpus all --rm -v "<this folder>":/work e2m-analyses python run_all.py
```

A GPU is used when available for the neural networks, and XGBoost in TCGA.

## Running

From this folder:

```
python run_all.py                                                 # everything
python run_all.py external singlecell                             # selected parts
python Figures/run_figures.py tcga external singlecell tables     # only the figures and tables, from the saved outputs
```

Parts run in this order: TCGA (then Figures 1-3), External (then Figure 4), Clinical drivers, Single-cell (then Figure 5), and the supplementary tables. Each run writes to its `output/` folder, figures to `Figures/FigureN/output/` (with the executed notebook) and tables to `Tables/output/main/`. External and Clinical drivers use the TCGA data, so TCGA runs first.

Two options change the analysis:

- `--normalization none`: TCGA counts are log1p-transformed without CPM (default `cpm`: counts per million, then log1p). TCGA, External, Figures 1-4 and the tables then write to `counts/` subfolders.
- `--external-methods combat`: Figure 4 and the tables use ComBat for the cohorts that ran it, and each cohort's main method for the others (default `main`: each cohort's first method in `Runs/External/config/e2m.yaml`). They write to `combat/` instead of `main/`.

## Using the saved models

The trained models and all outputs are on Zenodo (DOI). [`check_predictions.ipynb`](check_predictions.ipynb) loads a saved model, predicts with it, and checks that the saved models reproduce the saved predictions and metrics. In short:

```python
import e2m
model = e2m.E2MModel.load("Runs/TCGA/output/BRCA/model")
data = e2m.Dataset.from_tcga(["BRCA"], data_dir="Runs/TCGA/data", config="Runs/TCGA/config/e2m.yaml")
scores = model.predict(data)          # samples x mutation targets; also takes a samples x genes DataFrame
```

## Side analyses

Short scripts for single results, not run by `run_all.py`; run each after its run, from this folder:

| Analysis | Command | Output |
|---|---|---|
| SHAP stability in TCGA-BRCA (20 subsamples of 80%; Table S2c) | `python Runs/TCGA/src/shap_stability.py` | `Runs/TCGA/output/BRCA/shap_stability/` |
| BRAF in melanoma with one XGBoost model | `cd Runs/External && python src/braf_xgboost.py` | `Runs/External/output/braf_xgboost.csv` |
| Training and inference time | `python Runs/Benchmark/timing.py [--device cpu]` | `Runs/Benchmark/output/timing_<device>.csv` |

## Data

Most data are downloaded by the code on first use into each run's `data/` folder. Files to place by hand:

- External: CPTAC/CMI, and the supplementary tables of Hugo, Van Allen and ImmunoPOG (`Runs/External/README.md`).
- Single-cell: `CCLE_adata.h5ad` and DepMap `OmicsSomaticMutations.csv` in `Runs/SingleCell/data/CCLE` (`Runs/SingleCell/README.md`).
