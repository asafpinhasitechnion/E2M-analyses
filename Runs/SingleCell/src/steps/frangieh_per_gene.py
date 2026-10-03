"""Per-gene XGBoost cross-validation on Frangieh/Izar."""

from __future__ import annotations

from pathlib import Path

import scanpy as sc

import config
from models.per_gene import (
    get_genes_by_min_cells,
    infer_cell_line_name,
    run_per_gene_benchmark,
)


FILE_NAME = "FrangiehIzar2021_RNA.h5ad"
MIN_CELLS_PER_GENE = 800
NON_TARGET_LABEL = "control"
GENE_COL = "perturbation"
CONDITION_COL = "perturbation_2"
MODEL_TYPE = "xgboost"
RUN_TAG = "baseline"


def main() -> None:
    config.make_dirs()
    data_path = config.DATA_DIR / FILE_NAME
    if not data_path.exists():
        print(f"[skip] file missing: {data_path}")
        return

    adata = sc.read_h5ad(data_path)
    sc.pp.normalize_total(adata, target_sum=config.PREPROCESS_TARGET_SUM)
    sc.pp.log1p(adata)

    if GENE_COL not in adata.obs.columns:
        raise ValueError(f"{FILE_NAME}: missing obs column '{GENE_COL}'")
    if CONDITION_COL not in adata.obs.columns:
        raise ValueError(f"{FILE_NAME}: missing obs column '{CONDITION_COL}'")

    label_values = set(adata.obs[GENE_COL].astype(str).unique())
    if NON_TARGET_LABEL not in label_values:
        raise ValueError(f"non_target_label={NON_TARGET_LABEL!r} not found in obs[{GENE_COL!r}]")

    genes = get_genes_by_min_cells(
        adata=adata,
        min_cells_per_gene=MIN_CELLS_PER_GENE,
        gene_col=GENE_COL,
        exclude=(NON_TARGET_LABEL,),
    )
    if not genes:
        raise RuntimeError("No genes passed the min_cells_per_gene filter.")

    model_params = dict(config.XGBOOST_PARAMS)
    model_params.setdefault("tree_method", "hist")
    if config.USE_GPU:
        model_params["device"] = "cuda"

    stem = Path(FILE_NAME).stem
    run_name = f"{stem}_per_gene_mc{MIN_CELLS_PER_GENE}_{RUN_TAG}"

    print(
        f"[per_gene] {run_name} | genes={len(genes)} | cells={adata.n_obs} | "
        f"model={MODEL_TYPE} | params={model_params}"
    )
    run_per_gene_benchmark(
        adata=adata,
        genes=genes,
        output_root=config.OUTPUTS_ROOT,
        input_file=FILE_NAME,
        cell_line_name=infer_cell_line_name(FILE_NAME),
        model_type=MODEL_TYPE,
        gene_col=GENE_COL,
        negative_mode="all",
        non_target_label=NON_TARGET_LABEL,
        n_splits=config.CV_FOLDS,
        random_state=config.RANDOM_STATE,
        model_params=model_params,
        run_name=run_name,
        verbose=True,
        condition_col=CONDITION_COL,
    )
