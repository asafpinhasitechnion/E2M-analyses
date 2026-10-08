"""Multitask perturbation-prediction cross-validation on the scPerturb datasets."""

from __future__ import annotations

import os

import scanpy as sc

import config
from datasets.adamson import ADAMSON_CONTROL_LABELS, ADAMSON_PERTURBATION_TO_GENES
from models.multitask import infer_cell_line_name, run_multitask_benchmark
from models.presets import MODEL_PRESETS


# Each RUN: dict with keys
#   key                       : DATASET_FILES key
#   min_cells_per_gene
#   cv_group_col              : optional obs column for GroupKFold (e.g. cell_line, sample)
#   gene_col                  : None -> auto-detect from GENE_COL_CANDIDATES
#   control_labels            : labels treated as negative controls (default = generic scPerturb)
#   perturbation_to_genes     : optional construct -> [genes] map (multi-label mode; Adamson)
#   targets                   : optional perturbations to keep, with the controls
# First column present in obs is used; the scPerturb files used here carry 'perturbation'.
GENE_COL_CANDIDATES = ("gene", "perturbation", "target_gene", "perturbed_gene")
DEFAULT_CONTROL_LABELS = ("non-targeting", "control")

def _adamson_run(key: str) -> dict:
    """Adamson cohorts as independent multitask runs, cell-level 5-fold CV."""
    return {
        "key": key,
        "min_cells_per_gene": 300,
        "cv_group_col": None,
        "gene_col": None,
        "control_labels": ADAMSON_CONTROL_LABELS,
        "perturbation_to_genes": ADAMSON_PERTURBATION_TO_GENES,
    }


RUNS = [
    _adamson_run("adamson_10X001"),
    _adamson_run("adamson_10X005"),
    _adamson_run("adamson"),
    {"key": "mcfarland",     "min_cells_per_gene": 300, "cv_group_col": "cell_line"},
    {"key": "replogle_k562", "min_cells_per_gene": 300},
    {"key": "replogle_rpe1", "min_cells_per_gene": 300},
    {"key": "frangieh",      "min_cells_per_gene": 800, "condition_col": "perturbation_2"},
    # Zhao: the other four drugs were each given in one sample only, so CV grouped by sample cannot train and test them.
    {"key": "zhao",          "min_cells_per_gene": 300, "cv_group_col": "sample",
     "targets": ("etoposide", "panobinostat")},
]


def _selected_run_keys() -> set[str] | None:
    """Optional comma-separated filter for running a subset of stage01 datasets."""
    raw = os.environ.get("SINGLE_CELL_MULTITASK_KEYS", "").strip()
    if not raw:
        return None
    keys = {x.strip() for x in raw.split(",") if x.strip()}
    known = {run["key"] for run in RUNS}
    unknown = sorted(keys - known)
    if unknown:
        raise ValueError(
            "Unknown SINGLE_CELL_MULTITASK_KEYS values: "
            f"{unknown}. Known keys: {sorted(known)}"
        )
    return keys


def _resolve_gene_col(adata, file_name: str, requested: str | None) -> str:
    if requested is not None and requested in adata.obs.columns:
        return requested
    for cand in GENE_COL_CANDIDATES:
        if cand in adata.obs.columns:
            if requested is not None and requested != cand:
                print(f"  [multitask] {file_name}: '{requested}' not found, using '{cand}' instead")
            return cand
    raise ValueError(
        f"{file_name}: none of {GENE_COL_CANDIDATES} present in obs. "
        f"Available: {list(adata.obs.columns)[:15]}"
    )


def _run_one(run: dict, preset: str) -> None:
    file_name = config.DATASET_FILES[run["key"]]
    min_cells_per_gene = int(run["min_cells_per_gene"])
    cv_group_col = run.get("cv_group_col")
    gene_col_requested = run.get("gene_col")
    control_labels = tuple(run.get("control_labels", DEFAULT_CONTROL_LABELS))
    perturbation_to_genes = run.get("perturbation_to_genes")
    condition_col = run.get("condition_col")

    data_path = config.DATA_DIR / file_name
    if not data_path.exists():
        print(f"[skip] file missing: {data_path}")
        return

    adata = sc.read_h5ad(data_path)
    sc.pp.normalize_total(adata, target_sum=config.PREPROCESS_TARGET_SUM)
    sc.pp.log1p(adata)

    gene_col = _resolve_gene_col(adata, file_name, gene_col_requested)
    if run.get("targets") is not None:
        adata = adata[adata.obs[gene_col].astype(str).isin([*run["targets"], *control_labels])].copy()

    n_ctl_in_data = int(adata.obs[gene_col].astype(str).isin(control_labels).sum())
    print(
        f"  [multitask] {file_name}: gene_col='{gene_col}' "
        f"control_labels={len(control_labels)} -> matched {n_ctl_in_data} cells"
    )
    if n_ctl_in_data == 0:
        print(
            "  [warn] no negative-control cells matched; check control_labels for this dataset. "
            "Continuing (the multitask model can still train without negatives)."
        )

    stem = data_path.stem
    run_name = f"{stem}_mc{min_cells_per_gene}_{preset}"
    if cv_group_col is not None:
        safe = cv_group_col.replace("/", "_")
        run_name = f"{run_name}_cvgrp_{safe}"
    if stem.startswith("AdamsonWeissman2016"):
        run_name = f"admason_{run_name}"

    model_params = dict(MODEL_PRESETS[preset])
    non_target_subsample = model_params.pop("non_target_subsample", None)

    print(f"[multitask] {run_name} | cells={adata.n_obs}")
    run_multitask_benchmark(
        adata=adata,
        output_root=config.OUTPUTS_ROOT,
        input_file=file_name,
        cell_line_name=infer_cell_line_name(file_name),
        gene_col=gene_col,
        min_cells_per_gene=min_cells_per_gene,
        control_labels=control_labels,
        non_target_subsample=non_target_subsample,
        cv_group_col=cv_group_col,
        n_splits=config.CV_FOLDS,
        random_state=config.RANDOM_STATE,
        model_params=model_params,
        run_name=run_name,
        verbose=True,
        perturbation_to_genes=perturbation_to_genes,
        condition_col=condition_col,
    )


def main() -> None:
    config.make_dirs()
    selected_keys = _selected_run_keys()
    if selected_keys is not None:
        print(f"[multitask] filtering runs to keys: {sorted(selected_keys)}")
    for run in RUNS:
        if selected_keys is not None and run["key"] not in selected_keys:
            continue
        _run_one(run=run, preset=config.MULTITASK_PRESET)
