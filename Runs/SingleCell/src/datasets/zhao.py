"""Zhao PR-curve extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import average_precision_score, precision_recall_curve


def extract_zhao_pr_curves(
    run_dir: Path,
    data_file: Path,
    targets: Sequence[str] = ("etoposide", "panobinostat"),
) -> None:
    run_dir = Path(run_dir)
    meta = pd.read_json(run_dir / "run_metadata.json", typ="series")

    adata = sc.read_h5ad(data_file)
    gene_col = str(meta.get("gene_col", "perturbation"))

    # Reconstruct the same training set: cells whose perturbation has enough
    # cells in the full adata (or any of the configured control labels).
    sample_ids = adata.obs_names.to_numpy().astype(str)

    sid = pd.Index(sample_ids)
    probs = pd.read_csv(run_dir / "oof_probabilities.csv", index_col=0)
    folds = pd.read_csv(run_dir / "oof_cell_folds.csv", index_col=0)
    # Restrict to overlapping cells (training-set rows).
    common = sid.intersection(probs.index.astype(str))
    probs = probs.loc[common]
    folds = folds.loc[common]

    genes = list(probs.columns)
    if not all(t in genes for t in targets):
        missing = [t for t in targets if t not in genes]
        raise ValueError(f"Targets missing in OOF: {missing}")

    obs_gene = adata.obs.loc[common, gene_col].astype(str).to_numpy()
    points_rows = []
    summary_rows = []

    for tgt in targets:
        y_true = (obs_gene == tgt).astype(int)
        y_prob = probs[tgt].to_numpy(float)
        fold_ids = folds["fold"].to_numpy(int)
        baseline = float(np.mean(y_true)) if len(y_true) else float("nan")
        summary_rows.append({"target": tgt, "baseline_prevalence": baseline,
                             "n_cells": int(len(y_true)), "n_pos": int(y_true.sum())})

        for f in sorted(np.unique(fold_ids)):
            m = fold_ids == f
            if len(np.unique(y_true[m])) < 2:
                continue
            prec, rec, _ = precision_recall_curve(y_true[m], y_prob[m])
            ap = float(average_precision_score(y_true[m], y_prob[m]))
            for p, r in zip(prec, rec):
                points_rows.append({"target": tgt, "fold": int(f), "precision": float(p),
                                    "recall": float(r), "auprc": ap,
                                    "baseline_prevalence": baseline})

    # Selected-target predictions (compact)
    probs_sel = probs[list(targets)].copy()
    probs_sel.insert(0, "fold", folds["fold"].to_numpy())
    probs_sel.insert(1, "y_true_etoposide", (obs_gene == "etoposide").astype(int) if "etoposide" in targets else 0)
    if "panobinostat" in targets:
        probs_sel.insert(2, "y_true_panobinostat", (obs_gene == "panobinostat").astype(int))

    probs_sel.to_csv(run_dir / "oof_probabilities_selected_targets.csv")
    pd.DataFrame(points_rows).to_csv(run_dir / "pr_curve_points_selected_targets.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(run_dir / "pr_curve_summary_selected_targets.csv", index=False)
