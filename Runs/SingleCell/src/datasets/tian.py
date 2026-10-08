from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

from models.multitask import compute_binary_metrics


def _eligible_two_guide_targets(
    adata,
    gene_col: str,
    guide_col: str,
    min_cells_per_guide: int,
    control_labels: Tuple[str, ...],
) -> Dict[str, Tuple[str, str]]:
    obs = adata.obs[[gene_col, guide_col]].copy()
    obs[gene_col] = obs[gene_col].astype(str)
    obs[guide_col] = obs[guide_col].astype(str)

    ctl = set(control_labels)
    tgt = obs.loc[~obs[gene_col].isin(ctl)].copy()

    eligible: Dict[str, Tuple[str, str]] = {}
    for pert, g in tgt.groupby(gene_col):
        counts = g[guide_col].value_counts()
        if counts.shape[0] != 2:
            continue
        if int(counts.min()) < int(min_cells_per_guide):
            continue
        gs = sorted(counts.index.tolist())
        eligible[str(pert)] = (gs[0], gs[1])
    return eligible


def _build_direction_masks(
    obs: pd.DataFrame,
    eligible: Dict[str, Tuple[str, str]],
    gene_col: str,
    guide_col: str,
    control_labels: Tuple[str, ...],
    direction: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if direction not in (1, 2):
        raise ValueError("direction must be 1 or 2")

    gene_vals = obs[gene_col].astype(str).to_numpy()
    guide_vals = obs[guide_col].astype(str).to_numpy()
    n = obs.shape[0]

    train_target = np.zeros(n, dtype=bool)
    test_target = np.zeros(n, dtype=bool)

    idx = 0 if direction == 1 else 1
    jdx = 1 - idx
    for pert, (g1, g2) in eligible.items():
        tr_g = g1 if idx == 0 else g2
        te_g = g2 if jdx == 1 else g1
        m_pert = gene_vals == pert
        train_target |= m_pert & (guide_vals == tr_g)
        test_target |= m_pert & (guide_vals == te_g)

    # Split control rows into disjoint halves to avoid leakage.
    ctl = set(control_labels)
    is_ctl = np.array([v in ctl for v in gene_vals], dtype=bool)
    ctl_pos = np.flatnonzero(is_ctl)
    rng = np.random.RandomState(seed)
    ctl_perm = ctl_pos.copy()
    rng.shuffle(ctl_perm)
    half = len(ctl_perm) // 2
    ctl_a = ctl_perm[:half]
    ctl_b = ctl_perm[half:]

    train_ctl = np.zeros(n, dtype=bool)
    test_ctl = np.zeros(n, dtype=bool)
    if direction == 1:
        train_ctl[ctl_a] = True
        test_ctl[ctl_b] = True
    else:
        train_ctl[ctl_b] = True
        test_ctl[ctl_a] = True

    train_mask = train_target | train_ctl
    test_mask = test_target | test_ctl
    return train_mask, test_mask


def _run_per_gene_direction(
    adata,
    eligible: Dict[str, Tuple[str, str]],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    gene_col: str,
    control_labels: Tuple[str, ...],
    seed: int,
    direction_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = csr_matrix(adata.X, dtype=np.float32)
    gvals = adata.obs[gene_col].astype(str).to_numpy()
    sample_ids = adata.obs_names.to_numpy()
    ctl = set(control_labels)

    rows = []
    pred_rows = []
    for pert in sorted(eligible.keys()):
        tr_pos = train_mask & (gvals == pert)
        tr_neg = train_mask & np.isin(gvals, list(ctl))
        te_pos = test_mask & (gvals == pert)
        te_neg = test_mask & np.isin(gvals, list(ctl))

        tr_mask = tr_pos | tr_neg
        te_mask = te_pos | te_neg
        if int(tr_pos.sum()) == 0 or int(te_pos.sum()) == 0 or int(tr_neg.sum()) == 0 or int(te_neg.sum()) == 0:
            continue

        x_tr = x[tr_mask]
        y_tr = np.where(gvals[tr_mask] == pert, 1, 0).astype(int)
        x_te = x[te_mask]
        y_te = np.where(gvals[te_mask] == pert, 1, 0).astype(int)

        clf = LogisticRegression(
            solver="saga",
            penalty="l2",
            C=1.0,
            max_iter=1000,
            random_state=seed,
            n_jobs=1,
        )
        clf.fit(x_tr, y_tr)
        y_prob = clf.predict_proba(x_te)[:, 1]
        m = compute_binary_metrics(y_true=y_te, y_prob=y_prob, threshold=0.5)
        rows.append({"perturbation": pert, "direction": direction_name, "mode": "per-gene", **m})
        pred_rows.append(
            pd.DataFrame(
                {
                    "sample_id": sample_ids[te_mask],
                    "direction": direction_name,
                    "mode": "per-gene",
                    "perturbation": pert,
                    "y_true": y_te.astype(np.float32),
                    "y_prob": y_prob.astype(np.float32),
                }
            )
        )

    metrics_df = pd.DataFrame(rows)
    pred_df = (
        pd.concat(pred_rows, ignore_index=True)
        if pred_rows
        else pd.DataFrame(columns=["sample_id", "direction", "mode", "perturbation", "y_true", "y_prob"])
    )
    return metrics_df, pred_df


def run_one_file(args, input_file: str) -> None:
    adata = sc.read_h5ad(args.data_folder / input_file)
    target_sum = getattr(args, "preprocess_target_sum", 1e4)
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    if args.gene_col not in adata.obs.columns:
        raise ValueError(f"{input_file}: missing obs[{args.gene_col!r}]")
    if args.guide_col not in adata.obs.columns:
        raise ValueError(f"{input_file}: missing obs[{args.guide_col!r}]")

    control_labels = tuple(s.strip() for s in args.control_labels.split(",") if s.strip())
    eligible = _eligible_two_guide_targets(
        adata=adata,
        gene_col=args.gene_col,
        guide_col=args.guide_col,
        min_cells_per_guide=args.min_cells_per_guide,
        control_labels=control_labels,
    )
    if not eligible:
        raise ValueError(f"{input_file}: no perturbations with exactly 2 guides and >= {args.min_cells_per_guide} cells/guide")

    rows = []
    pred_rows = []
    for direction in (1, 2):
        dname = "A_to_B" if direction == 1 else "B_to_A"
        train_mask, test_mask = _build_direction_masks(
            obs=adata.obs,
            eligible=eligible,
            gene_col=args.gene_col,
            guide_col=args.guide_col,
            control_labels=control_labels,
            direction=direction,
            seed=args.seed + direction,
        )

        df, pred_df = _run_per_gene_direction(
            adata=adata,
            eligible=eligible,
            train_mask=train_mask,
            test_mask=test_mask,
            gene_col=args.gene_col,
            control_labels=control_labels,
            seed=args.seed + direction,
            direction_name=dname,
        )
        rows.append(df)
        pred_rows.append(pred_df)

    out_dir = args.output_root / f"{Path(input_file).stem}_two_guide_{args.mode}_mcg{args.min_cells_per_guide}_{args.preset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = pd.concat(rows, ignore_index=True)
    result.to_csv(out_dir / "directional_metrics_per_gene.csv", index=False)
    all_preds = pd.concat(pred_rows, ignore_index=True)
    all_preds.to_csv(out_dir / "directional_predictions_long.csv", index=False)

    summary = (
        result.groupby(["direction", "mode"], dropna=False)["average_precision"]
        .mean()
        .reset_index(name="mean_average_precision")
    )
    summary.to_csv(out_dir / "directional_summary.csv", index=False)

    meta = {
        "input_file": input_file,
        "mode": args.mode,
        "gene_col": args.gene_col,
        "guide_col": args.guide_col,
        "control_labels": list(control_labels),
        "min_cells_per_guide": int(args.min_cells_per_guide),
        "n_two_guide_targets": int(len(eligible)),
        "preset": args.preset,
        "seed": int(args.seed),
        "notes": [
            "Directional split by complementary guides per perturbation (A->B and B->A).",
            "Controls are split into disjoint halves per direction to avoid train/test leakage.",
        ],
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[{input_file}] targets={len(eligible)} -> {out_dir}")
    print(summary.to_string(index=False))
