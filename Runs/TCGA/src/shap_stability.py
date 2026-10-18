"""SHAP stability in one cohort: refit each SHAP target's XGBoost on resampled samples and keep the SHAP of every run.

Each run fits on a label-stratified subsample, with the features in the saved model's order and the samples in their
original order, and explains all samples, so runs differ only in which samples were fitted.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import e2m
from e2m.config import load_config
from e2m.interpretation import _explain_tree

RUN_ROOT = Path(__file__).resolve().parents[1]


def subsample(labels: np.ndarray, fraction: float, rng) -> np.ndarray:
    """Sorted row indices of a label-stratified subsample."""
    picked = [rng.choice(rows, size=round(fraction * len(rows)), replace=False)
              for rows in (np.flatnonzero(labels == 0), np.flatnonzero(labels == 1))]
    return np.sort(np.concatenate(picked))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=RUN_ROOT / "config" / "e2m.yaml")
    parser.add_argument("--data-dir", type=Path, default=RUN_ROOT / "data")
    parser.add_argument("--cohort", default="BRCA")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--fraction", type=float, default=0.8)
    args = parser.parse_args()

    config = load_config(args.config)
    method = config["interpretation"]["method"]
    cohort_output = RUN_ROOT / "output" / args.cohort
    output = cohort_output / "shap_stability"
    data = e2m.Dataset.from_tcga([args.cohort], data_dir=args.data_dir, config=args.config, with_tmb=False)
    model = e2m.E2MModel.load(cohort_output / "model")
    expression = model._align(data, 1.0)  # the model's feature order, as in model.explain
    metrics = pd.read_csv(cohort_output / "cv" / "metrics.csv", index_col=0)
    targets = metrics.index[metrics["normalized_auprc"] > config["run"]["shap_min_normalized_auprc"]]

    for target in targets:
        labels = data.mutations[target].reindex(expression.index).astype(int).to_numpy()
        rng = np.random.default_rng(config["preprocessing"]["random_state"])
        mean_abs, mean_signed, fitted = {}, {}, {}
        for run in ["full", *range(args.runs)]:
            rows = np.arange(len(labels)) if run == "full" else subsample(labels, args.fraction, rng)
            values, _ = _explain_tree(expression.iloc[rows], pd.Series(labels[rows]), expression, method,
                                      config["model"][method])
            name = run if run == "full" else f"run_{run}"
            mean_abs[name], mean_signed[name] = np.abs(values).mean(axis=0), values.mean(axis=0)
            fitted[name] = np.isin(np.arange(len(labels)), rows)
            print(f"{target} {name}", flush=True)

        target_output = output / target
        target_output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(mean_abs, index=expression.columns).rename_axis("feature").to_csv(target_output / "mean_abs_shap.csv.gz")
        pd.DataFrame(mean_signed, index=expression.columns).rename_axis("feature").to_csv(target_output / "mean_shap.csv.gz")
        pd.DataFrame(fitted, index=expression.index).rename_axis("sample_id").to_csv(target_output / "fitted_samples.csv.gz")


if __name__ == "__main__":
    main()
