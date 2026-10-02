from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

from preprocessing.star_counts_loader import StarCountsTCGALoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle)


def metric_summary(y_true, y_pred, min_n: int = 3) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[keep], y_pred[keep]
    out = {
        "n": len(y_true),
        "mae": np.nan,
        "rmse": np.nan,
        "spearman": np.nan,
        "spearman_p": np.nan,
        "pearson": np.nan,
        "pearson_p": np.nan,
    }
    if len(y_true) == 0:
        return out
    out["mae"] = mean_absolute_error(y_true, y_pred)
    out["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
    if len(y_true) >= min_n and np.std(y_true) > 0 and np.std(y_pred) > 0:
        sp = spearmanr(y_true, y_pred)
        pe = pearsonr(y_true, y_pred)
        out.update(
            {
                "spearman": sp.statistic,
                "spearman_p": sp.pvalue,
                "pearson": pe.statistic,
                "pearson_p": pe.pvalue,
            }
        )
    return out


def cohort_label(cancer_types: tuple[str, ...] | None) -> str:
    if not cancer_types or tuple(ct.upper() for ct in cancer_types) == ("ALL",):
        return "all"
    return "_".join(sorted(ct.upper() for ct in cancer_types))


def run_tmb_prediction(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    cancer_types: tuple[str, ...] | None = None,
    use_cache: bool = True,
    output_dir: str | Path | None = None,
) -> dict:
    import xgboost as xgb

    config_path = Path(config_path)
    config = load_config(config_path)
    loader = StarCountsTCGALoader(config_path=config_path, use_cache=use_cache)
    cohorts = loader.normalize_cancer_types(list(cancer_types) if cancer_types else None)

    print(f"Loading STAR-count expression for {cohorts or ['all default cohorts']}...", flush=True)
    expression = loader.load_expression_data(cancer_types=cohorts)
    print(f"Loading sample metadata for {cohorts or ['all default cohorts']}...", flush=True)
    meta = loader.load_sample_metadata(cancer_types=cohorts)
    common = expression.index.intersection(meta.index)
    if len(common) == 0:
        raise ValueError("No overlapping samples between expression and metadata.")

    expression = expression.loc[common].fillna(0.0)
    meta = meta.loc[common]
    print(
        f"Loading per-cohort MC3 event files for TMB target construction "
        f"({len(common):,} expression samples)...",
        flush=True,
    )
    tmb = loader.load_tmb_data(cancer_types=cohorts, sample_index=common)
    common = common.intersection(tmb.index)
    if len(common) == 0:
        raise ValueError("No samples remain after dropping samples without coding TMB events.")

    X = expression.loc[common]
    y = tmb.loc[common, "TMB_log2"]
    cancer = meta.loc[common, "Cancer"].astype(str).str.upper()

    label = cohort_label(cancer_types)
    if output_dir is None:
        output_root = Path(config.get("data", {}).get("output_path", "results_new"))
        if not output_root.is_absolute():
            output_root = PROJECT_ROOT / output_root
        expression_label = config.get("data", {}).get("expression_measure") or "star_counts"
        output_dir = output_root / "tmb_prediction" / str(expression_label) / label
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_splits = config.get("evaluation", {}).get("cv_folds", 5)
    random_state = config.get("preprocessing", {}).get("random_state", 42)
    if cancer.nunique() > 1:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(X, cancer)
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(X)

    oof = np.full(len(X), np.nan)
    fold_id = np.full(len(X), -1, dtype=int)
    fold_rows = []
    params = {"random_state": random_state}
    params.update(config.get("model", {}).get("xgboost", {}) or {})

    for fold, (train_idx, test_idx) in enumerate(split_iter, start=1):
        model = xgb.XGBRegressor(**params)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        oof[test_idx] = pred
        fold_id[test_idx] = fold
        row = {"fold": fold, **metric_summary(y.iloc[test_idx], pred)}
        fold_rows.append(row)
        print(f"Fold {fold}: n={row['n']:,}, Spearman={row['spearman']:.3f}, RMSE={row['rmse']:.3f}")

    pred_df = pd.DataFrame(
        {
            "sample": X.index,
            "Cancer": cancer.loc[X.index].values,
            "fold": fold_id,
            "TMB_log2_true": y.values,
            "TMB_log2_pred": oof,
            "TMB_true": np.power(2, y.values) - 1,
            "TMB_pred": np.maximum(np.power(2, oof) - 1, 0),
        }
    )

    summary_rows = [{"Cancer": "__OVERALL__", **metric_summary(pred_df["TMB_log2_true"], pred_df["TMB_log2_pred"])}]
    for cancer_type, sub in pred_df.groupby("Cancer"):
        summary_rows.append({"Cancer": cancer_type, **metric_summary(sub["TMB_log2_true"], sub["TMB_log2_pred"])})
    summary_df = pd.DataFrame(summary_rows)
    fold_df = pd.DataFrame(fold_rows)

    pred_df.to_csv(output_dir / "oof_predictions.csv", index=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    fold_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    with open(output_dir / "run_metadata.json", "w") as handle:
        json.dump(
            {
                "data_family": "xena_tcga_cohort_star_counts",
                "config": config,
                "cancer_types": cohorts or ["all"],
                "samples": int(X.shape[0]),
                "expression_features": int(X.shape[1]),
                "target": "log2(coding TMB + 1)",
                "zero_tmb_samples_retained": False,
                "samples_without_coding_mutation_events": "dropped",
            },
            handle,
            indent=2,
        )

    print(f"TMB outputs saved to: {output_dir}")
    return {"output_dir": output_dir, "predictions": pred_df, "summary": summary_df, "fold_metrics": fold_df}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TCGA expression-to-TMB prediction.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--cancer-types", nargs="+")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--job-id", type=int, default=0, help="Accepted for Condor compatibility.")
    args = parser.parse_args()

    run_tmb_prediction(
        config_path=args.config,
        cancer_types=tuple(args.cancer_types) if args.cancer_types else None,
        use_cache=not args.no_cache,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
