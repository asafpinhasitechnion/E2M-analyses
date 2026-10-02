from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from interpretation.shap_analysis import save_shap_summary
from preprocessing.star_counts_loader import StarCountsTCGALoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle)


def limit_mutation_genes_topk(mutation_data: pd.DataFrame, top_k: int | None) -> pd.DataFrame:
    if top_k is None or top_k <= 0 or mutation_data.shape[1] <= top_k:
        return mutation_data
    counts = mutation_data.apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=0)
    keep = counts.sort_values(ascending=False).head(top_k).index.tolist()
    return mutation_data.loc[:, keep]


def stable_int_from_str(value: str) -> int:
    return zlib.crc32(value.encode("utf-8")) & 0xFFFFFFFF


def cohort_label(cancer_types: tuple[str, ...] | None) -> str:
    if not cancer_types or tuple(ct.upper() for ct in cancer_types) == ("ALL",):
        return "all"
    return "_".join(sorted(ct.upper() for ct in cancer_types))


def select_good_genes(kfold_dir: Path, threshold: float) -> list[str]:
    summary_path = kfold_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing k-fold summary: {summary_path}")
    summary = pd.read_csv(summary_path, index_col=0)
    required = {"auprc_mean", "prevalence_mean"}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"summary.csv missing required columns: {sorted(missing)}")
    denominator = (1.0 - summary["prevalence_mean"]).replace(0, np.nan)
    norm_auprc = (summary["auprc_mean"] - summary["prevalence_mean"]) / denominator
    norm_auprc = norm_auprc.replace([np.inf, -np.inf], np.nan).dropna()
    return norm_auprc[norm_auprc > threshold].index.tolist()


def strip_gene_symbol(name: str) -> str:
    return name.split("|")[-1] if "|" in name else name


def save_kfold_summary_for_app(kfold_dir: Path, output_dir: Path) -> None:
    summary_path = kfold_dir / "summary.csv"
    if not summary_path.exists():
        return

    summary = pd.read_csv(summary_path, index_col=0)
    keep_cols = [col for col in ["auprc_mean", "auprc_std", "prevalence_mean"] if col in summary.columns]
    summary = summary.loc[:, keep_cols].copy()
    summary.index.name = "gene"
    summary.to_csv(output_dir / "kfold_summary.csv")


def save_beeswarm_parquet(
    output_dir: Path,
    gene: str,
    sample_ids: list[str],
    shap_values: np.ndarray,
    X_perm: np.ndarray,
    feature_symbols_perm: list[str],
    top_n: int,
) -> None:
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]

    beeswarm_data = {"sample_id": sample_ids}
    for feat_idx in top_idx:
        symbol = feature_symbols_perm[feat_idx]
        beeswarm_data[symbol] = shap_values[:, feat_idx].astype(np.float32)
        beeswarm_data[f"x_{symbol}"] = X_perm[:, feat_idx].astype(np.float32)

    try:
        pd.DataFrame(beeswarm_data).to_parquet(output_dir / f"beeswarm_{gene}.parquet", index=False)
    except ImportError as exc:
        raise ImportError("Saving beeswarm parquet files requires pyarrow or fastparquet.") from exc


def run_shap(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    cancer_types: tuple[str, ...] | None = None,
    kfold_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    target_genes: list[str] | None = None,
    use_cache: bool = True,
    job_id: int = 0,
    max_samples: int | None = None,
    save_app_outputs: bool | None = None,
) -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    label = cohort_label(cancer_types)

    output_root = Path(config.get("data", {}).get("output_path", "results_new"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    kfold_dir = Path(kfold_dir) if kfold_dir else output_root / "multitask_nn" / label / "kfold_prediction"
    output_dir = Path(output_dir) if output_dir else output_root / "shap" / label
    output_dir.mkdir(parents=True, exist_ok=True)

    if not target_genes:
        threshold = config.get("interpretation", {}).get("norm_score_threshold", 0.1)
        target_genes = select_good_genes(kfold_dir, threshold=threshold)
    print(f"SHAP target genes selected: {len(target_genes):,}")

    data_loader = StarCountsTCGALoader(config_path=config_path, use_cache=use_cache)
    cohorts = data_loader.normalize_cancer_types(list(cancer_types) if cancer_types else None)
    expression_data, mutation_data = data_loader.preprocess_data(cancer_types=cohorts)
    mutation_data = mutation_data.loc[expression_data.index]
    mutation_data = limit_mutation_genes_topk(
        mutation_data,
        top_k=config.get("data", {}).get("top_k_mutation_genes", 500),
    )

    target_genes = [gene for gene in target_genes if gene in mutation_data.columns]
    if not target_genes:
        raise ValueError("No selected SHAP target genes are present in mutation matrix.")

    if max_samples is not None and len(expression_data) > max_samples:
        rng = np.random.default_rng(config.get("preprocessing", {}).get("random_state", 42))
        keep_idx = rng.choice(len(expression_data), size=max_samples, replace=False)
        expression_data = expression_data.iloc[keep_idx].copy()
        mutation_data = mutation_data.loc[expression_data.index].copy()

    try:
        import shap
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("SHAP extraction requires shap and xgboost.") from exc

    X_values = expression_data.values
    feature_names = expression_data.columns.tolist()
    sample_ids = expression_data.index
    random_state = config.get("preprocessing", {}).get("random_state", 42)
    xgb_params = dict(config.get("model", {}).get("xgboost", {}) or {})
    cancer_seed = stable_int_from_str(label)
    interpretation_cfg = config.get("interpretation", {})
    if save_app_outputs is None:
        save_app_outputs = interpretation_cfg.get("save_shap_app_outputs", True)
    top_n_per_sample = interpretation_cfg.get("top_n_shap_features_per_sample", 20)
    top_n_beeswarm = interpretation_cfg.get("top_n_shap_beeswarm_features", top_n_per_sample)
    feature_symbols = [strip_gene_symbol(feature) for feature in feature_names]
    sample_id_list = sample_ids.astype(str).tolist()

    if save_app_outputs:
        save_kfold_summary_for_app(kfold_dir=kfold_dir, output_dir=output_dir)

    all_feature_rows = []
    app_matrix_rows = []
    for idx, gene in enumerate(target_genes, start=1):
        y = mutation_data[gene].astype(int).values
        if len(np.unique(y)) < 2:
            print(f"[{idx}/{len(target_genes)}] Skipping {gene}: one class present.")
            continue

        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        scale_pos_weight = neg / pos if pos else 1.0
        gene_seed = (random_state ^ job_id ^ cancer_seed ^ stable_int_from_str(gene)) & 0x7FFFFFFF
        perm = np.random.default_rng(gene_seed).permutation(X_values.shape[1])

        X_perm = X_values[:, perm]
        feature_names_perm = [feature_names[i] for i in perm]
        feature_symbols_perm = [feature_symbols[i] for i in perm]

        classifier_params = {
            **xgb_params,
            "scale_pos_weight": scale_pos_weight,
            "random_state": gene_seed,
            "eval_metric": xgb_params.get("eval_metric", "logloss"),
        }
        model = xgb.XGBClassifier(**classifier_params)
        model.fit(X_perm, y)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_perm)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        shap_values = np.asarray(shap_values)

        gene_dir = output_dir / gene
        save_shap_summary(
            shap_values=shap_values,
            feature_names=feature_names_perm,
            output_path=gene_dir / "shap_summary.csv",
            sample_ids=sample_ids,
            top_n=top_n_per_sample,
        )

        feature_summary = pd.read_csv(gene_dir / "shap_summary_feature_summary.csv")
        feature_summary.insert(0, "target_gene", gene)
        all_feature_rows.append(feature_summary)

        if save_app_outputs:
            save_beeswarm_parquet(
                output_dir=output_dir,
                gene=gene,
                sample_ids=sample_id_list,
                shap_values=shap_values,
                X_perm=X_perm,
                feature_symbols_perm=feature_symbols_perm,
                top_n=top_n_beeswarm,
            )
            mean_shap = shap_values.mean(axis=0)
            for feat_idx, mean_value in enumerate(mean_shap):
                if float(mean_value) != 0.0:
                    app_matrix_rows.append(
                        {
                            "feature": feature_symbols_perm[feat_idx],
                            "target": gene,
                            "mean_shap": float(mean_value),
                        }
                    )

        print(f"[{idx}/{len(target_genes)}] SHAP complete: {gene} (+{pos} / -{neg})")

    if all_feature_rows:
        combined = pd.concat(all_feature_rows, ignore_index=True)
        combined.to_csv(output_dir / "shap_feature_summary_all_genes.csv", index=False)

    if save_app_outputs and app_matrix_rows:
        app_matrix = pd.DataFrame(app_matrix_rows)
        app_matrix["mean_shap"] = app_matrix["mean_shap"].astype(np.float32)
        app_matrix.to_csv(output_dir / "shap_summary_feature_summary_matrix.csv", index=False)

    print(f"SHAP outputs saved to: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run per-gene XGBoost SHAP for TCGA mutation prediction.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--cancer-types", nargs="+")
    parser.add_argument("--kfold-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-genes", nargs="+")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--job-id", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--skip-shap-app-outputs", action="store_true")
    args = parser.parse_args()

    run_shap(
        config_path=args.config,
        cancer_types=tuple(args.cancer_types) if args.cancer_types else None,
        kfold_dir=args.kfold_dir,
        output_dir=args.output_dir,
        target_genes=args.target_genes,
        use_cache=not args.no_cache,
        job_id=args.job_id,
        max_samples=args.max_samples,
        save_app_outputs=False if args.skip_shap_app_outputs else None,
    )


if __name__ == "__main__":
    main()
