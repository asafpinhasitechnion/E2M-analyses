from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from models.model_factory import ModelFactory
from preprocessing.star_counts_loader import StarCountsTCGALoader
from training.extract_weights import extract_sample_embeddings, train_and_extract_head_weights
from training.trainer import run_kfold_training
from extract_shap import run_shap


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as handle:
        return yaml.safe_load(handle)


def cohort_label(cancer_types: tuple[str, ...] | None) -> str:
    if not cancer_types or tuple(ct.upper() for ct in cancer_types) == ("ALL",):
        return "all"
    return "_".join(sorted(ct.upper() for ct in cancer_types))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def limit_mutation_genes_topk(mutation_data: pd.DataFrame, top_k: int | None) -> pd.DataFrame:
    if top_k is None or top_k <= 0 or mutation_data.shape[1] <= top_k:
        return mutation_data
    counts = mutation_data.apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=0)
    keep = counts.sort_values(ascending=False).head(top_k).index.tolist()
    return mutation_data.loc[:, keep]


def run_mutation_prediction(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    cancer_types: tuple[str, ...] | None = None,
    use_cache: bool = True,
    keep_fold_dirs: bool = False,
    run_shap_step: bool = True,
    job_id: int = 0,
    max_shap_samples: int | None = None,
    embeddings_only: bool = False,
) -> dict:
    config_path = Path(config_path)
    config = load_config(config_path)
    config.setdefault("model", {})["name"] = "multitask_nn"
    if embeddings_only:
        run_shap_step = False

    loader = StarCountsTCGALoader(config_path=config_path, use_cache=use_cache)
    cohorts = loader.normalize_cancer_types(list(cancer_types) if cancer_types else None)
    expression_data, mutation_data = loader.preprocess_data(cancer_types=cohorts)

    mutation_data = mutation_data.loc[expression_data.index]
    top_k = config.get("data", {}).get("top_k_mutation_genes", 500)
    mutation_data = limit_mutation_genes_topk(mutation_data, top_k=top_k)
    print(f"Mutation targets kept: {mutation_data.shape[1]:,} genes (top_k={top_k})")
    if expression_data.shape[0] != mutation_data.shape[0]:
        raise ValueError("Expression/mutation sample mismatch after alignment.")

    output_root = Path(config.get("data", {}).get("output_path", "results_new"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    model_dir = output_root / "multitask_nn" / cohort_label(cancer_types)
    kfold_dir = model_dir / "kfold_prediction"
    embeddings_dir = model_dir / "gene_embeddings"
    sample_embeddings_dir = embeddings_dir / "sample_embeddings"
    model_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "data_family": "xena_tcga_cohort_star_counts",
        "config": config,
        "cancer_types": cohorts or ["all"],
        "samples": int(expression_data.shape[0]),
        "expression_features": int(expression_data.shape[1]),
        "mutation_targets": mutation_data.columns.tolist(),
        "top_k_mutation_genes": top_k,
        "collapse_duplicate_gene_symbols": bool(
            config.get("data", {}).get("collapse_duplicate_gene_symbols", True)
        ),
    }
    with open(model_dir / "run_metadata.json", "w") as handle:
        json.dump(metadata, handle, indent=2)

    model_factory = ModelFactory()
    model = model_factory.get_model(
        model_name="multitask_nn",
        input_size=expression_data.shape[1],
        output_size=mutation_data.shape[1],
        config=config,
    )

    cv_folds = config.get("evaluation", {}).get("cv_folds", 5)
    random_state = config.get("preprocessing", {}).get("random_state", 42)
    if not embeddings_only:
        set_seed(random_state)
        print(f"Running {cv_folds}-fold mutation prediction for {cohort_label(cancer_types)}...")
        run_kfold_training(
            model=model,
            X=expression_data,
            Y=mutation_data,
            k=cv_folds,
            output_dir=kfold_dir,
            config_meta=metadata,
            random_state=random_state,
            label="multitask_nn",
        )

        if not keep_fold_dirs:
            for fold_dir in kfold_dir.glob("fold_*"):
                if fold_dir.is_dir():
                    shutil.rmtree(fold_dir)

    set_seed(random_state)
    mt_cfg = config.get("model", {}).get("multitask_nn", {})
    head_weights = train_and_extract_head_weights(
        model_factory=model_factory,
        X=expression_data,
        Y=mutation_data,
        config=config,
        hidden_layers=mt_cfg.get("hidden_layers", [512, 256]),
        head_layers=mt_cfg.get("head_layers", []),
        output_dir=embeddings_dir,
        save_model=True,
    )
    with open(embeddings_dir / "features.json", "w") as handle:
        json.dump(expression_data.columns.tolist(), handle, indent=2)
    embeddings_result = extract_sample_embeddings(
        model=head_weights["model"],
        X=expression_data,
        sample_ids=expression_data.index,
        output_dir=sample_embeddings_dir,
        batch_size=mt_cfg.get("batch_size", 128),
    )

    shap_dir = model_dir / "shap"
    if run_shap_step:
        print("Running SHAP interpretation from k-fold summary...")
        run_shap(
            config_path=config_path,
            cancer_types=cancer_types,
            kfold_dir=kfold_dir,
            output_dir=shap_dir,
            use_cache=use_cache,
            job_id=job_id,
            max_samples=max_shap_samples,
        )
    else:
        print("Skipping SHAP interpretation.")

    print("Mutation prediction complete.")
    print(f"Results: {model_dir}")
    print(f"K-fold predictions: {kfold_dir / 'combined_predictions'}")
    print(f"Embeddings: {embeddings_dir}")
    print(f"SHAP: {shap_dir if run_shap_step else 'skipped'}")

    return {
        "model_dir": model_dir,
        "kfold_dir": kfold_dir,
        "embeddings_dir": embeddings_dir,
        "sample_embeddings_dir": sample_embeddings_dir,
        "shap_dir": shap_dir if run_shap_step else None,
        "sample_embeddings": embeddings_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TCGA expression-to-mutation prediction.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--cancer-types", nargs="+", help="TCGA cancer types, or all.")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--keep-fold-dirs", action="store_true")
    parser.add_argument("--skip-shap", action="store_true", help="Run prediction and embeddings but skip SHAP.")
    parser.add_argument("--max-shap-samples", type=int, help="Optional cap on samples used for SHAP.")
    parser.add_argument("--job-id", type=int, default=0, help="Condor process id used in SHAP feature permutations.")
    parser.add_argument("--embeddings-only", action="store_true", help="Only train the full-data model and extract embeddings.")
    args = parser.parse_args()

    run_mutation_prediction(
        config_path=args.config,
        cancer_types=tuple(args.cancer_types) if args.cancer_types else None,
        use_cache=not args.no_cache,
        keep_fold_dirs=args.keep_fold_dirs,
        run_shap_step=not args.skip_shap,
        job_id=args.job_id,
        max_shap_samples=args.max_shap_samples,
        embeddings_only=args.embeddings_only,
    )


if __name__ == "__main__":
    main()
