from __future__ import annotations

import argparse
from pathlib import Path

from main import DEFAULT_CONFIG_PATH, run_mutation_prediction
from preprocessing.star_counts_loader import DEFAULT_TCGA_COHORTS
from tmb_prediction import run_tmb_prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clean STAR-count prediction for each TCGA cohort and all cohorts.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--tasks", nargs="+", choices=["mutation", "tmb"], default=["mutation", "tmb"])
    parser.add_argument("--cohorts", nargs="+", default=DEFAULT_TCGA_COHORTS)
    parser.add_argument("--skip-all", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--keep-fold-dirs", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--max-shap-samples", type=int)
    parser.add_argument("--embeddings-only", action="store_true")
    args = parser.parse_args()

    cohort_jobs: list[tuple[str, ...] | None] = [(cohort,) for cohort in args.cohorts]
    if not args.skip_all:
        cohort_jobs.append(("all",))

    for cancer_types in cohort_jobs:
        label = "all" if cancer_types == ("all",) else "_".join(cancer_types or ["all"])
        print(f"\n===== {label} =====")
        if "mutation" in args.tasks:
            run_mutation_prediction(
                config_path=args.config,
                cancer_types=cancer_types,
                use_cache=not args.no_cache,
                keep_fold_dirs=args.keep_fold_dirs,
                run_shap_step=not args.skip_shap,
                max_shap_samples=args.max_shap_samples,
                embeddings_only=args.embeddings_only,
            )
        if "tmb" in args.tasks:
            run_tmb_prediction(
                config_path=args.config,
                cancer_types=cancer_types,
                use_cache=not args.no_cache,
            )


if __name__ == "__main__":
    main()
