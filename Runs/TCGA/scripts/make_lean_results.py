from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def copy_file(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_tree(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return True


def read_shap_feature_summary(path: Path, target_gene: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        df = pd.read_csv(path, index_col=0).reset_index().rename(columns={"index": "feature"})
    if target_gene is not None and "target_gene" not in df.columns:
        df.insert(0, "target_gene", target_gene)
    return df


def write_combined_long_shap(src_shap_dir: Path, dst_shap_dir: Path, cohort: str) -> dict:
    status = {
        "source": None,
        "combined_long": False,
        "rows": 0,
        "target_genes": 0,
    }
    if not src_shap_dir.exists():
        return status

    status["source"] = src_shap_dir.name
    combined_path = src_shap_dir / "shap_feature_summary_all_genes.csv"
    if combined_path.exists():
        combined = read_shap_feature_summary(combined_path)
    else:
        rows = []
        for gene_dir in sorted(path for path in src_shap_dir.iterdir() if path.is_dir()):
            summary_path = gene_dir / "shap_summary_feature_summary.csv"
            if summary_path.exists():
                rows.append(read_shap_feature_summary(summary_path, target_gene=gene_dir.name))
        if not rows:
            return status
        combined = pd.concat(rows, ignore_index=True)

    if "cohort" not in combined.columns:
        combined.insert(0, "cohort", cohort)
    if "target_gene" not in combined.columns:
        raise ValueError(f"Combined SHAP summary lacks target_gene column: {combined_path}")

    dst_shap_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(dst_shap_dir / "shap_feature_summary_long.csv", index=False)
    status["combined_long"] = True
    status["rows"] = int(combined.shape[0])
    status["target_genes"] = int(combined["target_gene"].nunique())
    return status


MODEL_FILES = ["multitask_nn.pt", "head_weights.npy", "gene_names.json", "features.json", "training_curve.csv"]
TMB_FILES = ["oof_predictions.csv", "summary.csv", "fold_metrics.csv", "run_metadata.json"]


def lean_one_cohort(cohort_dir: Path, dst_root: Path) -> dict:
    cohort = cohort_dir.name
    dst_cohort_dir = dst_root / cohort
    dst_cohort_dir.mkdir(parents=True, exist_ok=True)

    sample_embeddings_src = (
        cohort_dir / "gene_embeddings" / "sample_embeddings" / "sample_embeddings.csv"
    )
    sample_embeddings_dst = (
        dst_cohort_dir / "gene_embeddings" / "sample_embeddings" / "sample_embeddings.csv"
    )

    shap_src = cohort_dir / "shap"
    if not shap_src.exists():
        shap_src = cohort_dir / "shap_analysis"

    return {
        "cohort": cohort,
        "kfold_prediction": copy_tree(
            cohort_dir / "kfold_prediction",
            dst_cohort_dir / "kfold_prediction",
        ),
        "sample_embeddings_csv": copy_file(sample_embeddings_src, sample_embeddings_dst),
        "model_files": [
            name
            for name in MODEL_FILES
            if copy_file(cohort_dir / "gene_embeddings" / name, dst_cohort_dir / "model" / name)
        ],
        "run_metadata": copy_file(
            cohort_dir / "run_metadata.json",
            dst_cohort_dir / "run_metadata.json",
        ),
        "shap": write_combined_long_shap(shap_src, dst_cohort_dir / "shap", cohort=cohort),
    }


def make_lean_results(src_root: Path, dst_root: Path, tmb_dir: Path) -> list[dict]:
    if not src_root.exists():
        raise FileNotFoundError(f"Source multitask result root not found: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)
    manifest = [
        lean_one_cohort(cohort_dir, dst_root)
        for cohort_dir in sorted(path for path in src_root.iterdir() if path.is_dir())
    ]
    manifest.append(
        {"tmb": [name for name in TMB_FILES if copy_file(tmb_dir / name, dst_root / "tmb" / name)]}
    )

    with open(dst_root / "lean_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def resolve_project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a compact TCGA_E2M_prediction multitask result bundle."
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("output") / "multitask_nn",
        help="Source multitask result root. Relative paths are resolved from the run folder.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=Path("output") / "lean",
        help="Destination lean result root. Relative paths are resolved from the run folder.",
    )
    parser.add_argument(
        "--tmb-dir",
        type=Path,
        default=Path("output") / "tmb_prediction" / "counts" / "all",
        help="Pan-cancer TMB result folder. Relative paths are resolved from the run folder.",
    )
    args = parser.parse_args()

    manifest = make_lean_results(
        src_root=resolve_project_path(args.src_root),
        dst_root=resolve_project_path(args.dst_root),
        tmb_dir=resolve_project_path(args.tmb_dir),
    )
    tmb_row = manifest.pop()

    n_embeddings = sum(row["sample_embeddings_csv"] for row in manifest)
    n_shap_cohorts = sum(row["shap"]["combined_long"] for row in manifest)
    n_shap_rows = sum(row["shap"]["rows"] for row in manifest)
    print(f"Lean multitask results created: {resolve_project_path(args.dst_root)}")
    print(f"Cohorts processed: {len(manifest)}")
    print(f"Cohorts with full-cohort model files: {sum(bool(row['model_files']) for row in manifest)}")
    print(f"TMB files copied: {len(tmb_row['tmb'])}")
    print(f"Cohorts with sample embeddings: {n_embeddings}")
    print(f"Cohorts with combined SHAP long table: {n_shap_cohorts}")
    print(f"Combined SHAP long rows: {n_shap_rows:,}")


if __name__ == "__main__":
    main()
