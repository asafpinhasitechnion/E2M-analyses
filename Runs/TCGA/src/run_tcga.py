"""TCGA mutation prediction (cross-validation, full-data model, embeddings, SHAP) per cohort, and pan-cancer TMB, with E2M."""

import argparse
import json
from pathlib import Path

import pandas as pd

import e2m
from e2m.config import load_config

RUN_ROOT = Path(__file__).resolve().parents[1]


def run_cohort(name, cancers, tasks, args, config, overrides, output):
    output.mkdir(parents=True, exist_ok=True)
    data = e2m.Dataset.from_tcga(cancers, data_dir=args.data_dir, config=args.config, data_overrides=overrides, with_tmb=False)
    (output / "preprocessing_manifest.json").write_text(json.dumps(data.manifest, indent=2), encoding="utf-8")

    if "mutation" in tasks:
        e2m.E2MModel(config).cross_validate(data, output=output / "cv")
        model = e2m.E2MModel(config).fit(data)
        model.save(output / "model")
        model.embed(data).to_csv(output / "embeddings.csv")
        model.head_weights().to_csv(output / "head_weights.csv")

    if "shap" in tasks:
        model = e2m.E2MModel.load(output / "model")
        metrics = pd.read_csv(output / "cv" / "metrics.csv", index_col=0)
        targets = metrics.index[metrics["normalized_auprc"] > config["run"]["shap_min_normalized_auprc"]]
        print(f"{name}: SHAP for {len(targets)} of {len(metrics)} targets", flush=True)
        for target in targets:
            model.explain(data, target, method=config["interpretation"]["method"], output=output / "shap" / target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=RUN_ROOT / "config" / "e2m.yaml")
    parser.add_argument("--data-dir", type=Path, default=RUN_ROOT / "data")
    parser.add_argument("--normalization", choices=["cpm", "none"], help="Overrides data.normalization.")
    parser.add_argument("--cohorts", nargs="+", help='Cancer codes, and "all" for the pan-cancer run. Default: run.cohorts and all.')
    parser.add_argument("--tasks", nargs="+", choices=["mutation", "shap", "tmb"], default=["mutation", "shap", "tmb"])
    args = parser.parse_args()

    overrides = {"normalization": args.normalization} if args.normalization else {}
    config = load_config(args.config, {"data": overrides})
    all_cohorts = config["run"]["cohorts"]
    cohorts = [c.upper() if c != "all" else c for c in (args.cohorts or [*all_cohorts, "all"])]
    output = RUN_ROOT / "output" / ("cpm" if config["data"]["normalization"] == "cpm" else "counts")
    e2m.set_verbose()

    if {"mutation", "shap"} & set(args.tasks):
        for name in cohorts:
            cancers = all_cohorts if name == "all" else [name]
            run_cohort(name, cancers, args.tasks, args, config, overrides, output / name)

    if "tmb" in args.tasks:
        cancers = all_cohorts if "all" in cohorts else cohorts
        data = e2m.Dataset.from_tcga(cancers, data_dir=args.data_dir, config=args.config, data_overrides=overrides)
        e2m.TmbModel(config=config).cross_validate(data, output=output / "tmb")


if __name__ == "__main__":
    main()
