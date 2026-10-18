"""Build Supplementary Tables S1-S5 from the run and figure outputs.

Run after the Runs/ pipelines and the Figure 1, 4 and 5 notebooks. Writes one workbook per table to
Tables/output/<method>/ (Tables/output/counts/<method>/ for the variant without CPM), for the run variant and External method of Figures/constants.py
(E2M_NORMALIZATION, E2M_EXTERNAL_METHOD).
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
# Run variant and External method, as in Figures/constants.py
VARIANT = "" if os.environ.get("E2M_NORMALIZATION", "cpm") == "cpm" else "counts"
EXTERNAL_METHOD = os.environ.get("E2M_EXTERNAL_METHOD") or None
METHOD_LABEL = EXTERNAL_METHOD or "main"
FIG1 = ROOT / "Figures" / "Figure1" / "output" / VARIANT / "Supplemental"
FIG4 = ROOT / "Figures" / "Figure4" / "output" / VARIANT / METHOD_LABEL / "source_data"
FIG4_NOTEBOOK = ROOT / "Figures" / "Figure4" / "Figure4.ipynb"
EXTERNAL = ROOT / "Runs" / "External" / "output" / VARIANT
EXTERNAL_CONFIG = ROOT / "Runs" / "External" / "config" / "e2m.yaml"
DRIVERS = ROOT / "Runs" / "ClinicalDrivers" / "output" / "summary" / "clinical_driver_results.csv"
SINGLE_CELL = ROOT / "Runs" / "SingleCell" / "output" / "plot_inputs"
TCGA = ROOT / "Runs" / "TCGA" / "output" / VARIANT
DRIVER_RUNS = ROOT / "Runs" / "ClinicalDrivers" / "output"
SINGLE_CELL_RUNS = ROOT / "Runs" / "SingleCell" / "output" / "runs"
OUT =Path(__file__).resolve().parent / "output" / VARIANT / METHOD_LABEL

# External cohorts: display name, disease context, data source, and which clinical analyses were run.
EXTERNAL_COHORTS = {
    "CPTAC_CMI": ("CPTAC/CMI", "Pan-cancer", "NCI GDC: CPTAC-2, CPTAC-3, CMI-MBC, and CMI-MPC",
                  "Not available", "Time-to-event", "Not available"),
    "HUGO": ("Hugo", "Melanoma immunotherapy", "GEO GSE78220 and Hugo et al. 2016 Table S1",
             "Available", "Time-to-event", "Not available"),
    "IMMUNOPOG": ("ImmunoPOG", "Pan-cancer immunotherapy", "BC Cancer GSC (bcgsc.ca/downloads/immunoPOG) and Pender et al. 2021 Table S1",
                  "Available", "Time-to-event", "Time-to-event"),
    "LIU": ("Liu", "Melanoma immunotherapy", "github.com/vanallenlab/schadendorf-pd1",
            "Available", "Time-to-event", "Time-to-event"),
    "METABRIC": ("METABRIC", "Breast cancer", "cBioPortal brca_metabric",
                 "Not available", "Time-to-event", "Not available"),
    "MORRISON": ("Morrison", "Melanoma immunotherapy", "github.com/ParkerICI/MORRISON-1-public",
                 "Available", "Not available", "Not available"),
    "RIAZ": ("Riaz", "Melanoma immunotherapy", "github.com/riazn/bms038_analysis",
             "Available", "Time-to-event", "Time-to-event"),
    "VAN_ALLEN": ("Van Allen", "Melanoma immunotherapy",
                  "github.com/vanallenlab/VanAllen_CTLA4_Science_RNASeq_TPM and Van Allen et al. 2015 Tables S1-S2",
                  "Available", "Time-to-event", "Time-to-event"),
}
CLINICAL_COHORTS = ["HUGO", "IMMUNOPOG", "LIU", "METABRIC", "RIAZ", "VAN_ALLEN", "MORRISON"]
TCGA_PARAM_CANCERS = ["BRCA", "LUAD", "UCEC"]  # TCGA cancers shown in the main Figure 4 panels


def table_s1() -> dict[str, pd.DataFrame]:
    corr = pd.read_csv(FIG1 / "figure1c_tmb_per_cancer_spearman.csv")
    quart = pd.read_csv(FIG1 / "tmb_predicted_quartile_observed_tmb_summary.csv")
    purity = pd.read_csv(FIG1 / "tmb_purity_partial_correlation_summary.csv")

    table = (
        corr.rename(columns={
            "Cancer": "cancer_type", "n": "n_samples", "mean_tmb": "mean_observed_coding_tmb",
            "median_tmb": "median_observed_coding_tmb", "spearman": "spearman_rho", "pearson": "pearson_r",
        })
        .merge(quart[["Cancer", "stratification_auc", "mannwhitney_p", "mannwhitney_fdr"]]
               .rename(columns={"Cancer": "cancer_type"}), on="cancer_type", how="left")
        .merge(purity[["Cancer", "purity_adjusted_spearman", "purity_adjusted_p", "purity_adjusted_fdr",
                       "delta_after_purity_adjustment"]]
               .rename(columns={"Cancer": "cancer_type", "purity_adjusted_spearman": "purity_adjusted_spearman_rho",
                                "purity_adjusted_p": "purity_adjusted_spearman_p",
                                "purity_adjusted_fdr": "purity_adjusted_spearman_fdr"}), on="cancer_type", how="left")
    )
    columns = [
        "cancer_type", "n_samples", "mean_observed_coding_tmb", "median_observed_coding_tmb",
        "spearman_rho", "spearman_p", "spearman_fdr", "pearson_r", "pearson_p", "pearson_fdr",
        "stratification_auc", "mannwhitney_p", "mannwhitney_fdr",
        "purity_adjusted_spearman_rho", "purity_adjusted_spearman_p", "purity_adjusted_spearman_fdr",
        "delta_after_purity_adjustment",
    ]
    table = table[columns].sort_values("spearman_rho", ascending=False).reset_index(drop=True)
    return {"Table_S1": table}


def table_s2() -> dict[str, pd.DataFrame]:
    summary = pd.read_csv(FIG1 / "mutation_rank_sum_summary_by_cancer.csv").rename(columns={
        "cohort": "cancer_type", "n_fdr_lt_0_05": "n_within_cancer_fdr_lt_0_05",
        "fraction_fdr_lt_0_05": "fraction_within_cancer_fdr_lt_0_05", "median_prevalence": "median_mutation_prevalence",
    })
    summary = summary[[
        "cancer_type", "n_samples", "n_target_genes", "n_testable_genes", "n_within_cancer_fdr_lt_0_05",
        "fraction_within_cancer_fdr_lt_0_05", "median_mutation_prevalence", "median_auprc", "median_normalized_auprc",
    ]]
    per_gene = pd.read_csv(FIG1 / "mutation_gene_rank_sum_tests.csv")[[
        "cohort", "gene", "n_samples", "n_mutated", "prevalence", "auprc", "normalized_auprc",
        "roc_auc", "rank_sum_p", "rank_sum_fdr_within_cancer", "rank_sum_fdr_global",
    ]]
    return {"Table_S2": summary, "Table_S2_per_gene": per_gene}


def _external_run(cohort: str) -> Path:
    """A cohort's External result: EXTERNAL_METHOD if the cohort ran it, else its first integration method in the config."""
    methods = yaml.safe_load(EXTERNAL_CONFIG.read_text(encoding="utf-8"))["cohorts"][cohort]["methods"]
    return EXTERNAL / cohort / (EXTERNAL_METHOD if EXTERNAL_METHOD in methods else methods[0])


def _bh(p: pd.Series) -> np.ndarray:
    return multipletests(p, method="fdr_bh")[1]


def _cptac_per_site_genes() -> pd.DataFrame:
    """Per-site metrics plus one-sided Mann-Whitney tests (BH within site) for the tested CPTAC/CMI genes."""
    run = _external_run("CPTAC_CMI")
    prob = pd.read_csv(run / "probabilities.csv.gz", index_col=0)
    labels = pd.read_csv(run / "labels.csv.gz", index_col=0).loc[prob.index, prob.columns]  # missing = not profiled
    site = pd.read_csv(run / "clinical.csv.gz", index_col=0)["cases.primary_site"].reindex(prob.index)
    metrics = pd.read_csv(run / "site_metrics.csv").rename(columns={"target": "gene"})
    metrics = metrics[metrics["evaluable"]].copy()
    pvals = []
    for s, gene in zip(metrics["site"], metrics["gene"]):
        idx = site.index[site.eq(s)]
        y, p = labels.loc[idx, gene], prob.loc[idx, gene]
        pvals.append(mannwhitneyu(p[y.eq(1)], p[y.eq(0)], alternative="greater").pvalue)
    metrics["wilcoxon_p_greater"] = pvals
    metrics["wilcoxon_fdr_site"] = metrics.groupby("site")["wilcoxon_p_greater"].transform(_bh)
    tested = metrics[(metrics["prevalence"] >= 0.05) & (metrics["n_positive"] >= 10)].copy()
    return tested.assign(cohort="CPTAC/CMI")


def _notebook_params() -> tuple[dict, dict, dict]:
    """TCGA_PARAMS, CPTAC_SITE_PARAMS_DEFAULT and EXTERNAL_CLUSTER_PARAMS as defined in the Figure 4 notebook."""
    nb = json.loads(FIG4_NOTEBOOK.read_text(encoding="utf-8"))
    found = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in ("TCGA_PARAMS", "CPTAC_SITE_PARAMS_DEFAULT", "EXTERNAL_CLUSTER_PARAMS"):
                    found[name] = ast.literal_eval(node.value)
    return found["TCGA_PARAMS"], found["CPTAC_SITE_PARAMS_DEFAULT"], found["EXTERNAL_CLUSTER_PARAMS"]


def table_s3() -> dict[str, pd.DataFrame]:
    # S3a: cohort overview
    rows = []
    for cohort, (label, context, source, response, os_, pfs) in EXTERNAL_COHORTS.items():
        m = json.loads((_external_run(cohort) / "manifest.json").read_text(encoding="utf-8"))
        cancers = m["tcga"]["cancers"]
        rows.append({
            "cohort": label, "disease_context": context, "data_source": source,
            "n_external_samples": m["external"]["samples"],
            "n_profiled_samples": m["external"]["profiled_samples"],
            "tcga_training_cohort": "TCGA pan-cancer" if len(cancers) > 1 else f"TCGA-{cancers[0]}",
            "n_tcga_training_samples": m["tcga"]["samples"],
            "tcga_expression_measure": m["tcga"]["expression"],
            "normalization": m["method"],
            "n_shared_expression_features": m["features"],
            "n_model_targets": m["targets"]["n"],
            "n_evaluable_mutation_targets": m["metrics"]["n_evaluable"],
            "median_auprc": m["metrics"]["median_auprc"],
            "median_normalized_auprc": m["metrics"]["median_normalized_auprc"],
            "response_analysis": response, "os_analysis": os_, "pfs_analysis": pfs,
        })
    overview = pd.DataFrame(rows).sort_values("cohort", key=lambda s: s.str.lower()).reset_index(drop=True)

    # S3b: per-gene metrics (CPTAC/CMI per site with significance tests; other cohorts at cohort level)
    per_gene = [_cptac_per_site_genes()]
    for cohort in ("HUGO", "IMMUNOPOG", "LIU", "METABRIC", "RIAZ", "VAN_ALLEN"):
        metrics = pd.read_csv(_external_run(cohort) / "metrics.csv").rename(columns={"target": "gene"})
        per_gene.append(metrics[metrics["evaluable"]].assign(cohort=EXTERNAL_COHORTS[cohort][0], site=np.nan))
    per_gene = pd.concat(per_gene, ignore_index=True)[[
        "cohort", "site", "gene", "n_samples", "n_positive", "prevalence", "auprc", "normalized_auprc", "roc_auc",
        "wilcoxon_p_greater", "wilcoxon_fdr_site",
    ]].sort_values(["cohort", "site", "normalized_auprc"], ascending=[True, True, False], na_position="first")

    # S3c: clinical-driver validation, evaluable targets
    drivers = pd.read_csv(DRIVERS)
    drivers = drivers[drivers["evaluable"].fillna(False).astype(bool)].drop(columns="evaluable")

    # S3d: clinical associations
    assoc = []
    cptac = pd.read_csv(FIG4 / "figure4_cptac_site_os_km_summary.csv")
    assoc.append(pd.DataFrame({
        "cohort": "CPTAC/CMI", "site": cptac["site"], "representation": cptac["space"], "endpoint": "OS",
        "test": "Log-rank", "n_samples": cptac["n_os"], "n_groups": cptac["n_groups"],
        "p_value": cptac["p_value"], "p_fdr": cptac["p_fdr"], "status": cptac["status"],
    }))
    for cohort in CLINICAL_COHORTS:
        path = FIG4 / f"figure4_{cohort.lower()}_km_summary.csv"
        if not path.exists():
            continue
        km = pd.read_csv(path)
        status = km["error"].fillna("Saved") if "error" in km else "Saved"
        assoc.append(pd.DataFrame({
            "cohort": EXTERNAL_COHORTS[cohort][0], "representation": km["space"], "endpoint": km["endpoint"],
            "test": "Log-rank", "n_samples": km["n_samples"], "n_groups": km["n_groups"],
            "p_value": km["p_value"], "status": status,
        }))
    for cohort in ("HUGO", "IMMUNOPOG", "LIU", "MORRISON", "RIAZ", "VAN_ALLEN"):
        resp = pd.read_csv(FIG4 / f"figure4_{cohort.lower()}_response_summary.csv")
        assoc.append(pd.DataFrame({
            "cohort": EXTERNAL_COHORTS[cohort][0], "representation": resp["space"],
            "endpoint": resp["variable"].map({"Response": "Binary response", "Full_response": "Full response"}),
            "test": "Chi-square", "test_statistic": resp["chi2"], "degrees_of_freedom": resp["dof"],
            "p_value": resp["p_value"], "status": "Saved",
        }))
    assoc = pd.concat(assoc, ignore_index=True).reindex(columns=[
        "cohort", "site", "representation", "endpoint", "test", "n_samples", "n_groups", "test_statistic",
        "degrees_of_freedom", "p_value", "p_fdr", "status",
    ])

    # S3e: clustering parameters used for the Figure 4 panels (the over-clustering rule of figure4_helpers)
    rule = {node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse((ROOT / "Figures" / "Figure4" / "figure4_helpers.py").read_text(encoding="utf-8")).body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("MAX_MUTATION_CLUSTERS", "OVERCLUSTERED_MUTATION_NN", "MIN_OTHER_CLUSTER_SHARE")}
    MUTATION_RULE = (f"if more than {rule['MAX_MUTATION_CLUSTERS']} clusters: n_neighbors {rule['OVERCLUSTERED_MUTATION_NN']}, "
                     f"clusters under {rule['MIN_OTHER_CLUSTER_SHARE']:.0%} of the samples grouped as 'other'")
    tcga, cptac_default, external = _notebook_params()
    params = []
    for cancer in TCGA_PARAM_CANCERS:
        p = tcga[cancer]
        for rep, nn, res in (("expression", "expr_nn", "expr_leiden_res"), ("embedding", "emb_nn", "emb_leiden_res"),
                             ("mutation_profile", "mut_nn", "mut_leiden_res")):
            params.append(("TCGA", cancer, rep, p[nn], p.get("mut_min_dist", 0.3) if rep == "mutation_profile" else 0.3, p[res],
                           MUTATION_RULE if rep == "mutation_profile" else ""))
    for rep, key in (("embedding", "emb"), ("expression", "expr")):
        params.append(("CPTAC/CMI (per site)", "all sites (default)", rep, cptac_default[f"{key}_nn"],
                       cptac_default[f"{key}_min_dist"], cptac_default[f"{key}_leiden_res"], ""))
    for cohort, p in external.items():
        for rep in ("embedding", "expression"):  # one set per cohort for both (older notebooks: one per space)
            q = p.get(rep, p)
            params.append(("External cohort", cohort, rep, q["n_neighbors"], q["min_dist"], q["leiden_resolution"], ""))
    params = pd.DataFrame(params, columns=["panel_group", "cohort_or_site", "representation", "umap_n_neighbors",
                                           "umap_min_dist", "leiden_resolution", "note"])

    return {
        "S3a_cohort_overview": overview, "S3b_external_per_gene": per_gene, "S3c_driver_validation": drivers,
        "S3d_clinical_association": assoc, "S3e_clustering_params": params,
    }


def normalized_auprc(auprc, prevalence):
    """(AUPRC - prevalence) / (1 - prevalence), as for TCGA and the external cohorts."""
    return (auprc - prevalence) / (1 - prevalence)


def table_s4() -> dict[str, pd.DataFrame]:
    # S4a: multitask perturbation prediction, mean over folds (folds without positives have no AUPRC and are skipped)
    datasets = [
        ("Adamson (broad)", "Adamson_10X010", "genetic (CRISPRi)", "cell"),
        ("Adamson UPR", "Adamson_10X005", "genetic (CRISPRi)", "cell"),
        ("Frangieh/Izar (multitask)", "Frangieh", "genetic (CRISPR)", "cell"),
        ("McFarland/Tsherniak", "McFarland", "drug", "cell line"),
        ("Replogle K562", "Replogle_K562", "genetic (CRISPRi)", "cell"),
        ("Replogle RPE1", "Replogle_RPE1", "genetic (CRISPRi)", "cell"),
    ]
    rows = []
    for name, folder, modality, group in datasets:
        folds = pd.read_csv(SINGLE_CELL / folder / "metric_summary_per_gene_folds.csv")
        folds = folds[folds["prevalence"] > 0].assign(normalized_auprc=lambda f: normalized_auprc(f["average_precision"], f["prevalence"]))
        agg = folds.groupby("gene").agg(n_folds=("fold", "nunique"), prevalence=("prevalence", "mean"),
                                        mean_auprc=("average_precision", "mean"), mean_normalized_auprc=("normalized_auprc", "mean"),
                                        mean_roc_auc=("roc_auc", "mean"))
        rows.append(agg.reset_index().rename(columns={"gene": "perturbation"})
                    .assign(dataset=name, modality=modality, cv_group=group))
    # Zhao/Sims: precision-recall only, from the out-of-fold predictions of the two drugs
    zhao = pd.read_csv(SINGLE_CELL / "Zhao" / "oof_probabilities_selected_targets.csv")
    summary = pd.read_csv(SINGLE_CELL / "Zhao" / "pr_curve_summary_selected_targets.csv").set_index("target")
    for drug in summary.index:
        folds = [(average_precision_score(f[f"y_true_{drug}"], f[drug]), f[f"y_true_{drug}"].mean())
                 for _, f in zhao.groupby("fold") if f[f"y_true_{drug}"].any()]
        aps, prevalences = np.array(folds).T
        rows.append(pd.DataFrame([{
            "perturbation": drug, "n_folds": len(aps), "prevalence": prevalences.mean(), "mean_auprc": aps.mean(),
            "mean_normalized_auprc": normalized_auprc(aps, prevalences).mean(), "mean_roc_auc": np.nan,
            "dataset": "Zhao/Sims", "modality": "drug", "cv_group": "sample",
        }]))
    multitask = pd.concat(rows, ignore_index=True)
    multitask = multitask[["dataset", "perturbation", "modality", "cv_group", "n_folds", "prevalence", "mean_auprc",
                           "mean_normalized_auprc", "mean_roc_auc"]]
    multitask = multitask.sort_values(["dataset", "mean_auprc"], ascending=[True, False]).reset_index(drop=True)

    # S4b: Frangieh/Izar per-gene XGBoost
    per_gene = pd.read_csv(SINGLE_CELL / "Frangieh_per_gene" / "metric_summary_per_gene.csv")
    per_gene = pd.DataFrame({
        "gene": per_gene["gene"], "n_positives": per_gene["n_positives"], "prevalence": per_gene["prevalence_mean"],
        "mean_auprc": per_gene["average_precision_mean"], "auprc_std": per_gene["average_precision_std"],
        "oof_auprc": per_gene["oof_average_precision"], "mean_roc_auc": per_gene["roc_auc_mean"],
    })
    per_gene["normalized_auprc"] = normalized_auprc(per_gene["mean_auprc"], per_gene["prevalence"])
    per_gene = per_gene.sort_values("mean_auprc", ascending=False).reset_index(drop=True)

    # S4c: Frangieh/Izar per-gene XGBoost by condition
    cond = pd.read_csv(SINGLE_CELL / "Frangieh_per_gene" / "metric_summary_per_gene_by_condition.csv")
    cond = pd.DataFrame({
        "gene": cond["gene"], "condition": cond["condition"], "n_positives": cond["n_positives"],
        "prevalence": cond["oof_prevalence"], "auprc": cond["oof_average_precision"], "roc_auc": cond["oof_roc_auc"],
    })
    cond["normalized_auprc"] = normalized_auprc(cond["auprc"], cond["prevalence"])
    cond["_order"] = cond["gene"].map({g: i for i, g in enumerate(per_gene["gene"])})
    cond = cond.sort_values(["_order", "condition"]).drop(columns="_order").reset_index(drop=True)

    # S4d: Tian/Kampmann CRISPRa cross-guide transfer
    tian = pd.read_csv(SINGLE_CELL / "Tian" / "directional_metrics_per_gene.csv")
    tian = tian.rename(columns={"direction": "guide_direction", "average_precision": "auprc"})[
        ["perturbation", "guide_direction", "roc_auc", "auprc", "prevalence"]]

    return {
        "S4a_perturbation_multitask": multitask, "S4b_frangieh_per_gene": per_gene,
        "S4c_frangieh_by_condition": cond, "S4d_tian_cross_guide": tian,
    }


# S5 text columns: how each input scale and integration method is described
SCALE_TEXT = {"counts": "counts", "tpm": "TPM", "log": "log scale (as provided)"}
LOG_TEXT = {"counts": "CPM, log1p", "tpm": "log1p", "log": "none"}
METHOD_TEXT = {
    "sample_rank": "within-sample gene ranks, scaled to (0, 1]",
    "rank": "within-gene ranks across samples, per dataset",
    "combat": "ComBat (inmoose pycombat_norm), TCGA and cohort as batches",
}
# Single-cell runs: display name and run folder in Runs/SingleCell/output/runs
SINGLE_CELL_DATASETS = [
    ("Adamson/Weissman 10X001", "admason_AdamsonWeissman2016_GSM2406675_10X001_mc300_simple_wide_run"),
    ("Adamson/Weissman 10X005 (UPR)", "admason_AdamsonWeissman2016_GSM2406677_10X005_mc300_simple_wide_run"),
    ("Adamson/Weissman 10X010", "admason_AdamsonWeissman2016_GSM2406681_10X010_mc300_simple_wide_run"),
    ("Frangieh/Izar", "FrangiehIzar2021_RNA_mc300_simple_wide_run"),
    ("McFarland/Tsherniak", "McFarlandTsherniak2020_mc300_simple_wide_run_cvgrp_cell_line"),
    ("Replogle K562", "ReplogleWeissman2022_K562_essential_mc300_simple_wide_run"),
    ("Replogle RPE1", "ReplogleWeissman2022_rpe1_mc300_simple_wide_run"),
    ("Zhao/Sims", "ZhaoSims2021_mc300_simple_wide_run_cvgrp_sample"),
    ("Tian/Kampmann", "TianKampmann2021_CRISPRa_two_guide_per-gene_mcg100_simple_wide_run"),
]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def table_s5() -> dict[str, pd.DataFrame]:
    rows = []

    # TCGA: one row per cross-validation run, plus TMB
    tcga = {"source": "UCSC Xena GDC STAR counts; MC3 mutations; GENCODE v36",
            "input_representation": "STAR counts, as log2(count + 1)",
            "transformation": "counts, CPM, log1p" if VARIANT == "" else "counts, log1p", "integration": "none"}
    for path in sorted(TCGA.glob("*/cv/run_metadata.json")):
        m, cohort = _json(path), path.parents[1].name
        rows.append({**tcga, "analysis": "TCGA mutation prediction",
                     "dataset": "TCGA pan-cancer" if cohort == "all" else f"TCGA-{cohort}",
                     "n_samples": m["n_samples"], "n_features": m["n_features"], "n_targets": m["n_targets"],
                     "validation": "5-fold CV, stratified by cancer type" if cohort == "all" else "5-fold CV"})
    m = _json(TCGA / "tmb" / "run_metadata.json")
    rows.append({**tcga, "analysis": "TCGA TMB prediction", "dataset": "TCGA pan-cancer", "n_samples": m["n_samples"],
                 "n_features": m["n_features"], "n_targets": 1, "validation": "5-fold CV, stratified by cancer type"})

    # External cohorts: every configured integration method
    config = yaml.safe_load(EXTERNAL_CONFIG.read_text(encoding="utf-8"))["cohorts"]
    for cohort, (label, _, source, *_) in EXTERNAL_COHORTS.items():
        for method in config[cohort]["methods"]:
            m = _json(EXTERNAL / cohort / method / "manifest.json")
            scale, cancers = m["external"]["scale"], m["tcga"]["cancers"]
            integration = METHOD_TEXT[method.removesuffix("_shared")]
            if method.endswith("_shared"):
                integration += "; one model shared by Hugo, Liu, and Riaz"
            rows.append({
                "analysis": "External validation", "dataset": label, "source": source,
                "input_representation": SCALE_TEXT[scale], "transformation": LOG_TEXT[scale], "integration": integration,
                "tcga_training_data": ("TCGA pan-cancer" if len(cancers) > 1 else f"TCGA-{cancers[0]}") + f" ({m['tcga']['expression']})",
                "n_tcga_training_samples": m["tcga"]["samples"], "n_samples": m["external"]["samples"],
                "n_genes_before": m["external"]["loader"].get("n_expression_features"), "n_features": m["features"],
                "n_targets": m["targets"]["n"], "validation": "trained on TCGA, tested on the cohort",
            })

    # Clinical drivers: gene-level runs of each integration (alteration targets counted with them)
    for path in sorted(DRIVER_RUNS.glob("integration_*/gene/*/manifest.json")):
        m = _json(path)
        alteration = path.parents[2] / "alteration" / path.parent.name / "targets.csv"
        n_targets = sum(pd.read_csv(p)["selected"].sum() for p in (path.parent / "targets.csv", alteration) if p.exists())
        rows.append({
            "analysis": "Clinical driver validation", "dataset": m["label"], "source": m["source"],
            "input_representation": m["external_expression_type"],
            "transformation": "log1p" if m["log1p_external"] else "none (as provided)",
            "integration": METHOD_TEXT[m["integration"]],
            "tcga_training_data": "TCGA-" + "/".join(m["tcga_cancers"]) + f" ({m['tcga_expression_measure']})",
            "n_tcga_training_samples": m["tcga_samples"], "n_samples": m["external_samples"],
            "n_genes_before": m.get("n_expression_features"), "n_features": m["n_features"], "n_targets": int(n_targets),
            "validation": "trained on TCGA, tested on the cohort",
        })

    # Single-cell datasets
    single_cell = {"input_representation": "counts (adata.X)", "transformation": "normalize_total to 10,000, log1p",
                   "integration": "none"}
    for name, folder in SINGLE_CELL_DATASETS:
        m = _json(SINGLE_CELL_RUNS / folder / "run_metadata.json")
        group = m.get("cv_group_col")
        rows.append({
            **single_cell, "analysis": "Single-cell perturbation", "dataset": name,
            "source": f"scPerturb (Zenodo 7041849): {m['input_file']}", "n_samples": m.get("n_samples"),
            "n_features": m.get("n_features"), "n_targets": m.get("n_targets", m.get("n_two_guide_targets")),
            "validation": ("train on one guide, test on the other" if name == "Tian/Kampmann"
                           else f"5-fold CV, grouped by {group}" if group else "5-fold CV, cell level"),
        })
    m = _json(SINGLE_CELL_RUNS / "ccle_tmb_and_multitask" / "run_metadata.json")
    rows.append({
        **single_cell, "transformation": "genes in fewer than 3 cells removed; normalize_total to 10,000, log1p",
        "analysis": "Single-cell mutation", "dataset": "CCLE", "source": "Broad Single Cell Portal SCP542; DepMap mutations",
        "n_samples": m["n_cells"], "n_features": m["n_features"], "n_targets": m["n_targets"],
        "validation": "5-fold CV, grouped by cell line",
    })

    manifest = pd.DataFrame(rows).reindex(columns=[
        "analysis", "dataset", "source", "input_representation", "transformation", "integration", "tcga_training_data",
        "n_tcga_training_samples", "n_samples", "n_genes_before", "n_features", "n_targets", "validation",
    ])
    counts = ["n_tcga_training_samples", "n_samples", "n_genes_before", "n_features", "n_targets"]
    manifest[counts] = manifest[counts].astype("Int64")  # whole numbers, blank where not applicable
    return {"Table_S5": manifest}


def write(name: str, sheets: dict[str, pd.DataFrame]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT / f"{name}.xlsx") as writer:
        for sheet, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet, index=False)
    print(f"Wrote {OUT / f'{name}.xlsx'}")


def main() -> None:
    write("Supplementary_Table1", table_s1())
    write("Supplementary_Table2", table_s2())
    write("Supplementary_Table3", table_s3())
    write("Supplementary_Table4", table_s4())
    write("Supplementary_Table5", table_s5())


if __name__ == "__main__":
    main()
