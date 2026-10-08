"""Build Supplementary Tables S1-S4 from the run and figure outputs.

Run after the Runs/ pipelines and the Figure 1, 4 and 5 notebooks. Writes one workbook per table to Tables/output/.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
FIG1 = ROOT / "Figures" / "Figure1" / "output" / "Supplemental"
FIG4 = ROOT / "Figures" / "Figure4" / "output" / "source_data"
FIG4_NOTEBOOK = ROOT / "Figures" / "Figure4" / "Figure4.ipynb"
EXTERNAL = ROOT / "Runs" / "External" / "output"
EXTERNAL_CONFIG = ROOT / "Runs" / "External" / "config" / "e2m.yaml"
DRIVERS = ROOT / "Runs" / "ClinicalDrivers" / "output" / "summary" / "clinical_driver_results.csv"
SINGLE_CELL = ROOT / "Runs" / "SingleCell" / "output" / "plot_inputs"
OUT = Path(__file__).resolve().parent / "output"

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
    """A cohort's main External result: its first integration method in the run's config."""
    methods = yaml.safe_load(EXTERNAL_CONFIG.read_text(encoding="utf-8"))["cohorts"][cohort]["methods"]
    return EXTERNAL / cohort / methods[0]


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

    # S3e: clustering parameters used for the Figure 4 panels
    tcga, cptac_default, external = _notebook_params()
    params = []
    for cancer in TCGA_PARAM_CANCERS:
        p = tcga[cancer]
        for rep, nn, res in (("expression", "expr_nn", "expr_leiden_res"), ("embedding", "emb_nn", "emb_leiden_res"),
                             ("mutation_profile", "mut_nn", "mut_leiden_res")):
            params.append(("TCGA", cancer, rep, p[nn], p.get("mut_min_dist", 0.3) if rep == "mutation_profile" else 0.3, p[res]))
    for rep, key in (("embedding", "emb"), ("expression", "expr")):
        params.append(("CPTAC/CMI (per site)", "all sites (default)", rep, cptac_default[f"{key}_nn"],
                       cptac_default[f"{key}_min_dist"], cptac_default[f"{key}_leiden_res"]))
    for cohort, p in external.items():
        for rep in ("embedding", "expression"):
            params.append(("External cohort", cohort, rep, p[rep]["n_neighbors"], p[rep]["min_dist"], p[rep]["leiden_resolution"]))
    params = pd.DataFrame(params, columns=["panel_group", "cohort_or_site", "representation", "umap_n_neighbors",
                                           "umap_min_dist", "leiden_resolution"])

    return {
        "S3a_cohort_overview": overview, "S3b_external_per_gene": per_gene, "S3c_driver_validation": drivers,
        "S3d_clinical_association": assoc, "S3e_clustering_params": params,
    }


def table_s4() -> dict[str, pd.DataFrame]:
    # S4a: multitask perturbation prediction, mean over folds
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
        agg = folds.groupby("gene").agg(n_folds=("fold", "nunique"), prevalence=("prevalence", "mean"),
                                        mean_auprc=("average_precision", "mean"), mean_roc_auc=("roc_auc", "mean"))
        rows.append(agg.reset_index().rename(columns={"gene": "perturbation"})
                    .assign(dataset=name, modality=modality, cv_group=group))
    # Zhao/Sims: precision-recall only, from the out-of-fold predictions of the two drugs
    zhao = pd.read_csv(SINGLE_CELL / "Zhao" / "oof_probabilities_selected_targets.csv")
    summary = pd.read_csv(SINGLE_CELL / "Zhao" / "pr_curve_summary_selected_targets.csv").set_index("target")
    for drug in summary.index:
        aps = [average_precision_score(f[f"y_true_{drug}"], f[drug]) for _, f in zhao.groupby("fold")]
        rows.append(pd.DataFrame([{
            "perturbation": drug, "n_folds": zhao["fold"].nunique(), "prevalence": summary.loc[drug, "baseline_prevalence"],
            "mean_auprc": np.mean(aps), "mean_roc_auc": np.nan, "dataset": "Zhao/Sims", "modality": "drug", "cv_group": "sample",
        }]))
    multitask = pd.concat(rows, ignore_index=True)
    multitask["auprc_over_prevalence"] = multitask["mean_auprc"] / multitask["prevalence"]
    multitask = multitask[["dataset", "perturbation", "modality", "cv_group", "n_folds", "prevalence", "mean_auprc",
                           "auprc_over_prevalence", "mean_roc_auc"]]
    multitask = multitask.sort_values(["dataset", "mean_auprc"], ascending=[True, False]).reset_index(drop=True)

    # S4b: Frangieh/Izar per-gene XGBoost
    per_gene = pd.read_csv(SINGLE_CELL / "Frangieh_per_gene" / "metric_summary_per_gene.csv")
    per_gene = pd.DataFrame({
        "gene": per_gene["gene"], "n_positives": per_gene["n_positives"], "prevalence": per_gene["prevalence_mean"],
        "mean_auprc": per_gene["average_precision_mean"], "auprc_std": per_gene["average_precision_std"],
        "oof_auprc": per_gene["oof_average_precision"], "mean_roc_auc": per_gene["roc_auc_mean"],
    })
    per_gene["auprc_over_prevalence"] = per_gene["mean_auprc"] / per_gene["prevalence"]
    per_gene = per_gene.sort_values("mean_auprc", ascending=False).reset_index(drop=True)

    # S4c: Frangieh/Izar per-gene XGBoost by condition
    cond = pd.read_csv(SINGLE_CELL / "Frangieh_per_gene" / "metric_summary_per_gene_by_condition.csv")
    cond = pd.DataFrame({
        "gene": cond["gene"], "condition": cond["condition"], "n_positives": cond["n_positives"],
        "prevalence": cond["oof_prevalence"], "auprc": cond["oof_average_precision"], "roc_auc": cond["oof_roc_auc"],
    })
    cond["auprc_over_prevalence"] = cond["auprc"] / cond["prevalence"]
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


if __name__ == "__main__":
    main()
