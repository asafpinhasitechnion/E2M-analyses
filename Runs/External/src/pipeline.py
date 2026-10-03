"""Normalization, target selection, training and evaluation for TCGA-to-external transfer."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from data import (
    collapse_duplicate_columns,
    load_cptac_cmi_data,
    load_hugo_data,
    load_immunopog_data,
    load_liu_data,
    load_metabric_data,
    load_morrison_data,
    load_riaz_data,
    load_tcga_training_data,
    load_van_allen_data,
    sample_patient_id,
)

RUN_ROOT = Path(__file__).resolve().parents[1]
TCGA_SRC = RUN_ROOT.parent / "TCGA" / "src"
if str(TCGA_SRC) not in sys.path:
    sys.path.insert(0, str(TCGA_SRC))

# cohort -> (loader, ComBat batch column in clinical or None, per-site column in clinical or None)
COHORT_LOADERS = {
    "CPTAC_CMI": (load_cptac_cmi_data, "project.project_id", "cases.primary_site"),
    "METABRIC": (load_metabric_data, None, "CANCER_TYPE_DETAILED"),
    "IMMUNOPOG": (load_immunopog_data, None, "Cancer"),
    "HUGO": (load_hugo_data, None, None),
    "LIU": (load_liu_data, None, None),
    "RIAZ": (load_riaz_data, None, None),
    "VAN_ALLEN": (load_van_allen_data, None, None),
    "MORRISON": (load_morrison_data, None, None),
}


@dataclass
class CohortData:
    cohort: str
    tcga_expression: pd.DataFrame
    tcga_mutations: pd.DataFrame
    tcga_meta: dict[str, Any]
    external_expression: pd.DataFrame
    external_mutations: pd.DataFrame | None
    clinical: pd.DataFrame
    external_meta: dict[str, Any]
    test_batches: pd.Series | None
    site_col: str | None


@dataclass
class ExternalPredictionResult:
    predictions: pd.DataFrame
    probabilities: pd.DataFrame
    train_embeddings: pd.DataFrame | None
    test_embeddings: pd.DataFrame | None


def read_config(path: str | Path = RUN_ROOT / "config" / "config.yaml") -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else RUN_ROOT / path



def write_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, default=str)
    return path


def write_dataframe(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


def reset_index_named(df: pd.DataFrame, name: str = "sample") -> pd.DataFrame:
    out = df.copy()
    index_name = out.index.name or name
    if name in out.columns:
        out = out.copy()
        out[name] = out.index.astype(str)
        return out.reset_index(drop=True)
    return out.reset_index().rename(columns={index_name: name})


def available_device() -> str:
    try:
        import torch
    except Exception:
        return "torch unavailable"
    return "cuda" if torch.cuda.is_available() else "cpu"


def print_gpu_status() -> None:
    try:
        import torch
    except Exception as exc:
        print(f"Torch unavailable: {exc}")
        return
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0))


def load_cohort_data(config: dict[str, Any], cohort: str) -> CohortData:
    ccfg = config["cohorts"][cohort]
    tcga_cfg = config["tcga"]
    tcga_expression, tcga_mutations, tcga_meta = load_tcga_training_data(
        expression_path=run_path(tcga_cfg["expression_dir"]),
        gene_name_mapping_path=run_path(tcga_cfg["gene_name_mapping"]),
        gene_annotation_path=run_path(tcga_cfg["gene_annotation"]),
        mutation_path=run_path(tcga_cfg["mutation_dir"]),
        mutation_file_template=tcga_cfg["mutation_file_template"],
        cancer_types=ccfg.get("tcga_cancer_types"),
        expression_measure=ccfg.get("tcga_expression_measure", "counts"),
        expression_log_transformed=bool(tcga_cfg.get("expression_log_transformed", True)),
        only_protein_coding=bool(tcga_cfg.get("only_protein_coding", True)),
        min_mutations_per_gene=int(tcga_cfg.get("min_mutations_per_gene", 1)),
    )

    loader, batch_col, site_col = COHORT_LOADERS[cohort]
    expr, muts, clinical, meta = loader(run_path(config["data"]["external_root"]) / ccfg["data_dir"], coding_only=True)
    batches = clinical[batch_col].astype(str) if batch_col else None

    return CohortData(
        cohort=cohort,
        tcga_expression=tcga_expression,
        tcga_mutations=tcga_mutations,
        tcga_meta=tcga_meta,
        external_expression=expr,
        external_mutations=muts,
        clinical=clinical,
        external_meta=meta,
        test_batches=batches,
        site_col=site_col,
    )


def rank_top_genes(rank_file: Path, top_n: int) -> pd.DataFrame:
    """Top genes of a TCGA cross-validation summary by normalized AUPRC."""
    df = pd.read_csv(rank_file)
    df = df.rename(columns={df.columns[0]: "gene"})
    df["gene"] = df["gene"].astype(str)
    if "normalized_auprc" not in df.columns:
        auprc = pd.to_numeric(df["auprc_mean"], errors="coerce")
        prevalence = pd.to_numeric(df["prevalence_mean"], errors="coerce")
        df["normalized_auprc"] = (auprc - prevalence) / (1.0 - prevalence)
    keep = [c for c in ("gene", "normalized_auprc", "auprc_mean", "prevalence_mean", "roc_auc_mean") if c in df.columns]
    ranked = (
        df[keep]
        .dropna(subset=["normalized_auprc"])
        .sort_values(["normalized_auprc"], ascending=False)
        .reset_index(drop=True)
    )
    return ranked.head(int(top_n))


def target_gene_list(config: dict[str, Any], cohort: str) -> list[str] | None:
    """Explicit target list for cohorts configured with `target_ranking`, else None."""
    ranking = config["cohorts"][cohort].get("target_ranking")
    if not ranking:
        return None
    return rank_top_genes(run_path(ranking["file"]), ranking["top_n"])["gene"].tolist()


def binarize_labels(labels: pd.DataFrame, target_genes: list[str]) -> pd.DataFrame:
    out = labels.reindex(columns=target_genes, fill_value=0)
    return (out.apply(pd.to_numeric, errors="coerce").fillna(0) > 0).astype("int8")


def select_target_genes(
    train_mutations: pd.DataFrame,
    *,
    external_mutations: pd.DataFrame | None = None,
    min_train_positive: int = 1,
    min_external_positive: int = 0,
    max_genes: int | None = None,
) -> tuple[list[str], pd.DataFrame]:
    """Select mutation targets using explicit count thresholds."""
    train_binary = (train_mutations.apply(pd.to_numeric, errors="coerce").fillna(0) > 0).astype("int8")
    train_pos = train_binary.sum(axis=0).astype(int)
    table = pd.DataFrame(
        {
            "gene": train_pos.index.astype(str),
            "tcga_n_positive": train_pos.to_numpy(),
            "tcga_prevalence": train_pos.to_numpy() / max(train_binary.shape[0], 1),
        }
    )

    if external_mutations is not None:
        external_binary = binarize_labels(external_mutations, table["gene"].tolist())
        external_pos = external_binary.sum(axis=0).astype(int)
        table["external_n_positive"] = external_pos.reindex(table["gene"]).to_numpy()
        table["external_prevalence"] = table["external_n_positive"] / max(external_binary.shape[0], 1)
    else:
        table["external_n_positive"] = pd.NA
        table["external_prevalence"] = pd.NA

    table["passes_min_train_positive"] = table["tcga_n_positive"] >= int(min_train_positive)
    table["passes_min_external_positive"] = (
        True
        if int(min_external_positive) <= 0 or external_mutations is None
        else table["external_n_positive"] >= int(min_external_positive)
    )
    table["selected"] = table["passes_min_train_positive"] & table["passes_min_external_positive"]
    table = table.sort_values(
        ["selected", "tcga_n_positive", "external_n_positive", "gene"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)

    selected = table.loc[table["selected"], "gene"].tolist()
    if max_genes is not None and int(max_genes) > 0:
        selected = selected[: int(max_genes)]
        table["selected"] = table["gene"].isin(selected)
    return selected, table


def choose_target_genes(config: dict[str, Any], data: CohortData) -> tuple[list[str], pd.DataFrame]:
    tcfg = config["target_selection"]
    min_train_positive = int(tcfg.get("min_tcga_mutated_samples", 10))
    min_train_prevalence = float(tcfg.get("min_tcga_mutation_prevalence", 0.0) or 0.0)
    if min_train_prevalence > 0:
        min_train_positive = max(
            min_train_positive,
            int(np.ceil(min_train_prevalence * max(data.tcga_mutations.shape[0], 1))),
        )
    selected, table = select_target_genes(
        data.tcga_mutations,
        external_mutations=data.external_mutations,
        min_train_positive=min_train_positive,
    )
    table.insert(3, "effective_min_tcga_mutated_samples", min_train_positive)
    table.insert(4, "min_tcga_mutation_prevalence", min_train_prevalence)

    restrict_to_genes = target_gene_list(config, data.cohort)
    if restrict_to_genes:
        # Explicit target list (melanoma cohorts): keep listed genes present in TCGA, in list order,
        # instead of the prevalence threshold.
        available = set(table["gene"].astype(str))
        selected = [str(gene) for gene in restrict_to_genes if str(gene) in available]
        table["passes_restrict_to_genes"] = table["gene"].astype(str).isin(selected)
        table["selected"] = table["passes_restrict_to_genes"]
        rank_lookup = {str(gene): i for i, gene in enumerate(restrict_to_genes)}
        table["restrict_rank"] = table["gene"].astype(str).map(rank_lookup)
        table = table.sort_values(
            ["selected", "restrict_rank", "gene"],
            ascending=[False, True, True],
            na_position="last",
        ).reset_index(drop=True)
    return selected, table


def log1p_counts(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return np.log1p(np.clip(numeric, a_min=0.0, a_max=None))


def robust_rank_transform(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Column-wise rank transform scaled to [0, 1]."""
    numeric = expr_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    ranked = numeric.rank(axis=0, method="average")
    return ranked / max(ranked.shape[0], 1)


def rank_normalize_train_test(
    train_expression: pd.DataFrame,
    test_expression: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """Match shared genes and rank-normalize train and test matrices independently."""
    train = collapse_duplicate_columns(train_expression.copy(), method="mean").fillna(0.0)
    test = collapse_duplicate_columns(test_expression.copy(), method="mean").fillna(0.0)
    shared = pd.Index(train.columns.intersection(test.columns)).sort_values()
    train_norm = robust_rank_transform(train.loc[:, shared])
    test_norm = robust_rank_transform(test.loc[:, shared])
    return train_norm, test_norm, shared


def combat_normalize_train_test(
    train_expression: pd.DataFrame,
    test_expression: pd.DataFrame,
    test_batches: pd.Series,
    *,
    train_batch_name: str = "TCGA",
    log1p_train: bool = True,
    log1p_test: bool = True,
    ref_batch: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index, pd.Series]:
    """Optionally log1p-transform train/test, match genes, and ComBat-correct batches."""
    try:
        from inmoose.pycombat import pycombat_norm
    except ImportError as exc:
        raise ImportError(
            "Install inmoose to run ComBat normalization (pip install inmoose)."
        ) from exc

    train = log1p_counts(train_expression) if log1p_train else train_expression.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    test = log1p_counts(test_expression) if log1p_test else test_expression.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    train = collapse_duplicate_columns(train, method="mean")
    test = collapse_duplicate_columns(test, method="mean")
    shared = pd.Index(train.columns.intersection(test.columns)).sort_values()
    train = train.loc[:, shared]
    test = test.loc[:, shared]

    batch = pd.concat(
        [
            pd.Series(train_batch_name, index=train.index, name="batch"),
            test_batches.reindex(test.index).fillna("External").astype(str).rename("batch"),
        ]
    )
    combined = pd.concat([train, test], axis=0)
    combined = combined.apply(pd.to_numeric, errors="coerce")
    finite_mask = np.isfinite(combined).all(axis=0)
    var_mask = combined.var(axis=0, ddof=0) > 0
    batch_var_mask = pd.Series(True, index=combined.columns)
    for batch_name in batch.dropna().unique():
        idx = batch[batch == batch_name].index
        if len(idx) >= 2:
            batch_var_mask &= combined.loc[idx].var(axis=0, ddof=0) > 0
    keep = combined.columns[finite_mask & var_mask & batch_var_mask]
    combined = combined.loc[:, keep]
    if combined.shape[1] == 0:
        raise ValueError(
            "No genes survived the per-batch finite + non-zero-variance filter for pycombat_norm."
        )

    expr_gxs = combined.T.to_numpy(dtype=np.float64, copy=False)
    batch_labels = batch.to_numpy()
    if ref_batch is not None and ref_batch not in set(batch_labels):
        raise ValueError(
            f"ref_batch={ref_batch!r} is not present in the batch vector; "
            f"available batches: {sorted(set(batch_labels))}"
        )
    corrected_gxs = pycombat_norm(
        counts=expr_gxs,
        batch=batch_labels,
        ref_batch=ref_batch,
    )
    corrected = pd.DataFrame(
        np.asarray(corrected_gxs).T,
        index=combined.index,
        columns=combined.columns,
    )
    return corrected.loc[train.index], corrected.loc[test.index], corrected.columns, batch


def combat_seq_normalize_train_test(
    train_expression: pd.DataFrame,
    test_expression: pd.DataFrame,
    test_batches: pd.Series,
    *,
    train_batch_name: str = "TCGA",
    ref_batch: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index, pd.Series]:
    """ComBat-seq batch correction on raw integer counts, then log1p for model input."""
    try:
        from inmoose.pycombat import pycombat_seq
    except ImportError as exc:
        raise ImportError(
            "Install inmoose to run ComBat-seq normalization (pip install inmoose)."
        ) from exc

    train = collapse_duplicate_columns(train_expression, method="mean")
    test = collapse_duplicate_columns(test_expression, method="mean")
    train = train.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0).round().astype(np.int32)
    test = test.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0).round().astype(np.int32)

    shared = pd.Index(train.columns.intersection(test.columns)).sort_values()
    train = train.loc[:, shared]
    test = test.loc[:, shared]

    batch = pd.concat(
        [
            pd.Series(train_batch_name, index=train.index, name="batch"),
            test_batches.reindex(test.index).fillna("External").astype(str).rename("batch"),
        ]
    )
    combined = pd.concat([train, test], axis=0)

    # ComBat-seq requires every batch to contribute non-zero expression for each gene.
    # Drop genes that are all-zero in any batch so the dispersion estimate is well-defined.
    keep_mask = pd.Series(True, index=combined.columns)
    for batch_name in batch.dropna().unique():
        idx = batch.index[batch == batch_name]
        if len(idx) >= 1:
            keep_mask &= combined.loc[idx].sum(axis=0) > 0
    combined = combined.loc[:, combined.columns[keep_mask]]
    if combined.shape[1] == 0:
        raise ValueError("No genes survived the per-batch non-zero filter for ComBat-seq.")

    counts_gxs = combined.T.to_numpy(dtype=np.int64, copy=False)
    batch_labels = batch.to_numpy()
    if ref_batch is not None and ref_batch not in set(batch_labels):
        raise ValueError(
            f"ref_batch={ref_batch!r} is not present in the batch vector; "
            f"available batches: {sorted(set(batch_labels))}"
        )
    corrected_gxs = pycombat_seq(
        counts=counts_gxs,
        batch=batch_labels,
        ref_batch=ref_batch,
    )
    corrected_counts = pd.DataFrame(
        np.asarray(corrected_gxs).T,
        index=combined.index,
        columns=combined.columns,
    ).clip(lower=0.0)
    corrected_log = np.log1p(corrected_counts)
    return (
        corrected_log.loc[train.index],
        corrected_log.loc[test.index],
        corrected_log.columns,
        batch,
    )


def normalize_for_method(config: dict[str, Any], data: CohortData, method: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index, str]:
    """Return normalized TCGA and external matrices on shared genes, and a label for the manifest."""
    batches = data.test_batches
    if batches is None:
        batches = pd.Series(data.cohort, index=data.external_expression.index, name="batch")
    norm_cfg = config["cohorts"][data.cohort].get("normalization", {}) or {}
    if method == "combat":
        log1p_train = bool(norm_cfg.get("log1p_train", True))
        log1p_test = bool(norm_cfg.get("log1p_test", True))
        train, test, features, _ = combat_normalize_train_test(
            data.tcga_expression,
            data.external_expression,
            test_batches=batches,
            train_batch_name="TCGA",
            log1p_train=log1p_train,
            log1p_test=log1p_test,
        )
        parts = ["combat"]
        if log1p_train:
            parts.append("log1p_train")
        if log1p_test:
            parts.append("log1p_test")
        return train, test, features, "_".join(parts)
    if method == "combat_seq":
        train, test, features, _ = combat_seq_normalize_train_test(
            data.tcga_expression,
            data.external_expression,
            test_batches=batches,
            train_batch_name="TCGA",
        )
        return train, test, features, "combat_seq_log1p"
    if method == "gene_rank":
        train, test, features = rank_normalize_train_test(data.tcga_expression, data.external_expression)
        return train, test, features, "shared_genes_independent_gene_rank"
    raise ValueError(f"Unknown normalization method {method!r}")


def align_external_mutations_to_expression_samples(
    external_mutations: pd.DataFrame,
    target_genes: list[str],
    expression_index: pd.Index,
) -> pd.DataFrame:
    """Align external mutation labels to expression samples, expanding patient-level labels when needed."""
    labels = binarize_labels(external_mutations, target_genes)
    expression_index = pd.Index(expression_index).astype(str)
    labels.index = labels.index.astype(str)
    if expression_index.isin(labels.index).all():
        return labels.reindex(expression_index).fillna(0).astype("int8")

    patient_ids = pd.Series([sample_patient_id(sample) for sample in expression_index], index=expression_index)
    if patient_ids.isin(labels.index).any():
        aligned = labels.reindex(patient_ids.to_numpy()).fillna(0).astype("int8")
        aligned.index = expression_index
        return aligned

    missing = expression_index.difference(labels.index)
    raise KeyError(
        "External mutation labels cannot be aligned to expression samples. "
        f"Missing {len(missing)} expression IDs; first examples: {missing[:5].tolist()}"
    )


def align_mutations(data: CohortData, target_genes: list[str], train_expr: pd.DataFrame, test_expr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    train_mut = binarize_labels(data.tcga_mutations, target_genes).loc[train_expr.index]
    if data.external_mutations is None:
        return train_mut, None
    test_mut = align_external_mutations_to_expression_samples(data.external_mutations, target_genes, test_expr.index)
    return train_mut, test_mut


def model_config(config: dict[str, Any]) -> dict[str, Any]:
    return {"model": {"multitask_nn": config["model"]["multitask_nn"]}}


def train_external_multitask_model(
    train_expression: pd.DataFrame,
    train_mutations: pd.DataFrame,
    test_expression: pd.DataFrame,
    target_genes: list[str],
    config: dict,
    *,
    random_seed: int = 42,
    extract_embeddings: bool = True,
) -> ExternalPredictionResult:
    """Train the multitask model on TCGA and predict the external samples."""
    from models.model_factory import ModelFactory

    try:
        import torch
    except ImportError as exc:
        raise ImportError("Install torch to train the multitask external validation model.") from exc

    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    target_genes = list(target_genes)
    train_mutations = train_mutations.reindex(columns=target_genes, fill_value=0)
    model = ModelFactory().get_model(
        model_name="multitask_nn",
        input_size=train_expression.shape[1],
        output_size=len(target_genes),
        config=config,
    )
    model.fit(train_expression, train_mutations)
    pred, prob = model.predict(test_expression)
    predictions = pd.DataFrame(pred, index=test_expression.index, columns=target_genes)
    probabilities = pd.DataFrame(prob, index=test_expression.index, columns=target_genes)

    train_embeddings = test_embeddings = None
    if extract_embeddings and hasattr(model, "get_sample_embeddings"):
        train_embeddings = pd.DataFrame(
            model.get_sample_embeddings(train_expression).detach().cpu().numpy(),
            index=train_expression.index,
        )
        test_embeddings = pd.DataFrame(
            model.get_sample_embeddings(test_expression).detach().cpu().numpy(),
            index=test_expression.index,
        )

    return ExternalPredictionResult(
        predictions=predictions,
        probabilities=probabilities,
        train_embeddings=train_embeddings,
        test_embeddings=test_embeddings,
    )


def compute_gene_metrics(
    true_mutations: pd.DataFrame,
    predictions: pd.DataFrame,
    probabilities: pd.DataFrame,
    target_genes: list[str],
) -> pd.DataFrame:
    """Compute metrics for all requested genes; one-class genes retain count rows with NaN AUPRC."""
    common = true_mutations.index.intersection(predictions.index).intersection(probabilities.index)
    rows = []
    for gene in target_genes:
        if gene not in predictions.columns or gene not in probabilities.columns:
            continue
        y_true = (
            true_mutations.reindex(index=common, columns=[gene], fill_value=0)
            .iloc[:, 0]
            .astype(int)
            .to_numpy()
        )
        y_pred = predictions.loc[common, gene].astype(int).to_numpy()
        y_prob = probabilities.loc[common, gene].astype(float).to_numpy()
        n_pos = int(y_true.sum())
        n_neg = int(len(y_true) - n_pos)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        has_two_classes = np.unique(y_true).size == 2
        prevalence = float(n_pos / len(y_true)) if len(y_true) else np.nan
        auprc = float(average_precision_score(y_true, y_prob)) if has_two_classes else np.nan
        roc_auc = float(roc_auc_score(y_true, y_prob)) if has_two_classes else np.nan
        norm_auprc = (
            float((auprc - prevalence) / (1.0 - prevalence))
            if has_two_classes and prevalence < 1.0
            else np.nan
        )
        rows.append(
            {
                "gene": gene,
                "n_samples": int(len(y_true)),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "prevalence": prevalence,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "accuracy": float((y_pred == y_true).mean()) if len(y_true) else np.nan,
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "mcc": float(matthews_corrcoef(y_true, y_pred)) if has_two_classes else np.nan,
                "roc_auc": roc_auc,
                "auprc": auprc,
                "normalized_auprc": norm_auprc,
                "evaluable": bool(has_two_classes),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["normalized_auprc", "auprc", "n_positive"], ascending=[False, False, False], na_position="last")
        .reset_index(drop=True)
    )


def compute_site_metrics(
    true_mutations: pd.DataFrame,
    predictions: pd.DataFrame,
    probabilities: pd.DataFrame,
    clinical: pd.DataFrame,
    target_genes: list[str],
    *,
    site_col: str = "cases.primary_site",
    min_site_n: int = 20,
) -> pd.DataFrame:
    """Compute per-site metrics without any site-level prevalence prefilter."""
    common = (
        true_mutations.index
        .intersection(predictions.index)
        .intersection(probabilities.index)
        .intersection(clinical.index)
    )
    clinical_sub = clinical.loc[common]
    rows = []
    for site, site_idx in clinical_sub.groupby(site_col).groups.items():
        site_idx = pd.Index(site_idx)
        if len(site_idx) < min_site_n:
            continue
        metrics = compute_gene_metrics(
            true_mutations.loc[site_idx],
            predictions.loc[site_idx],
            probabilities.loc[site_idx],
            target_genes,
        )
        metrics.insert(0, "site", site)
        metrics.insert(1, "site_n_samples", int(len(site_idx)))
        rows.append(metrics)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def integration_coordinates(
    train_expression: pd.DataFrame,
    test_expression: pd.DataFrame,
    *,
    test_batches: pd.Series | None = None,
    external_label: str = "External",
    max_features: int = 5000,
    n_neighbors: int = 30,
    min_dist: float = 0.25,
    random_state: int = 42,
) -> pd.DataFrame:
    """UMAP coordinates of TCGA and external samples on their shared most-variable genes."""
    import umap

    train_source = collapse_duplicate_columns(train_expression.copy(), method="mean")
    test_source = collapse_duplicate_columns(test_expression.copy(), method="mean")
    shared = pd.Index(train_source.columns.intersection(test_source.columns)).sort_values()
    train = train_source.loc[:, shared].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    test = test_source.loc[:, shared].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if train.shape[1] > max_features:
        variances = pd.concat([train, test], axis=0).var(axis=0).sort_values(ascending=False)
        keep = variances.head(max_features).index
        train = train.loc[:, keep]
        test = test.loc[:, keep]

    matrix = pd.concat([train, test], axis=0)
    scaled = StandardScaler().fit_transform(matrix)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=int(random_state),
    )
    coords = reducer.fit_transform(scaled)
    coord_df = pd.DataFrame(coords, index=matrix.index, columns=["UMAP1", "UMAP2"])
    coord_df["dataset"] = ["TCGA"] * len(train) + [external_label] * len(test)
    if test_batches is not None:
        batch = pd.concat([
            pd.Series("TCGA", index=train.index),
            test_batches.reindex(test.index).fillna("External").astype(str),
        ])
        coord_df["batch"] = batch.values
    else:
        coord_df["batch"] = coord_df["dataset"]
    return coord_df
