"""TCGA-to-external XGBoost transfer for clinically actionable driver genes and alterations."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import e2m
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from alterations import alteration_labels_for_primary_targets, build_alteration_matrix
from driver_genes import CANCER_DRIVER_GENES

RUN_ROOT = Path(__file__).resolve().parents[1]


CODING_VARIANT_CLASSIFICATIONS = {
    "Missense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Nonsense_Mutation",
    "Nonstop_Mutation",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Translation_Start_Site",
    "Splice_Site",
}


@dataclass(frozen=True)
class CohortPlan:
    cohort_id: str
    source_cohort_id: str
    label: str
    cancer: str
    tcga_cancers: tuple[str, ...]
    tcga_expression_measure: str
    external_expression_type: str
    log1p_train: bool
    log1p_external: bool
    primary_targets: tuple[str, ...]
    tumor_only: bool = False
    external_filter_column: str | None = None
    external_filter_values: tuple[str, ...] = ()


def read_config(path: Path = RUN_ROOT / "config" / "config.yaml") -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def run_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else RUN_ROOT / path


def cohort_plans(config: dict[str, Any]) -> list[CohortPlan]:
    plans = []
    for cohort_id, ccfg in config["cohorts"].items():
        filt = ccfg.get("filter") or {}
        plans.append(
            CohortPlan(
                cohort_id=cohort_id,
                source_cohort_id=ccfg.get("source", cohort_id),
                label=str(ccfg["label"]),
                cancer=str(ccfg["cancer"]),
                tcga_cancers=tuple(ccfg["tcga_cancers"]),
                tcga_expression_measure=str(ccfg["tcga_expression_measure"]),
                external_expression_type=str(ccfg["external_expression_type"]),
                log1p_train=bool(ccfg["log1p_tcga"]),
                log1p_external=bool(ccfg["log1p_external"]),
                primary_targets=tuple(CANCER_DRIVER_GENES[ccfg["drivers"]]),
                tumor_only=bool(ccfg.get("tumor_only", False)),
                external_filter_column=filt.get("column"),
                external_filter_values=tuple(str(v) for v in filt.get("values", ())),
            )
        )
    return plans


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def load_metabric_external(config: dict[str, Any], *, label_mode: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = run_path(config["data"]["metabric_dir"])
    expression_path = data_dir / "data_mrna_illumina_microarray.txt"
    mutation_path = data_dir / "data_mutations.txt"
    sample_clinical_path = data_dir / "data_clinical_sample.txt"
    patient_clinical_path = data_dir / "data_clinical_patient.txt"

    if not expression_path.exists():
        raise FileNotFoundError(f"Missing METABRIC expression file: {expression_path}")
    if not mutation_path.exists():
        raise FileNotFoundError(f"Missing METABRIC mutation file: {mutation_path}")

    expression_raw = pd.read_csv(expression_path, sep="\t", comment="#", low_memory=False)
    if "Hugo_Symbol" not in expression_raw.columns:
        raise ValueError("METABRIC expression file is missing Hugo_Symbol.")
    expression = expression_raw.set_index("Hugo_Symbol").drop(columns=["Entrez_Gene_Id"], errors="ignore").T
    expression.index = expression.index.astype(str)
    expression.index.name = "sample"
    expression = collapse_duplicate_columns(expression)

    sample_clinical = pd.read_csv(sample_clinical_path, sep="\t", comment="#", low_memory=False)
    patient_clinical = pd.read_csv(patient_clinical_path, sep="\t", comment="#", low_memory=False)
    sample_clinical["SAMPLE_ID"] = sample_clinical["SAMPLE_ID"].astype(str)
    patient_clinical["PATIENT_ID"] = patient_clinical["PATIENT_ID"].astype(str)
    clinical = sample_clinical.merge(patient_clinical, on="PATIENT_ID", how="left")
    clinical = clinical.drop_duplicates(subset=["SAMPLE_ID"]).set_index("SAMPLE_ID")
    clinical.index = clinical.index.astype(str)
    clinical.index.name = "sample"

    maf = pd.read_csv(mutation_path, sep="\t", comment="#", low_memory=False)
    maf["Tumor_Sample_Barcode"] = maf["Tumor_Sample_Barcode"].astype(str)
    maf["Hugo_Symbol"] = maf["Hugo_Symbol"].astype(str).str.upper()
    if "Variant_Classification" in maf.columns:
        maf = maf[maf["Variant_Classification"].fillna("").isin(CODING_VARIANT_CLASSIFICATIONS)]
    maf = maf.dropna(subset=["Tumor_Sample_Barcode", "Hugo_Symbol"])

    mutation_gene = (
        maf.groupby(["Tumor_Sample_Barcode", "Hugo_Symbol"]).size().unstack(fill_value=0) > 0
    ).astype(np.int8)
    mutation_gene.index = mutation_gene.index.astype(str)
    mutation_gene.index.name = "sample"

    common = expression.index.intersection(mutation_gene.index).intersection(clinical.index)
    expression = expression.loc[common]
    clinical = clinical.loc[common]
    if label_mode == "alteration":
        mutations = build_alteration_matrix(maf, common)
    else:
        mutations = mutation_gene.loc[common]
    return expression, mutations, clinical


def load_external_alteration_labels(cohort_dir: Path, sample_ids: pd.Index) -> pd.DataFrame:
    path = cohort_dir / "mutations_long.csv.gz"
    if not path.exists():
        return build_alteration_matrix(pd.DataFrame(), sample_ids)
    events = pd.read_csv(path, compression="gzip", low_memory=False)
    return build_alteration_matrix(events, sample_ids)


def load_external_standardized(
    config: dict[str, Any],
    plan: CohortPlan,
    *,
    label_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if plan.source_cohort_id == "metabric":
        return load_metabric_external(config, label_mode=label_mode)

    cohort_dir = run_path(config["data"]["standardized_dir"]) / plan.source_cohort_id
    expression = read_table(cohort_dir / "expression.csv.gz")
    mutations = read_table(cohort_dir / "mutations_gene_level.csv.gz")
    clinical = read_table(cohort_dir / "clinical_sample.csv.gz")

    expression.index = expression.index.astype(str)
    mutations.index = mutations.index.astype(str)
    clinical.index = clinical.index.astype(str)
    expression = expression[~expression.index.duplicated(keep="first")]
    mutations = mutations[~mutations.index.duplicated(keep="first")]
    clinical = clinical[~clinical.index.duplicated(keep="first")]
    expression = expression.apply(pd.to_numeric, errors="coerce")
    mutations = mutations.apply(pd.to_numeric, errors="coerce")

    common = expression.index.intersection(mutations.index)
    expression = expression.loc[common]
    mutations = mutations.loc[common]
    clinical = clinical.reindex(common)

    if plan.external_filter_column and plan.external_filter_values:
        if plan.external_filter_column not in clinical.columns:
            raise KeyError(f"Missing filter column {plan.external_filter_column!r} in {plan.source_cohort_id} clinical table.")
        values = clinical[plan.external_filter_column].fillna("").astype(str)
        keep = values.isin(set(plan.external_filter_values))
        expression = expression.loc[keep]
        mutations = mutations.loc[keep]
        clinical = clinical.loc[keep]

    if plan.tumor_only:
        labels_path = cohort_dir / "driver_labels.csv.gz"
        if labels_path.exists():
            labels = read_table(labels_path)
            labels.index = labels.index.astype(str)
            is_tumor = labels.reindex(expression.index).get("is_tumor")
            if is_tumor is not None:
                keep = is_tumor.astype("boolean").fillna(False)
                expression = expression.loc[keep]
                mutations = mutations.loc[keep]
                clinical = clinical.loc[keep]

    if label_mode == "alteration":
        mutations = load_external_alteration_labels(cohort_dir, expression.index)

    return expression, mutations, clinical


def tcga_event_sample_id(barcode: str) -> str:
    parts = str(barcode).split("-")
    if len(parts) >= 4 and parts[0] == "TCGA":
        sample_type = parts[3][:3] if len(parts[3]) >= 3 else f"{parts[3][:2]}A"
        return "-".join([parts[0], parts[1], parts[2], sample_type])
    return str(barcode)


def load_tcga_alteration_labels(config: dict[str, Any], sample_ids: pd.Index) -> pd.DataFrame:
    usecols = ["sample", "chr", "start", "end", "gene", "effect", "Amino_Acid_Change"]
    # The per-cancer MC3 event files in Runs/TCGA/data/mutation_events ({cancer}_mc3.txt.gz) have the same columns and could
    # replace this pan-cancer file; not yet checked that they give identical labels.
    events = pd.read_csv(run_path(config["data"]["tcga_mc3_events"]), sep="\t", usecols=usecols, low_memory=False)
    events["sample"] = events["sample"].map(tcga_event_sample_id)
    events = events[events["sample"].isin(set(sample_ids.astype(str)))]
    return build_alteration_matrix(events, sample_ids)


def load_tcga_training(
    config: dict[str, Any],
    plan: CohortPlan,
    *,
    label_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """TCGA training data from E2M: linear counts or TPM of protein-coding genes, every mutated gene as a label,
    and the cancer type of each sample as its ComBat batch."""
    tcga_cfg = config["tcga"]
    overrides = {
        "expression_dataset": f"star_{plan.tcga_expression_measure}",
        "expression_transform": "raw",       # linear scale; log1p is applied later per the cohort's config
        "normalization": "none",
        "min_mutation_prevalence": 0,
        "min_mutation_positives": 1,
        "max_mutation_targets": 0,
        "targets": None,
    }
    tcga = e2m.Dataset.from_tcga(list(plan.tcga_cancers), data_dir=run_path(tcga_cfg["data_dir"]),
                                 config=run_path(tcga_cfg["config"]), data_overrides=overrides, with_tmb=False)
    expression = tcga.expression.astype(float)
    mutations = load_tcga_alteration_labels(config, expression.index) if label_mode == "alteration" else tcga.mutations
    train_batches = "TCGA_" + tcga.cancer.astype(str)
    return expression, mutations, train_batches


def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def collapse_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = numeric_frame(df)
    if not df.columns.duplicated().any():
        return df
    return df.T.groupby(level=0).mean().T


def maybe_log1p(df: pd.DataFrame, do_log: bool) -> pd.DataFrame:
    out = numeric_frame(df).fillna(0.0)
    if not do_log:
        return out
    return np.log1p(out.clip(lower=0.0))


def shared_prepared_matrices(
    train_expression: pd.DataFrame,
    external_expression: pd.DataFrame,
    *,
    log1p_train: bool,
    log1p_external: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = collapse_duplicate_columns(maybe_log1p(train_expression, log1p_train))
    external = collapse_duplicate_columns(maybe_log1p(external_expression, log1p_external))
    shared = pd.Index(train.columns.intersection(external.columns)).sort_values()
    train = train.loc[:, shared]
    external = external.loc[:, shared]
    combined = pd.concat([train, external], axis=0)
    finite = np.isfinite(combined).all(axis=0)
    variable = combined.var(axis=0, ddof=0) > 0
    keep = combined.columns[finite & variable]
    return train.loc[:, keep], external.loc[:, keep]


def normalize_combat(
    train_expression: pd.DataFrame,
    external_expression: pd.DataFrame,
    train_batches: pd.Series,
    external_batches: pd.Series,
    *,
    log1p_train: bool,
    log1p_external: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    from inmoose.pycombat import pycombat_norm

    train, external = shared_prepared_matrices(
        train_expression,
        external_expression,
        log1p_train=log1p_train,
        log1p_external=log1p_external,
    )
    batch = pd.concat(
        [
            train_batches.reindex(train.index).fillna("TCGA").astype(str),
            external_batches.reindex(external.index).fillna("External").astype(str),
        ]
    )
    combined = pd.concat([train, external], axis=0)

    batch_var_mask = pd.Series(True, index=combined.columns)
    for batch_name in batch.unique():
        idx = batch[batch == batch_name].index
        if len(idx) >= 2:
            batch_var_mask &= combined.loc[idx].var(axis=0, ddof=0) > 0
    combined = combined.loc[:, batch_var_mask]

    # inmoose takes genes x samples, as in Runs/External
    corrected = pycombat_norm(counts=combined.T.to_numpy(dtype=np.float64), batch=batch.to_numpy())
    corrected = pd.DataFrame(np.asarray(corrected).T, index=combined.index, columns=combined.columns)
    return corrected.loc[train.index], corrected.loc[external.index], batch


def normalize_train_external(
    plan: CohortPlan,
    train_expression: pd.DataFrame,
    external_expression: pd.DataFrame,
    train_batches: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, str, pd.Series]:
    external_batches = pd.Series(plan.cohort_id, index=external_expression.index)
    train, external, batches = normalize_combat(
        train_expression,
        external_expression,
        train_batches,
        external_batches,
        log1p_train=plan.log1p_train,
        log1p_external=plan.log1p_external,
    )
    return train, external, "combat", batches


def select_targets(
    primary_targets: tuple[str, ...],
    train_mutations: pd.DataFrame,
    external_mutations: pd.DataFrame,
    *,
    min_train_positive: int,
    min_external_positive: int,
) -> tuple[list[str], pd.DataFrame]:
    rows = []
    selected = []
    train_binary = (numeric_frame(train_mutations).fillna(0.0) > 0).astype(np.int8)
    for order, gene in enumerate(primary_targets, start=1):
        in_train = gene in train_binary.columns
        in_external = gene in external_mutations.columns
        tcga_pos = int(train_binary[gene].sum()) if in_train else 0
        external_series = pd.to_numeric(external_mutations[gene], errors="coerce") if in_external else pd.Series(dtype=float)
        external_evaluable = int(external_series.notna().sum()) if in_external else 0
        external_pos = int((external_series.fillna(0) > 0).sum()) if in_external else 0

        reasons = []
        if not in_train:
            reasons.append("missing_tcga_label")
        if not in_external:
            reasons.append("missing_external_label")
        if in_train and tcga_pos < min_train_positive:
            reasons.append("too_few_tcga_positives")
        if in_external and external_pos < min_external_positive:
            reasons.append("too_few_external_positives")
        is_selected = not reasons
        if is_selected:
            selected.append(gene)

        rows.append(
            {
                "target_order": order,
                "gene": gene,
                "selected": is_selected,
                "reason": ";".join(reasons),
                "tcga_n_samples": int(train_binary.shape[0]),
                "tcga_n_positive": tcga_pos,
                "tcga_prevalence": tcga_pos / max(train_binary.shape[0], 1),
                "external_n_samples": int(external_mutations.shape[0]),
                "external_n_evaluable": external_evaluable,
                "external_n_positive": external_pos,
                "external_prevalence": external_pos / max(external_evaluable, 1) if external_evaluable else np.nan,
            }
        )

    return selected, pd.DataFrame(rows)


def fit_predict_xgboost(
    train_expression: pd.DataFrame,
    train_mutations: pd.DataFrame,
    external_expression: pd.DataFrame,
    target_genes: list[str],
    xgb_params: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost is required for this workflow.") from exc

    probabilities = pd.DataFrame(index=external_expression.index)
    predictions = pd.DataFrame(index=external_expression.index)
    base_params = dict(xgb_params)
    base_params.setdefault("eval_metric", "logloss")

    X_train = train_expression.to_numpy(dtype=np.float32, copy=False)
    X_external = external_expression.to_numpy(dtype=np.float32, copy=False)

    for idx, gene in enumerate(target_genes, start=1):
        y = (pd.to_numeric(train_mutations[gene], errors="coerce").fillna(0) > 0).astype(np.int8).to_numpy()
        if np.unique(y).size < 2:
            continue
        pos = int(y.sum())
        neg = int(len(y) - pos)
        params = dict(base_params)
        params["scale_pos_weight"] = neg / pos if pos else 1.0
        model = xgb.XGBClassifier(**params)
        print(f"[{idx}/{len(target_genes)}] training {gene} ({pos}/{len(y)} TCGA positives; {X_train.shape[1]} features)", flush=True)
        model.fit(X_train, y)
        prob = model.predict_proba(X_external)[:, 1]
        probabilities[gene] = prob
        predictions[gene] = (prob >= 0.5).astype(np.int8)

    return predictions, probabilities


def compute_metrics(
    external_mutations: pd.DataFrame,
    predictions: pd.DataFrame,
    probabilities: pd.DataFrame,
    target_genes: list[str],
) -> pd.DataFrame:
    rows = []
    for gene in target_genes:
        if gene not in probabilities.columns or gene not in external_mutations.columns:
            continue
        y_raw = pd.to_numeric(external_mutations[gene], errors="coerce")
        evaluable = y_raw.notna()
        y_true = (y_raw.loc[evaluable] > 0).astype(np.int8)
        y_prob = probabilities.loc[y_true.index, gene].astype(float)
        y_pred = predictions.loc[y_true.index, gene].astype(np.int8)
        n_pos = int(y_true.sum())
        n_neg = int(len(y_true) - n_pos)
        has_two_classes = y_true.nunique() == 2
        prevalence = n_pos / len(y_true) if len(y_true) else np.nan
        auprc = average_precision_score(y_true, y_prob) if has_two_classes else np.nan
        roc_auc = roc_auc_score(y_true, y_prob) if has_two_classes else np.nan
        normalized_auprc = (auprc - prevalence) / (1.0 - prevalence) if has_two_classes and prevalence < 1.0 else np.nan
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel() if len(y_true) else (0, 0, 0, 0)
        rows.append(
            {
                "gene": gene,
                "n_evaluable": int(len(y_true)),
                "n_positive": n_pos,
                "n_negative": n_neg,
                "prevalence": prevalence,
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
                "accuracy": accuracy_score(y_true, y_pred) if len(y_true) else np.nan,
                "precision": precision_score(y_true, y_pred, zero_division=0) if len(y_true) else np.nan,
                "recall": recall_score(y_true, y_pred, zero_division=0) if len(y_true) else np.nan,
                "f1": f1_score(y_true, y_pred, zero_division=0) if len(y_true) else np.nan,
                "mcc": matthews_corrcoef(y_true, y_pred) if has_two_classes else np.nan,
                "roc_auc": roc_auc,
                "auprc": auprc,
                "normalized_auprc": normalized_auprc,
                "evaluable": bool(has_two_classes),
            }
        )
    return pd.DataFrame(rows).sort_values(["normalized_auprc", "auprc"], ascending=[False, False], na_position="last")


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if path.suffix == ".gz" else None
    last_error: Exception | None = None
    for _ in range(3):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, compression=compression)
            return
        except (FileNotFoundError, PermissionError) as exc:
            last_error = exc
            time.sleep(1.0)
    if last_error is not None:
        raise last_error


def collect_manifests(output_root: Path) -> list[dict[str, Any]]:
    manifests = []
    for path in sorted(output_root.glob("*/manifest.json")):
        try:
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            print(f"Skipping invalid manifest: {path}", flush=True)
    return manifests


def write_run_summary(output_root: Path, manifests: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)
    all_manifests = collect_manifests(output_root)
    if manifests:
        by_cohort = {str(row.get("cohort_id")): row for row in all_manifests}
        by_cohort.update({str(row.get("cohort_id")): row for row in manifests})
        all_manifests = [by_cohort[key] for key in sorted(by_cohort)]
    summary = pd.DataFrame(all_manifests)
    summary.to_csv(output_root / "run_summary.csv", index=False)
    return summary


def run_cohort(
    config: dict[str, Any],
    plan: CohortPlan,
    *,
    output_root: Path,
    label_mode: str,
) -> dict[str, Any]:
    start = time.time()
    out_dir = output_root / plan.cohort_id
    for subdir in ("predictions", "metrics", "qc"):
        (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    external_expression, external_mutations, external_clinical = load_external_standardized(config, plan, label_mode=label_mode)
    tcga_expression, tcga_mutations, tcga_batches = load_tcga_training(config, plan, label_mode=label_mode)
    primary_targets = (
        alteration_labels_for_primary_targets(plan.primary_targets)
        if label_mode == "alteration"
        else plan.primary_targets
    )
    targets, target_table = select_targets(
        primary_targets,
        tcga_mutations,
        external_mutations,
        min_train_positive=int(config["target_selection"]["min_tcga_positive"]),
        min_external_positive=int(config["target_selection"]["min_external_positive"]),
    )
    write_dataframe(target_table, out_dir / "qc" / "target_selection.csv")

    manifest: dict[str, Any] = {
        "cohort_id": plan.cohort_id,
        "source_cohort_id": plan.source_cohort_id,
        "label": plan.label,
        "cancer": plan.cancer,
        "tcga_cancers": list(plan.tcga_cancers),
        "tcga_expression_measure": plan.tcga_expression_measure,
        "external_expression_type": plan.external_expression_type,
        "log1p_train": plan.log1p_train,
        "log1p_external": plan.log1p_external,
        "tumor_only": plan.tumor_only,
        "external_filter_column": plan.external_filter_column,
        "external_filter_values": list(plan.external_filter_values),
        "external_shape": list(external_expression.shape),
        "tcga_shape": list(tcga_expression.shape),
        "label_mode": label_mode,
        "n_targets_configured": len(primary_targets),
        "configured_targets": list(primary_targets),
        "n_targets_selected": len(targets),
        "selected_targets": targets,
    }

    if not targets:
        manifest["elapsed_s"] = round(time.time() - start, 3)
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    train_norm, external_norm, normalization_label, batches = normalize_train_external(
        plan,
        tcga_expression,
        external_expression,
        tcga_batches,
    )
    manifest["normalization_applied"] = normalization_label
    manifest["n_features"] = int(train_norm.shape[1])
    pd.Series(train_norm.columns, name="gene").to_csv(out_dir / "qc" / "features.csv", index=False)
    batches.value_counts().rename_axis("batch").reset_index(name="n_samples").to_csv(out_dir / "qc" / "batch_counts.csv", index=False)

    xgb_params = dict(config["xgboost"])
    manifest["xgboost_params"] = xgb_params

    predictions, probabilities = fit_predict_xgboost(
        train_norm,
        tcga_mutations,
        external_norm,
        targets,
        xgb_params,
    )
    metrics = compute_metrics(external_mutations, predictions, probabilities, targets)

    write_dataframe(predictions, out_dir / "predictions" / "predictions.csv.gz")
    write_dataframe(probabilities, out_dir / "predictions" / "probabilities.csv.gz")
    write_dataframe(metrics, out_dir / "metrics" / "gene_metrics.csv")

    manifest["n_predictions"] = int(probabilities.shape[1])
    manifest["n_evaluable_targets"] = int(metrics["evaluable"].sum()) if not metrics.empty else 0
    manifest["elapsed_s"] = round(time.time() - start, 3)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
