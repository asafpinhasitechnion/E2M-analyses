"""BRAF in melanoma with one XGBoost model (the TCGA run's settings): CV in TCGA SKCM on the TCGA run's folds, then trained on
TCGA SKCM and applied to each melanoma cohort with its main integration, as in run_external.py.
Writes output/braf_xgboost.csv (ROC AUC and normalized AUPRC per cohort)."""

import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

import e2m
from e2m.config import load_config
from e2m.evaluation import normalized_auprc
from run_external import LOADERS, RUN_ROOT, integrate, mutation_labels

config = yaml.safe_load((RUN_ROOT / "config" / "e2m.yaml").read_text(encoding="utf-8"))
base = load_config(RUN_ROOT / config["run"]["tcga_config"], {"data": config["data"]})
run = config["run"]
rows = []


def score(name, labels, probabilities):
    rows.append({"cohort": name, "n": len(labels), "n_braf": int(labels.sum()), "roc_auc": roc_auc_score(labels, probabilities),
                 "normalized_auprc": normalized_auprc(average_precision_score(labels, probabilities), labels.mean())})
    print(rows[-1], flush=True)


def predict(x_train, y_train, x_test):
    model = xgb.XGBClassifier(**base["model"]["xgboost"], scale_pos_weight=(1 - y_train.mean()) / y_train.mean())
    return model.fit(x_train.to_numpy(), y_train.to_numpy()).predict_proba(x_test.to_numpy())[:, 1]


def tcga(expression_dataset=None):
    overrides = {**config["data"], "expression_dataset": expression_dataset, "expression_transform": "raw", "normalization": "none"}
    return e2m.Dataset.from_tcga(["SKCM"], data_dir=RUN_ROOT / run["tcga_data_dir"], config=RUN_ROOT / run["tcga_config"],
                                 data_overrides=overrides if expression_dataset else None, with_tmb=False)


data = tcga()
y = data.mutations["BRAF"].astype(int)
folds = pd.read_csv(RUN_ROOT / "../TCGA/output/SKCM/cv/fold_assignments.csv", index_col="sample_id")["fold"].reindex(y.index)
oof = pd.Series(0.0, index=y.index)
for fold in folds.unique():
    oof[folds == fold] = predict(data.expression[folds != fold], y[folds != fold], data.expression[folds == fold])
score("TCGA SKCM (CV)", y, oof)

for cohort in ["HUGO", "LIU", "RIAZ", "VAN_ALLEN", "MORRISON"]:
    s = config["cohorts"][cohort]
    expression, mutations, _, _, profiled = LOADERS[cohort][0](RUN_ROOT / run["external_root"] / s["data_dir"])
    reference = tcga(s["tcga_expression"])
    train, test = integrate(reference.expression, expression, s["methods"][0], external_batches=None, external_scale=s["scale"],
                            tcga_scale="counts" if s["tcga_expression"] == "star_counts" else "tpm",
                            cpm=base["data"]["normalization"] == "cpm")
    p = pd.Series(predict(train, reference.mutations.loc[train.index, "BRAF"].astype(int), test), index=test.index)
    labels = mutation_labels(mutations, profiled, test.index, ["BRAF"])["BRAF"].dropna().astype(int)
    score(cohort, labels, p[labels.index])

table = pd.DataFrame(rows)
table.to_csv(RUN_ROOT / "output" / "braf_xgboost.csv", index=False)
print(table.round(3).to_string(index=False))
print("Median over the external cohorts:", table.iloc[1:][["roc_auc", "normalized_auprc"]].median().round(2).to_dict())
