"""Training and inference time of one E2M model (no cross-validation) on TCGA-BRCA, pan-cancer TCGA, and Replogle K562.

    python Runs/Benchmark/timing.py                  # GPU if available
    python Runs/Benchmark/timing.py --device cpu

Data loading and preprocessing are not timed. Inference is the prediction of all training samples.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import scanpy as sc
import torch

import e2m
from e2m.config import load_config

RUNS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNS / "SingleCell" / "src"))
import config as sc_config  # noqa: E402
from models.multitask import build_multitask_dataset  # noqa: E402
from preprocess import preprocess  # noqa: E402

TCGA_CONFIG = RUNS / "TCGA" / "config" / "e2m.yaml"
# The single-cell runs' settings (Runs/SingleCell/src/models/presets.py): inputs already log-normalized
SINGLE_CELL_PARAMS = dict(learning_rate=1.5e-4, weight_decay=8e-4, batch_size=1024, validation_split=0.2, standardize_inputs=False)


def timed(device, fn):
    start = time.perf_counter()
    result = fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


def benchmark(name, device, data, mutations=None, features=None, **params):
    model = e2m.E2MModel(load_config(TCGA_CONFIG), **params)
    model.device = device
    _, train_s = timed(device, lambda: model.fit(data, mutations, features=features))
    _, predict_s = timed(device, lambda: model.predict(data))
    n_samples, n_features = model.metadata["n_samples"], model.metadata["n_features"]
    print(f"{name}: {n_samples} samples, {n_features} genes, {len(model.targets)} targets, "
          f"{len(model.history)} epochs, train {train_s:.1f} s, predict {predict_s:.2f} s", flush=True)
    return {"dataset": name, "device": device.type, "samples": n_samples, "genes": n_features, "targets": len(model.targets),
            "epochs": len(model.history), "train_s": round(train_s, 1), "predict_s": round(predict_s, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    tcga = load_config(TCGA_CONFIG)

    rows = []
    for name, cancers in (("TCGA-BRCA", ["BRCA"]), ("TCGA pan-cancer", tcga["run"]["cohorts"])):
        data = e2m.Dataset.from_tcga(cancers, data_dir=RUNS / "TCGA" / "data", config=TCGA_CONFIG, with_tmb=False)
        rows.append(benchmark(name, device, data))

    adata = preprocess(sc.read_h5ad(sc_config.DATA_DIR / sc_config.DATASET_FILES["replogle_k562"]))
    x, y, _, perturbations, *_ = build_multitask_dataset(adata, gene_col="perturbation", min_cells_per_gene=300, verbose=False)
    rows.append(benchmark("Replogle K562", device, x, pd.DataFrame(y, columns=perturbations), features=adata.var_names,
                          **SINGLE_CELL_PARAMS))

    out = Path(__file__).resolve().parent / "output"
    out.mkdir(exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / f"timing_{device.type}.csv", index=False)
    print(table.to_markdown(index=False))


if __name__ == "__main__":
    main()
