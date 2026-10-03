"""Run the single-cell analysis.

Downloads missing scPerturb files from Zenodo, then runs the steps in order:

    ccle                    CCLE TMB regression + mutation multitask
    perturbation_multitask  multitask CV on the scPerturb datasets
    frangieh_per_gene       per-gene XGBoost on Frangieh/Izar
    tian_guide_transfer     Tian/Kampmann two-guide transfer
    frangieh_conditions     Frangieh metrics per immune condition   (needs the two Frangieh steps)
    zhao_pr_curves          Zhao/Sims PR curves                     (needs perturbation_multitask)
    plot_inputs             copy the files used by Figure 5 and Table S4

Usage (from Runs/SingleCell):
    python src/run_all.py
    python src/run_all.py --only ccle
    python src/run_all.py --skip ccle --no-download
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from datasets import download  # noqa: E402

STEPS = [
    "ccle",
    "perturbation_multitask",
    "frangieh_per_gene",
    "tian_guide_transfer",
    "frangieh_conditions",
    "zhao_pr_curves",
    "plot_inputs",
]


def _file_status(name: str, expected_size: int) -> str:
    """Return one of: 'missing', 'truncated', 'ok'."""
    dst = config.DATA_DIR / name
    if not dst.exists():
        return "missing"
    if expected_size and dst.stat().st_size < expected_size:
        return "truncated"
    return "ok"


def ensure_data_downloaded(check_md5: bool = False) -> None:
    """Download any missing or truncated scPerturb files from Zenodo."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        record = download.fetch_record_metadata()
        all_files = download.resolve_requested_files(record, list(config.DATASET_FILES.keys()))
    except Exception as exc:
        print(f"[warn] Zenodo JSON metadata unreachable ({exc}); using direct URLs.")
        all_files = download._files_from_fallback(list(config.DATASET_FILES.keys()))

    expected_size = {item["key"]: int(item.get("size") or 0) for item in all_files}

    todo = []
    for item in all_files:
        name = item["key"]
        status = _file_status(name, expected_size.get(name, 0))
        if status == "ok":
            print(f"[ok]        {name}")
        elif status == "missing":
            print(f"[missing]   {name}")
            todo.append(item)
        else:
            on_disk = (config.DATA_DIR / name).stat().st_size
            exp = expected_size.get(name, 0)
            print(
                f"[truncated] {name}: on disk={on_disk / 1e9:.2f} GB < expected {exp / 1e9:.2f} GB; "
                f"removing and re-downloading"
            )
            (config.DATA_DIR / name).unlink()
            todo.append(item)

    if not todo:
        print("[download] all perturbation files already present and complete.")
        return

    for item in todo:
        name = item["key"]
        dst = config.DATA_DIR / name
        size = int(item.get("size") or 0)
        print(f"[download] {name}")
        download.download_file(item["links"]["self"], dst, expected_size=size)

        if check_md5:
            checksum = str(item.get("checksum", ""))
            expected_md5 = checksum.split(":", 1)[1] if checksum.startswith("md5:") else None
            if expected_md5:
                observed = download.file_md5(dst)
                if observed != expected_md5:
                    raise RuntimeError(f"MD5 mismatch for {dst}: {observed} != {expected_md5}")
                print(f"[md5 ok] {name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=STEPS, help="Run only these steps.")
    p.add_argument("--skip", nargs="+", choices=STEPS, default=(), help="Skip these steps.")
    p.add_argument("--no-download", action="store_true", help="Don't download missing scPerturb files.")
    p.add_argument("--check-md5", action="store_true", help="Verify md5 of downloaded files.")
    args = p.parse_args()

    print(f"=== {config.describe_runtime()} ===")
    if not args.no_download:
        print("=== download ===")
        ensure_data_downloaded(check_md5=args.check_md5)

    for step in STEPS:
        if (args.only and step not in args.only) or step in args.skip:
            continue
        print(f"\n=== {step} ===")
        importlib.import_module(f"steps.{step}").main()


if __name__ == "__main__":
    main()
