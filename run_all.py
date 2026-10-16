"""Run the whole analysis in order: each run, then the figures that read it, then the supplementary tables.

Usage (from this folder):
    python run_all.py                        # everything, TCGA counts with CPM
    python run_all.py --normalization none   # everything, TCGA counts without CPM (log1p of counts)
    python run_all.py external singlecell    # selected parts, in this order

--normalization picks the run variant: TCGA and External write to output/ (CPM) or output/counts/, and the figures and
tables of that variant read them. Clinical drivers and single-cell do not depend on it. Figure 4 and the tables are
made once per External method in --external-methods ("main" = each cohort's first method in the External config).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Each part: its folder, its script, and the figure group that reads its outputs (Figures/run_figures.py).
PARTS = {
    "tcga": ("Runs/TCGA", "src/run_tcga.py", "tcga"),
    "external": ("Runs/External", "src/run_external.py", "external"),
    "clinicaldrivers": ("Runs/ClinicalDrivers", "src/run_drivers.py", None),
    "singlecell": ("Runs/SingleCell", "src/run_all.py", "singlecell"),
}
VARIANT_PARTS = ("tcga", "external")  # take --normalization


def run(folder, *args):
    subprocess.run([sys.executable, *args], cwd=ROOT / folder, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Python 3.10 checks the default of an optional positional against choices, so the default is a choice itself
    parser.add_argument("parts", nargs="*", choices=["all", *PARTS, "tables"], default="all")
    parser.add_argument("--normalization", choices=["cpm", "none"], default="cpm")
    parser.add_argument("--external-methods", nargs="+", default=["main"])
    args = parser.parse_args()
    os.environ["E2M_NORMALIZATION"] = args.normalization  # read by the figures and the tables
    for part in [*PARTS, "tables"] if "all" in args.parts else args.parts:
        if part == "tables":
            for method in args.external_methods:
                run("Figures", "run_figures.py", "tables", "--external-method", method)
            continue
        folder, script, figures = PARTS[part]
        if part == "clinicaldrivers":
            run(folder, "src/prepare_cohorts.py")
        run(folder, script, *(["--normalization", args.normalization] if part in VARIANT_PARTS else []))
        if figures == "external":
            for method in args.external_methods:
                run("Figures", "run_figures.py", figures, "--external-method", method)
        elif figures:
            run("Figures", "run_figures.py", figures)


if __name__ == "__main__":
    main()
