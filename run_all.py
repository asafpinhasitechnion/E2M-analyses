"""Run the whole analysis in order: each run, then the figures that read it, then the supplementary tables.

Usage (from Code_and_Analyses):
    python run_all.py                        # everything
    python run_all.py external singlecell    # selected parts, in this order
"""

import argparse
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


def run(folder, *args):
    subprocess.run([sys.executable, *args], cwd=ROOT / folder, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parts", nargs="*", choices=[*PARTS, "tables"], default=[*PARTS, "tables"])
    args = parser.parse_args()
    for part in args.parts:
        if part == "tables":
            run("Tables", "supplementary_tables.py")
            continue
        folder, script, figures = PARTS[part]
        if part == "clinicaldrivers":
            run(folder, "src/prepare_cbioportal.py", "--all")
            run(folder, "src/prepare_geo.py", "--all")
        run(folder, script)
        if figures:
            run("Figures", "run_figures.py", figures)


if __name__ == "__main__":
    main()
