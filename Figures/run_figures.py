"""Run the figure notebooks, and the supplementary tables, once their analyses have finished.

    python Figures/run_figures.py tcga         # Figures 1-3, after Runs/TCGA
    python Figures/run_figures.py external     # Figure 4, after Runs/TCGA and Runs/External
    python Figures/run_figures.py singlecell   # Figure 5, after Runs/SingleCell
    python Figures/run_figures.py tables       # Supplementary tables, after all runs and figures

    python Figures/run_figures.py tcga external tables --normalization none --external-method sample_rank

--normalization picks the run variant (cpm: Runs/*/output, none: Runs/*/output/counts) and
--external-method the External method of Figure 4 and the tables (default: each cohort's main method).
They are passed to the notebooks and the tables as E2M_NORMALIZATION and E2M_EXTERNAL_METHOD
(Figures/constants.py). Each executed notebook is saved next to its outputs, in its output folder.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import nbclient
import nbformat

FIGURES = Path(__file__).resolve().parent
GROUPS = {"tcga": [1, 2, 3], "external": [4], "singlecell": [5]}


def output_folder(number: int) -> Path:
    """Where a figure writes its panels: output/ (Figures 1-3), output/<method>/ (Figure 4), with counts/ before them
    for the variant without CPM; output/ (Figure 5, which does not depend on the variant)."""
    variant = "" if os.environ.get("E2M_NORMALIZATION", "cpm") == "cpm" else "counts"
    folder = FIGURES / f"Figure{number}" / "output"
    if number == 5:
        return folder
    if number == 4:
        return folder / variant / (os.environ.get("E2M_EXTERNAL_METHOD") or "main")
    return folder / variant


def run_notebook(number: int) -> None:
    # The kernel is this Python's own environment, and runs in the notebook's folder, which the notebooks use
    # to find the project; it inherits E2M_NORMALIZATION and E2M_EXTERNAL_METHOD. The executed notebook is saved
    # with its outputs in the figure's output folder; the notebook itself is left unchanged.
    notebook = FIGURES / f"Figure{number}" / f"Figure{number}.ipynb"
    executed = output_folder(number) / notebook.name
    print(f"Running {notebook.name} -> {executed.relative_to(FIGURES)}", flush=True)
    nb = nbformat.read(notebook, as_version=4)
    try:
        nbclient.NotebookClient(nb, kernel_name="python3", timeout=None,
                                resources={"metadata": {"path": str(notebook.parent)}}).execute()
    finally:
        executed.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, executed)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("groups", nargs="+", choices=[*GROUPS, "tables"])
    parser.add_argument("--normalization", choices=["cpm", "none"], help="Run variant (default: E2M_NORMALIZATION, else cpm).")
    parser.add_argument("--external-method", help='External method for Figure 4 and the tables, e.g. sample_rank; "main" = '
                                                  "each cohort's first method (default: E2M_EXTERNAL_METHOD, else main).")
    args = parser.parse_args()
    if args.normalization:
        os.environ["E2M_NORMALIZATION"] = args.normalization
    if args.external_method:
        os.environ["E2M_EXTERNAL_METHOD"] = "" if args.external_method == "main" else args.external_method
    for group in args.groups:
        if group == "tables":
            subprocess.run([sys.executable, str(FIGURES.parent / "Tables" / "supplementary_tables.py")], check=True)
        else:
            for number in GROUPS[group]:
                run_notebook(number)


if __name__ == "__main__":
    main()
