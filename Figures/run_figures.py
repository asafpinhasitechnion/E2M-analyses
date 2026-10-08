"""Run the figure notebooks, and the supplementary tables, once their analyses have finished.

    python Figures/run_figures.py tcga         # Figures 1-3, after Runs/TCGA
    python Figures/run_figures.py external     # Figure 4, after Runs/TCGA and Runs/External
    python Figures/run_figures.py singlecell   # Figure 5, after Runs/SingleCell
    python Figures/run_figures.py tables       # Supplementary tables, after all runs and figures
"""

import argparse
import subprocess
import sys
from pathlib import Path

import nbclient
import nbformat

FIGURES = Path(__file__).resolve().parent
GROUPS = {"tcga": [1, 2, 3], "external": [4], "singlecell": [5]}


def run_notebook(number: int) -> None:
    # The kernel is this Python's own environment, and runs in the notebook's folder, which the notebooks use
    # to find the project. The executed notebook is saved with its outputs.
    notebook = FIGURES / f"Figure{number}" / f"Figure{number}.ipynb"
    print(f"Running {notebook.name}", flush=True)
    nb = nbformat.read(notebook, as_version=4)
    try:
        nbclient.NotebookClient(nb, kernel_name="python3", timeout=None,
                                resources={"metadata": {"path": str(notebook.parent)}}).execute()
    finally:
        nbformat.write(nb, notebook)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("groups", nargs="+", choices=[*GROUPS, "tables"])
    args = parser.parse_args()
    for group in args.groups:
        if group == "tables":
            subprocess.run([sys.executable, str(FIGURES.parent / "Tables" / "supplementary_tables.py")], check=True)
        else:
            for number in GROUPS[group]:
                run_notebook(number)


if __name__ == "__main__":
    main()
