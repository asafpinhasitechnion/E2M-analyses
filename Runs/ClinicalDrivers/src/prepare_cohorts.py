"""Download the external cohorts of config/config.yaml and write each in one format to data/standardized/<source>/:
expression.csv.gz (samples x genes), mutations_gene_level.csv.gz (samples x driver genes: 1/0, missing if not sequenced),
clinical_sample.csv.gz, and for cBioPortal studies mutations_long.csv.gz (the coding events of the driver genes).

Usage (from Runs/ClinicalDrivers):
    python src/prepare_cohorts.py
    python src/prepare_cohorts.py --only difg_glass GSE31210
"""

from __future__ import annotations

import argparse
import gzip
import io
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yaml

from alterations import TEXT_COLUMNS
from driver_genes import CODING_VARIANT_CLASSES, DRIVER_GENES

RUN_ROOT = Path(__file__).resolve().parents[1]

# cBioPortal datahub at a fixed commit (2026-05-13; the next commit lost the LFS file of the paad_qcmg_uq_2016 mutations)
DATAHUB_COMMIT = "a766d41024e4229c6abf017eda000a3a2a40a293"
DATAHUB_LFS = f"https://media.githubusercontent.com/media/cBioPortal/datahub/{DATAHUB_COMMIT}/"
DATAHUB_RAW = f"https://raw.githubusercontent.com/cBioPortal/datahub/{DATAHUB_COMMIT}/"
DATAHUB_LISTING = f"https://api.github.com/repos/cBioPortal/datahub/contents/public/{{study}}?ref={DATAHUB_COMMIT}"
GEO_MATRIX = "https://ftp.ncbi.nlm.nih.gov/geo/series/{group}/{accession}/matrix/{accession}_series_matrix.txt.gz"
GPL570_ANNOTATION = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz"

MAF_COLUMNS = ["Tumor_Sample_Barcode", "Hugo_Symbol", "Start_Position", *TEXT_COLUMNS]


def fetch(url: str, path: Path) -> Path:
    """Download `url` to `path` unless the file is already there."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")  # renamed only when complete, so an interrupted download is not reused
        with requests.get(url, stream=True, timeout=300) as response:
            response.raise_for_status()
            with open(part, "wb") as f:
                for chunk in response.iter_content(1 << 20):
                    f.write(chunk)
        part.replace(path)
    return path


def datahub(path: str, local: Path) -> Path:
    """A datahub file at the fixed commit. Large files are in Git LFS: raw gives a pointer, so they come from media."""
    fetch(DATAHUB_RAW + path, local)
    with open(local, "rb") as f:
        is_pointer = f.read(40).startswith(b"version https://git-lfs")
    if is_pointer:
        local.unlink()
        fetch(DATAHUB_LFS + path, local)
    return local


def key_values(lines) -> dict[str, str]:
    """'key: value' lines as a dict."""
    return {key.strip(): value.strip() for key, value in (line.split(":", 1) for line in lines if ":" in line)}


def prepare_cbioportal(study: str, raw_dir: Path):
    """Expression, gene labels, clinical table and coding driver events of a cBioPortal study."""
    names = [f["name"] for f in requests.get(DATAHUB_LISTING.format(study=study), timeout=60).json() if f["type"] == "file"]
    get = lambda name: datahub(f"public/{study}/{name}", raw_dir / Path(name).name)
    read = lambda name: pd.read_csv(get(name), sep="\t", comment="#", low_memory=False)
    metas = [key_values(get(name).read_text().splitlines()) for name in names if name.startswith("meta_")]

    # Expression: the continuous mRNA file (z-scores only when there is no other); duplicate symbols averaged
    mrna = [m for m in metas if m.get("genetic_alteration_type") == "MRNA_EXPRESSION"]
    expression_file = min(mrna, key=lambda m: (m.get("datatype") != "CONTINUOUS", "zscore" in m["data_filename"].lower(), m["data_filename"]))["data_filename"]
    table = read(expression_file).dropna(subset=["Hugo_Symbol"])
    expression = table.drop(columns="Entrez_Gene_Id", errors="ignore").set_index("Hugo_Symbol").apply(pd.to_numeric, errors="coerce")
    expression = expression.groupby(level=0).mean().T
    expression.index.name = "sample"

    # Coding events of the driver genes
    mutation_file = next(m["data_filename"] for m in metas if m.get("genetic_alteration_type") == "MUTATION_EXTENDED" and m.get("datatype") == "MAF")
    events = pd.concat(
        chunk[chunk["Variant_Classification"].isin(CODING_VARIANT_CLASSES) & chunk["Hugo_Symbol"].isin(DRIVER_GENES)]
        for chunk in pd.read_csv(get(mutation_file), sep="\t", comment="#", usecols=lambda c: c in MAF_COLUMNS,
                                 chunksize=200_000, low_memory=False)
    )

    # Gene labels: 1/0 for the sequenced samples (the study's sequenced case list), missing for the others
    sequenced = key_values(get("case_lists/cases_sequenced.txt").read_text().splitlines())["case_list_ids"].split("\t")
    labels = pd.crosstab(events["Tumor_Sample_Barcode"], events["Hugo_Symbol"]).gt(0)
    labels = labels.reindex(index=expression.index, columns=DRIVER_GENES, fill_value=False).astype(float)
    labels.loc[~labels.index.isin(sequenced)] = np.nan
    # Targeted panels (METABRIC): genes outside a sample's panel were not sequenced
    if "data_gene_panel_matrix.txt" in names:
        panels = read("data_gene_panel_matrix.txt").set_index("SAMPLE_ID")["mutations"].reindex(labels.index)
        for panel in panels.dropna().unique():
            if panel in ("WXS", "WGS", "WXS/WGS"):  # whole exome or genome: all genes
                continue
            panel_file = datahub(f"reference_data/gene_panels/data_gene_panel_{panel.lower()}.txt", raw_dir / f"{panel}.txt")
            genes = key_values(panel_file.read_text().splitlines())["gene_list"].split()
            labels.loc[panels.eq(panel), ~labels.columns.isin(genes)] = np.nan

    clinical = read("data_clinical_sample.txt").set_index("SAMPLE_ID")
    return expression, labels, clinical, events


def read_series_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Probe x sample values and the sample characteristics ('key: value' cells) of a GEO series matrix."""
    # Header lines up to the table; the table itself is read straight from the file
    header = []
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("!series_matrix_table_begin"):
                break
            header.append(line.rstrip("\n"))
    probes = pd.read_csv(path, sep="\t", index_col=0, skiprows=len(header) + 1).drop("!series_matrix_table_end")
    characteristics = {sample: {} for sample in probes.columns}
    for line in header:
        if line.startswith("!Sample_characteristics"):
            for sample, cell in zip(probes.columns, line.split("\t")[1:]):
                characteristics[sample].update(key_values([cell.strip('"')]))
    return probes, pd.DataFrame.from_dict(characteristics, orient="index")


def prepare_geo(accession: str, raw_dir: Path):
    """Expression, gene labels and clinical table of a GPL570 GEO series, tumour samples only."""
    matrix = fetch(GEO_MATRIX.format(group=accession[:-3] + "nnn", accession=accession), raw_dir / f"{accession}_series_matrix.txt.gz")
    probes, clinical = read_series_matrix(matrix)

    # Probes to the first gene symbol of the GPL570 annotation, averaged per gene
    with gzip.open(fetch(GPL570_ANNOTATION, raw_dir / "GPL570.annot.gz"), "rt", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    begin, end = lines.index("!platform_table_begin"), lines.index("!platform_table_end")
    annotation = pd.read_csv(io.StringIO("\n".join(lines[begin + 1:end])), sep="\t", index_col="ID", usecols=["ID", "Gene symbol"])
    symbols = annotation["Gene symbol"].dropna().str.split(" /// ").str[0]
    expression = probes.loc[probes.index.isin(symbols.index)].groupby(symbols).mean().T
    expression.index.name = "sample"

    if accession == "GSE39582":
        tumor = clinical["dataset"].ne("Non Tumoral")
        # 1 for M, 0 for WT, other values missing
        labels = pd.DataFrame({gene: clinical[f"{gene.lower()}.mutation"].str.upper().map({"M": 1.0, "WT": 0.0}) for gene in ["TP53", "KRAS", "BRAF"]})
    elif accession == "GSE31210":
        tumor = clinical["tissue"].eq("primary lung tumor")
        status = clinical["gene alteration status"]
        labels = pd.DataFrame({gene: status.eq(value).astype(float) for gene, value in
                               {"EGFR": "EGFR mutation +", "KRAS": "KRAS mutation +", "ALK": "ALK-fusion +"}.items()})
    else:
        raise ValueError(f"No label parsing for {accession}")
    return expression.loc[tumor[tumor].index], labels.loc[tumor], clinical.loc[tumor]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", help="Prepare only these sources (cBioPortal study IDs or GEO accessions).")
    args = parser.parse_args()
    config = yaml.safe_load((RUN_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))

    # Each source once (the POG570 cohorts share one study); GEO series are the GSE sources, the rest are cBioPortal studies
    sources = dict.fromkeys(cohort.get("source", key) for key, cohort in config["cohorts"].items())
    for source in sources:
        if args.only and source not in args.only:
            continue
        print(f"Preparing {source}", flush=True)
        raw_dir = RUN_ROOT / "data" / "raw" / source
        out_dir = RUN_ROOT / config["data"]["standardized_dir"] / source
        out_dir.mkdir(parents=True, exist_ok=True)
        if source.startswith("GSE"):
            expression, labels, clinical = prepare_geo(source, raw_dir)
        else:
            expression, labels, clinical, events = prepare_cbioportal(source, raw_dir)
            events.to_csv(out_dir / "mutations_long.csv.gz", index=False)
        expression.to_csv(out_dir / "expression.csv.gz")
        labels.to_csv(out_dir / "mutations_gene_level.csv.gz")
        clinical.to_csv(out_dir / "clinical_sample.csv.gz")
        shutil.rmtree(raw_dir, ignore_errors=True)  # OneDrive can hold the emptied folder
        print(f"  {expression.shape[0]} samples ({labels.notna().any(axis=1).sum()} sequenced), {expression.shape[1]} genes", flush=True)


if __name__ == "__main__":
    main()
