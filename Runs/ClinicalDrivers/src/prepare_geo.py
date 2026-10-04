from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import shutil
import urllib.request
from collections import defaultdict
from pathlib import Path

import pandas as pd


GPL570_ANNOT_URL = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz"

ANNOTATION_PATTERNS = {
    "KRAS": r"\bKRAS\b",
    "BRAF": r"\bBRAF\b",
    "TP53": r"\bTP53\b|\bp53\b",
    "PIK3CA": r"\bPIK3CA\b",
    "APC": r"\bAPC\b",
    "EGFR": r"\bEGFR\b",
    "ALK": r"\bALK\b",
    "MMR": r"\bMMR\b|microsatellite|MSI|MSS",
    "CIMP": r"\bCIMP\b",
    "mutation": r"mutat",
    "wildtype": r"wild.?type|\bWT\b",
}


def download_file(url: str, target: Path, timeout: int = 300) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return
    tmp = target.with_suffix(target.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Final-E2M clinical-driver-validation"})
    with urllib.request.urlopen(req, timeout=timeout) as response, tmp.open("wb") as out:
        shutil.copyfileobj(response, out)
    tmp.replace(target)


def geo_series_group(accession: str) -> str:
    match = re.fullmatch(r"GSE(\d+)", accession.upper())
    if not match:
        raise ValueError(f"Unsupported GEO accession: {accession}")
    digits = match.group(1)
    prefix = digits[:-3] if len(digits) > 3 else ""
    return f"GSE{prefix}nnn"


def series_matrix_url(accession: str) -> str:
    accession = accession.upper()
    group = geo_series_group(accession)
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{group}/{accession}/matrix/{accession}_series_matrix.txt.gz"


def strip_geo_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1]
    return value


def parse_series_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata: dict[str, list[list[str]]] = defaultdict(list)
    table_lines: list[str] = []
    in_table = False
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line == "!series_matrix_table_begin":
                in_table = True
                continue
            if line == "!series_matrix_table_end":
                in_table = False
                continue
            if in_table:
                table_lines.append(line)
                continue
            if line.startswith("!Sample_"):
                parts = line.split("\t")
                key = parts[0].lstrip("!")
                values = [strip_geo_value(part) for part in parts[1:]]
                metadata[key].append(values)

    if not table_lines:
        raise RuntimeError(f"No series matrix table found in {path}")

    probe_df = pd.read_csv(io.StringIO("\n".join(table_lines)), sep="\t", low_memory=False)
    id_col = "ID_REF" if "ID_REF" in probe_df.columns else probe_df.columns[0]
    probe_df = probe_df.rename(columns={id_col: "probe_id"}).set_index("probe_id")
    probe_df = probe_df.apply(pd.to_numeric, errors="coerce")

    sample_ids = list(probe_df.columns.astype(str))
    clinical = pd.DataFrame(index=sample_ids)
    clinical.index.name = "sample_id"
    clinical["sample_id"] = sample_ids
    for key, value_lists in metadata.items():
        for idx, values in enumerate(value_lists, start=1):
            col = key if len(value_lists) == 1 else f"{key}_{idx}"
            if len(values) == len(sample_ids):
                clinical[col] = values

    return probe_df, clinical


def load_gpl570_mapping(path: Path) -> pd.DataFrame:
    rows = []
    in_table = False
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line == "!platform_table_begin":
                in_table = True
                continue
            if line == "!platform_table_end":
                break
            if in_table:
                rows.append(line)
    if not rows:
        raise RuntimeError(f"No GPL table found in {path}")
    annot = pd.read_csv(io.StringIO("\n".join(rows)), sep="\t", low_memory=False)
    symbol_col = next((c for c in annot.columns if c.lower() in {"gene symbol", "gene_symbol", "symbol"}), None)
    if symbol_col is None:
        raise RuntimeError(f"No gene-symbol column found in GPL annotation columns: {list(annot.columns)}")
    mapping = annot[["ID", symbol_col]].rename(columns={"ID": "probe_id", symbol_col: "gene_symbol"}).copy()
    mapping["gene_symbol"] = (
        mapping["gene_symbol"]
        .astype(str)
        .str.split(" /// ")
        .str[0]
        .str.strip()
    )
    mapping = mapping[
        mapping["probe_id"].notna()
        & mapping["gene_symbol"].notna()
        & ~mapping["gene_symbol"].isin(["", "---", "nan"])
    ]
    return mapping.drop_duplicates("probe_id")


def collapse_probes_to_genes(probe_expression: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    joined = probe_expression.join(mapping.set_index("probe_id"), how="inner")
    gene_symbols = joined.pop("gene_symbol")
    gene_expression = joined.groupby(gene_symbols).mean()
    gene_expression.index.name = "gene"
    return gene_expression.T


def scan_annotation_terms(clinical: pd.DataFrame) -> pd.DataFrame:
    text_cols = [c for c in clinical.columns if c != "sample_id"]
    joined = clinical[text_cols].fillna("").astype(str).agg(" | ".join, axis=1)
    rows = []
    for term, pattern in ANNOTATION_PATTERNS.items():
        mask = joined.str.contains(pattern, case=False, regex=True, na=False)
        for sample_id, value in joined[mask].items():
            rows.append({"sample_id": sample_id, "term": term, "annotation_text": value[:1000]})
    return pd.DataFrame(rows)


def values_after_colon(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.split(":", n=1).str[-1].str.strip()


def find_characteristic_column(clinical: pd.DataFrame, prefix: str) -> str | None:
    prefix = prefix.lower()
    for col in clinical.columns:
        values = clinical[col].dropna().astype(str).str.lower()
        if not values.empty and values.str.startswith(prefix).any():
            return col
    return None


def map_mutation_status(series: pd.Series) -> pd.Series:
    values = values_after_colon(series).str.upper()
    mapped = values.map({"M": 1, "WT": 0})
    return mapped.astype("Int64")


def extract_geo_driver_labels(accession: str, clinical: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = pd.DataFrame({"sample_id": clinical["sample_id"].astype(str)})
    mutation_cols: dict[str, pd.Series] = {}

    if accession == "GSE39582":
        dataset_col = find_characteristic_column(clinical, "dataset:")
        if dataset_col:
            dataset = values_after_colon(clinical[dataset_col])
            labels["dataset"] = dataset
            labels["is_tumor"] = dataset.ne("Non Tumoral").astype("Int64")

        for gene in ["TP53", "KRAS", "BRAF"]:
            status_col = find_characteristic_column(clinical, f"{gene.lower()}.mutation:")
            protein_col = find_characteristic_column(clinical, f"{gene.lower()}.mutation.protein:")
            dna_col = find_characteristic_column(clinical, f"{gene.lower()}.mutation.dna:")
            if status_col:
                labels[f"{gene}_mutation"] = map_mutation_status(clinical[status_col])
                mutation_cols[gene] = labels[f"{gene}_mutation"]
            if protein_col:
                labels[f"{gene}_protein_change"] = values_after_colon(clinical[protein_col])
            if dna_col:
                labels[f"{gene}_dna_change"] = values_after_colon(clinical[dna_col])

        mmr_col = find_characteristic_column(clinical, "mmr.status:")
        if mmr_col:
            values = values_after_colon(clinical[mmr_col])
            labels["MMR_deficient"] = values.map({"dMMR": 1, "pMMR": 0}).astype("Int64")
            labels["MMR_status"] = values

        cimp_col = find_characteristic_column(clinical, "cimp.status:")
        if cimp_col:
            values = values_after_colon(clinical[cimp_col])
            labels["CIMP_positive"] = values.map({"+": 1, "-": 0}).astype("Int64")
            labels["CIMP_status"] = values

    elif accession == "GSE31210":
        status_col = find_characteristic_column(clinical, "gene alteration status:")
        if status_col:
            values = values_after_colon(clinical[status_col])
            labels["gene_alteration_status"] = values
            for gene, pattern in {
                "EGFR": "EGFR mutation +",
                "KRAS": "KRAS mutation +",
                "ALK": "ALK-fusion +",
            }.items():
                label = values.eq(pattern).astype("Int64")
                labels[f"{gene}_alteration"] = label
                mutation_cols[gene] = label
            labels["triple_negative_EGFR_KRAS_ALK"] = values.eq("EGFR/KRAS/ALK -").astype("Int64")

    mutation_matrix = pd.DataFrame(index=clinical["sample_id"].astype(str))
    for gene, values in mutation_cols.items():
        mutation_matrix[gene] = values.to_numpy()
    mutation_matrix.index.name = "sample_id"
    return labels, mutation_matrix


def target_gene_counts(mutation_matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = mutation_matrix.shape[0]
    for gene in mutation_matrix.columns:
        values = mutation_matrix[gene]
        evaluable = int(values.notna().sum())
        positives = int(values.fillna(0).sum())
        rows.append(
            {
                "gene": gene,
                "n_samples": n,
                "n_evaluable": evaluable,
                "n_positive": positives,
                "prevalence": positives / evaluable if evaluable else pd.NA,
            }
        )
    return pd.DataFrame(rows)


def load_dataset_config(root: Path) -> dict:
    return json.loads((root / "config" / "datasets.json").read_text(encoding="utf-8"))


def geo_accessions(root: Path) -> list[str]:
    config = load_dataset_config(root)
    return [record["accession"] for record in config["phase2_geo"]]


def process_geo_accession(accession: str, root: Path, cleanup_raw: bool = True) -> dict[str, object]:
    accession = accession.upper()
    raw_dir = root / "data" / "raw" / "geo" / accession
    out_dir = root / "data" / "standardized" / accession
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = raw_dir / f"{accession}_series_matrix.txt.gz"
    gpl_path = root / "data" / "raw" / "geo" / "GPL570.annot.gz"
    download_file(series_matrix_url(accession), matrix_path)
    download_file(GPL570_ANNOT_URL, gpl_path)

    probe_expression, clinical = parse_series_matrix(matrix_path)
    mapping = load_gpl570_mapping(gpl_path)
    expression = collapse_probes_to_genes(probe_expression, mapping)
    term_hits = scan_annotation_terms(clinical)
    labels, mutation_matrix = extract_geo_driver_labels(accession, clinical)
    counts = target_gene_counts(mutation_matrix)

    expression.to_csv(out_dir / "expression.csv.gz", compression="gzip")
    clinical.to_csv(out_dir / "clinical_sample.csv.gz", compression="gzip", index=False)
    labels.to_csv(out_dir / "driver_labels.csv.gz", compression="gzip", index=False)
    mutation_matrix.to_csv(out_dir / "mutations_gene_level.csv.gz", compression="gzip")
    counts.to_csv(out_dir / "target_gene_counts.csv", index=False)
    term_hits.to_csv(out_dir / "annotation_term_hits.csv", index=False)

    qc = {
        "accession": accession,
        "expression_samples": expression.shape[0],
        "expression_genes": expression.shape[1],
        "probe_rows": probe_expression.shape[0],
        "clinical_samples": clinical.shape[0],
        "clinical_columns": clinical.shape[1],
        "parsed_driver_label_columns": labels.shape[1] - 1,
        "parsed_gene_label_columns": mutation_matrix.shape[1],
        "annotation_term_hits": term_hits.shape[0],
        "annotation_terms_detected": ";".join(sorted(term_hits["term"].unique())) if not term_hits.empty else "",
        "raw_dir": str(raw_dir),
        "standardized_dir": str(out_dir),
    }
    pd.DataFrame([qc]).to_csv(out_dir / "qc_summary.csv", index=False)

    if cleanup_raw:
        shutil.rmtree(raw_dir, ignore_errors=True)
    return qc


def aggregate_geo_qc(root: Path) -> Path:
    rows = []
    for accession in geo_accessions(root):
        qc_path = root / "data" / "standardized" / accession.upper() / "qc_summary.csv"
        if qc_path.exists():
            rows.append(pd.read_csv(qc_path))
    out_path = root / "data" / "standardized" / "phase2_geo_qc_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out_path, index=False)
    else:
        pd.DataFrame().to_csv(out_path, index=False)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and standardize GEO microarray cohorts.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--accession", action="append", help="GEO accession to process; may be supplied multiple times.")
    parser.add_argument("--all", action="store_true", help="Process all phase-2 GEO accessions.")
    parser.add_argument("--keep-raw", action="store_true", help="Keep raw downloads after standardization (default: delete them).")
    parser.add_argument("--summarize-only", action="store_true", help="Only rebuild aggregate GEO QC summary.")
    args = parser.parse_args(argv)

    if args.summarize_only:
        out_path = aggregate_geo_qc(args.root)
        print(f"[clinical-driver-validation] wrote {out_path}", flush=True)
        return 0

    accessions = args.accession or []
    if args.all:
        accessions = geo_accessions(args.root)
    if not accessions:
        parser.error("provide --accession ACCESSION or --all")

    for accession in accessions:
        print(f"[clinical-driver-validation] processing {accession}", flush=True)
        process_geo_accession(accession, args.root, cleanup_raw=not args.keep_raw)
    out_path = aggregate_geo_qc(args.root)
    print(f"[clinical-driver-validation] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
