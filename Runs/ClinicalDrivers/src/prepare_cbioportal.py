from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from driver_genes import HOTSPOT_PATTERNS, NONSILENT_VARIANT_CLASSES, TARGET_DRIVER_GENES


CBIO_API_BASE = "https://www.cbioportal.org/api"
DATAHUB_GITHUB_API = "https://api.github.com/repos/cBioPortal/datahub/contents/public/{study_id}?ref=master"
DATAHUB_GITHUB_MEDIA = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/{study_id}/{filename}"
DATAHUB_GITHUB_RAW = "https://github.com/cBioPortal/datahub/raw/master/public/{study_id}/{filename}"
DOWNLOAD_BASES: list[str] = []


@dataclass(frozen=True)
class StudyPaths:
    study_id: str
    root: Path

    @property
    def raw_dir(self) -> Path:
        return self.root / "data" / "raw" / "cbioportal" / self.study_id

    @property
    def tar_path(self) -> Path:
        return self.root / "data" / "raw" / "cbioportal" / f"{self.study_id}.tar.gz"

    @property
    def standardized_dir(self) -> Path:
        return self.root / "data" / "standardized" / self.study_id


def request_json(url: str, timeout: int = 60):
    # cBioPortal refuses requests without a User-Agent
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "Final-E2M clinical-driver-validation"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload, timeout: int = 120):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Accept": "application/json", "Content-Type": "application/json", "User-Agent": "Final-E2M clinical-driver-validation"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, target: Path, timeout: int = 300, retries: int = 3) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Final-E2M clinical-driver-validation"})
            with urllib.request.urlopen(req, timeout=timeout) as response, tmp.open("wb") as out:
                shutil.copyfileobj(response, out)
            tmp.replace(target)
            return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if tmp.exists():
                tmp.unlink()
            time.sleep(min(2 * attempt, 10))
    raise RuntimeError(f"failed to download {url}: {last_error}") from last_error


def fetch_study_metadata(study_id: str) -> dict:
    return request_json(f"{CBIO_API_BASE}/studies/{study_id}")


def fetch_molecular_profiles(study_id: str) -> list[dict]:
    return request_json(f"{CBIO_API_BASE}/studies/{study_id}/molecular-profiles")


def mutation_profile_id(study_id: str, meta_records: list[dict[str, str]]) -> str | None:
    for record in meta_records:
        if record.get("genetic_alteration_type") == "MUTATION_EXTENDED":
            stable_id = record.get("stable_id")
            if stable_id:
                return f"{study_id}_{stable_id}"
    return None


def fetch_gene_map(genes: Iterable[str]) -> dict[str, int]:
    records = post_json(
        f"{CBIO_API_BASE}/genes/fetch?geneIdType=HUGO_GENE_SYMBOL&projection=SUMMARY",
        sorted(set(genes)),
    )
    return {record["hugoGeneSymbol"]: int(record["entrezGeneId"]) for record in records}


def api_mutations_to_maf(records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in records:
        gene = record.get("gene") or {}
        rows.append(
            {
                "Hugo_Symbol": gene.get("hugoGeneSymbol"),
                "Entrez_Gene_Id": record.get("entrezGeneId"),
                "Variant_Classification": record.get("mutationType"),
                "Variant_Type": record.get("variantType"),
                "Tumor_Sample_Barcode": record.get("sampleId"),
                "HGVSp_Short": record.get("proteinChange"),
                "HGVSp": record.get("proteinChange"),
                "Chromosome": record.get("chr"),
                "Start_Position": record.get("startPosition"),
                "End_Position": record.get("endPosition"),
                "Reference_Allele": record.get("referenceAllele"),
                "Tumor_Seq_Allele2": record.get("variantAllele"),
                "Mutation_Status": record.get("mutationStatus"),
                "Validation_Status": record.get("validationStatus"),
            }
        )
    return pd.DataFrame(rows)


def fetch_target_mutations_from_api(study_id: str, meta_records: list[dict[str, str]]) -> pd.DataFrame:
    profile_id = mutation_profile_id(study_id, meta_records)
    if profile_id is None:
        return pd.DataFrame()
    gene_map = fetch_gene_map(TARGET_DRIVER_GENES)
    sample_list_id = f"{study_id}_sequenced"
    records: list[dict] = []
    for gene in TARGET_DRIVER_GENES:
        entrez_id = gene_map.get(gene)
        if entrez_id is None:
            continue
        url = (
            f"{CBIO_API_BASE}/molecular-profiles/{profile_id}/mutations"
            f"?sampleListId={urllib.parse.quote(sample_list_id)}"
            f"&entrezGeneId={entrez_id}"
            "&projection=DETAILED&pageSize=10000000"
        )
        gene_records = request_json(url, timeout=120)
        records.extend(gene_records)
    maf = api_mutations_to_maf(records)
    return nonsilent_mutations(maf)


def is_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 512:
        return False
    try:
        first_line = path.open("r", encoding="utf-8", errors="replace").readline().strip()
    except OSError:
        return False
    return first_line == "version https://git-lfs.github.com/spec/v1"


def download_datahub_files_from_github(study_id: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    listing = request_json(DATAHUB_GITHUB_API.format(study_id=study_id))
    files = [item for item in listing if item.get("type") == "file"]
    if not files:
        raise RuntimeError(f"No top-level DataHub files listed for {study_id}")
    for item in files:
        filename = item["name"]
        target = raw_dir / filename
        if target.exists() and target.stat().st_size > 0 and not is_lfs_pointer(target):
            continue
        fallback_url = item.get("download_url")
        raw_url = DATAHUB_GITHUB_RAW.format(study_id=study_id, filename=urllib.parse.quote(filename))
        if filename.startswith("data_"):
            quoted = urllib.parse.quote(filename)
            url = DATAHUB_GITHUB_MEDIA.format(study_id=study_id, filename=quoted)
        else:
            url = fallback_url
        if not url:
            continue
        print(f"[clinical-driver-validation] downloading {study_id}/{filename}", flush=True)
        try:
            download_file(url, target)
        except RuntimeError:
            try:
                download_file(raw_url, target)
            except RuntimeError:
                if fallback_url and fallback_url not in {url, raw_url}:
                    download_file(fallback_url, target)
                else:
                    raise
    return raw_dir


def ensure_datahub_study(paths: StudyPaths, force_download: bool = False) -> Path:
    if force_download and paths.raw_dir.exists():
        shutil.rmtree(paths.raw_dir)
    if (
        paths.raw_dir.exists()
        and any(paths.raw_dir.glob("meta_*.txt"))
        and not any(is_lfs_pointer(path) for path in paths.raw_dir.glob("data_*.txt"))
    ):
        return paths.raw_dir

    if force_download and paths.tar_path.exists():
        paths.tar_path.unlink()
    if not paths.tar_path.exists():
        errors = []
        for template in DOWNLOAD_BASES:
            url = template.format(study_id=paths.study_id)
            try:
                download_file(url, paths.tar_path)
                break
            except RuntimeError as exc:
                errors.append(str(exc))
        else:
            print(
                "[clinical-driver-validation] tarball download failed; "
                "falling back to GitHub media files",
                flush=True,
            )
            print("\n".join(errors), flush=True)
            return download_datahub_files_from_github(paths.study_id, paths.raw_dir)

    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(paths.tar_path, "r:gz") as tar:
        def safe_members():
            for member in tar.getmembers():
                member_path = paths.raw_dir / member.name
                if not member_path.resolve().is_relative_to(paths.raw_dir.resolve()):
                    raise RuntimeError(f"unsafe tar member: {member.name}")
                yield member

        tar.extractall(paths.raw_dir, members=safe_members())

    nested = paths.raw_dir / paths.study_id
    if nested.exists() and nested.is_dir():
        for item in nested.iterdir():
            target = paths.raw_dir / item.name
            if target.exists():
                continue
            item.replace(target)
        try:
            nested.rmdir()
        except OSError:
            pass
    return paths.raw_dir


def parse_meta_file(path: Path) -> dict[str, str]:
    data = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def collect_meta(raw_dir: Path) -> list[dict[str, str]]:
    records = []
    for meta_file in sorted(raw_dir.glob("meta_*.txt")):
        record = parse_meta_file(meta_file)
        record["_meta_file"] = meta_file.name
        records.append(record)
    return records


def choose_data_file(meta_records: Iterable[dict[str, str]], *, alteration_type: str, datatype: str | None = None) -> Path | None:
    candidates = []
    for record in meta_records:
        if record.get("genetic_alteration_type") != alteration_type:
            continue
        if datatype is not None and record.get("datatype") != datatype:
            continue
        filename = record.get("data_filename")
        if filename:
            candidates.append(filename)
    if not candidates:
        return None
    # Prefer raw continuous mRNA over z-scores if both are present.
    candidates = sorted(candidates, key=lambda name: ("zscore" in name.lower(), name))
    return Path(candidates[0])


def read_cbio_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", comment="#", low_memory=False)


def load_expression(raw_dir: Path, meta_records: list[dict[str, str]]) -> pd.DataFrame:
    rel = choose_data_file(meta_records, alteration_type="MRNA_EXPRESSION", datatype="CONTINUOUS")
    if rel is None:
        rel = choose_data_file(meta_records, alteration_type="MRNA_EXPRESSION")
    if rel is None:
        raise RuntimeError("No MRNA_EXPRESSION data file found")

    df = read_cbio_table(raw_dir / rel)
    gene_col = "Hugo_Symbol" if "Hugo_Symbol" in df.columns else df.columns[0]
    drop_cols = [c for c in [gene_col, "Entrez_Gene_Id"] if c in df.columns]
    expr = df.drop(columns=drop_cols)
    expr.index = df[gene_col].astype(str)
    expr = expr.loc[expr.index.notna() & (expr.index != "") & (expr.index != "nan")]
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.groupby(expr.index).mean()
    return expr.T


def load_sample_clinical(raw_dir: Path) -> pd.DataFrame:
    candidates = [
        raw_dir / "data_clinical_sample.txt",
        raw_dir / "data_clinical_samples.txt",
    ]
    existing = next((p for p in candidates if p.exists()), None)
    if existing is None:
        return pd.DataFrame()
    clinical = read_cbio_table(existing)
    sample_col = "SAMPLE_ID" if "SAMPLE_ID" in clinical.columns else clinical.columns[0]
    clinical = clinical.rename(columns={sample_col: "sample_id"})
    return clinical.set_index("sample_id", drop=False)


def mutation_file_path(raw_dir: Path, meta_records: list[dict[str, str]]) -> Path | None:
    rel = choose_data_file(meta_records, alteration_type="MUTATION_EXTENDED", datatype="MAF")
    if rel is None:
        rel = choose_data_file(meta_records, alteration_type="MUTATION_EXTENDED")
    return raw_dir / rel if rel is not None else None


def nonsilent_mutations(maf: pd.DataFrame) -> pd.DataFrame:
    if maf.empty:
        return maf
    if "Variant_Classification" not in maf.columns:
        return maf
    return maf[maf["Variant_Classification"].isin(NONSILENT_VARIANT_CLASSES)].copy()


def load_nonsilent_target_mutations(raw_dir: Path, meta_records: list[dict[str, str]], study_id: str) -> pd.DataFrame:
    path = mutation_file_path(raw_dir, meta_records)
    if path is None:
        return fetch_target_mutations_from_api(study_id, meta_records)
    if is_lfs_pointer(path):
        return fetch_target_mutations_from_api(study_id, meta_records)

    header = pd.read_csv(path, sep="\t", comment="#", nrows=0).columns.tolist()
    wanted = [
        "Hugo_Symbol",
        "Variant_Classification",
        "Variant_Type",
        "Tumor_Sample_Barcode",
        "HGVSc",
        "HGVSp",
        "HGVSp_Short",
        "Protein_Change",
        "Amino_Acid_Change",
        "Hotspot",
        "Chromosome",
        "Start_Position",
        "End_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele2",
    ]
    usecols = [col for col in wanted if col in header]
    required = {"Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode"}
    if not required.issubset(usecols):
        missing = ", ".join(sorted(required - set(usecols)))
        raise RuntimeError(f"Mutation file lacks required columns: {missing}")

    target_genes = set(TARGET_DRIVER_GENES) | set(HOTSPOT_PATTERNS)
    chunks = []
    for chunk in pd.read_csv(path, sep="\t", comment="#", usecols=usecols, chunksize=200_000, low_memory=False):
        filtered = chunk[
            chunk["Variant_Classification"].isin(NONSILENT_VARIANT_CLASSES)
            & chunk["Hugo_Symbol"].isin(target_genes)
        ].copy()
        if not filtered.empty:
            chunks.append(filtered)
    if not chunks:
        return pd.DataFrame(columns=usecols)
    return pd.concat(chunks, ignore_index=True)


def gene_level_matrix(maf: pd.DataFrame, sample_ids: list[str]) -> pd.DataFrame:
    if maf.empty:
        return pd.DataFrame(index=sample_ids)
    required = {"Tumor_Sample_Barcode", "Hugo_Symbol"}
    if not required.issubset(maf.columns):
        missing = ", ".join(sorted(required - set(maf.columns)))
        raise RuntimeError(f"Mutation file lacks required columns: {missing}")
    pairs = maf[["Tumor_Sample_Barcode", "Hugo_Symbol"]].dropna().drop_duplicates()
    pairs["Tumor_Sample_Barcode"] = pairs["Tumor_Sample_Barcode"].astype(str)
    pairs["Hugo_Symbol"] = pairs["Hugo_Symbol"].astype(str)
    pairs["value"] = 1
    mat = pairs.pivot_table(
        index="Tumor_Sample_Barcode",
        columns="Hugo_Symbol",
        values="value",
        fill_value=0,
        aggfunc="max",
    )
    mat = mat.reindex(sample_ids, fill_value=0).astype("int8")
    mat.columns.name = None
    return mat


def select_hotspot_records(maf: pd.DataFrame) -> pd.DataFrame:
    if maf.empty:
        return maf
    protein_cols = [c for c in ["HGVSp_Short", "Protein_Change", "Amino_Acid_Change", "HGVSp"] if c in maf.columns]
    if not protein_cols or "Hugo_Symbol" not in maf.columns:
        return pd.DataFrame(columns=list(maf.columns) + ["hotspot_pattern"])
    records = []
    for gene, patterns in HOTSPOT_PATTERNS.items():
        sub = maf[maf["Hugo_Symbol"].eq(gene)]
        if sub.empty:
            continue
        protein_text = sub[protein_cols].fillna("").astype(str).agg(" ".join, axis=1)
        mask = pd.Series(False, index=sub.index)
        hit = pd.Series("", index=sub.index, dtype="object")
        for pattern in patterns:
            pattern_mask = protein_text.str.contains(pattern, case=False, regex=False)
            mask |= pattern_mask
            hit.loc[pattern_mask & hit.eq("")] = pattern
        selected = sub.loc[mask].copy()
        selected["hotspot_pattern"] = hit.loc[mask]
        records.append(selected)
    if not records:
        return pd.DataFrame(columns=list(maf.columns) + ["hotspot_pattern"])
    return pd.concat(records, ignore_index=True)


def target_gene_counts(mutation_matrix: pd.DataFrame, sample_ids: list[str]) -> pd.DataFrame:
    rows = []
    n = len(sample_ids)
    for gene in TARGET_DRIVER_GENES:
        positives = int(mutation_matrix[gene].sum()) if gene in mutation_matrix.columns else 0
        rows.append(
            {
                "gene": gene,
                "n_samples": n,
                "n_positive": positives,
                "prevalence": positives / n if n else pd.NA,
                "present_in_mutation_matrix": gene in mutation_matrix.columns,
            }
        )
    return pd.DataFrame(rows)


def process_study(study_id: str, root: Path, force_download: bool = False, cleanup_raw: bool = True) -> dict[str, object]:
    paths = StudyPaths(study_id=study_id, root=root)
    paths.standardized_dir.mkdir(parents=True, exist_ok=True)

    metadata = fetch_study_metadata(study_id)
    profiles = fetch_molecular_profiles(study_id)
    raw_dir = ensure_datahub_study(paths, force_download=force_download)
    meta_records = collect_meta(raw_dir)

    expression = load_expression(raw_dir, meta_records)
    clinical = load_sample_clinical(raw_dir)
    maf_ns = load_nonsilent_target_mutations(raw_dir, meta_records, study_id)

    sample_ids = list(expression.index.astype(str))
    # Samples without mutation sequencing (not in the study's "sequenced" list) get missing labels, not wild-type
    sequenced = set(request_json(f"{CBIO_API_BASE}/sample-lists/{study_id}_sequenced/sample-ids"))
    mutation_matrix = gene_level_matrix(maf_ns, sample_ids).astype(float)
    mutation_matrix.loc[~mutation_matrix.index.isin(sequenced)] = float("nan")
    hotspots = select_hotspot_records(maf_ns)
    counts = target_gene_counts(mutation_matrix, [s for s in sample_ids if s in sequenced])

    expression.to_csv(paths.standardized_dir / "expression.csv.gz", compression="gzip")
    mutation_matrix.to_csv(paths.standardized_dir / "mutations_gene_level.csv.gz", compression="gzip")
    clinical.to_csv(paths.standardized_dir / "clinical_sample.csv.gz", compression="gzip", index=False)
    maf_ns.to_csv(paths.standardized_dir / "mutations_long.csv.gz", compression="gzip", index=False)
    hotspots.to_csv(paths.standardized_dir / "mutations_hotspot_long.csv.gz", compression="gzip", index=False)
    counts.to_csv(paths.standardized_dir / "target_gene_counts.csv", index=False)

    qc = {
        "study_id": study_id,
        "name": metadata.get("name"),
        "all_sample_count_api": metadata.get("allSampleCount"),
        "sequenced_sample_count_api": metadata.get("sequencedSampleCount"),
        "mrna_rnaseq_sample_count_api": metadata.get("mrnaRnaSeqSampleCount"),
        "mrna_rnaseq_v2_sample_count_api": metadata.get("mrnaRnaSeqV2SampleCount"),
        "complete_sample_count_api": metadata.get("completeSampleCount"),
        "expression_samples": expression.shape[0],
        "expression_samples_sequenced": int(mutation_matrix.index.isin(sequenced).sum()),
        "expression_genes": expression.shape[1],
        "clinical_samples": clinical.shape[0],
        "target_nonsilent_mutation_records": maf_ns.shape[0],
        "target_mutated_genes_observed": mutation_matrix.shape[1],
        "hotspot_records": hotspots.shape[0],
        "target_genes_with_positive": int((counts["n_positive"] > 0).sum()),
        "raw_dir": str(raw_dir),
        "standardized_dir": str(paths.standardized_dir),
    }
    pd.DataFrame([qc]).to_csv(paths.standardized_dir / "qc_summary.csv", index=False)
    (paths.standardized_dir / "api_study_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (paths.standardized_dir / "api_molecular_profiles.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")

    if cleanup_raw:
        shutil.rmtree(paths.raw_dir, ignore_errors=True)
        paths.tar_path.unlink(missing_ok=True)
    return qc


def load_dataset_config(root: Path) -> dict:
    config_path = root / "config" / "datasets.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def phase1_study_ids(root: Path) -> list[str]:
    config = load_dataset_config(root)
    return [record["study_id"] for record in config["phase1_cbioportal"]]


def aggregate_phase1_qc(root: Path) -> Path:
    rows = []
    for study_id in phase1_study_ids(root):
        qc_path = root / "data" / "standardized" / study_id / "qc_summary.csv"
        if qc_path.exists():
            rows.append(pd.read_csv(qc_path))
    out_path = root / "data" / "standardized" / "phase1_qc_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out_path, index=False)
    else:
        pd.DataFrame().to_csv(out_path, index=False)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and standardize cBioPortal/DataHub studies.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--study", action="append", help="Study ID to process; may be supplied multiple times.")
    parser.add_argument("--all", action="store_true", help="Process all phase-1 cBioPortal studies.")
    parser.add_argument("--force-download", action="store_true", help="Re-download and re-extract raw DataHub tarballs.")
    parser.add_argument("--keep-raw", action="store_true", help="Keep raw downloads after standardization (default: delete them).")
    parser.add_argument("--summarize-only", action="store_true", help="Only rebuild the aggregate phase-1 QC summary.")
    args = parser.parse_args(argv)

    if args.summarize_only:
        out_path = aggregate_phase1_qc(args.root)
        print(f"[clinical-driver-validation] wrote {out_path}", flush=True)
        return 0

    studies = args.study or []
    if args.all:
        studies = phase1_study_ids(args.root)
    if not studies:
        parser.error("provide --study STUDY_ID or --all")

    for study_id in studies:
        print(f"[clinical-driver-validation] processing {study_id}", flush=True)
        process_study(study_id, args.root, force_download=args.force_download, cleanup_raw=not args.keep_raw)
    out_path = aggregate_phase1_qc(args.root)
    print(f"[clinical-driver-validation] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
