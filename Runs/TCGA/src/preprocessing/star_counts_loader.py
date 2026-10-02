from __future__ import annotations

import gzip
import logging
import shutil
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import requests
import yaml


DEFAULT_TCGA_COHORTS = [
    "ACC",
    "BLCA",
    "BRCA",
    "CESC",
    "CHOL",
    "COAD",
    "DLBC",
    "ESCA",
    "GBM",
    "HNSC",
    "KICH",
    "KIRC",
    "KIRP",
    "LAML",
    "LGG",
    "LIHC",
    "LUAD",
    "LUSC",
    "MESO",
    "OV",
    "PAAD",
    "PCPG",
    "PRAD",
    "READ",
    "SARC",
    "SKCM",
    "STAD",
    "TGCT",
    "THCA",
    "THYM",
    "UCEC",
    "UCS",
    "UVM",
]

COMPOSITE_XENA_COHORTS = {"COADREAD", "FPPP", "GBMLGG", "LUNG", "PANCAN"}


class StarCountsTCGALoader:
    """Focused TCGA loader for per-cohort Xena STAR counts plus MC3 labels/TMB."""

    def __init__(self, config_path: str | Path | None = None, use_cache: bool = True):
        self.project_root = Path(__file__).resolve().parents[2]
        config_path = Path(config_path or self.project_root / "config" / "config.yaml")
        if not config_path.is_absolute():
            config_path = config_path.resolve() if config_path.exists() else (self.project_root / config_path).resolve()
        self.config_path = config_path

        with open(config_path, "r") as handle:
            self.config = yaml.safe_load(handle)
        self.data_cfg = self.config.get("data", {})

        self.data_dir = self._resolve_path(self.data_cfg.get("data_dir", "data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = self._resolve_project_path(self.data_cfg.get("cache_dir", "cache"))
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = cache_dir
        self.use_cache = use_cache and bool(self.data_cfg.get("use_cache", True))

        self._gene_gtf: pd.DataFrame | None = None
        self._protein_gene_ids: set[str] | None = None
        self._protein_gene_symbols: set[str] | None = None
        self._gene_symbol_map: pd.Series | None = None

        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            logging.basicConfig(level=logging.INFO)

    def load_expression_data(self, cancer_types: Optional[list[str]] = None) -> pd.DataFrame:
        cohorts = self.normalize_cancer_types(cancer_types)
        expression_cache = self._expression_cache_path(cohorts)
        if self.use_cache and expression_cache.exists():
            print(f"Loading cached STAR-count expression matrix:\n{expression_cache}", flush=True)
            return joblib.load(expression_cache)

        files = self._expression_files(cohorts)
        symbol_map = self.gene_id_to_symbol()
        protein_ids = self.protein_coding_gene_ids() if self.data_cfg.get("only_protein_coding_expression", True) else None

        frames = []
        start_all = time.perf_counter()
        for i, file_path in enumerate(files, start=1):
            start = time.perf_counter()
            cancer = file_path.name.split(".")[0].replace("TCGA-", "").upper()
            print(f"[expression {i}/{len(files)}] Reading {cancer}: {file_path.name}", flush=True)
            frame = pd.read_csv(file_path, sep="\t", index_col=0).T
            print(
                f"[expression {i}/{len(files)}] {cancer}: raw {frame.shape[0]:,} samples x "
                f"{frame.shape[1]:,} genes",
                flush=True,
            )
            frame.columns = frame.columns.astype(str)
            if self.data_cfg.get("expression_log2_count_plus_one", True):
                frame = (2.0 ** frame) - 1.0
            if self.data_cfg.get("expression_log1p", False):
                frame = np.log1p(frame.clip(lower=0))

            gene_ids = pd.Index(frame.columns.astype(str))
            gene_base = gene_ids.str.split(".").str[0]
            if protein_ids is not None:
                keep = gene_base.isin(protein_ids)
                frame = frame.loc[:, keep]
                gene_base = gene_base[keep]
                gene_ids = gene_ids[keep]

            symbols = pd.Index([symbol_map.get(gene_id, gene_id) for gene_id in gene_ids])
            symbols = symbols.where(symbols.notna(), gene_base)
            frame.columns = symbols.astype(str)
            frames.append(frame)
            print(
                f"[expression {i}/{len(files)}] {cancer}: retained {frame.shape[1]:,} genes "
                f"in {time.perf_counter() - start:.1f}s",
                flush=True,
            )

        print(f"Concatenating {len(frames)} expression matrices...", flush=True)
        expression = pd.concat(frames, axis=0)
        expression.index = expression.index.astype(str)
        expression.index.name = "sample"
        expression = expression[expression.index.str.startswith("TCGA-", na=False)]
        expression = expression[~expression.index.duplicated(keep="first")]
        expression = expression.apply(pd.to_numeric, errors="coerce")

        if self.data_cfg.get("collapse_duplicate_gene_symbols", True):
            print("Collapsing duplicate expression gene symbols...", flush=True)
            expression = expression.T.groupby(level=0, sort=False).mean().T

        expression = expression.fillna(0.0)
        print(
            f"STAR-count expression: {expression.shape[0]:,} samples x {expression.shape[1]:,} genes "
            f"in {time.perf_counter() - start_all:.1f}s",
            flush=True,
        )
        if self.use_cache:
            print(f"Caching STAR-count expression matrix:\n{expression_cache}", flush=True)
            joblib.dump(expression, expression_cache)
        return expression

    def load_mutation_data(self, cancer_types: Optional[list[str]] = None) -> pd.DataFrame:
        cohorts = self.normalize_cancer_types(cancer_types)
        files = self._mutation_files(cohorts)
        frames = []
        for file_path in files:
            matrix = pd.read_csv(file_path, sep="\t", compression="gzip", index_col=0).T
            frames.append(matrix)
        labels = pd.concat(frames, axis=0)

        if self.data_cfg.get("add_a_suffix_to_mutation_samples", True):
            labels.index = labels.index.astype(str) + "A"
        else:
            labels.index = labels.index.astype(str)
        labels.index.name = "sample"
        labels = labels[~labels.index.duplicated(keep="first")]
        labels = labels.apply(pd.to_numeric, errors="coerce").fillna(0)

        if self.data_cfg.get("only_protein_coding_mutations", True):
            protein_symbols = self.protein_coding_gene_symbols()
            labels = labels.loc[:, labels.columns.astype(str).isin(protein_symbols)]

        labels = (labels > 0).astype(np.int8)
        labels = self._filter_min_mutations(labels)
        print(f"MC3 gene labels: {labels.shape[0]:,} samples x {labels.shape[1]:,} genes", flush=True)
        return labels

    def load_sample_metadata(self, cancer_types: Optional[list[str]] = None) -> pd.DataFrame:
        cohorts = self.normalize_cancer_types(cancer_types)
        rows = []
        for file_path in self._expression_files(cohorts):
            cancer = file_path.name.split(".")[0].replace("TCGA-", "").upper()
            header = pd.read_csv(file_path, sep="\t", nrows=0).columns.astype(str).tolist()
            for sample in header[1:]:
                if sample.startswith("TCGA-"):
                    rows.append({"sample": sample, "Cancer": cancer, "primary_disease": f"TCGA-{cancer}"})
        if not rows:
            raise ValueError("No TCGA sample metadata found in STAR-count expression headers.")
        return pd.DataFrame(rows).drop_duplicates("sample").set_index("sample")

    def load_tmb_data(
        self,
        cancer_types: Optional[list[str]] = None,
        sample_index: Optional[pd.Index] = None,
    ) -> pd.DataFrame:
        cohorts = self.normalize_cancer_types(cancer_types)
        mut = self.load_mc3_events(cancer_types=cohorts)
        variant_classes = set(self.data_cfg.get("coding_variant_classes", []))
        if variant_classes:
            mut = mut[mut["variant"].isin(variant_classes)]
        mut["sample"] = mut["sample"].map(self.legacy_tcga_sample_id)

        if sample_index is None:
            meta = self.load_sample_metadata(cancer_types)
            sample_index = meta.index
        sample_index = pd.Index(sample_index).astype(str)
        mut = mut[mut["sample"].isin(set(sample_index))]

        tmb = mut.groupby("sample").size().rename("TMB").to_frame()
        tmb["TMB_log2"] = np.log2(tmb["TMB"].astype(float) + 1.0)
        dropped = int(len(sample_index) - len(tmb.index.intersection(sample_index)))
        if dropped:
            print(
                f"Dropping {dropped:,} expression samples with no coding MC3 mutation event for TMB.",
                flush=True,
            )
        print(f"TMB targets with coding mutation events: {len(tmb):,} samples", flush=True)
        return tmb

    def preprocess_data(self, cancer_types: Optional[list[str]] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        cohorts = self.normalize_cancer_types(cancer_types)
        expr_cache, mut_cache = self._cache_paths(cohorts)
        if self.use_cache and expr_cache.exists() and mut_cache.exists():
            print("Loading cached expression/mutation matrices:")
            print(expr_cache)
            print(mut_cache)
            return joblib.load(expr_cache), joblib.load(mut_cache)

        expression = self.load_expression_data(cohorts)
        mutation = self.load_mutation_data(cohorts)
        expression, mutation = self.align_expression_mutation(expression, mutation)

        if self.use_cache:
            joblib.dump(expression, expr_cache)
            joblib.dump(mutation, mut_cache)
        return expression, mutation

    def align_expression_mutation(
        self,
        expression: pd.DataFrame,
        mutation: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        common = expression.index.intersection(mutation.index)
        if len(common) == 0:
            raise ValueError("No overlapping samples between STAR-count expression and MC3 labels.")
        return expression.loc[common].fillna(0.0), mutation.loc[common].fillna(0).astype(np.int8)

    def load_mc3_events(self, cancer_types: Optional[list[str]] = None) -> pd.DataFrame:
        frames = []
        for mc3_path in self._mc3_event_files(cancer_types):
            frames.append(self._read_mc3_event_file(mc3_path))
        mut = pd.concat(frames, axis=0, ignore_index=True)
        print(f"MC3 event records: {len(mut):,}", flush=True)
        return mut.dropna(subset=["sample", "gene", "variant"])

    def _read_mc3_event_file(self, mc3_path: Path) -> pd.DataFrame:
        mut = pd.read_csv(mc3_path, sep="\t", low_memory=False)
        sample_col = "sample" if "sample" in mut.columns else "Tumor_Sample_Barcode"
        gene_col = "gene" if "gene" in mut.columns else "Hugo_Symbol"
        variant_col = "effect" if "effect" in mut.columns else "Variant_Classification"
        missing = [col for col in [sample_col, gene_col, variant_col] if col not in mut.columns]
        if missing:
            raise ValueError(f"Missing expected MC3 event columns {missing} in {mc3_path}")
        mut = mut[[sample_col, gene_col, variant_col]].rename(
            columns={sample_col: "sample", gene_col: "gene", variant_col: "variant"}
        )
        mut["sample"] = mut["sample"].map(self.tcga_sample_id)
        mut["gene"] = mut["gene"].astype(str)
        return mut

    def normalize_cancer_types(self, cancer_types: Optional[list[str]]) -> Optional[list[str]]:
        if cancer_types:
            clean = [str(cancer).upper() for cancer in cancer_types]
            if clean == ["ALL"]:
                return None
            composites = sorted(set(clean).intersection(COMPOSITE_XENA_COHORTS))
            if composites:
                raise ValueError(
                    "Composite Xena cohorts are not valid for this prediction workflow: "
                    f"{composites}. Use the individual TCGA disease cohorts instead."
                )
            return clean
        configured = self.data_cfg.get("cancer_types") or []
        return [str(cancer).upper() for cancer in configured] if configured else None

    def requested_default_cohorts(self) -> list[str]:
        configured = self.data_cfg.get("download_cancer_types") or DEFAULT_TCGA_COHORTS
        return [str(cancer).upper() for cancer in configured]

    def gene_id_to_symbol(self) -> pd.Series:
        if self._gene_symbol_map is not None:
            return self._gene_symbol_map
        mapping_path = self._ensure_input(
            self.data_cfg.get("gene_name_mapping_file", "downloads/gencode.v36.annotation.gtf.gene.probemap"),
            self.data_cfg.get("gene_name_mapping_url"),
        )
        mapping = pd.read_csv(mapping_path, sep="\t")
        if not {"id", "gene"}.issubset(mapping.columns):
            raise ValueError(f"Expected columns 'id' and 'gene' in probemap: {mapping_path}")
        by_full = pd.Series(mapping["gene"].astype(str).values, index=mapping["id"].astype(str).values)
        by_base = pd.Series(
            mapping["gene"].astype(str).values,
            index=mapping["id"].astype(str).str.split(".").str[0].values,
        )
        self._gene_symbol_map = pd.concat([by_full, by_base[~by_base.index.isin(by_full.index)]])
        return self._gene_symbol_map

    def protein_coding_gene_ids(self) -> set[str]:
        if self._protein_gene_ids is not None:
            return self._protein_gene_ids
        genes = self._gene_rows()
        gene_id = genes.str.extract('gene_id "([^"]*)"', expand=False).astype(str)
        gene_type = genes.str.extract('gene_type "([^"]*)"', expand=False)
        protein = gene_type.eq("protein_coding")
        self._protein_gene_ids = set(gene_id[protein].str.split(".").str[0])
        return self._protein_gene_ids

    def protein_coding_gene_symbols(self) -> set[str]:
        if self._protein_gene_symbols is not None:
            return self._protein_gene_symbols
        genes = self._gene_rows()
        gene_name = genes.str.extract('gene_name "([^"]*)"', expand=False)
        gene_type = genes.str.extract('gene_type "([^"]*)"', expand=False)
        protein = gene_type.eq("protein_coding")
        self._protein_gene_symbols = set(gene_name[protein].dropna().astype(str))
        return self._protein_gene_symbols

    def _gene_rows(self) -> pd.Series:
        gtf = self._load_gene_gtf()
        return gtf.loc[gtf["feature"].eq("gene"), "attribute"]

    def _load_gene_gtf(self) -> pd.DataFrame:
        if self._gene_gtf is not None:
            return self._gene_gtf
        gtf_path = self._ensure_input(
            self.data_cfg.get("gene_annotation_file", "downloads/gencode.v36.annotation.gtf.gz"),
            self.data_cfg.get("gene_annotation_url"),
        )
        self._gene_gtf = pd.read_csv(
            gtf_path,
            sep="\t",
            compression="gzip",
            names=["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"],
            comment="#",
        )
        return self._gene_gtf

    def _filter_min_mutations(self, labels: pd.DataFrame) -> pd.DataFrame:
        min_mutations = self.data_cfg.get("min_mutations_per_gene", 1)
        counts = labels.sum(axis=0)
        if isinstance(min_mutations, float) and 0 < min_mutations < 1:
            min_count = labels.shape[0] * min_mutations
        else:
            min_count = float(min_mutations)
        return labels.loc[:, counts >= min_count]

    def _expression_files(self, cancer_types: Optional[list[str]]) -> list[Path]:
        measure = self.data_cfg.get("expression_measure", "counts")
        expression_dir = self._data_path(self.data_cfg.get("expression_dir", "downloads"))
        cohorts = cancer_types or self.requested_default_cohorts()
        files = [expression_dir / f"TCGA-{cancer}.star_{measure}.tsv.gz" for cancer in cohorts]
        self._download_missing_files(
            files,
            self.data_cfg.get("expression_url_template"),
            cohorts=cohorts,
            measure=measure,
        )
        return files

    def _mutation_files(self, cancer_types: Optional[list[str]]) -> list[Path]:
        mutation_dir = self._data_path(self.data_cfg.get("mutation_dir", "tcga_xena_mutations"))
        template = self.data_cfg.get("mutation_file_template", "mc3_gene_level%2F{cancer}_mc3_gene_level.txt.gz")
        cohorts = cancer_types or self.requested_default_cohorts()
        files = [mutation_dir / template.format(cancer=cancer) for cancer in cohorts]
        self._download_missing_files(
            files,
            self._url_templates("mutation_url_template", "mutation_url_fallback_templates"),
            cohorts=cohorts,
        )
        return files

    def _mc3_event_files(self, cancer_types: Optional[list[str]]) -> list[Path]:
        event_dir = self._data_path(self.data_cfg.get("mc3_event_dir", "mc3"))
        template = self.data_cfg.get("mc3_event_file_template", "{cancer}_mc3.txt.gz")
        cohorts = cancer_types or self.requested_default_cohorts()
        files = [event_dir / template.format(cancer=cancer) for cancer in cohorts]
        self._download_missing_files(
            files,
            self._url_templates("mc3_event_url_template", "mc3_event_url_fallback_templates"),
            cohorts=cohorts,
        )
        return files

    def _download_missing_files(
        self,
        files: list[Path],
        url_template: str | list[str] | tuple[str, ...] | None,
        *,
        cohorts: Optional[list[str]] = None,
        measure: str | None = None,
    ) -> None:
        missing = [path for path in files if not path.exists()]
        if not missing:
            return
        if not self.data_cfg.get("download_missing", True) or not url_template:
            raise FileNotFoundError(f"Missing required files: {[str(path) for path in missing]}")
        print(f"Downloading {len(missing):,} missing input file(s)...", flush=True)
        cohort_by_path = {path: str(cohort).upper() for path, cohort in zip(files, cohorts or [])}
        for path in missing:
            cancer = cohort_by_path.get(path, self._cancer_from_filename(path))
            urls = [
                template.format(cancer=cancer, measure=measure)
                for template in self._as_url_template_list(url_template)
            ]
            self._ensure_input(path, urls)

    def _ensure_input(self, filename: str | Path, url: str | list[str] | tuple[str, ...] | None = None) -> Path:
        path = Path(filename)
        if not path.is_absolute():
            path = self.data_dir / path
        if path.exists():
            return path
        if path.suffix == ".gz":
            uncompressed = path.with_suffix("")
            if uncompressed.exists():
                print(f"Compressing {uncompressed}\n -> {path}")
                with open(uncompressed, "rb") as src, gzip.open(path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                return path
        urls = self._as_url_template_list(url)
        if not self.data_cfg.get("download_missing", True) or not urls:
            raise FileNotFoundError(f"Required input not found: {path}")

        path.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for i, candidate_url in enumerate(urls, start=1):
            try:
                return self._download_input(path, candidate_url, attempt=i, n_attempts=len(urls))
            except requests.HTTPError as exc:
                last_error = exc
                status = exc.response.status_code if exc.response is not None else "unknown"
                print(f"Download failed with HTTP {status}: {candidate_url}", flush=True)
                if path.with_name(path.name + ".part").exists():
                    path.with_name(path.name + ".part").unlink(missing_ok=True)
                raw_tmp = path.with_suffix(path.with_suffix("").suffix + ".part")
                if raw_tmp.exists():
                    raw_tmp.unlink(missing_ok=True)
        if last_error is not None:
            raise last_error
        raise FileNotFoundError(f"Required input not found: {path}")

    def _download_input(self, path: Path, url: str, *, attempt: int, n_attempts: int) -> Path:
        prefix = f"Downloading [{attempt}/{n_attempts}]" if n_attempts > 1 else "Downloading"
        print(f"{prefix} {url}\n -> {path}", flush=True)
        if path.suffix == ".gz" and not str(url).endswith(".gz"):
            raw_tmp = path.with_suffix(path.with_suffix("").suffix + ".part")
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with open(raw_tmp, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        if chunk:
                            handle.write(chunk)
            with open(raw_tmp, "rb") as src, gzip.open(path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            raw_tmp.unlink(missing_ok=True)
            return path

        tmp = path.with_name(path.name + ".part")
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with open(tmp, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
        tmp.replace(path)
        return path

    def _url_templates(self, primary_key: str, fallback_key: str) -> list[str]:
        return self._as_url_template_list(self.data_cfg.get(primary_key)) + self._as_url_template_list(
            self.data_cfg.get(fallback_key)
        )

    @staticmethod
    def _as_url_template_list(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value if item]
        return [str(value)]

    def _transform_label(self) -> str:
        if not self.data_cfg.get("expression_log2_count_plus_one", True):
            return "xena"
        return "log1p" if self.data_cfg.get("expression_log1p", False) else "raw"

    def _cache_paths(self, cancer_types: Optional[list[str]]) -> tuple[Path, Path]:
        cohort = "all" if not cancer_types else "-".join(sorted(cancer_types))
        protein = "pc" if self.data_cfg.get("only_protein_coding_expression", True) else "allgenes"
        dedup = "dedup" if self.data_cfg.get("collapse_duplicate_gene_symbols", True) else "rawsymbols"
        key = f"star_counts_{self._transform_label()}_{protein}_{dedup}_{cohort}"
        return (
            self.cache_dir / f"expression_aligned_{key}.pkl",
            self.cache_dir / f"mutation_aligned_{key}.pkl",
        )

    def _expression_cache_path(self, cancer_types: Optional[list[str]]) -> Path:
        cohort = "all" if not cancer_types else "-".join(sorted(cancer_types))
        protein = "pc" if self.data_cfg.get("only_protein_coding_expression", True) else "allgenes"
        dedup = "dedup" if self.data_cfg.get("collapse_duplicate_gene_symbols", True) else "rawsymbols"
        measure = self.data_cfg.get("expression_measure", "counts")
        return self.cache_dir / f"expression_{measure}_{self._transform_label()}_{protein}_{dedup}_{cohort}.pkl"

    def _data_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.data_dir / path).resolve()

    @staticmethod
    def _cancer_from_filename(path: Path) -> str:
        name = path.name
        if name.startswith("TCGA-"):
            return name.split(".")[0].replace("TCGA-", "").upper()
        stem = name.split("_mc3_gene_level")[0]
        if "%2F" in stem:
            stem = stem.split("%2F")[-1]
        if "/" in stem:
            stem = stem.split("/")[-1]
        return stem.upper()

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.project_root / path).resolve()

    def _resolve_project_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (self.project_root / path).resolve()

    @staticmethod
    def tcga_sample_id(barcode: str) -> str:
        parts = str(barcode).split("-")
        if len(parts) >= 4 and parts[0] == "TCGA":
            sample_type = parts[3][:2]
            return "-".join([parts[0], parts[1], parts[2], sample_type])
        return str(barcode)

    @staticmethod
    def legacy_tcga_sample_id(barcode: str) -> str:
        parts = str(barcode).split("-")
        if len(parts) >= 4 and parts[0] == "TCGA":
            sample_type = parts[3][:3] if len(parts[3]) >= 3 else f"{parts[3][:2]}A"
            return "-".join([parts[0], parts[1], parts[2], sample_type])
        return str(barcode)
