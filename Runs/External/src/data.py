"""TCGA training data and external cohort loaders for external validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd



CODING_VARIANT_CLASSIFICATIONS = {
    "Missense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Nonsense_Mutation",
    "Nonstop_Mutation",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Translation_Start_Site",
    "Splice_Site",
}


HUGO_CODING_VARIANT_CLASSIFICATIONS = CODING_VARIANT_CLASSIFICATIONS | {
    "Start_Codon_SNP",
    "Start_Codon_Del",
    "Stop_Codon_Del",
    "De_novo_Start_InFrame",
    "De_novo_Start_OutOfFrame",
}


SEQUENCE_ONTOLOGY_CODING_CONSEQUENCES = {
    "missense_variant",
    "frameshift_variant",
    "stop_gained",
    "stop_lost",
    "start_lost",
    "initiator_codon_variant",
    "inframe_insertion",
    "inframe_deletion",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "protein_altering_variant",
}


HUGO_GSM_TO_PATIENT = {
    "GSM2069823": "Pt1",
    "GSM2069824": "Pt2",
    "GSM2069825": "Pt4",
    "GSM2069826": "Pt5",
    "GSM2069827": "Pt6",
    "GSM2069828": "Pt7",
    "GSM2069829": "Pt8",
    "GSM2069830": "Pt9",
    "GSM2069831": "Pt10",
    "GSM2069832": "Pt12",
    "GSM2069833": "Pt13",
    "GSM2069834": "Pt14",
    "GSM2069835": "Pt15",
    "GSM2069836": "Pt16",
    "GSM2069837": "Pt19",
    "GSM2069838": "Pt20",
    "GSM2069839": "Pt22",
    "GSM2069840": "Pt23",
    "GSM2069841": "Pt25",
    "GSM2069842": "Pt27A",
    "GSM2069843": "Pt27B",
    "GSM2069844": "Pt28",
    "GSM2069845": "Pt29",
    "GSM2069846": "Pt31",
    "GSM2069847": "Pt32",
    "GSM2069848": "Pt35",
    "GSM2069849": "Pt37",
    "GSM2069850": "Pt38",
}


GDC_CODING_EFFECTS = {
    "missense_variant",
    "frameshift_variant",
    "stop_gained",
    "stop_lost",
    "start_lost",
    "initiator_codon_variant",
    "inframe_deletion",
    "inframe_insertion",
    "splice_acceptor_variant",
    "splice_donor_variant",
    "splice_region_variant",
}


def collapse_duplicate_columns(df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    if df.columns.is_unique:
        return df
    if method == "mean":
        return df.T.groupby(level=0).mean().T
    if method == "sum":
        return df.T.groupby(level=0).sum().T
    raise ValueError("method must be 'mean' or 'sum'.")


def load_tcga_expression(
    *,
    expression_path: str | Path,
    gene_name_mapping_path: str | Path,
    gene_annotation_path: str | Path | None = None,
    cancer_types: list[str] | None = None,
    measure: str = "counts",
    log_transformed: bool = True,
    only_protein_coding: bool = True,
) -> pd.DataFrame:
    """Load per-cancer TCGA Xena STAR expression (counts, TPM or FPKM) on the linear scale."""
    expression_path = Path(expression_path)
    gene_name_mapping_path = Path(gene_name_mapping_path)
    if cancer_types:
        files = [expression_path / f"TCGA-{str(cancer).upper()}.star_{measure}.tsv.gz" for cancer in cancer_types]
    else:
        files = sorted(expression_path.glob(f"TCGA-*.star_{measure}.tsv.gz"))
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing TCGA expression files: {missing}")
    if not files:
        raise FileNotFoundError(f"No TCGA expression files found under {expression_path} for measure={measure!r}.")

    gene_name_mapping_df = pd.read_csv(gene_name_mapping_path, sep="\t", index_col=0)
    dfs = []
    for file_path in files:
        df = pd.read_csv(file_path, sep="\t", index_col=0).T
        if log_transformed:
            df = 2**df - 1
        dfs.append(df)
    expression = pd.concat(dfs, axis=0)

    mapped_symbols = expression.columns.map(gene_name_mapping_df["gene"])
    expression.columns = expression.columns.astype(str) + "|" + mapped_symbols.astype(str)

    if only_protein_coding and gene_annotation_path is not None:
        gene_annotation_df = pd.read_csv(
            gene_annotation_path,
            sep="\t",
            compression="gzip",
            names=["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"],
            comment="#",
        )
        gene_annotation_df["gene_id"] = gene_annotation_df["attribute"].str.extract('gene_id "([^"]*)"')
        gene_annotation_df["gene_type"] = gene_annotation_df["attribute"].str.extract('gene_type "([^"]*)"')
        annotation_map = dict(zip(gene_annotation_df["gene_id"], gene_annotation_df["gene_type"]))
        gene_ids = expression.columns.str.split("|").str[0]
        gene_types = gene_ids.map(annotation_map)
        expression = expression.loc[:, gene_types == "protein_coding"]

    expression.columns = expression.columns.str.split("|").str[1]
    expression.index = expression.index.astype(str)
    expression.index.name = "sample"
    expression = expression.apply(pd.to_numeric, errors="coerce")
    return expression


def load_tcga_mutations(
    *,
    mutation_path: str | Path,
    cancer_types: list[str] | None = None,
    file_template: str = "{cancer}_mc3_gene_level.txt.gz",
    gene_annotation_path: str | Path | None = None,
    only_protein_coding_mutations: bool = True,
    min_mutations_per_gene: int | float = 1,
    add_a_suffix_to_samples: bool = True,
) -> pd.DataFrame:
    """Load per-cancer MC3 gene-level mutation matrices as binary labels."""
    mutation_path = Path(mutation_path)
    if cancer_types:
        files = [mutation_path / file_template.format(cancer=str(cancer).upper()) for cancer in cancer_types]
    else:
        files = sorted(mutation_path.glob("**/*gene_level.txt.gz"))
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing TCGA mutation files: {missing}")
    if not files:
        raise FileNotFoundError(f"No TCGA mutation gene-level files found under {mutation_path}.")

    matrices = []
    for file_path in files:
        mutation_df = pd.read_csv(file_path, sep="\t", compression="gzip", index_col=0).T
        matrices.append(mutation_df)
    mutations = pd.concat(matrices, axis=0)
    if add_a_suffix_to_samples:
        mutations.index = mutations.index.astype(str) + "A"
    else:
        mutations.index = mutations.index.astype(str)
    mutations.index.name = "sample"
    mutations = mutations.apply(pd.to_numeric, errors="coerce").fillna(0)

    if only_protein_coding_mutations and gene_annotation_path is not None:
        gene_annotation_df = pd.read_csv(
            gene_annotation_path,
            sep="\t",
            compression="gzip",
            names=["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"],
            comment="#",
        )
        gene_annotation_df["gene_type"] = gene_annotation_df["attribute"].str.extract('gene_type "([^"]*)"')
        gene_annotation_df["gene_name"] = gene_annotation_df["attribute"].str.extract('gene_name "([^"]*)"')
        annotation_map = dict(zip(gene_annotation_df["gene_name"], gene_annotation_df["gene_type"]))
        gene_types = mutations.columns.map(annotation_map)
        mutations = mutations.loc[:, gene_types == "protein_coding"]

    mutation_counts = mutations.sum(axis=0)
    if isinstance(min_mutations_per_gene, float) and 0 < min_mutations_per_gene < 1:
        min_sample_count = mutations.shape[0] * min_mutations_per_gene
    else:
        min_sample_count = min_mutations_per_gene
    mutations = mutations.loc[:, mutation_counts >= min_sample_count]
    return (mutations > 0).astype("int8")


def load_tcga_training_data(
    *,
    expression_path: str | Path,
    gene_name_mapping_path: str | Path,
    gene_annotation_path: str | Path,
    mutation_path: str | Path,
    mutation_file_template: str = "{cancer}_mc3_gene_level.txt.gz",
    cancer_types: list[str] | None = None,
    expression_measure: str = "counts",
    expression_log_transformed: bool = True,
    only_protein_coding: bool = True,
    min_mutations_per_gene: int | float = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load TCGA expression and mutation labels for the samples present in both."""
    expression = load_tcga_expression(
        expression_path=expression_path,
        gene_name_mapping_path=gene_name_mapping_path,
        gene_annotation_path=gene_annotation_path,
        cancer_types=cancer_types,
        measure=expression_measure,
        log_transformed=expression_log_transformed,
        only_protein_coding=only_protein_coding,
    )
    mutations = load_tcga_mutations(
        mutation_path=mutation_path,
        cancer_types=cancer_types,
        file_template=mutation_file_template,
        gene_annotation_path=gene_annotation_path,
        only_protein_coding_mutations=only_protein_coding,
        min_mutations_per_gene=min_mutations_per_gene,
        add_a_suffix_to_samples=True,
    )
    common = expression.index.intersection(mutations.index)
    expression = expression.loc[common].copy()
    mutations = mutations.loc[common].copy()

    metadata = {
        "source": "xena_star_mc3",
        "expression_path": str(expression_path),
        "gene_name_mapping_path": str(gene_name_mapping_path),
        "gene_annotation_path": str(gene_annotation_path),
        "mutation_path": str(mutation_path),
        "cancer_types": cancer_types or ["all"],
        "expression_measure": expression_measure,
        "expression_log_transformed": bool(expression_log_transformed),
        "only_protein_coding": bool(only_protein_coding),
        "min_mutations_per_gene": min_mutations_per_gene,
        "n_samples": int(expression.shape[0]),
        "n_expression_features": int(expression.shape[1]),
        "n_mutation_targets": int(mutations.shape[1]),
    }
    return expression, mutations, metadata


def sample_patient_id(sample_id: str) -> str:
    """Return the patient ID from an expression sample ID (e.g. Riaz `Pt1_Pre_AD101148-6`)."""
    return str(sample_id).split("_")[0].split(".")[0]


def expand_patient_table_to_expression_samples(
    patient_table: pd.DataFrame,
    expression_index: pd.Index,
) -> pd.DataFrame:
    """Expand a patient-indexed table to one row per expression sample."""
    patients = pd.Series([sample_patient_id(sample) for sample in expression_index], index=expression_index)
    keep = patients.isin(patient_table.index.astype(str))
    expanded = patient_table.copy()
    expanded.index = expanded.index.astype(str)
    out = expanded.reindex(patients.loc[keep].to_numpy())
    out.index = pd.Index(patients.loc[keep].index.astype(str), name="sample")
    return out


def entrez_to_symbols(
    entrez_ids,
    *,
    cache_file: str | Path | None = None,
    species: str = "human",
) -> list[str | None]:
    """Map Entrez IDs to gene symbols, persisting a cache for reproducible external processing."""
    entrez_ids = [str(value) for value in entrez_ids]
    unique_ids = pd.Index(entrez_ids).dropna().astype(str).unique().tolist()
    mapping: dict[str, str] = {}
    cached_ids: set[str] = set()
    cache_path = Path(cache_file) if cache_file is not None else None
    if cache_path is not None and cache_path.exists():
        cached = pd.read_csv(cache_path, dtype=str)
        if {"entrez_id", "gene_symbol"}.issubset(cached.columns):
            cached = cached.dropna(subset=["entrez_id"])
            cached_ids = set(cached["entrez_id"].astype(str))
            cached_mapped = cached.dropna(subset=["gene_symbol"])
            cached_mapped = cached_mapped[cached_mapped["gene_symbol"].astype(str).str.len() > 0]
            mapping.update(cached_mapped.set_index("entrez_id")["gene_symbol"].to_dict())

    missing = [entrez_id for entrez_id in unique_ids if entrez_id not in cached_ids]
    if missing:
        try:
            import mygene
        except ImportError as exc:
            raise ImportError(
                "mygene is required to map Hugo Entrez IDs unless an entrez_gene_symbol_cache.csv file already exists."
            ) from exc

        mg = mygene.MyGeneInfo()
        rows = mg.querymany(
            missing,
            scopes="entrezgene",
            fields="symbol",
            species=species,
            as_dataframe=False,
        )
        for row in rows:
            entrez_id = str(row.get("query", ""))
            symbol = row.get("symbol")
            if symbol:
                mapping[entrez_id] = str(symbol)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_df = pd.DataFrame(
            [
                {"entrez_id": entrez_id, "gene_symbol": mapping.get(entrez_id, "")}
                for entrez_id in sorted(set(cached_ids).union(unique_ids))
            ]
        )
        cache_df = cache_df.drop_duplicates("entrez_id").sort_values("entrez_id")
        cache_df.to_csv(cache_path, index=False)

    return [mapping.get(entrez_id) for entrez_id in entrez_ids]


def ensembl_to_symbols(
    ensembl_ids,
    *,
    cache_file: str | Path | None = None,
    species: str = "human",
) -> list[str | None]:
    """Map Ensembl gene IDs to gene symbols with a persistent cache."""
    ensembl_ids = [str(value).split(".")[0] for value in ensembl_ids]
    unique_ids = pd.Index(ensembl_ids).dropna().astype(str).unique().tolist()
    mapping: dict[str, str] = {}
    cached_ids: set[str] = set()
    cache_path = Path(cache_file) if cache_file is not None else None
    if cache_path is not None and cache_path.exists():
        cached = pd.read_csv(cache_path, dtype=str)
        if {"ensembl_id", "gene_symbol"}.issubset(cached.columns):
            cached = cached.dropna(subset=["ensembl_id"])
            cached_ids = set(cached["ensembl_id"].astype(str))
            cached_mapped = cached.dropna(subset=["gene_symbol"])
            cached_mapped = cached_mapped[cached_mapped["gene_symbol"].astype(str).str.len() > 0]
            mapping.update(cached_mapped.set_index("ensembl_id")["gene_symbol"].to_dict())

    missing = [ensembl_id for ensembl_id in unique_ids if ensembl_id not in cached_ids]
    if missing:
        try:
            import mygene
        except ImportError as exc:
            raise ImportError(
                "mygene is required to map Van Allen Ensembl IDs unless an ensembl_gene_symbol_cache.csv file already exists."
            ) from exc
        mg = mygene.MyGeneInfo()
        rows = mg.querymany(
            missing,
            scopes="ensembl.gene",
            fields="symbol",
            species=species,
            as_dataframe=False,
        )
        for row in rows:
            ensembl_id = str(row.get("query", ""))
            symbol = row.get("symbol")
            if symbol:
                mapping[ensembl_id] = str(symbol)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_df = pd.DataFrame(
            [
                {"ensembl_id": ensembl_id, "gene_symbol": mapping.get(ensembl_id, "")}
                for ensembl_id in sorted(set(cached_ids).union(unique_ids))
            ]
        )
        cache_df = cache_df.drop_duplicates("ensembl_id").sort_values("ensembl_id")
        cache_df.to_csv(cache_path, index=False)

    return [mapping.get(ensembl_id) for ensembl_id in ensembl_ids]


def load_cptac_cmi_data(
    data_dir: str | Path,
    *,
    expression_file: str = "expression_counts.tsv.gz",
    coding_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load CPTAC/CMI expression, mutation, and clinical data without prevalence filtering."""
    data_dir = Path(data_dir)
    clinical_df = pd.read_csv(data_dir / "clinical.tsv", sep="\t", low_memory=False)
    clinical_df["cases.case_id"] = clinical_df["cases.case_id"].astype(str)
    clinical_df["cases.submitter_id"] = clinical_df["cases.submitter_id"].astype(str)

    vital = clinical_df["demographic.vital_status"].astype(str).str.strip().str.lower()
    clinical_df["OS_time"] = np.where(
        vital.eq("dead"),
        pd.to_numeric(clinical_df.get("demographic.days_to_death", np.nan), errors="coerce"),
        pd.to_numeric(clinical_df.get("diagnoses.days_to_last_follow_up", np.nan), errors="coerce"),
    )
    clinical_df["OS_event"] = vital.eq("dead").astype(int)
    clinical_df["OS_time"] = pd.to_numeric(clinical_df["OS_time"], errors="coerce")
    clinical_df.loc[clinical_df["OS_time"] < 0, "OS_time"] = np.nan
    clinical = clinical_df.drop_duplicates(subset=["cases.case_id"]).set_index("cases.case_id")
    clinical.index.name = "case_uuid"

    expression_raw = pd.read_csv(
        data_dir / expression_file,
        sep="\t",
        compression="gzip",
        index_col=0,
    )
    expression_raw.columns = expression_raw.columns.astype(str)

    manifest_files = sorted(data_dir.glob("gdc_sample_sheet*.tsv"))
    if not manifest_files:
        raise FileNotFoundError(f"No gdc_sample_sheet*.tsv found in {data_dir}")
    manifest_df = pd.read_csv(manifest_files[-1], sep="\t", low_memory=False)
    manifest_df["expr_uuid"] = manifest_df["File Name"].astype(str).str.split(".").str[0]
    manifest_df["Case ID"] = manifest_df["Case ID"].astype(str)
    manifest_expr = (
        manifest_df[manifest_df["expr_uuid"].isin(expression_raw.columns)]
        .copy()
        .rename(columns={"Case ID": "case_submitter_id"})
    )

    clinical_case_map = (
        clinical_df[["cases.case_id", "cases.submitter_id"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"cases.case_id": "case_uuid", "cases.submitter_id": "case_submitter_id"})
    )
    expr_uuid_to_case = (
        manifest_expr[["expr_uuid", "case_submitter_id"]]
        .drop_duplicates()
        .merge(clinical_case_map, on="case_submitter_id", how="inner")
    )
    if expr_uuid_to_case.empty:
        raise ValueError("Expression-to-case mapping is empty for CPTAC/CMI.")

    uuid_to_case = expr_uuid_to_case.set_index("expr_uuid")["case_uuid"]
    expr_cols = expression_raw.columns[expression_raw.columns.isin(uuid_to_case.index)]
    expr = expression_raw.loc[:, expr_cols]
    expr_case = expr.T.groupby(uuid_to_case.reindex(expr.columns)).mean(numeric_only=True).T
    expression = expr_case.T.select_dtypes(include=[np.number])
    expression.index.name = "case_uuid"
    expression = collapse_duplicate_columns(expression, method="mean")

    maf = pd.read_csv(
        data_dir / "mutations_merged.maf.gz",
        sep="\t",
        compression="gzip",
        low_memory=False,
    )
    maf["case_id"] = maf["case_id"].astype(str)
    maf["Hugo_Symbol"] = maf["Hugo_Symbol"].astype(str)
    if coding_only:
        maf = maf[maf["Variant_Classification"].fillna("").isin(CODING_VARIANT_CLASSIFICATIONS)]
    maf = maf.dropna(subset=["case_id", "Hugo_Symbol"])
    mutation_binary = (
        maf.groupby(["case_id", "Hugo_Symbol"]).size().unstack(fill_value=0) > 0
    ).astype("int8")
    mutation_binary.index.name = "case_uuid"
    mutation_binary.index = mutation_binary.index.astype(str)

    common = pd.Index(sorted(set(expression.index) & set(mutation_binary.index) & set(clinical.index)))
    if common.empty:
        raise ValueError("No common CPTAC/CMI samples across expression, mutations, and clinical.")

    metadata = {
        "data_dir": str(data_dir),
        "expression_file": expression_file,
        "coding_only": bool(coding_only),
        "n_expression_samples_before_intersection": int(expression.shape[0]),
        "n_mutation_samples_before_intersection": int(mutation_binary.shape[0]),
        "n_clinical_samples_before_intersection": int(clinical.shape[0]),
        "n_common_samples": int(len(common)),
        "n_expression_features": int(expression.shape[1]),
        "n_observed_mutated_genes": int(mutation_binary.shape[1]),
    }
    return expression.loc[common], mutation_binary.loc[common], clinical.loc[common], metadata


def load_immunopog_data(
    data_dir: str | Path,
    *,
    expression_file: str = "ImmunoPOG_expression_raw_counts_all.txt.gz",
    mutation_file: str = "ImmunoPOG_coding_mutations.txt.gz",
    clinical_file: str = "10780432ccr201163-sup-240190_4_supp_6604687_qh8c2m (1).xlsx",
    sequencing_file: str = "10780432ccr201163-sup-240190_4_supp_6604688_qh8c2k (1).xlsx",
    gips_file: str = "MSK-GI_JP_PUCH_clinical_info_with_GIPS.xlsx",
    coding_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load Pan-ImmunoPOG raw-count expression, coding mutations, and clinical metadata."""
    data_dir = Path(data_dir)

    expression_raw = pd.read_csv(data_dir / expression_file, sep="\t", low_memory=False)
    sample_cols = [c for c in expression_raw.columns if c not in {"Ensembl_ID", "HGNC_name"}]
    gene_names = expression_raw["HGNC_name"].astype("string").fillna("")
    keep = gene_names.ne("") & ~gene_names.str.startswith("ENSG", na=False)
    expression = expression_raw.loc[keep, ["HGNC_name", *sample_cols]].copy()
    expression = expression.groupby("HGNC_name", sort=False).mean(numeric_only=True).T
    expression.index = expression.index.astype(str)
    expression.index.name = "sample"
    expression.columns = expression.columns.astype(str)
    expression = expression.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    expression = collapse_duplicate_columns(expression, method="mean")

    mutations_long = pd.read_csv(data_dir / mutation_file, sep="\t", low_memory=False)
    mutations_long["sample"] = mutations_long["patient_id"].astype(str)
    mutations_long["gene"] = mutations_long["gene_id"].astype(str)
    if coding_only and "consequence_type" in mutations_long.columns:
        consequence = mutations_long["consequence_type"].astype(str)
        coding_mask = consequence.apply(
            lambda value: any(
                term.strip() in SEQUENCE_ONTOLOGY_CODING_CONSEQUENCES
                for term in str(value).replace("&", ",").replace(";", ",").replace("+", ",").split(",")
            )
        )
        mutations_long = mutations_long.loc[coding_mask].copy()
    mutations_long = mutations_long.dropna(subset=["sample", "gene"])
    mutations_long = mutations_long[mutations_long["gene"].ne("") & mutations_long["gene"].ne("nan")]

    mutation_binary = (
        mutations_long.assign(value=1)
        .pivot_table(index="sample", columns="gene", values="value", aggfunc="max", fill_value=0)
        .astype("int8")
    )
    mutation_binary.index = mutation_binary.index.astype(str)
    mutation_binary.index.name = "sample"

    clinical_raw = pd.read_excel(data_dir / clinical_file)
    clinical_raw["sample"] = clinical_raw["Anonymous ID"].astype(str)
    clinical = clinical_raw.drop_duplicates("sample").set_index("sample", drop=False)
    clinical.index.name = "sample"

    clinical["Cancer"] = clinical["Tumour type"].astype("string")
    clinical["cancer_type"] = clinical["Tumour type"].astype("string")
    clinical["Cohort"] = clinical["Cohort"].astype("string")
    clinical["Treatment"] = clinical.get("Immunotherapy regimen", pd.Series(pd.NA, index=clinical.index)).astype("string")
    clinical["Line_of_therapy"] = clinical.get("Line of therapy", pd.Series(pd.NA, index=clinical.index)).astype("string")
    clinical["Sex"] = clinical.get("Sex", pd.Series(pd.NA, index=clinical.index)).astype("string")
    clinical["Age"] = pd.to_numeric(clinical.get("Age at advanced disease diagnosis"), errors="coerce")
    clinical["OS_time"] = pd.to_numeric(clinical["Overall survival (days)"], errors="coerce")
    clinical["OS_event"] = pd.to_numeric(clinical["Alive_0"], errors="coerce").astype("Int64")
    clinical["PFS_time"] = pd.to_numeric(clinical["Time to progression (days)"], errors="coerce")
    clinical["PFS_event"] = pd.to_numeric(clinical["Progression_1"], errors="coerce").astype("Int64")

    benefit = clinical["Clinical benefit"].astype("string")
    clinical["Full_response"] = clinical["Best response"].astype("string")
    clinical["Response"] = benefit.replace({"DCB": "R", "NCB": "NR"})
    clinical["Clinical_benefit"] = benefit
    clinical["TMB_exome"] = pd.to_numeric(clinical.get("Exome mut per mb"), errors="coerce")
    clinical["TMB_genome"] = pd.to_numeric(clinical.get("Genome mut per mb"), errors="coerce")
    clinical["CD8_score"] = pd.to_numeric(clinical.get("CD8+ T cell score"), errors="coerce")
    clinical["CD274_expression"] = pd.to_numeric(clinical.get("CD274 expression"), errors="coerce")
    clinical["M1M2_expression"] = pd.to_numeric(clinical.get("M1M2 expression"), errors="coerce")
    clinical["Lymph_related"] = clinical.get("Lymph related", pd.Series(pd.NA, index=clinical.index)).astype("string")
    clinical["Study"] = "Pan_ImmunoPOG"
    clinical["Tumor_type"] = "Metastatic"
    clinical["Sequencing"] = "RNA-seq+WES"

    try:
        sequencing_raw = pd.read_excel(data_dir / sequencing_file)
        sequencing_raw["sample"] = sequencing_raw["PATIENT_ID"].astype(str)
        seq_summary = sequencing_raw.groupby("sample").agg(
            dna_libraries=("SAMPLE_ID", lambda x: ";".join(pd.Series(x).dropna().astype(str))),
            dna_library_types=("LIBRARY_TYPE", lambda x: ";".join(pd.Series(x).dropna().astype(str))),
            n_dna_libraries=("SAMPLE_ID", "nunique"),
            treatment_statuses=("TREATMENT_STATUS", lambda x: ";".join(pd.Series(x).dropna().astype(str).unique())),
        )
        clinical = clinical.join(seq_summary, how="left")
    except Exception:
        sequencing_raw = pd.DataFrame()

    gips_overlap_n = 0
    try:
        gips_tables: list[pd.DataFrame] = []
        for sheet_name, table in pd.read_excel(data_dir / gips_file, sheet_name=None).items():
            table = table.copy()
            table["gips_source_sheet"] = sheet_name
            if "Sample_ID" not in table.columns:
                continue
            table["sample"] = table["Sample_ID"].astype(str)
            gips_tables.append(table)
        if gips_tables:
            gips_all = pd.concat(gips_tables, ignore_index=True, sort=False)
            gips_overlap = (
                gips_all[gips_all["sample"].isin(clinical.index)]
                .drop_duplicates("sample")
                .set_index("sample")
            )
            gips_cols = [
                c for c in ["GIPS", "GIPS_Score", "TMB_Score", "MSI", "DCB", "gips_source_sheet"]
                if c in gips_overlap.columns
            ]
            if gips_cols:
                clinical = clinical.join(gips_overlap[gips_cols].add_prefix("msk_"), how="left")
            gips_overlap_n = int(len(gips_overlap))
    except Exception:
        pass

    common = pd.Index(sorted(set(expression.index) & set(mutation_binary.index) & set(clinical.index)))
    if common.empty:
        raise ValueError("No common IMMUNOPOG samples across expression, mutations, and clinical.")
    expression = expression.loc[common]
    clinical = clinical.loc[common]
    mutation_binary = mutation_binary.reindex(common).fillna(0).astype("int8")

    clinical["RNA"] = clinical.index.isin(expression.index)
    clinical["coding_mutation_event_count"] = (
        mutations_long.groupby("sample").size().reindex(common).fillna(0).astype(int)
    )
    clinical["mutated_gene_count"] = mutation_binary.sum(axis=1).astype(int)

    metadata = {
        "data_dir": str(data_dir),
        "expression_file": expression_file,
        "mutation_file": mutation_file,
        "clinical_file": clinical_file,
        "sequencing_file": sequencing_file,
        "gips_file": gips_file,
        "coding_only": bool(coding_only),
        "expression_measure": "raw_counts",
        "n_expression_samples_before_intersection": int(len(sample_cols)),
        "n_mutation_samples_before_intersection": int(mutation_binary.shape[0]),
        "n_clinical_samples_before_intersection": int(clinical_raw.shape[0]),
        "n_common_samples": int(len(common)),
        "n_expression_features": int(expression.shape[1]),
        "n_observed_mutated_genes": int(mutation_binary.shape[1]),
        "n_gips_rows_joined": gips_overlap_n,
        "n_cancer_types": int(clinical["Cancer"].nunique(dropna=True)),
    }
    return expression, mutation_binary, clinical, metadata


def load_metabric_data(
    data_dir: str | Path,
    *,
    expression_file: str = "data_mrna_illumina_microarray.txt",
    sample_clinical_file: str = "data_clinical_sample.txt",
    patient_clinical_file: str = "data_clinical_patient.txt",
    mutations_file: str = "data_mutations.txt",
    coding_only: bool = True,
    min_prevalence: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load METABRIC expression, mutation labels, and merged clinical tables without target filtering."""
    data_dir = Path(data_dir)
    clinical_sample = pd.read_csv(data_dir / sample_clinical_file, sep="\t", comment="#", low_memory=False)
    clinical_patient = pd.read_csv(data_dir / patient_clinical_file, sep="\t", comment="#", low_memory=False)
    clinical_sample["SAMPLE_ID"] = clinical_sample["SAMPLE_ID"].astype(str)
    clinical_patient["PATIENT_ID"] = clinical_patient["PATIENT_ID"].astype(str)
    clinical = clinical_sample.merge(clinical_patient, on="PATIENT_ID", how="left")
    clinical["OS_event"] = (
        clinical.get("OS_STATUS", pd.Series(index=clinical.index, dtype="object"))
        .astype(str)
        .str.split(":")
        .str[0]
        .pipe(pd.to_numeric, errors="coerce")
    )
    clinical["OS_time"] = pd.to_numeric(clinical.get("OS_MONTHS", np.nan), errors="coerce") * 30.4375
    clinical = clinical.drop_duplicates(subset=["SAMPLE_ID"]).set_index("SAMPLE_ID")
    clinical.index.name = "sample"

    expression_raw = pd.read_csv(data_dir / expression_file, sep="\t", comment="#", low_memory=False)
    if "Hugo_Symbol" not in expression_raw.columns:
        raise ValueError(f"{expression_file} is missing Hugo_Symbol.")
    expression = expression_raw.set_index("Hugo_Symbol").drop(columns=["Entrez_Gene_Id"], errors="ignore").T
    expression.index = expression.index.astype(str)
    expression.index.name = "sample"
    expression = expression.apply(pd.to_numeric, errors="coerce")
    expression = collapse_duplicate_columns(expression, method="mean")

    maf = pd.read_csv(data_dir / mutations_file, sep="\t", comment="#", low_memory=False)
    maf["Tumor_Sample_Barcode"] = maf["Tumor_Sample_Barcode"].astype(str)
    maf["Hugo_Symbol"] = maf["Hugo_Symbol"].astype(str)
    if coding_only and "Variant_Classification" in maf.columns:
        maf = maf[maf["Variant_Classification"].fillna("").isin(CODING_VARIANT_CLASSIFICATIONS)]
    maf = maf.dropna(subset=["Tumor_Sample_Barcode", "Hugo_Symbol"])
    mutation_binary = (
        maf.groupby(["Tumor_Sample_Barcode", "Hugo_Symbol"]).size().unstack(fill_value=0) > 0
    ).astype("int8")
    mutation_binary.index = mutation_binary.index.astype(str)
    mutation_binary.index.name = "sample"
    if min_prevalence is not None:
        mutation_binary = mutation_binary.loc[:, mutation_binary.mean(axis=0) > float(min_prevalence)]

    common = pd.Index(sorted(set(expression.index) & set(mutation_binary.index) & set(clinical.index)))
    if common.empty:
        raise ValueError("No common METABRIC samples across expression, mutations, and clinical.")

    metadata = {
        "data_dir": str(data_dir),
        "expression_file": expression_file,
        "coding_only": bool(coding_only),
        "min_prevalence": min_prevalence,
        "n_expression_samples_before_intersection": int(expression.shape[0]),
        "n_mutation_samples_before_intersection": int(mutation_binary.shape[0]),
        "n_clinical_samples_before_intersection": int(clinical.shape[0]),
        "n_common_samples": int(len(common)),
        "n_expression_features": int(expression.shape[1]),
        "n_observed_mutated_genes": int(mutation_binary.shape[1]),
    }
    return expression.loc[common], mutation_binary.loc[common], clinical.loc[common], metadata


def load_hugo_data(
    data_dir: str | Path,
    *,
    counts_file: str = "GSE78220_raw_counts_GRCh38.p13_NCBI.tsv.gz",
    workbook_file: str = "NIHMS765463-supplement-5.xlsx",
    mutation_sheet: str = "S1D",
    clinical_sheet1: str = "S1A",
    clinical_sheet2: str = "S1B",
    coding_only: bool = True,
    entrez_symbol_cache_file: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load Hugo melanoma raw-count expression, WES mutations, and clinical metadata."""
    data_dir = Path(data_dir)
    if entrez_symbol_cache_file is None:
        entrez_symbol_cache_file = data_dir / "entrez_gene_symbol_cache.csv"

    expression_raw = pd.read_csv(data_dir / counts_file, sep="\t", index_col=0)
    expression = expression_raw.T
    expression.index = expression.index.astype(str).map(HUGO_GSM_TO_PATIENT)
    expression = expression.loc[expression.index.notna()].copy()
    expression.index = expression.index.astype(str)
    expression.index.name = "sample"
    expression.columns = entrez_to_symbols(
        expression.columns,
        cache_file=entrez_symbol_cache_file,
    )
    expression = expression.loc[:, pd.Index(expression.columns).notna()].copy()
    expression = expression.apply(pd.to_numeric, errors="coerce")
    expression = collapse_duplicate_columns(expression, method="mean")

    mutation_df = pd.read_excel(
        data_dir / workbook_file,
        sheet_name=mutation_sheet,
        skiprows=2,
    )
    maf = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": mutation_df["Sample"].astype(str),
            "Hugo_Symbol": mutation_df["Gene"].astype(str).str.strip(),
            "Variant_Classification": mutation_df["MutType"].astype(str),
        }
    )
    if coding_only:
        maf = maf[maf["Variant_Classification"].isin(HUGO_CODING_VARIANT_CLASSIFICATIONS)].copy()
    gene_as_dt = pd.to_datetime(maf["Hugo_Symbol"], errors="coerce")
    maf = maf.loc[gene_as_dt.isna()].dropna(subset=["Tumor_Sample_Barcode", "Hugo_Symbol"])
    mutation_binary = (
        maf[["Tumor_Sample_Barcode", "Hugo_Symbol"]]
        .drop_duplicates()
        .assign(value=1)
        .pivot(index="Tumor_Sample_Barcode", columns="Hugo_Symbol", values="value")
        .fillna(0)
        .astype("int8")
    )
    mutation_binary.index = mutation_binary.index.astype(str)
    mutation_binary.index.name = "sample"

    clinical_1 = pd.read_excel(data_dir / workbook_file, sheet_name=clinical_sheet1, skiprows=2)
    clinical_2 = pd.read_excel(data_dir / workbook_file, sheet_name=clinical_sheet2, skiprows=2)
    clinical_1 = clinical_1[clinical_1["Patient ID"].astype("string").str.startswith("Pt").fillna(False)]
    clinical_2 = clinical_2[clinical_2["Patient ID"].astype("string").str.startswith("Pt").fillna(False)]
    clinical_2 = clinical_2[
        [
            "Patient ID",
            "Response",
            "Purity",
            "Ploidy",
            "TotalNonSyn_Exp",
            "TotalIndelNonSyn",
            "TotalIndelNonSyn_Exp",
            "Total",
        ]
    ]
    clinical_raw = clinical_1.merge(clinical_2, on="Patient ID", how="inner")
    clinical = pd.DataFrame(index=clinical_raw["Patient ID"].astype(str))
    clinical.index.name = "sample"
    clinical["Cohort"] = "Hugo"
    clinical["OS_time"] = pd.to_numeric(clinical_raw["Overall Survival"], errors="coerce").to_numpy()
    clinical["OS_event"] = clinical_raw["Vital Status"].replace({"Dead": 1, "Alive": 0}).to_numpy()
    clinical["Response"] = clinical_raw["Response"].astype("string").to_numpy()
    clinical["Full_response"] = (
        clinical_raw["irRECIST"]
        .replace({"Partial Response": "PR", "Progressive Disease": "PD", "Complete Response": "CR"})
        .astype("string")
        .to_numpy()
    )
    clinical["Treatment"] = clinical_raw["Treatment"].astype("string").to_numpy()
    clinical["Age"] = pd.to_numeric(clinical_raw["Age"], errors="coerce").to_numpy()
    clinical["Gender"] = clinical_raw["Gender"].astype("string").to_numpy()
    clinical["Purity"] = pd.to_numeric(clinical_raw["Purity"], errors="coerce").to_numpy()
    clinical["Ploidy"] = pd.to_numeric(clinical_raw["Ploidy"], errors="coerce").to_numpy()
    clinical["Tumor_type"] = "Metastatic"
    clinical["Stage"] = clinical_raw["Disease Status"].astype("string").to_numpy()
    clinical["Study"] = "Melanoma_Hugo"
    clinical["Cancer"] = "Melanoma"
    clinical["Sequencing"] = "WES"
    clinical["RNA"] = clinical_raw["RNAseq"].replace({1: True, 0: False}).to_numpy()
    clinical["TMB"] = (
        clinical_raw["TotalNonSyn_Exp"].astype(str) + "/" + clinical_raw["Total"].astype(str)
    ).to_numpy()

    common = pd.Index(sorted(set(expression.index) & set(mutation_binary.index) & set(clinical.index)))
    if common.empty:
        raise ValueError("No common Hugo samples across expression, mutations, and clinical.")

    metadata = {
        "data_dir": str(data_dir),
        "counts_file": counts_file,
        "workbook_file": workbook_file,
        "coding_only": bool(coding_only),
        "entrez_symbol_cache_file": str(entrez_symbol_cache_file),
        "n_expression_samples_before_intersection": int(expression.shape[0]),
        "n_mutation_samples_before_intersection": int(mutation_binary.shape[0]),
        "n_clinical_samples_before_intersection": int(clinical.shape[0]),
        "n_common_samples": int(len(common)),
        "n_expression_features": int(expression.shape[1]),
        "n_observed_mutated_genes": int(mutation_binary.shape[1]),
    }
    return expression.loc[common], mutation_binary.loc[common], clinical.loc[common], metadata


def load_liu_data(
    data_dir: str | Path,
    *,
    counts_file: str = "rnaseq_rawcounts.txt",
    mutation_file: str = "all_muts_12_1_2020_ref_alt_counts_added.maf",
    clinical_file: str = "Supplemental_Table_1_wAge.tsv",
    coding_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load Liu melanoma raw counts, WES mutations, and clinical metadata."""
    data_dir = Path(data_dir)

    expression_raw = pd.read_csv(data_dir / counts_file, sep="\t", index_col=0)
    expression = expression_raw.T
    expression.index = expression.index.astype(str)
    expression.index.name = "sample"
    expression.columns = expression.columns.astype(str)
    expression = expression.apply(pd.to_numeric, errors="coerce")
    expression = collapse_duplicate_columns(expression, method="mean")

    mutation_raw = pd.read_csv(data_dir / mutation_file, sep="\t", low_memory=False)
    maf = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": mutation_raw["Patient"].astype(str),
            "Hugo_Symbol": mutation_raw["Hugo_Symbol"].astype(str).str.strip(),
            "Variant_Classification": mutation_raw["Variant_Classification"].astype(str),
        }
    )
    if coding_only:
        maf = maf.loc[maf["Variant_Classification"].isin(CODING_VARIANT_CLASSIFICATIONS)].copy()
    maf = maf.loc[~maf["Hugo_Symbol"].astype(str).str.match(r"^\d{4}-\d{1,2}-\d{1,2}$")]
    maf = maf.dropna(subset=["Tumor_Sample_Barcode", "Hugo_Symbol"])
    mutation_binary = (
        maf[["Tumor_Sample_Barcode", "Hugo_Symbol"]]
        .drop_duplicates()
        .assign(value=1)
        .pivot(index="Tumor_Sample_Barcode", columns="Hugo_Symbol", values="value")
        .fillna(0)
        .astype("int8")
    )
    mutation_binary.index = mutation_binary.index.astype(str)
    mutation_binary.index.name = "sample"

    clinical_raw = pd.read_csv(data_dir / clinical_file, sep="\t")
    clinical = pd.DataFrame(index=clinical_raw["Sample ID"].astype(str))
    clinical.index.name = "sample"
    clinical["Sample"] = clinical.index
    clinical["Cohort"] = "Liu"
    clinical["Treatment"] = clinical_raw["Tx"].astype("string").to_numpy()
    clinical["OS_event"] = pd.to_numeric(clinical_raw["dead"], errors="coerce").to_numpy()
    clinical["OS_time"] = pd.to_numeric(clinical_raw["OS"], errors="coerce").to_numpy()
    clinical["PFS_time"] = pd.to_numeric(clinical_raw["PFS"], errors="coerce").to_numpy()
    clinical["PFS_event"] = pd.to_numeric(clinical_raw["progressed"], errors="coerce").to_numpy()
    clinical["Age"] = pd.to_numeric(clinical_raw["Age"], errors="coerce").to_numpy()
    clinical["Gender"] = clinical_raw["gender (Male=1, Female=0)"].replace({1: "M", 0: "F"}).to_numpy()
    clinical["Full_response"] = clinical_raw["BR"].astype("string").to_numpy()
    clinical["Response"] = (
        clinical_raw["BR"]
        .replace({"CR": "R", "PR": "R", "MR": "R", "SD": "NR", "PD": "NR"})
        .astype("string")
        .to_numpy()
    )
    clinical["Stage"] = (
        clinical_raw["Mstage (IIIC=0, M1a=1, M1b=2, M1c=3)"]
        .replace({0: "IIIC", 1: "M1a", 2: "M1b", 3: "M1c"})
        .astype("string")
        .to_numpy()
    )
    clinical["TMB"] = pd.to_numeric(clinical_raw["nonsyn_muts"], errors="coerce").to_numpy()
    clinical["Purity"] = pd.to_numeric(clinical_raw["purity"], errors="coerce").to_numpy()
    clinical["Ploidy"] = pd.to_numeric(clinical_raw["ploidy"], errors="coerce").to_numpy()
    clinical["Biopsy_site"] = clinical_raw["biopsy site"].astype("string").to_numpy()
    clinical["Biopsy_context"] = clinical_raw["biopsyContext (1=Pre-Ipi; 2=On-Ipi; 3=Pre-PD1; 4=On-PD1)"].to_numpy()
    clinical["Study"] = "Melanoma_Liu"
    clinical["Cancer"] = "Melanoma"
    clinical["Tumor_type"] = "Metastatic"
    clinical["Sequencing"] = "WES"
    clinical["RNA"] = clinical.index.isin(expression.index)

    common = expression.index.intersection(clinical.index)
    expression = expression.loc[common]
    clinical = clinical.loc[common]

    metadata = {
        "data_dir": str(data_dir),
        "counts_file": counts_file,
        "mutation_file": mutation_file,
        "clinical_file": clinical_file,
        "coding_only": bool(coding_only),
        "expression_measure": "raw_counts",
        "n_expression_samples_before_clinical_intersection": int(expression_raw.shape[1]),
        "n_expression_samples": int(expression.shape[0]),
        "n_mutation_samples": int(mutation_binary.shape[0]),
        "n_clinical_samples_before_intersection": int(clinical_raw.shape[0]),
        "n_expression_features": int(expression.shape[1]),
        "n_observed_mutated_genes": int(mutation_binary.shape[1]),
    }
    return expression, mutation_binary, clinical, metadata


def load_riaz_data(
    data_dir: str | Path,
    *,
    expression_file: str = "CountData.BMS038.txt",
    mutation_file: str = "1-s2.0-S0092867417311224-mmc3.xlsx",
    mutation_sheet: str = "Table S3",
    clinical_file: str = "1-s2.0-S0092867417311224-mmc2.xlsx",
    clinical_sheet: str = "Table S2",
    sample_file: str = "1-s2.0-S0092867417311224-mmc4.xlsx",
    sample_sheet: str = "Table S4",
    coding_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load Riaz melanoma raw counts, patient-level WES mutations, and clinical metadata."""
    data_dir = Path(data_dir)

    expression_raw = pd.read_csv(data_dir / expression_file, sep="\t", index_col=0)
    expression = expression_raw.T
    expression.index = expression.index.astype(str)
    if "HUGO" in expression.index:
        expression.columns = expression.loc["HUGO"]
        expression = expression.drop(index="HUGO")
    expression = expression.loc[:, pd.Index(expression.columns).notna()].copy()
    expression.columns = expression.columns.astype(str)
    expression = expression.loc[:, expression.columns.str.len() > 0].copy()
    expression = expression.apply(pd.to_numeric, errors="coerce")
    expression = collapse_duplicate_columns(expression, method="mean")
    expression.index.name = "sample"

    mutation_raw = pd.read_excel(data_dir / mutation_file, sheet_name=mutation_sheet, skiprows=3)
    maf = pd.DataFrame(
        {
            "Patient": mutation_raw["Patient"].astype(str),
            "Hugo_Symbol": mutation_raw["Hugo Symbol"].astype(str).str.strip(),
            "Variant_Classification": mutation_raw["Variant Classification"].astype(str),
        }
    )
    if coding_only:
        is_coding = maf["Variant_Classification"].apply(
            lambda value: any(effect in str(value) for effect in GDC_CODING_EFFECTS)
        )
        maf = maf.loc[is_coding].copy()
    maf = maf.loc[~maf["Hugo_Symbol"].astype(str).str.match(r"^\d{4}-\d{1,2}-\d{1,2}$")]
    maf = maf.dropna(subset=["Patient", "Hugo_Symbol"])
    mutation_binary = (
        maf[["Patient", "Hugo_Symbol"]]
        .drop_duplicates()
        .assign(value=1)
        .pivot(index="Patient", columns="Hugo_Symbol", values="value")
        .fillna(0)
        .astype("int8")
    )
    mutation_binary.index = mutation_binary.index.astype(str)
    mutation_binary.index.name = "patient"

    clinical_raw = pd.read_excel(data_dir / clinical_file, sheet_name=clinical_sheet, skiprows=2)
    sample_raw = pd.read_excel(data_dir / sample_file, sheet_name=sample_sheet, skiprows=2)
    sample_raw["RNAseq"] = (
        sample_raw[["Pre-treatment RNA-Seq", "On-treatment RNA-Seq"]]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .sum(axis=1)
        > 0
    )
    clinical_raw = clinical_raw.merge(sample_raw[["Patient", "RNAseq"]], on="Patient", how="left")

    clinical_patient = pd.DataFrame(index=clinical_raw["Patient"].astype(str))
    clinical_patient.index.name = "patient"
    clinical_patient["Patient"] = clinical_patient.index
    clinical_patient["Cohort"] = "Riaz"
    clinical_patient["Trial_Cohort"] = clinical_raw["Cohort"].astype("string").to_numpy()
    clinical_patient["OS_time"] = pd.to_numeric(clinical_raw["Time to Death\n(weeks)"], errors="coerce").to_numpy() * 7.0
    clinical_patient["OS_event"] = clinical_raw["Dead/Alive\n(Dead = True)"].map({True: 1, False: 0}).astype("Int64").to_numpy()
    clinical_patient["Full_response"] = clinical_raw["Response"].astype("string").to_numpy()
    clinical_patient["Response"] = (
        clinical_raw["Response"]
        .replace({"CR": "R", "PR": "R", "SD": "NR", "PD": "NR"})
        .astype("string")
        .to_numpy()
    )
    clinical_patient["Stage"] = clinical_raw["M Stage"].astype("string").to_numpy()
    clinical_patient["Subtype"] = clinical_raw["Subtype"].astype("string").to_numpy()
    clinical_patient["Mutational_Subtype"] = clinical_raw["Mutational\n Subtype"].astype("string").to_numpy()
    clinical_patient["Treatment"] = "Nivolumab"
    clinical_patient["Mutation_Load"] = pd.to_numeric(clinical_raw["Mutation Load"], errors="coerce").to_numpy()
    clinical_patient["Neo_antigen_Load"] = pd.to_numeric(clinical_raw["Neo-antigen Load"], errors="coerce").to_numpy()
    clinical_patient["Neo_peptide_Load"] = pd.to_numeric(clinical_raw["Neo-peptide Load"], errors="coerce").to_numpy()
    clinical_patient["Cytolytic_Score"] = pd.to_numeric(clinical_raw["Cytolytic Score"], errors="coerce").to_numpy()
    clinical_patient["TMB"] = clinical_patient["Mutation_Load"]
    clinical_patient["Study"] = "Melanoma_Riaz"
    clinical_patient["Cancer"] = "Melanoma"
    clinical_patient["Tumor_type"] = "Advanced/Metastatic"
    clinical_patient["Sequencing"] = "WES"
    clinical_patient["RNA"] = clinical_raw["RNAseq"].fillna(False).to_numpy()

    clinical = expand_patient_table_to_expression_samples(clinical_patient, expression.index)
    clinical.insert(0, "Expression_sample", clinical.index.astype(str))
    clinical["Expression_timepoint"] = clinical.index.astype(str).str.extract(r"_(Pre|On)", expand=False).astype("string")

    expression = expression.loc[clinical.index]
    metadata = {
        "data_dir": str(data_dir),
        "expression_file": expression_file,
        "mutation_file": mutation_file,
        "clinical_file": clinical_file,
        "sample_file": sample_file,
        "coding_only": bool(coding_only),
        "expression_measure": "raw_counts",
        "n_expression_samples_before_clinical_intersection": int(expression_raw.shape[1] - 1 if "HUGO" in expression_raw.columns else expression_raw.shape[1]),
        "n_expression_samples": int(expression.shape[0]),
        "n_expression_patients": int(pd.Series([sample_patient_id(x) for x in expression.index]).nunique()),
        "n_mutation_patients": int(mutation_binary.shape[0]),
        "n_clinical_patients": int(clinical_patient.shape[0]),
        "n_expression_features": int(expression.shape[1]),
        "n_observed_mutated_genes": int(mutation_binary.shape[1]),
    }
    return expression, mutation_binary, clinical, metadata


def load_van_allen_data(
    data_dir: str | Path,
    *,
    expression_file: str = "TPM_RSEM_VAScience2015.txt",
    clinical_file: str = "tables2.clinical_and_genome_characteristics_each_patient.xlsx",
    clinical_sheet: str = "exome analysis (n=110)",
    mutations_file: str = "tables1.mutation_list_all_patients.xlsx",
    mutations_sheet: str = "S1_MEL-All-Muts.csv",
    coding_only: bool = True,
    ensembl_symbol_cache_file: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load Van Allen melanoma TPM expression, mutations, and clinical metadata."""
    data_dir = Path(data_dir)
    if ensembl_symbol_cache_file is None:
        ensembl_symbol_cache_file = data_dir / "ensembl_gene_symbol_cache.csv"

    expression_raw = pd.read_csv(data_dir / expression_file, sep="\t", index_col=0)
    expression_raw.index = expression_raw.index.astype(str)
    expression_raw["hugo_symbol"] = ensembl_to_symbols(
        expression_raw.index,
        cache_file=ensembl_symbol_cache_file,
    )
    expression_raw = expression_raw.dropna(subset=["hugo_symbol"]).set_index("hugo_symbol")
    expression = expression_raw.T
    expression.index = expression.index.astype(str)
    expression.index.name = "sample"
    expression = expression.apply(pd.to_numeric, errors="coerce")
    expression = collapse_duplicate_columns(expression, method="mean")

    clinical_raw = pd.read_excel(data_dir / clinical_file, sheet_name=clinical_sheet)
    clinical = pd.DataFrame()
    clinical["sample"] = "MEL-IPI_" + clinical_raw["patient"].astype(str)
    clinical["Patient"] = clinical_raw["patient"].astype(str)
    clinical["Cohort"] = "Van_Allen"
    clinical["Gender"] = clinical_raw["gender"].replace({"male": "M", "female": "F"})
    clinical["Age"] = pd.to_numeric(clinical_raw["age_start"], errors="coerce")
    clinical["OS_event"] = pd.to_numeric(clinical_raw["dead"], errors="coerce")
    clinical["OS_time"] = pd.to_numeric(clinical_raw["overall_survival"], errors="coerce")
    clinical["PFS_time"] = pd.to_numeric(clinical_raw["progression_free"], errors="coerce")
    clinical["PFS_event"] = pd.to_numeric(clinical_raw["progression"], errors="coerce")
    clinical["Full_response"] = clinical_raw["RECIST"].astype("string")
    clinical["Response"] = clinical["Full_response"].replace({"CR": "R", "PR": "R", "SD": "NR", "PD": "NR"})
    mask_x = clinical["Response"].astype("string") == "X"
    if mask_x.any():
        clinical.loc[mask_x, "Response"] = clinical_raw.loc[mask_x, "group"].replace({"response": "R", "nonresponse": "NR"}).values
    clinical["TMB"] = pd.to_numeric(clinical_raw["nonsynonymous"], errors="coerce")
    clinical["Study"] = "Melanoma_Van_Allen"
    clinical["Stage"] = clinical_raw["stage"].astype("string") + ";" + clinical_raw["M"].astype("string")
    clinical["Treatment"] = "Ipilimumab"
    clinical["Sequencing"] = "WES"
    clinical["Cancer"] = "Melanoma"
    clinical["Tumor_type"] = "Metastatic"
    clinical = clinical.drop_duplicates("sample").set_index("sample")
    clinical.index.name = "sample"
    clinical["RNA"] = clinical.index.isin(expression.index)

    try:
        mutation_raw = pd.read_excel(data_dir / mutations_file, sheet_name=mutations_sheet)
    except Exception:
        mutation_raw = pd.read_excel(data_dir / mutations_file, sheet_name=0)
    maf = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": "MEL-IPI_" + mutation_raw["patient"].astype(str),
            "Hugo_Symbol": mutation_raw["Hugo_Symbol"].astype(str).str.strip(),
            "Variant_Classification": mutation_raw["Variant_Classification"].astype(str),
        }
    )
    if coding_only:
        maf = maf[maf["Variant_Classification"].isin(HUGO_CODING_VARIANT_CLASSIFICATIONS)].copy()
    maf = maf.dropna(subset=["Tumor_Sample_Barcode", "Hugo_Symbol"])
    mutation_binary = (
        maf[["Tumor_Sample_Barcode", "Hugo_Symbol"]]
        .drop_duplicates()
        .assign(value=1)
        .pivot(index="Tumor_Sample_Barcode", columns="Hugo_Symbol", values="value")
        .fillna(0)
        .astype("int8")
    )
    mutation_binary.index = mutation_binary.index.astype(str)
    mutation_binary.index.name = "sample"

    common = pd.Index(sorted(set(expression.index) & set(mutation_binary.index) & set(clinical.index)))
    if common.empty:
        raise ValueError("No common Van Allen samples across expression, mutations, and clinical.")

    metadata = {
        "data_dir": str(data_dir),
        "expression_file": expression_file,
        "clinical_file": clinical_file,
        "mutations_file": mutations_file,
        "coding_only": bool(coding_only),
        "ensembl_symbol_cache_file": str(ensembl_symbol_cache_file),
        "expression_measure": "RSEM_TPM",
        "n_expression_samples_before_intersection": int(expression.shape[0]),
        "n_mutation_samples_before_intersection": int(mutation_binary.shape[0]),
        "n_clinical_samples_before_intersection": int(clinical.shape[0]),
        "n_common_samples": int(len(common)),
        "n_expression_features": int(expression.shape[1]),
        "n_observed_mutated_genes": int(mutation_binary.shape[1]),
    }
    return expression.loc[common], mutation_binary.loc[common], clinical.loc[common], metadata


def load_morrison_data(
    data_dir: str | Path,
    *,
    expression_file: str = "RNA-CancerCell-MORRISON1-combat_batch_corrected-logcpm-all_samples.tsv",
    rna_meta_file: str = "RNA-CancerCell-MORRISON1-metadata.tsv",
    subjects_file: str = "Subjects-CancerCell-MORRISON1-metadata.tsv",
    wes_meta_file: str = "WES-CancerCell-MORRISON1-metadata.tsv",
    variants_file: str = "WES-CancerCell-MORRISON1-variants.tsv",
    coding_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load Morrison melanoma expression, WES mutation labels, and clinical metadata."""
    data_dir = Path(data_dir)
    expression_raw = pd.read_csv(data_dir / expression_file, sep="\t", index_col=0)
    expression_raw.columns = expression_raw.columns.astype(str)
    expression_raw.index = expression_raw.index.astype(str)

    rna_meta = pd.read_csv(data_dir / rna_meta_file, sep="\t")
    subject_meta = pd.read_csv(data_dir / subjects_file, sep="\t")
    wes_meta = pd.read_csv(data_dir / wes_meta_file, sep="\t")
    variants = pd.read_csv(data_dir / variants_file, sep="\t", low_memory=False)

    for frame in (rna_meta, subject_meta, wes_meta, variants):
        if "subject.id" in frame.columns:
            frame["subject.id"] = frame["subject.id"].astype(str)
    rna_meta["sample.id"] = rna_meta["sample.id"].astype(str)

    available_samples = pd.Index(rna_meta["sample.id"].astype(str)).intersection(expression_raw.columns)
    expression = expression_raw.loc[:, available_samples].T
    expression.index = expression.index.astype(str)
    expression.index.name = "sample"
    expression.columns = expression.columns.astype(str)
    expression = expression.apply(pd.to_numeric, errors="coerce")
    expression = collapse_duplicate_columns(expression, method="mean")

    wes_by_subject = wes_meta.sort_values("subject.id").groupby("subject.id", as_index=False).first()
    clinical_raw = rna_meta.merge(
        subject_meta[
            [
                "subject.id",
                "sample.tumor.type",
                "meddra.disease.preferred.name",
                "subject.age",
                "subject.sex",
                "treatment.regimen.name",
                "bor",
                "pfs",
                "os",
                "previous.treatment",
                "cohort",
            ]
        ],
        on="subject.id",
        how="left",
        suffixes=("", "_subject"),
    ).merge(
        wes_by_subject[["subject.id", "ploidy", "purity", "tmb"]],
        on="subject.id",
        how="left",
    )

    def _first_existing(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
        for col in candidates:
            if col in frame.columns:
                return frame[col]
        return pd.Series(pd.NA, index=frame.index)

    bor = _first_existing(clinical_raw, ["bor", "response", "bor_subject"]).astype("string")
    response = bor.replace({"CR": "R", "PR": "R", "MR": "R", "SD": "NR", "PD": "NR", "CRPR": "R", "PDSD": "NR"})
    os_months = pd.to_numeric(_first_existing(clinical_raw, ["os"]), errors="coerce")
    pfs_months = pd.to_numeric(_first_existing(clinical_raw, ["pfs"]), errors="coerce")

    clinical = pd.DataFrame(index=clinical_raw["sample.id"].astype(str))
    clinical.index.name = "sample"
    clinical["Sample"] = clinical.index
    clinical["Subject"] = clinical_raw["subject.id"].astype(str).to_numpy()
    clinical["Full_response"] = bor.to_numpy()
    clinical["Response"] = response.astype("string").to_numpy()
    clinical["Treatment"] = _first_existing(clinical_raw, ["treatment.regimen.name", "treatment.regimen.name_subject"]).astype("string").to_numpy()
    clinical["Age"] = pd.to_numeric(_first_existing(clinical_raw, ["subject.age", "subject.age_subject"]), errors="coerce").to_numpy()
    clinical["Gender"] = _first_existing(clinical_raw, ["subject.sex", "subject.sex_subject"]).replace({"male": "M", "female": "F"}).to_numpy()
    clinical["Cohort"] = _first_existing(clinical_raw, ["cohort", "cohort_subject"]).astype("string").to_numpy()
    clinical["Tumor_type"] = _first_existing(clinical_raw, ["sample.tumor.type", "sample.tumor.type_subject"]).astype("string").to_numpy()
    clinical["Cancer"] = _first_existing(clinical_raw, ["meddra.disease.preferred.name"]).astype("string").to_numpy()
    clinical["Study"] = "Melanoma_Morrison"
    clinical["Sequencing"] = "WES"
    clinical["Ploidy"] = pd.to_numeric(clinical_raw.get("ploidy"), errors="coerce").to_numpy()
    clinical["Purity"] = pd.to_numeric(clinical_raw.get("purity"), errors="coerce").to_numpy()
    clinical["TMB"] = pd.to_numeric(clinical_raw.get("tmb"), errors="coerce").to_numpy()
    clinical["OS_months"] = os_months.to_numpy()
    clinical["PFS_months"] = pfs_months.to_numpy()
    clinical["OS_time"] = (os_months * 30.4375).to_numpy()
    clinical["PFS_time"] = (pfs_months * 30.4375).to_numpy()
    clinical["RNA"] = clinical.index.isin(expression.index)

    maf = variants.dropna(subset=["subject.id", "gene.hgnc.symbol"]).copy()
    maf["gene.hgnc.symbol"] = maf["gene.hgnc.symbol"].astype(str).str.strip()
    if coding_only and "collapsed.consequences" in maf.columns:
        consequence = maf["collapsed.consequences"].astype(str)
        coding_mask = consequence.apply(
            lambda value: any(
                term.strip() in SEQUENCE_ONTOLOGY_CODING_CONSEQUENCES
                for term in str(value).replace("&", ",").replace(";", ",").split(",")
            )
        )
        maf = maf.loc[coding_mask].copy()
    subject_gene_matrix = (
        maf[["subject.id", "gene.hgnc.symbol"]]
        .drop_duplicates()
        .assign(value=1)
        .pivot(index="subject.id", columns="gene.hgnc.symbol", values="value")
        .fillna(0)
        .astype("int8")
    )
    sample_to_subject = rna_meta.set_index("sample.id")["subject.id"]
    mutation_binary = pd.DataFrame(0, index=expression.index, columns=subject_gene_matrix.columns, dtype="int8")
    for sample_id in mutation_binary.index:
        subject_id = sample_to_subject.get(sample_id)
        if pd.notna(subject_id) and subject_id in subject_gene_matrix.index:
            mutation_binary.loc[sample_id, :] = subject_gene_matrix.loc[subject_id].astype("int8").to_numpy()
    mutation_binary.index.name = "sample"

    common = expression.index.intersection(clinical.index)
    expression = expression.loc[common]
    mutation_binary = mutation_binary.loc[common]
    clinical = clinical.loc[common]

    metadata = {
        "data_dir": str(data_dir),
        "expression_file": expression_file,
        "rna_meta_file": rna_meta_file,
        "subjects_file": subjects_file,
        "wes_meta_file": wes_meta_file,
        "variants_file": variants_file,
        "coding_only": bool(coding_only),
        "expression_measure": "combat_batch_corrected_logcpm",
        "n_expression_samples_before_clinical_intersection": int(len(available_samples)),
        "n_expression_samples": int(expression.shape[0]),
        "n_expression_features": int(expression.shape[1]),
        "n_observed_mutated_genes": int(mutation_binary.shape[1]),
        "n_subjects_with_coding_mutations": int(subject_gene_matrix.shape[0]),
    }
    return expression, mutation_binary, clinical, metadata
