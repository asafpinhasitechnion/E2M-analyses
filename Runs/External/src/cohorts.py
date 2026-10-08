"""External cohort loaders. Each returns expression, mutation labels, clinical data, metadata, and the profiled samples.

Tumour RNA samples with clinical data are kept (CPTAC/CMI: one per patient). A sample is profiled if it (or its patient) was
sequenced: from the sequencing metadata where the cohort has it (METABRIC panel matrix, Morrison WES samples), otherwise
the samples in the mutation data, which equal the sequenced lists of these cohorts. Profiled samples are 1/0 per gene from
the coding calls, the others have missing labels. Duplicate gene symbols are summed (counts, TPM), except on a log scale
(METABRIC microarray, Morrison logCPM), where summing is meaningless and they are averaged.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from e2m.config import load_config


# The coding MAF classes used for the TCGA labels and TMB (E2M config); the other vocabularies map to them.
CODING_VARIANT_CLASSIFICATIONS = set(load_config()["tmb"]["coding_variant_classes"])


# Oncotator names: start-codon changes are Translation_Start_Site, stop-codon deletions are Nonstop.
HUGO_CODING_VARIANT_CLASSIFICATIONS = CODING_VARIANT_CLASSIFICATIONS | {
    "Start_Codon_SNP",
    "Start_Codon_Del",
    "Start_Codon_Ins",
    "Stop_Codon_Del",
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
    "disruptive_inframe_insertion",  # SnpEff names for in-frame indels (ImmunoPOG)
    "disruptive_inframe_deletion",
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


def is_coding_consequence(value) -> bool:
    """True if any term of a Sequence Ontology consequence string (joined by & , ; or +) is coding."""
    terms = str(value).replace("&", ",").replace(";", ",").replace("+", ",").split(",")
    return any(term.strip() in SEQUENCE_ONTOLOGY_CODING_CONSEQUENCES for term in terms)


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


def mutation_labels(mutations: pd.DataFrame, profiled: pd.Index, samples: pd.Index, targets: list[str]) -> pd.DataFrame:
    """Labels for each expression sample: 1/0 if its sample or patient ID was profiled, else NaN (not profiled)."""
    profiled = pd.Index(profiled).astype(str)
    binary = (mutations.reindex(columns=targets).fillna(0) > 0).astype(float)
    binary.index = binary.index.astype(str)
    binary = binary.reindex(profiled).fillna(0.0)
    keys = [s if s in profiled else sample_patient_id(s) for s in pd.Index(samples).astype(str)]
    labels = binary.reindex(keys)
    labels.index = pd.Index(samples, name="sample")
    return labels


def collapse_duplicate_columns(df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    if df.columns.is_unique:
        return df
    if method == "mean":
        return df.T.groupby(level=0).mean().T
    if method == "sum":
        return df.T.groupby(level=0).sum().T
    raise ValueError("method must be 'mean' or 'sum'.")


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


HEAD_AND_NECK = ("larynx", "tongue", "mouth", "lip", "gum", "oropharynx", "pharynx", "tonsil")
CPTAC_TYPE_BY_SITE = {"Uterus, NOS": "UCEC", "Breast": "BRCA", "Stomach": "STAD", "Pancreas": "PAAD", "Colon": "COAD", "Prostate gland": "PRAD"}


def cptac_tumour_type(site, diagnosis) -> str:
    """TCGA-style tumour type from GDC primary site + primary diagnosis (GDC has no such code for CPTAC/CMI).

    Lung and kidney are split by histology; all head-and-neck sites are HNSC; serous carcinomas of the ovary,
    genital tract and peritoneum are OV; breast and pancreas cases without a diagnosis go by site; AML is LAML.
    """
    site, dx = str(site), str(diagnosis).lower()
    if "myeloid leukemia" in dx:
        return "LAML"
    if site == "Bronchus and lung":
        return "LUSC" if "squamous" in dx else "LUAD" if "adenocarcinoma" in dx else "Other"
    if site == "Kidney":
        return {"renal cell carcinoma, nos": "KIRC", "papillary renal cell carcinoma": "KIRP", "renal cell carcinoma, chromophobe type": "KICH"}.get(dx, "Other")
    if site == "Brain":
        return "GBM" if ("glioblastoma" in dx or "gliosarcoma" in dx) else "LGG" if ("glioma" in dx or "astrocytoma" in dx) else "Other"
    if "squamous" in dx and any(k in site.lower() for k in HEAD_AND_NECK):
        return "HNSC"
    if "serous" in dx:
        return "OV"
    if "in situ" in dx:
        return "Other"
    return CPTAC_TYPE_BY_SITE.get(site, "Other")


def load_cptac_cmi_data(
    data_dir: str | Path,
    *,
    expression_file: str = "expression_counts.tsv.gz",
    coding_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.Index]:
    """CPTAC/CMI: one row per patient (case UUID), a tumour RNA sample with the mutations of the same tumour sample where possible.

    The GDC sample sheet links each file to its sample(s); sample.tsv says what each sample is. RNA files of the same
    sample(s) are repeat sequencing and are averaged. An RNA sample is used with the MAF(s) of the same tumour sample;
    without one, with all of the patient's MAFs, but only if it is the patient's only RNA sample; otherwise it is not profiled.
    A patient with several RNA samples keeps one with mutation data, then a Primary tumour, then the first sample ID.
    """
    data_dir = Path(data_dir)
    sheet = pd.read_csv(next(data_dir.glob("gdc_sample_sheet*.tsv")), sep="\t")
    samples = pd.read_csv(data_dir / "sample.tsv", sep="\t", index_col="samples.submitter_id",
                          usecols=["samples.submitter_id", "cases.case_id", "samples.tissue_type", "samples.sample_type", "samples.tumor_descriptor"])
    clinical_raw = pd.read_csv(data_dir / "clinical.tsv", sep="\t", low_memory=False).replace("'--", np.nan)
    counts = pd.read_csv(data_dir / expression_file, sep="\t", index_col=0)
    maf_rows = pd.read_csv(data_dir / "mutations_merged.maf.gz", sep="\t", low_memory=False,
                           usecols=["file_name", "Hugo_Symbol", "Variant_Classification"])

    def tumour_files(data_type):
        # A file can list several samples; a MAF lists its tumour and its matched normal. Keep each file's tumour samples.
        files = sheet[sheet["Data Type"].eq(data_type)]
        per_sample = files.assign(sample=files["Sample ID"].str.split(", ")).explode("sample").join(samples, on="sample")
        tumour = per_sample[per_sample["samples.tissue_type"].eq("Tumor")]
        return tumour.groupby("File Name").agg(case=("cases.case_id", "first"), samples=("sample", frozenset),
                                               sample_type=("samples.sample_type", "first"), descriptor=("samples.tumor_descriptor", "first"))

    # RNA samples: tumour RNA files, with files of the same sample(s) merged into one row.
    rna = tumour_files("Gene Expression Quantification")
    rna["uuid"] = rna.index.str.split(".").str[0]
    rna_rows = rna.groupby(["case", "samples"]).agg(uuids=("uuid", list), sample_type=("sample_type", "first"), descriptor=("descriptor", "first")).reset_index()
    rna_rows["sample_id"] = rna_rows["samples"].map(lambda s: "+".join(sorted(s)))

    # MAF files: tumour sample(s) and coding genes.
    mafs = tumour_files("Masked Somatic Mutation")
    coding = maf_rows[maf_rows["Variant_Classification"].isin(CODING_VARIANT_CLASSIFICATIONS)] if coding_only else maf_rows
    genes = coding.groupby("file_name")["Hugo_Symbol"].agg(frozenset)
    mafs["genes"] = [genes.get(f, frozenset()) for f in mafs.index]
    mafs["n_variants"] = maf_rows["file_name"].value_counts().reindex(mafs.index, fill_value=0)
    mafs = mafs[mafs["n_variants"].gt(0)]          # a MAF without any call does not count as mutation data

    # Each RNA row's MAFs: those of the same tumour sample; without one, all of the patient's MAFs if this is the patient's only RNA row.
    same_sample = [
        list(mafs.index[mafs["case"].eq(case) & mafs["samples"].map(lambda s: bool(s & row_samples))])
        for case, row_samples in zip(rna_rows["case"], rna_rows["samples"])
    ]
    only_row = rna_rows["case"].map(rna_rows["case"].value_counts()).eq(1)
    rna_rows["mafs"] = [same or (list(mafs.index[mafs["case"].eq(case)]) if only else [])
                        for same, case, only in zip(same_sample, rna_rows["case"], only_row)]
    rna_rows["match"] = ["sample" if same else ("case" if m else "none") for same, m in zip(same_sample, rna_rows["mafs"])]

    # One row per patient: a row with mutation data first, then a Primary tumour, then the first sample ID.
    patients = (rna_rows.assign(_no_dna=rna_rows["match"].eq("none"), _not_primary=rna_rows["descriptor"].ne("Primary"))
                .sort_values(["case", "_no_dna", "_not_primary", "sample_id"]).drop_duplicates("case").set_index("case"))
    patients.index.name = "case_uuid"
    patients["wes_variants"] = patients["mafs"].map(lambda f: int(mafs.loc[f, "n_variants"].sum()))

    # Mutation labels only for patients with mutation data (the others are not profiled).
    profiled = patients.index[patients["match"].ne("none")]
    patient_genes = patients.loc[profiled, "mafs"].map(lambda f: frozenset().union(*mafs.loc[f, "genes"]))
    mutation_binary = pd.DataFrame([dict.fromkeys(g, 1) for g in patient_genes], index=profiled).fillna(0).astype("int8")
    expression = pd.DataFrame({case: counts[uuids].mean(axis=1) for case, uuids in patients["uuids"].items()}).T
    expression.index.name = "case_uuid"

    # Clinical per case, combined over the case's rows (one per diagnosis / treatment / follow-up):
    # OS: Dead -> days to death, otherwise the longest follow-up. Recurrence (PFS_*): any "yes" -> earliest days to recurrence, otherwise the longest follow-up.
    c = clinical_raw.assign(
        OS_event=clinical_raw["demographic.vital_status"].str.lower().map({"dead": 1, "alive": 0}),
        PFS_event=clinical_raw["diagnoses.progression_or_recurrence"].str.lower().map({"yes": 1, "no": 0}),
        death=pd.to_numeric(clinical_raw["demographic.days_to_death"], errors="coerce"),
        follow_up=pd.to_numeric(clinical_raw["diagnoses.days_to_last_follow_up"], errors="coerce"),
        recurrence=pd.to_numeric(clinical_raw["diagnoses.days_to_recurrence"], errors="coerce"),
    )
    per_case = c.groupby("cases.case_id").agg(**{
        "cases.submitter_id": ("cases.submitter_id", "first"), "project.project_id": ("project.project_id", "first"),
        "cases.primary_site": ("cases.primary_site", "first"), "diagnosis": ("diagnoses.primary_diagnosis", "first"),
        "OS_event": ("OS_event", "max"), "PFS_event": ("PFS_event", "max"),
        "death": ("death", "max"), "follow_up": ("follow_up", "max"), "recurrence": ("recurrence", "min"),
    })
    per_case["OS_time"] = per_case["death"].where(per_case["OS_event"].eq(1), per_case["follow_up"])
    per_case["PFS_time"] = per_case["recurrence"].where(per_case["PFS_event"].eq(1), per_case["follow_up"])
    per_case["cancer"] = [cptac_tumour_type(site, dx) for site, dx in zip(per_case["cases.primary_site"], per_case["diagnosis"])]

    clinical = patients[["sample_id", "sample_type", "descriptor", "match", "wes_variants"]].join(
        per_case.drop(columns=["death", "follow_up", "recurrence"]))
    metadata = {
        "data_dir": str(data_dir),
        "expression_file": expression_file,
        "coding_only": bool(coding_only),
        "n_tumour_rna_files": int(len(rna)),
        "n_tumour_rna_samples": int(len(rna_rows)),
        "n_maf_files": int(len(mafs)),
        "n_patients": int(len(patients)),
        "match_counts": patients["match"].value_counts().to_dict(),
        "n_expression_features": int(expression.shape[1]),
        "n_observed_mutated_genes": int(mutation_binary.shape[1]),
    }
    return expression, mutation_binary, clinical, metadata, profiled


IMMUNOPOG_URL = "https://www.bcgsc.ca/downloads/immunoPOG/"


def load_immunopog_data(
    data_dir: str | Path,
    *,
    clinical_file: str = "10780432ccr201163-sup-240190_4_supp_6604687_qh8c2m.xlsx",
    coding_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.Index]:
    """Pan-ImmunoPOG (Pender et al. 2021): counts and coding mutations from BCGSC (downloaded), clinical from
    Supplementary Table 1 of the paper (placed manually). All 98 patients have RNA, WES and clinical data."""
    data_dir = Path(data_dir)
    counts = pd.read_csv(fetch(IMMUNOPOG_URL + "ImmunoPOG_expression_raw_counts_all.txt.gz", data_dir / "ImmunoPOG_expression_raw_counts_all.txt.gz"), sep="\t")
    mutations = pd.read_csv(fetch(IMMUNOPOG_URL + "ImmunoPOG_coding_mutations.txt.gz", data_dir / "ImmunoPOG_coding_mutations.txt.gz"), sep="\t")
    clinical_raw = pd.read_excel(data_dir / clinical_file)

    # Expression: genes by symbol (duplicate symbols summed).
    expression = collapse_duplicate_columns(counts.drop(columns="Ensembl_ID").set_index("HGNC_name").T.astype(float), method="sum")
    expression.index = expression.index.astype(str)
    expression.index.name = "sample"

    # Mutations: every patient in the file was sequenced; genes with a coding call (one patient has two tumour libraries, combined).
    mutations["patient_id"] = mutations["patient_id"].astype(str)
    profiled = pd.Index(mutations["patient_id"].unique())
    if coding_only:
        mutations = mutations[mutations["consequence_type"].map(is_coding_consequence)]
    mutation_binary = pd.crosstab(mutations["patient_id"], mutations["gene_id"]).gt(0).reindex(profiled, fill_value=False).astype("int8")
    mutation_binary.index.name = "sample"

    # Clinical.
    raw = clinical_raw.assign(sample=clinical_raw["Anonymous ID"].astype(str)).set_index("sample")
    clinical = pd.DataFrame({
        "Cancer": raw["Tumour type"],
        "Cohort": raw["Cohort"],
        "Treatment": raw["Immunotherapy regimen"],
        "Line_of_therapy": raw["Line of therapy"],
        "Sex": raw["Sex"],
        "Age": raw["Age at advanced disease diagnosis"],
        "OS_time": raw["Overall survival (days)"],
        "OS_event": raw["Alive_0"],
        "PFS_time": raw["Time to progression (days)"],
        "PFS_event": raw["Progression_1"],
        "Full_response": raw["Best response"],
        "Response": raw["Clinical benefit"].map({"DCB": "R", "NCB": "NR"}),
        "TMB_exome": raw["Exome mut per mb"],
        "TMB_genome": raw["Genome mut per mb"],
    })
    common = expression.index.intersection(clinical.index)

    metadata = {
        "data_dir": str(data_dir),
        "source": IMMUNOPOG_URL,
        "clinical_file": clinical_file,
        "coding_only": bool(coding_only),
        "n_samples": int(len(common)),
        "n_profiled_samples": int(len(common.intersection(profiled))),
        "n_expression_features": int(expression.shape[1]),
        "n_observed_mutated_genes": int(mutation_binary.shape[1]),
    }
    return expression.loc[common], mutation_binary, clinical.loc[common], metadata, profiled


METABRIC_URL = "https://media.githubusercontent.com/media/cBioPortal/datahub/a022eca324072ce6575586e5ea5fda305fc90a90/"


def load_metabric_data(data_dir: str | Path, *, coding_only: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.Index]:
    """METABRIC, cBioPortal study brca_metabric (cBioPortal datahub, commit a022eca).

    Expression: Illumina microarray (log-scaled). Mutations: targeted sequencing of 173 genes; the samples in the
    gene panel matrix were sequenced (1/0), the others have missing labels, and genes outside the panel are not measured
    (returned in metadata["measured_genes"]).
    """
    data_dir = Path(data_dir)
    read = lambda path, **kw: pd.read_csv(fetch(METABRIC_URL + path, data_dir / Path(path).name), sep="\t", comment="#", low_memory=False, **kw)
    study = "public/brca_metabric/"
    sample = read(study + "data_clinical_sample.txt")
    patient = read(study + "data_clinical_patient.txt")
    maf = read(study + "data_mutations.txt")
    panel_matrix = read(study + "data_gene_panel_matrix.txt")
    expression_raw = read(study + "data_mrna_illumina_microarray.txt")
    panel_file = fetch(METABRIC_URL + "reference_data/gene_panels/data_gene_panel_metabric_173.txt", data_dir / "data_gene_panel_metabric_173.txt")
    panel_genes = next(line for line in panel_file.read_text().splitlines() if line.startswith("gene_list:")).split()[1:]

    # Expression: genes by symbol (duplicate symbols averaged).
    expression = collapse_duplicate_columns(expression_raw.drop(columns="Entrez_Gene_Id").set_index("Hugo_Symbol").T.astype(float))
    expression.index.name = "sample"

    # Clinical: sample and patient tables; OS months -> days.
    clinical = sample.merge(patient, on="PATIENT_ID", how="left").set_index("SAMPLE_ID")
    clinical.index.name = "sample"
    clinical["OS_event"] = clinical["OS_STATUS"].str.split(":").str[0].astype(float)
    clinical["OS_time"] = clinical["OS_MONTHS"] * 30.4375
    common = expression.index.intersection(clinical.index)

    # Mutations: samples sequenced with the panel; genes with a coding call.
    profiled = pd.Index(panel_matrix.loc[panel_matrix["mutations"].eq("METABRIC_173"), "SAMPLE_ID"])
    if coding_only:
        maf = maf[maf["Variant_Classification"].isin(CODING_VARIANT_CLASSIFICATIONS)]
    mutation_binary = pd.crosstab(maf["Tumor_Sample_Barcode"], maf["Hugo_Symbol"]).gt(0).reindex(index=profiled, columns=panel_genes, fill_value=False).astype("int8")
    mutation_binary.index.name = "sample"

    metadata = {
        "data_dir": str(data_dir),
        "source": METABRIC_URL + study,
        "coding_only": bool(coding_only),
        "n_samples": int(len(common)),
        "n_profiled_samples": int(len(common.intersection(profiled))),
        "n_expression_features": int(expression.shape[1]),
        "measured_genes": panel_genes,
    }
    return expression.loc[common], mutation_binary, clinical.loc[common], metadata, profiled


HUGO_COUNTS_URL = "https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc=GSE78220&format=file&file="


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.Index]:
    """Hugo et al. 2016 (anti-PD-1, melanoma): GEO GSE78220 NCBI-generated counts (downloaded), Table S1 (clinical, mutations)."""
    data_dir = Path(data_dir)
    fetch(HUGO_COUNTS_URL + counts_file, data_dir / counts_file)
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
    expression = collapse_duplicate_columns(expression, method="sum")

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
    profiled = pd.Index(maf["Tumor_Sample_Barcode"].dropna().unique())
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
            "TotalNonSyn",
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
    clinical["TMB"] = pd.to_numeric(clinical_raw["TotalNonSyn"], errors="coerce").to_numpy()  # nonsynonymous mutations

    # Pt27 has two RNA samples (GEO): Pt27A, the 1st biopsy (R upper arm, the location given in S1A), and Pt27B,
    # the 2nd (R upper back). Both get Pt27's clinical data; only Pt27A gets its mutations.
    clinical = pd.concat([clinical, clinical.loc[["Pt27", "Pt27"]].set_axis(pd.Index(["Pt27A", "Pt27B"], name="sample"))])
    mutation_binary = pd.concat([mutation_binary, mutation_binary.loc[["Pt27"]].set_axis(pd.Index(["Pt27A"], name="sample"))])
    profiled = profiled.append(pd.Index(["Pt27A"]))

    common = pd.Index(sorted(set(expression.index) & set(clinical.index)))

    metadata = {
        "data_dir": str(data_dir),
        "counts_file": counts_file,
        "source": HUGO_COUNTS_URL + counts_file,
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
    return expression.loc[common], mutation_binary, clinical.loc[common], metadata, profiled


LIU_URL = "https://raw.githubusercontent.com/vanallenlab/schadendorf-pd1/536010aa7f39c59f17e4ff0827afdf38f2ccddf5/data/"


def load_liu_data(
    data_dir: str | Path,
    *,
    counts_file: str = "rnaseq_rawcounts.txt",
    mutation_file: str = "all_muts_12_1_2020_ref_alt_counts_added.maf",
    clinical_file: str = "Supplemental_Table_1_wAge.tsv",
    coding_only: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.Index]:
    """Liu et al. 2019 (PD-1, melanoma), from the authors' repository (github.com/vanallenlab/schadendorf-pd1)."""
    data_dir = Path(data_dir)

    # The counts and the MAF are in the repository's data/addData.zip; the clinical table is next to it.
    fetch(LIU_URL + clinical_file, data_dir / clinical_file)
    if not (data_dir / counts_file).exists() or not (data_dir / mutation_file).exists():
        archive = fetch(LIU_URL + "addData.zip", data_dir / "addData.zip")
        with zipfile.ZipFile(archive) as z:
            z.extract(counts_file, data_dir)
            z.extract(mutation_file, data_dir)
        archive.unlink()

    expression_raw = pd.read_csv(data_dir / counts_file, sep="\t", index_col=0)
    expression = expression_raw.T
    expression.index = expression.index.astype(str)
    expression.index.name = "sample"
    expression.columns = expression.columns.astype(str)
    expression = expression.apply(pd.to_numeric, errors="coerce")
    expression = collapse_duplicate_columns(expression, method="sum")

    mutation_raw = pd.read_csv(data_dir / mutation_file, sep="\t", low_memory=False)
    maf = pd.DataFrame(
        {
            "Tumor_Sample_Barcode": mutation_raw["Patient"].astype(str),
            "Hugo_Symbol": mutation_raw["Hugo_Symbol"].astype(str).str.strip(),
            "Variant_Classification": mutation_raw["Variant_Classification"].astype(str),
        }
    )
    profiled = pd.Index(maf["Tumor_Sample_Barcode"].dropna().unique())
    if coding_only:
        maf = maf.loc[maf["Variant_Classification"].isin(HUGO_CODING_VARIANT_CLASSIFICATIONS)].copy()  # Oncotator classes
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
        "source": LIU_URL,
        "coding_only": bool(coding_only),
        "expression_measure": "raw_counts",
        "n_expression_samples_before_clinical_intersection": int(expression_raw.shape[1]),
        "n_expression_samples": int(expression.shape[0]),
        "n_mutation_samples": int(mutation_binary.shape[0]),
        "n_clinical_samples_before_intersection": int(clinical_raw.shape[0]),
        "n_expression_features": int(expression.shape[1]),
        "n_observed_mutated_genes": int(mutation_binary.shape[1]),
    }
    return expression, mutation_binary, clinical, metadata, profiled


RIAZ_URL = "https://raw.githubusercontent.com/riazn/bms038_analysis/137111c38a32f8626ee6a436e14ac667bd31b1e7/data/"


def load_riaz_data(data_dir: str | Path, *, coding_only: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.Index]:
    """Riaz et al. 2017 (BMS038, nivolumab), from the authors' repository (github.com/riazn/bms038_analysis).

    RNA samples are those of the authors' sample table (it leaves out the ".2" libraries and Pt68_On). The mutations
    are from the pre-treatment exome, so only pre-treatment RNA samples of exome patients get labels; every RNA sample
    with clinical data keeps OS, PFS and response.
    """
    data_dir = Path(data_dir)
    counts = pd.read_csv(fetch(RIAZ_URL + "CountData.BMS038.txt", data_dir / "CountData.BMS038.txt"), sep="\t", index_col=0)
    sample_table = pd.read_csv(fetch(RIAZ_URL + "SampleTableCorrected.9.19.16.csv", data_dir / "SampleTableCorrected.9.19.16.csv"), index_col=0)
    maf = pd.read_csv(fetch(RIAZ_URL + "pre_therapy_nonsynonmous_mutations.csv", data_dir / "pre_therapy_nonsynonmous_mutations.csv"))
    patients = pd.read_csv(fetch(RIAZ_URL + "bms038_clinical_data.csv", data_dir / "bms038_clinical_data.csv")).set_index("PatientID")

    # Expression: the authors' RNA samples, genes by symbol (duplicate symbols summed).
    counts = counts.dropna(subset=["HUGO"]).set_index("HUGO")
    expression = collapse_duplicate_columns(counts[[s for s in sample_table.index if s in counts.columns]].T.astype(float), method="sum")
    expression.index.name = "sample"

    # Clinical per patient. OS_SOR / PFS_SOR are censoring flags (1 = censored), so the event is 1 - flag.
    clinical_patient = pd.DataFrame({
        "Cohort": patients["Cohort"],
        "Subtype": patients["SubtypeEZ"],
        "Full_response": patients["BOR"],
        "Response": patients["BOR"].map({"CR": "R", "PR": "R", "SD": "NR", "PD": "NR"}),
        "OS_time": patients["OS"], "OS_event": 1 - patients["OS_SOR"],
        "PFS_time": patients["PFS"], "PFS_event": 1 - patients["PFS_SOR"],
        "Treatment": "Nivolumab",
    })
    keep = [s for s in expression.index if sample_table.loc[s, "PatientID"] in clinical_patient.index]
    expression = expression.loc[keep]
    clinical = clinical_patient.loc[sample_table.loc[keep, "PatientID"]].set_axis(pd.Index(keep, name="sample"))
    clinical.insert(0, "Patient", sample_table.loc[keep, "PatientID"].to_numpy())
    clinical.insert(1, "Timepoint", sample_table.loc[keep, "PreOn"].to_numpy())

    # Mutations (pre-treatment exome): profiled patients are those in the mutation table; labels go to their pre-treatment RNA.
    exome_patients = set(maf["Patient"].astype(str))
    if coding_only:
        maf = maf[maf["Variant Classification"].map(is_coding_consequence)]
    by_patient = (maf.assign(value=1).pivot_table(index="Patient", columns="Hugo Symbol", values="value", aggfunc="max", fill_value=0) > 0).astype("int8")
    pre = [s for s in keep if clinical.loc[s, "Timepoint"] == "Pre" and clinical.loc[s, "Patient"] in exome_patients]
    mutation_binary = by_patient.reindex(clinical.loc[pre, "Patient"]).fillna(0).astype("int8").set_axis(pd.Index(pre, name="sample"))

    metadata = {
        "data_dir": str(data_dir),
        "source": RIAZ_URL,
        "coding_only": bool(coding_only),
        "n_rna_samples": int(len(keep)),
        "n_patients": int(clinical["Patient"].nunique()),
        "n_exome_patients": int(len(exome_patients)),
        "n_labelled_samples": int(len(pre)),
        "n_expression_features": int(expression.shape[1]),
        "n_observed_mutated_genes": int(mutation_binary.shape[1]),
    }
    return expression, mutation_binary, clinical, metadata, pd.Index(pre)


VAN_ALLEN_URL = "https://raw.githubusercontent.com/vanallenlab/VanAllen_CTLA4_Science_RNASeq_TPM/ca459ce4a3eaeffdcd47454064afc2cb319b74ce/"


def load_van_allen_data(
    data_dir: str | Path,
    *,
    expression_file: str = "TPM_RSEM_VAScience2015.txt",
    clinical_file: str = "tables2.clinical_and_genome_characteristics_each_patient.xlsx",
    clinical_sheet: str = "exome analysis (n=110)",
    transcriptome_sheet: str = "transcriptome analysis (n=42)",
    mutations_file: str = "tables1.mutation_list_all_patients.xlsx",
    mutations_sheet: str = "S1_MEL-All-Muts.csv",
    coding_only: bool = True,
    ensembl_symbol_cache_file: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.Index]:
    """Van Allen et al. 2015 (CTLA-4, melanoma): RSEM TPM from the authors' repository
    (github.com/vanallenlab/VanAllen_CTLA4_Science_RNASeq_TPM), mutations and clinical data from Tables S1 and S2."""
    data_dir = Path(data_dir)
    fetch(VAN_ALLEN_URL + expression_file, data_dir / expression_file)
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
    expression = collapse_duplicate_columns(expression, method="sum")

    # Clinical: the exome sheet, plus the RNA patients without exome from the transcriptome sheet (same columns, no TMB).
    exome = pd.read_excel(data_dir / clinical_file, sheet_name=clinical_sheet)
    transcriptome = pd.read_excel(data_dir / clinical_file, sheet_name=transcriptome_sheet)
    clinical_raw = pd.concat([exome, transcriptome[~transcriptome["patient"].isin(exome["patient"])]], ignore_index=True)
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
    profiled = pd.Index(maf["Tumor_Sample_Barcode"].dropna().unique())
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

    common = pd.Index(sorted(set(expression.index) & set(clinical.index)))
    if common.empty:
        raise ValueError("No common Van Allen samples across expression, mutations, and clinical.")

    metadata = {
        "data_dir": str(data_dir),
        "expression_file": expression_file,
        "source": VAN_ALLEN_URL,
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
    return expression.loc[common], mutation_binary, clinical.loc[common], metadata, profiled


MORRISON_URL = "https://raw.githubusercontent.com/ParkerICI/MORRISON-1-public/a4ac597bab359006f27a221318114b18e0a6135d/"


def load_morrison_data(data_dir: str | Path, *, coding_only: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.Index]:
    """Campbell et al. 2023 (MORRISON-1), from the authors' repository (github.com/ParkerICI/MORRISON-1-public).

    Six cohorts combined by the authors (CheckMate 038/064/067, Gide, Liu, Van Allen); expression is their ComBat
    batch-corrected logCPM. An RNA sample gets the mutations of the WES sample of the same subject and timepoint;
    without one, those of the subject's WES samples if it is the subject's only RNA sample. Others have missing labels.
    """
    data_dir = Path(data_dir)
    expression_file = "RNA-CancerCell-MORRISON1-combat_batch_corrected-logcpm-all_samples.tsv"
    if not (data_dir / expression_file).exists():
        archive = fetch(MORRISON_URL + "RNASeq/data/" + expression_file + ".zip", data_dir / (expression_file + ".zip"))
        with zipfile.ZipFile(archive) as z:
            z.extract(expression_file, data_dir)
        archive.unlink()
    read = lambda path: pd.read_csv(fetch(MORRISON_URL + path, data_dir / Path(path).name), sep="\t", low_memory=False)
    rna = read("RNASeq/RNA-CancerCell-MORRISON1-metadata.tsv").set_index("sample.id")
    subjects = read("Clinical/Subjects-CancerCell-MORRISON1-metadata.tsv").set_index("subject.id")
    wes = read("WES/WES-CancerCell-MORRISON1-metadata.tsv").set_index("sample.id")
    variants = read("WES/data/WES-CancerCell-MORRISON1-variants.tsv")

    # Expression: genes by symbol (duplicate symbols averaged).
    counts = pd.read_csv(data_dir / expression_file, sep="\t", index_col=0)
    expression = collapse_duplicate_columns(counts[rna.index.intersection(counts.columns)].T.astype(float))
    expression.index.name = "sample"
    rna = rna.loc[expression.index]

    # Mutations per WES sample (genes with a coding call).
    if coding_only:
        variants = variants[variants["collapsed.consequences"].map(is_coding_consequence)]
    by_wes = pd.crosstab(variants["sample.id"], variants["gene.hgnc.symbol"]).gt(0).reindex(wes.index, fill_value=False)

    # Each RNA sample's WES samples: same subject and timepoint; otherwise the subject's WES if this is its only RNA sample.
    rna_per_subject = rna["subject.id"].value_counts()
    def matched_wes(subject, timepoint):
        same = wes.index[wes["subject.id"].eq(subject) & wes["timepoint.id"].eq(timepoint)]
        if len(same) or rna_per_subject[subject] > 1:
            return list(same)
        return list(wes.index[wes["subject.id"].eq(subject)])
    rna["wes"] = [matched_wes(s, t) for s, t in zip(rna["subject.id"], rna["timepoint.id"])]
    profiled = rna.index[rna["wes"].map(bool)]
    mutation_binary = pd.DataFrame({s: by_wes.loc[rna.loc[s, "wes"]].any() for s in profiled}).T.astype("int8")
    mutation_binary.index.name = "sample"

    # Clinical per sample. OS/PFS have no event status. The repository says months, but the Liu and Van Allen
    # subjects are in days (as in their own studies); all are converted to days.
    times = subjects[["os", "pfs"]].reindex(rna["subject.id"]).set_axis(rna.index)
    to_days = np.where(rna["cohort"].isin(["liu", "va"]), 1.0, 30.4375)
    clinical = pd.DataFrame({
        "Subject": rna["subject.id"],
        "Cohort": rna["cohort"],
        "Timepoint": rna["timepoint.id"],
        "Treatment": rna["treatment.regimen.name"],
        "Previous_treatment": rna["previous.treatment"],
        "Tumor_type": rna["sample.tumor.type"],
        "Age": rna["subject.age"],
        "Gender": rna["subject.sex"].replace({"male": "M", "female": "F"}),
        "Full_response": rna["bor"],
        "Response": rna["bor"].map({"CR": "R", "PR": "R", "SD": "NR", "PD": "NR"}),
        "OS_time": times["os"] * to_days,
        "PFS_time": times["pfs"] * to_days,
        "TMB": [wes.loc[w, "tmb"].mean() if w else np.nan for w in rna["wes"]],
    })

    metadata = {
        "data_dir": str(data_dir),
        "source": MORRISON_URL,
        "coding_only": bool(coding_only),
        "n_rna_samples": int(len(rna)),
        "n_subjects": int(rna["subject.id"].nunique()),
        "n_wes_samples": int(len(wes)),
        "n_labelled_samples": int(len(profiled)),
        "n_expression_features": int(expression.shape[1]),
        "n_observed_mutated_genes": int(mutation_binary.shape[1]),
    }
    return expression, mutation_binary, clinical, metadata, profiled
