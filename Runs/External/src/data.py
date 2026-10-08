"""Old TCGA loader, used only by Runs/ClinicalDrivers until it moves to E2M (the external cohorts are in cohorts.py)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


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
