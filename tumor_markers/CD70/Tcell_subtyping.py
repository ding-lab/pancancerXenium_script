#!/usr/bin/env python3
"""Analyze CD27/CD70 across ccRCC cell types and recluster ccRCC T cells.

The nine user-specified rare-RCC datasets are excluded. The large RCC h5ad is
opened in backed mode: only two genes are loaded for the all-cell-type dotplot,
then only T cells are loaded for the full Harmony reclustering workflow.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import harmonypy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


RCC_DIR = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/RCC"
)
DEFAULT_INPUT = RCC_DIR / "annotated.Yuwei.slim.h5ad"
DEFAULT_OUTPUT = (
    RCC_DIR / "analysis/ccRCC_CD27_CD70_and_Tcell_subtyping"
)

RARE_RCC_DATASETS = {
    "HT934N1-S2H1Fp1U1",
    "HT870N1-S1H1Fp1U2",
    "C3N-03021-11U1",
    "HT870N1-S1H1Fp1U3",
    "HT588N1-H1Fp1U1",
    "HT902N1-S1H1Fp1U2",
    "C3N-01175-11U1",
    "C3L-01469-11U1",
    "HT1094N-S1H1Fp1U1",
}

CELLTYPE_ORDER = [
    "Tumor_ccRCC",
    "Tumor_pRCC",
    "Tumor_chRCC",
    "T_cell",
    "NK",
    "B_cell",
    "Plasma",
    "Macrophage",
    "Granulocyte",
    "cDC",
    "pDC",
    "Endothelial",
    "Lymphatic_Endothelial",
    "Fibroblast",
    "Pericyte/vSMC",
    "Necrosis",
]

T_CELL_MARKERS = {
    "T-cell identity": ["CD3D", "CD3E", "TRAC", "CD247"],
    "CD4/CD8": ["CD4", "CD8A", "CD8B"],
    "CD8/cytotoxic T": [
        "CD8A",
        "CD8B",
        "CCL5",
        "GZMK",
        "GZMA",
        "GZMB",
        "GZMH",
        "PRF1",
    ],
    "Naive T/TCM": ["TCF7", "CCR7", "SELL", "LEF1", "IL7R", "LTB", "MAL"],
    "Treg": ["FOXP3", "IL2RA", "CTLA4", "TIGIT", "TNFRSF18"],
    "CD4 helper/Th17-like": [
        "CD4",
        "KLRB1",
        "DPP4",
        "IL7R",
        "CCR6",
        "RORC",
    ],
    "Tfh-like": ["ICOS", "CXCL13", "PDCD1", "SH2D1A", "BCL6"],
    "Exhausted T": ["HAVCR2", "LAG3", "PDCD1", "TOX", "TIGIT"],
    "Trm": ["ITGAE", "ZNF683", "CXCR6", "CD69", "ITGA1"],
    "IFN-T": ["IFIT1", "IFIT3", "ISG15", "MX1"],
    "Proliferating T": ["MKI67", "TOP2A", "STMN1", "TUBA1B"],
    "NK contamination": ["NCAM1", "KLRD1", "NKG7", "GNLY"],
    "B-cell contamination": ["CD19", "CD79A", "MS4A1", "CD37"],
    "Myeloid contamination": [
        "LST1",
        "TYROBP",
        "FCER1G",
        "CD68",
        "CD163",
        "C1QA",
        "C1QB",
        "C1QC",
    ],
    "Endothelial contamination": [
        "PECAM1",
        "VWF",
        "KDR",
        "EMCN",
        "CLDN5",
        "ENG",
    ],
    "SMC/pericyte contamination": [
        "ACTA2",
        "TAGLN",
        "MYH11",
        "CNN1",
        "RGS5",
        "PDGFRB",
    ],
    "Epithelial contamination": ["EPCAM", "KRT8", "KRT18", "KRT19"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--n-hvg", type=int, default=2_000)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--top-deg", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=1)
    return parser.parse_args()


def dense_vector(matrix, column: int) -> np.ndarray:
    values = matrix[:, column]
    if sparse.issparse(values):
        return values.toarray().ravel()
    return np.asarray(values).ravel()


def cell_ids(adata: sc.AnnData) -> pd.Index:
    for column in ("barcode", "cell_id"):
        if column in adata.obs:
            return pd.Index(adata.obs[column].astype(str), name="cell_id")
    return pd.Index(adata.obs_names.astype(str), name="cell_id")


def numeric_order(values: pd.Series) -> list[str]:
    return sorted(
        set(values.astype(str)),
        key=lambda value: (
            (0, int(value))
            if value.lstrip("-").isdigit()
            else (1, value)
        ),
    )


def save_umap(
    adata: sc.AnnData,
    color: str,
    output: Path,
    title: str,
    legend_loc: str = "right margin",
) -> None:
    sc.pl.umap(
        adata,
        color=color,
        legend_loc=legend_loc,
        frameon=False,
        title=title,
        show=False,
    )
    plt.gcf().savefig(
        output,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close("all")
    print(f"Wrote: {output}", flush=True)


def run_harmony(
    pca: np.ndarray,
    metadata: pd.DataFrame,
    batch_key: str,
    random_state: int,
) -> np.ndarray:
    result = harmonypy.run_harmony(
        np.asarray(pca, dtype=np.float64),
        metadata,
        batch_key,
        max_iter_harmony=20,
        random_state=random_state,
    )
    corrected = result.Z_corr
    if hasattr(corrected, "detach"):
        corrected = corrected.detach().cpu().numpy()
    corrected = np.asarray(corrected, dtype=np.float32)
    if corrected.shape == pca.shape[::-1]:
        corrected = corrected.T
    if corrected.shape != pca.shape:
        raise RuntimeError(
            f"Unexpected Harmony shape {corrected.shape}; "
            f"expected {pca.shape}"
        )
    return corrected


def save_all_celltype_cd27_cd70(
    backed: sc.AnnData,
    ccrcc_mask: np.ndarray,
    output_dir: Path,
) -> None:
    genes = ["CD27", "CD70"]
    missing = [gene for gene in genes if gene not in backed.var_names]
    if missing:
        raise KeyError(f"Missing genes: {missing}")
    adata = backed[ccrcc_mask, genes].to_memory()
    try:
        observed = set(adata.obs["cell_type"].astype(str))
        order = [label for label in CELLTYPE_ORDER if label in observed]
        extras = sorted(observed - set(order))
        order.extend(extras)
        adata.obs["cell_type"] = pd.Categorical(
            adata.obs["cell_type"].astype(str),
            categories=order,
            ordered=True,
        )

        rows = []
        for cell_type in order:
            mask = adata.obs["cell_type"].astype(str).eq(cell_type).to_numpy()
            for gene in genes:
                index = int(adata.var_names.get_loc(gene))
                expression = dense_vector(adata.X, index).astype(float)
                values = expression[mask]
                rows.append(
                    {
                        "cell_type": cell_type,
                        "gene": gene,
                        "n_cells": int(mask.sum()),
                        "mean_log_normalized_expression": (
                            float(np.mean(values)) if len(values) else np.nan
                        ),
                        "fraction_expressing": (
                            float(np.mean(values > 0)) if len(values) else np.nan
                        ),
                    }
                )
        pd.DataFrame(rows).to_csv(
            output_dir / "ccRCC_CD27_CD70_expression_by_cell_type.tsv",
            sep="\t",
            index=False,
        )

        for standardized in (False, True):
            dotplot = sc.pl.dotplot(
                adata,
                var_names=genes,
                groupby="cell_type",
                categories_order=order,
                standard_scale="var" if standardized else None,
                use_raw=False,
                dendrogram=False,
                show=False,
                return_fig=True,
            )
            suffix = (
                "standardized_by_var" if standardized else "unscaled"
            )
            output = (
                output_dir
                / f"dotplot_ccRCC_all_celltypes_CD27_CD70_{suffix}.pdf"
            )
            dotplot.savefig(output)
            plt.close("all")
            print(f"Wrote: {output}", flush=True)
    finally:
        del adata
        gc.collect()


def present_markers(adata: sc.AnnData) -> dict[str, list[str]]:
    genes = set(adata.var_names.astype(str))
    result = {}
    for group, markers in T_CELL_MARKERS.items():
        present = [gene for gene in markers if gene in genes]
        missing = [gene for gene in markers if gene not in genes]
        if present:
            result[group] = present
        if missing:
            print(f"{group} missing: {', '.join(missing)}", flush=True)
    return result


def save_dotplot(
    adata: sc.AnnData,
    genes: dict[str, list[str]] | list[str],
    cluster_order: list[str],
    output: Path,
) -> None:
    dotplot = sc.pl.dotplot(
        adata,
        var_names=genes,
        groupby="leiden_ccRCC_T",
        categories_order=cluster_order,
        standard_scale="var",
        use_raw=False,
        dendrogram=False,
        show=False,
        return_fig=True,
    )
    dotplot.savefig(output)
    plt.close("all")
    print(f"Wrote: {output}", flush=True)


def recluster_tcells(
    backed: sc.AnnData,
    tcell_mask: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    print(f"Loading ccRCC T cells: {int(tcell_mask.sum()):,}", flush=True)
    adata = backed[tcell_mask].to_memory()
    adata.raw = None
    if "counts" not in adata.layers:
        raise KeyError("layers['counts'] is required for T-cell reclustering")
    adata.X = adata.layers["counts"].copy()
    adata.uns.pop("log1p", None)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    adata.var.drop(
        columns=[
            column
            for column in adata.var.columns
            if column.startswith("highly_variable")
            or column in {"means", "dispersions", "dispersions_norm"}
        ],
        inplace=True,
        errors="ignore",
    )
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=min(args.n_hvg, adata.n_vars),
        flavor="seurat",
    )
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    n_pcs = min(args.n_pcs, adata_hvg.n_obs - 1, adata_hvg.n_vars - 1)
    sc.tl.pca(
        adata_hvg,
        n_comps=n_pcs,
        svd_solver="arpack",
        random_state=args.random_state,
    )
    corrected = run_harmony(
        adata_hvg.obsm["X_pca"],
        adata_hvg.obs,
        "dataset",
        args.random_state,
    )
    adata.obsm["X_pca"] = np.asarray(
        adata_hvg.obsm["X_pca"],
        dtype=np.float32,
    )
    adata.obsm["X_pca_harmony"] = corrected
    for key in ("neighbors", "umap"):
        adata.uns.pop(key, None)
    for key in ("connectivities", "distances"):
        adata.obsp.pop(key, None)
    adata.obsm.pop("X_umap", None)
    sc.pp.neighbors(
        adata,
        use_rep="X_pca_harmony",
        n_neighbors=args.n_neighbors,
        random_state=args.random_state,
    )
    sc.tl.umap(adata, random_state=args.random_state)
    sc.tl.leiden(
        adata,
        resolution=args.resolution,
        key_added="leiden_ccRCC_T",
        random_state=args.random_state,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
    adata.obs["leiden_ccRCC_T"] = adata.obs[
        "leiden_ccRCC_T"
    ].astype("category")
    cluster_order = numeric_order(adata.obs["leiden_ccRCC_T"])

    save_umap(
        adata,
        "leiden_ccRCC_T",
        output_dir / "umap_ccRCC_Tcell_leiden_ondata.png",
        f"ccRCC T cells (Leiden {args.resolution:g})",
        legend_loc="on data",
    )
    save_umap(
        adata,
        "dataset",
        output_dir / "umap_ccRCC_Tcell_dataset.png",
        "ccRCC T cells by dataset",
    )

    markers = present_markers(adata)
    save_dotplot(
        adata,
        markers,
        cluster_order,
        output_dir
        / "dotplot_ccRCC_Tcell_subtype_and_contamination_markers_"
        "standardized_by_var.pdf",
    )

    print("Calculating T-cell cluster DEGs", flush=True)
    sc.tl.rank_genes_groups(
        adata,
        groupby="leiden_ccRCC_T",
        method="wilcoxon",
        use_raw=False,
        pts=True,
    )
    deg = sc.get.rank_genes_groups_df(adata, group=None)
    deg.to_csv(
        output_dir / "ccRCC_Tcell_leiden_all_DEG.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    top_genes = []
    for cluster in cluster_order:
        frame = deg.loc[
            deg["group"].astype(str).eq(cluster)
            & deg["names"].notna()
            & deg["names"].isin(adata.var_names)
        ].sort_values(["scores", "logfoldchanges"], ascending=False)
        for gene in frame["names"].head(args.top_deg):
            if gene not in top_genes:
                top_genes.append(gene)
    if top_genes:
        save_dotplot(
            adata,
            top_genes,
            cluster_order,
            output_dir
            / f"dotplot_ccRCC_Tcell_top{args.top_deg}_DEG_"
            "standardized_by_var.pdf",
        )

    annotation = pd.DataFrame(
        {
            "cell_id": cell_ids(adata),
            "dataset": adata.obs["dataset"].astype(str).to_numpy(),
            "original_cell_type": (
                adata.obs["cell_type"].astype(str).to_numpy()
            ),
            "leiden_ccRCC_T": (
                adata.obs["leiden_ccRCC_T"].astype(str).to_numpy()
            ),
        }
    )
    annotation.to_csv(
        output_dir / "ccRCC_Tcell_harmony_cluster_annotation.tsv",
        sep="\t",
        index=False,
    )
    annotation.to_csv(
        output_dir / "ccRCC_Tcell_harmony_cluster_annotation.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    adata.write_h5ad(
        output_dir / "ccRCC_Tcell_full_harmony.h5ad",
        compression="gzip",
    )


def main() -> None:
    args = parse_args()
    all_cell_dir = args.output_dir / "CD27_CD70_all_celltypes"
    tcell_dir = args.output_dir / "Tcell_full_harmony"
    all_cell_dir.mkdir(parents=True, exist_ok=True)
    tcell_dir.mkdir(parents=True, exist_ok=True)
    sc.set_figure_params(
        dpi=100,
        dpi_save=300,
        frameon=False,
        fontsize=10,
        facecolor="white",
    )

    print(f"Opening backed h5ad: {args.input}", flush=True)
    backed = sc.read_h5ad(args.input, backed="r")
    try:
        if not {"dataset", "cell_type"}.issubset(backed.obs):
            raise KeyError("obs must contain dataset and cell_type")
        datasets = backed.obs["dataset"].astype(str)
        ccrcc_mask = (~datasets.isin(RARE_RCC_DATASETS)).to_numpy()
        observed_rare = set(datasets).intersection(RARE_RCC_DATASETS)
        missing_rare = RARE_RCC_DATASETS - observed_rare
        if missing_rare:
            raise ValueError(
                "Rare-RCC datasets absent from h5ad: "
                + ", ".join(sorted(missing_rare))
            )
        sample_summary_frame = pd.DataFrame(
            {
                "dataset": datasets.to_numpy(),
                "excluded_rare_RCC": (
                    datasets.isin(RARE_RCC_DATASETS).to_numpy()
                ),
            }
        )
        sample_summary = (
            sample_summary_frame
            .groupby(["dataset", "excluded_rare_RCC"], observed=True)
            .size()
            .rename("n_cells")
            .reset_index()
        )
        sample_summary.to_csv(
            args.output_dir / "RCC_dataset_inclusion_summary.tsv",
            sep="\t",
            index=False,
        )
        print(
            f"Keeping presumed ccRCC cells: {int(ccrcc_mask.sum()):,}/"
            f"{backed.n_obs:,}; excluded rare-RCC cells: "
            f"{int((~ccrcc_mask).sum()):,}",
            flush=True,
        )

        save_all_celltype_cd27_cd70(backed, ccrcc_mask, all_cell_dir)
        tcell_mask = (
            ccrcc_mask
            & backed.obs["cell_type"].astype(str).eq("T_cell").to_numpy()
        )
        if tcell_mask.sum() < 50:
            raise ValueError(f"Only {int(tcell_mask.sum())} T cells retained")
        recluster_tcells(backed, tcell_mask, tcell_dir, args)
    finally:
        backed.file.close()

    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
