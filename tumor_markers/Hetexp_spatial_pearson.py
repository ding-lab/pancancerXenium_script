#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

ROOT = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/01_cell_typing"
)
PLOT_DIR = ROOT / "01_analysis" / "00_tumor_gene_module" / "plot"
DEFAULT_GENE_LIST = PLOT_DIR / "selected_shared_and_cancer_specific_heterogeneous_genes.tsv"
DEFAULT_OUTPUT_DIR = PLOT_DIR / "tumor_smoothed_pearson_results"

ANNOTATION_OVERRIDES = {
    "PDAC": ROOT / "PDAC" / "PDAC_cell_annotations.updated.csv",
}
H5AD_SUBDIR_OVERRIDES = {
    "PDAC": "adata_by_singlesample_allprobe",
}

NEG_CORR_THRESHOLD = -0.15
POS_CORR_THRESHOLD = 0.15
PVAL_THRESHOLD = 0.05
KNN_NEIGHBORS = 50


def knn_mean_smooth(
    coordinates: np.ndarray,
    values: np.ndarray,
    k: int,
) -> tuple[np.ndarray, int]:
    values = np.asarray(values)
    effective_k = min(int(k), len(coordinates))
    if effective_k < 1:
        raise ValueError("k must be >= 1")
    adjacency = normalize(
        NearestNeighbors(n_neighbors=effective_k)
        .fit(coordinates)
        .kneighbors_graph(coordinates),
        norm="l1",
        axis=1,
    )
    return np.asarray(adjacency.dot(values)), effective_k


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate smoothed spatial Pearson correlation for one sample."
    )
    parser.add_argument("--cancer", required=True, help="Cancer folder, e.g. PDAC")
    parser.add_argument("--sample", required=True, help="Sample ID without .h5ad")
    parser.add_argument("--gene-list-path", type=Path, default=DEFAULT_GENE_LIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--annotation-path",
        type=Path,
        help="External annotation CSV. Defaults to PDAC updated CSV for PDAC.",
    )
    parser.add_argument(
        "--h5ad-subdir",
        help="Subdirectory under cancer folder containing per-sample h5ad files.",
    )
    parser.add_argument("--tumor-regex", default="^Tumor$")
    parser.add_argument("--min-tumor-cells", type=int, default=50)
    return parser.parse_args()


def resolve_annotation_path(cancer: str, override: Path | None = None) -> Path | None:
    if override is not None:
        return override
    return ANNOTATION_OVERRIDES.get(cancer)


def resolve_h5ad_subdir(cancer: str, override: str | None = None) -> str:
    if override:
        return override
    return H5AD_SUBDIR_OVERRIDES.get(cancer, "adata_by_sample")


def load_cancer_specific_genes(gene_list_path: Path) -> dict[str, list[str]]:
    gene_df = pd.read_csv(gene_list_path, sep="\t")
    return {
        cancer: sub["gene"].tolist()
        for cancer, sub in gene_df.groupby("top_cancer")
    }


def annotation_schema(path: Path) -> dict[str, str | None]:
    columns = pd.read_csv(path, nrows=0).columns.tolist()
    if {"index", "dataset", "cell_type"}.issubset(columns):
        return {
            "barcode": "index",
            "cell_id": None,
            "sample": "dataset",
            "cell_type": "cell_type",
        }
    if {"barcode", "cell_id", "cell_type"}.issubset(columns):
        return {
            "barcode": "barcode",
            "cell_id": "cell_id",
            "sample": None,
            "cell_type": "cell_type",
        }
    raise ValueError(f"Unsupported annotation format in {path}. Columns: {columns}")


def load_sample_annotation(path: Path, sample: str) -> pd.DataFrame:
    schema = annotation_schema(path)
    usecols = [
        column
        for column in (
            schema["barcode"],
            schema["cell_id"],
            schema["sample"],
            schema["cell_type"],
        )
        if column is not None
    ]
    suffix = f"_{sample}"
    chunks = []
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype=str,
        chunksize=500_000,
        keep_default_na=False,
    ):
        if schema["sample"] is not None:
            mask = chunk[schema["sample"]].eq(sample)
        else:
            mask = chunk[schema["barcode"]].str.endswith(suffix, na=False)
        if mask.any():
            chunks.append(chunk.loc[mask].copy())
    if not chunks:
        raise RuntimeError(f"No annotation rows found for sample {sample} in {path}")

    annotation = pd.concat(chunks, ignore_index=True).rename(
        columns={
            schema["barcode"]: "annotation_barcode",
            schema["cell_type"]: "cell_type",
        }
    )
    if schema["cell_id"] is None:
        annotation["cell_id"] = annotation["annotation_barcode"].str.removesuffix(suffix)
    else:
        annotation = annotation.rename(columns={schema["cell_id"]: "cell_id"})
    return annotation[["annotation_barcode", "cell_id", "cell_type"]].drop_duplicates(
        subset=["annotation_barcode"]
    )


def attach_tumor_mask(
    adata: sc.AnnData,
    annotation: pd.DataFrame,
    tumor_regex: str,
) -> tuple[pd.Series, str]:
    obs_names = pd.Index(adata.obs_names.astype(str))
    full_map = annotation.drop_duplicates("annotation_barcode").set_index(
        "annotation_barcode"
    )["cell_type"]
    local_map = annotation.drop_duplicates("cell_id").set_index("cell_id")["cell_type"]

    full_labels = pd.Series(obs_names.map(full_map), index=adata.obs_names)
    local_labels = pd.Series(obs_names.map(local_map), index=adata.obs_names)
    if full_labels.notna().sum() >= local_labels.notna().sum():
        labels, match_key = full_labels, "annotation_barcode"
    else:
        labels, match_key = local_labels, "cell_id"

    tumor_mask = labels.astype(str).str.contains(
        tumor_regex,
        case=False,
        na=False,
        regex=True,
    )
    return tumor_mask, match_key


def process_single_sample(
    base_cancer: str,
    sample_id: str,
    cancer_specific_genes: dict[str, list[str]],
    output_dir: Path,
    annotation_path: Path | None,
    h5ad_subdir: str,
    tumor_regex: str,
    min_tumor_cells: int,
) -> list[dict]:
    h5ad_path = ROOT / base_cancer / h5ad_subdir / f"{sample_id}.h5ad"
    if not h5ad_path.is_file():
        print(f"[Error] File not found: {h5ad_path}")
        return []

    print(f"[Info] Processing {sample_id} | cancer={base_cancer}")
    if base_cancer not in cancer_specific_genes:
        print(f"[Warn] No cancer-specific genes listed for {base_cancer}.")
        return []

    target_genes = cancer_specific_genes[base_cancer]
    try:
        adata = sc.read_h5ad(h5ad_path)
    except Exception as exc:
        print(f"[Error] Failed to read h5ad: {exc}")
        return []

    if annotation_path is not None:
        annotation = load_sample_annotation(annotation_path, sample_id)
        tumor_mask, match_key = attach_tumor_mask(adata, annotation, tumor_regex)
        print(f"[Info] Tumor cells from external annotation ({match_key}).")
    elif "cell_type_refine" in adata.obs.columns:
        tumor_mask = adata.obs["cell_type_refine"].astype(str).str.contains(
            tumor_regex,
            case=False,
            na=False,
            regex=True,
        )
        print("[Info] Tumor cells from obs['cell_type_refine'].")
    else:
        print("[Warn] No external annotation and no cell_type_refine column.")
        return []

    n_tumor = int(tumor_mask.sum())
    if n_tumor < min_tumor_cells:
        print(f"[Warn] Too few tumor cells ({n_tumor} < {min_tumor_cells}).")
        return []

    adata_tumor = adata[tumor_mask].copy()
    valid_genes = [gene for gene in target_genes if gene in adata_tumor.var_names]
    if len(valid_genes) < 2:
        print("[Warn] Fewer than 2 target genes present in this sample.")
        return []

    print(f"[Info] Using {n_tumor} tumor cells and {len(valid_genes)} genes.")

    if "counts" in adata_tumor.layers:
        adata_tumor.X = adata_tumor.layers["counts"].copy()
    sc.pp.normalize_total(adata_tumor, target_sum=1e4)
    sc.pp.log1p(adata_tumor)

    if "X_spatial" in adata_tumor.obsm:
        coords = adata_tumor.obsm["X_spatial"]
    elif "spatial" in adata_tumor.obsm:
        coords = adata_tumor.obsm["spatial"]
    elif {"x_centroid", "y_centroid"}.issubset(adata_tumor.obs.columns):
        coords = adata_tumor.obs[["x_centroid", "y_centroid"]].to_numpy()
    else:
        coords = np.zeros((adata_tumor.n_obs, 2))

    x_coords, y_coords = coords[:, 0], coords[:, 1]

    print(f"[Info] Spatial KNN smoothing (k={KNN_NEIGHBORS}, NearestNeighbors)...")
    raw_expr_mat = adata_tumor[:, valid_genes].X.toarray()
    smoothed_expr_mat, _ = knn_mean_smooth(
        coords, raw_expr_mat, KNN_NEIGHBORS
    )
    expr_df = pd.DataFrame(
        smoothed_expr_mat,
        columns=valid_genes,
        index=adata_tumor.obs_names,
    )

    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    results = []
    sig_count = 0
    for g1, g2 in combinations(valid_genes, 2):
        v1 = expr_df[g1].values
        v2 = expr_df[g2].values
        corr, pval = pearsonr(v1, v2)
        results.append(
            {
                "sample_id": sample_id,
                "cancertype": base_cancer,
                "gene_A": g1,
                "gene_B": g2,
                "smoothed_pearson_r": corr,
                "p_value": pval,
                "n_tumor_cells": len(v1),
            }
        )

        if (corr <= NEG_CORR_THRESHOLD or corr >= POS_CORR_THRESHOLD) and pval <= PVAL_THRESHOLD:
            sig_count += 1
            rel_type, suffix = (
                ("Mutual Exclusivity", "exclusive")
                if corr < 0
                else ("Positive Co-expression", "coexpressed")
            )
            plot_file = plots_dir / f"{base_cancer}_{sample_id}_{g1}_vs_{g2}_{suffix}.pdf"
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
            ax1.scatter(v1, v2, alpha=0.5, s=8, c="gray")
            sns.kdeplot(x=v1, y=v2, color="darkred", alpha=0.7, levels=5, ax=ax1)
            ax1.set_title(f"{rel_type} (Smoothed)\nPearson R = {corr:.3f}")
            ax1.set_xlabel(f"{g1} (Smoothed Log1p)")
            ax1.set_ylabel(f"{g2} (Smoothed Log1p)")

            ax2.scatter(x_coords, y_coords, color="lightgray", s=1, alpha=0.3)
            thresh1 = np.median(v1[v1 > 0]) if (v1 > 0).sum() > 0 else 0
            thresh2 = np.median(v2[v2 > 0]) if (v2 > 0).sum() > 0 else 0
            ax2.scatter(
                x_coords[v1 > thresh1],
                y_coords[v1 > thresh1],
                color="red",
                s=3,
                alpha=0.7,
                label=f"{g1} High",
            )
            ax2.scatter(
                x_coords[v2 > thresh2],
                y_coords[v2 > thresh2],
                color="blue",
                s=3,
                alpha=0.7,
                label=f"{g2} High",
            )
            ax2.set_title("Spatial Distribution (Smoothed)")
            ax2.set_aspect("equal")
            ax2.axis("off")
            ax2.legend(loc="upper right", markerscale=3)
            plt.tight_layout()
            plt.savefig(plot_file, dpi=150)
            plt.close()

    print(f"[Success] Finished gene pairs; wrote {sig_count} significant PDFs.")
    return results


def main() -> None:
    args = parse_args()
    annotation_path = resolve_annotation_path(args.cancer, args.annotation_path)
    h5ad_subdir = resolve_h5ad_subdir(args.cancer, args.h5ad_subdir)
    cancer_specific_genes = load_cancer_specific_genes(args.gene_list_path)

    print("-" * 50)
    print(f"Target cancer      : {args.cancer}")
    print(f"Target sample      : {args.sample}")
    print(f"Gene list          : {args.gene_list_path}")
    print(f"H5AD subdir        : {h5ad_subdir}")
    print(f"Annotation path    : {annotation_path or 'obs[cell_type_refine]'}")
    print("-" * 50)

    results = process_single_sample(
        base_cancer=args.cancer,
        sample_id=args.sample,
        cancer_specific_genes=cancer_specific_genes,
        output_dir=args.output_dir,
        annotation_path=annotation_path,
        h5ad_subdir=h5ad_subdir,
        tumor_regex=args.tumor_regex,
        min_tumor_cells=args.min_tumor_cells,
    )

    if results:
        final_df = pd.DataFrame(results)
        final_df["abs_r"] = final_df["smoothed_pearson_r"].abs()
        final_df = final_df.sort_values(by="abs_r", ascending=False).drop(columns=["abs_r"])
        csv_path = args.output_dir / "tables" / f"{args.cancer}_{args.sample}_results.csv"
        final_df.to_csv(csv_path, index=False)
        print(f"[Finished] Saved table: {csv_path}")
    else:
        print("[Finished] No valid gene-pair results.")


if __name__ == "__main__":
    main()
