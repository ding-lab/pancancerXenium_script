#!/usr/bin/env python3
"""Extract TLS-core T cells, then Harmony + Leiden + UMAP + T/Tfh dotplots.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_tls_tcell_filtered_dotplot import T_CELL_MARKERS, save_grouped_dotplot
from tls_fibroblast_dc_harmony_umap import (
    SLIM_DIR,
    filter_by_cluster,
    load_lineage_subset,
    load_tls_cells,
    run_dimred,
    save_gene_dotplot,
    save_umap,
)


DEFAULT_T = SLIM_DIR / "NK_T.slim.h5ad"
DEFAULT_TLS_ROOT = Path(
    "/storage1/fs1/dinglab/Active/Projects/pdc/projects/general/x1000/"
    "tls/tls_define/gen12/outputs/tls"
)
DEFAULT_OUT = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/01_analysis/03_TLS/tls_tcell_harmony"
)
T_GENES = [
    "CD3E",
    "CD4",
    "CD8A",
    "FOXP3",
    "CXCL13",
    "CXCR5",
    "BCL6",
    "ICOS",
    "IL21",
    "PDCD1",
    "CCR7",
    "TCF7",
    "GZMB",
    "MKI67",
]
T_IDENTITY = ["CD3E"]
STRICT_CONTAMINANTS = [
    "EPCAM",
    "CD68",
    "KRT8",
    "KRT18",
    "KRT19",
    "LYZ",
    "C1QA",
    "PDGFRA",
    "PDGFRB",
    "FAP",
    "PECAM1",
    "VWF",
    "CDH5",
    "RGS5",
    "MYH11",
    "TAGLN",
    "MCAM",
]
# Follicle T / Tfh sit next to B cells, so MS4A1/CD19 spillover is expected.
# Keep these for QC plots only; do not use them to drop T clusters.
B_CONTAMINANTS = ["CD19", "MS4A1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t-h5ad", type=Path, default=DEFAULT_T)
    parser.add_argument("--tls-root", type=Path, default=DEFAULT_TLS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--include-halo100", action="store_true")
    parser.add_argument("--exclude-lymph-nodes", action="store_true")
    parser.add_argument(
        "--keep-nk",
        action="store_true",
        help="Keep cell_type == NK. Default drops explicit NK cells.",
    )
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--n-pcs", type=int, default=40)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--leiden-resolution", type=float, default=0.5)
    parser.add_argument(
        "--leiden-resolutions",
        nargs="+",
        type=float,
        default=[0.8, 1.0, 1.5, 2.0],
    )
    parser.add_argument("--harmony-theta", type=float, default=2.0)
    parser.add_argument("--harmony-max-iter", type=int, default=20)
    parser.add_argument("--batch-key", default="dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-harmony", action="store_true")
    parser.add_argument("--t-genes", nargs="+", default=T_GENES)
    parser.add_argument(
        "--skip-contaminant-filter",
        action="store_true",
        help="Keep contaminant-like Leiden clusters.",
    )
    parser.add_argument("--cluster-key", default="leiden_res0.8")
    parser.add_argument("--min-identity-frac", type=float, default=0.5)
    parser.add_argument("--max-contaminant-frac", type=float, default=0.4)
    return parser.parse_args()


def filter_t_contaminants(
    adata,
    cluster_key: str = "leiden_res0.8",
    min_identity_frac: float = 0.5,
    max_contaminant_frac: float = 0.4,
):
    return filter_by_cluster(
        adata,
        cluster_key=cluster_key,
        identity_genes=["CD3E"],
        contaminant_genes=STRICT_CONTAMINANTS,
        min_identity_frac=min_identity_frac,
        max_contaminant_frac=max_contaminant_frac,
    )


def save_feature_umap(adata, genes: list[str], out_dir: Path, filename: str) -> None:
    names = adata.raw.var_names if adata.raw is not None else adata.var_names
    found = [g for g in genes if g in names]
    if not found:
        print("[WARN] no T genes for feature UMAP", flush=True)
        return
    sc.pl.umap(
        adata,
        color=found,
        use_raw=True,
        ncols=4,
        frameon=False,
        show=False,
    )
    fig = plt.gcf()
    fig.savefig(out_dir / filename, bbox_inches="tight")
    fig.savefig(Path(out_dir / filename).with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close("all")
    print(f"[UMAP] {filename} + png genes={found}", flush=True)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = str(args.out_dir)
    sc.settings.verbosity = 2

    tls = load_tls_cells(args.tls_root, args.include_halo100)
    tls_ids = set(tls["cell_id"])
    tls_meta = tls.set_index("cell_id")[["tls_id"]]
    adata = load_lineage_subset(
        args.t_h5ad,
        "T_cell",
        tls_ids,
        args.exclude_lymph_nodes,
    )
    if "cell_type" in adata.obs.columns and not args.keep_nk:
        before = adata.n_obs
        adata = adata[~adata.obs["cell_type"].astype(str).eq("NK")].copy()
        print(f"[FILTER] dropped NK: {before - adata.n_obs} cells", flush=True)
    overlap = adata.obs_names.duplicated()
    if overlap.any():
        print(f"[WARN] dropping {int(overlap.sum())} duplicated barcodes", flush=True)
        adata = adata[~overlap].copy()
    adata.obs["tls_id"] = adata.obs_names.map(tls_meta["tls_id"])
    adata.obs["sample"] = adata.obs["dataset"].astype(str)

    inventory = (
        adata.obs.groupby(["cancer_type", "dataset", "cell_type"], observed=True)
        .size()
        .reset_index(name="n_cells")
        .sort_values(["cancer_type", "dataset", "cell_type"])
    )
    inventory.to_csv(args.out_dir / "cell_inventory.tsv", sep="\t", index=False)
    print(adata.obs["cell_type"].astype(str).value_counts().to_string())
    print(f"[T] {adata.n_obs} TLS T cells x {adata.n_vars} genes", flush=True)
    if adata.n_obs < 20:
        raise RuntimeError("Too few TLS T cells to cluster")

    adata.write_h5ad(args.out_dir / "TLS_tcell_rawsubset.h5ad")
    analyzed = run_dimred(adata, args)
    analyzed.write_h5ad(args.out_dir / "TLS_tcell_harmony_leiden.h5ad")
    if not args.skip_contaminant_filter:
        cluster_key = args.cluster_key
        if cluster_key not in analyzed.obs.columns:
            cluster_key = analyzed.uns.get("leiden_keys", ["leiden"])[0]
        analyzed, dropped, summary = filter_t_contaminants(
            analyzed,
            cluster_key=cluster_key,
            min_identity_frac=args.min_identity_frac,
            max_contaminant_frac=args.max_contaminant_frac,
        )
        dropped.to_csv(args.out_dir / "dropped_contaminant_cells.tsv", sep="\t")
        summary.to_csv(args.out_dir / "dropped_cluster_summary.tsv", sep="\t", index=False)

    leiden_keys = analyzed.uns.get("leiden_keys", ["leiden"])
    count_parts = []
    for key in leiden_keys:
        part = (
            analyzed.obs.groupby([key, "cell_type", "cancer_type"], observed=True)
            .size()
            .reset_index(name="n_cells")
            .rename(columns={key: "leiden"})
        )
        part.insert(0, "leiden_key", key)
        count_parts.append(part)
        save_umap(analyzed, args.out_dir, key, f"umap_{key}.pdf")
        save_gene_dotplot(
            analyzed,
            args.t_genes,
            args.out_dir,
            key,
            f"dotplot_T_{key}",
            f"TLS T cells {key}: lineage / Tfh / cytotoxic genes",
        )
        save_grouped_dotplot(
            analyzed,
            T_CELL_MARKERS,
            key,
            args.out_dir,
            f"dotplot_T_states_{key}",
            f"TLS T cells {key}: functional state markers",
        )
    pd.concat(count_parts, ignore_index=True).to_csv(
        args.out_dir / "cluster_counts.tsv", sep="\t", index=False
    )
    meta_cols = ["dataset", "cancer_type", "lineage", "cell_type", "tls_id", "leiden"]
    meta_cols += [k for k in leiden_keys if k != "leiden"]
    keep = [c for c in meta_cols if c in analyzed.obs.columns]
    analyzed.obs[keep].to_csv(args.out_dir / "cell_metadata.tsv", sep="\t")
    save_umap(analyzed, args.out_dir, "cancer_type", "umap_cancer_type.pdf")
    save_umap(analyzed, args.out_dir, "cell_type", "umap_cell_type.pdf")
    save_umap(analyzed, args.out_dir, "dataset", "umap_dataset.pdf")
    save_feature_umap(analyzed, args.t_genes, args.out_dir, "umap_T_genes.pdf")
    print(f"Wrote outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
