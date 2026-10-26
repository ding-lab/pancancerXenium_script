#!/usr/bin/env python3
"""Extract TLS-core Fibroblast and DC cells, then Harmony + Leiden + UMAP.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import anndata as ad
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix

try:
    from anndata.io import read_elem
except ImportError:
    from anndata.experimental import read_elem


SLIM_DIR = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "00_makeh5ad/01_x5k/00_integration/00_finalslim_subsets"
)
DEFAULT_FIBRO = SLIM_DIR / "Fibroblast.slim.h5ad"
DEFAULT_DC = SLIM_DIR / "DC.slim.h5ad"
DEFAULT_TLS_ROOT = Path(
    "/storage1/fs1/dinglab/Active/Projects/pdc/projects/general/x1000/"
    "tls/tls_define/gen12/outputs/tls"
)
DEFAULT_OUT = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/01_analysis/03_TLS/tls_fibroblast_dc_harmony"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fibroblast-h5ad", type=Path, default=DEFAULT_FIBRO)
    parser.add_argument("--dc-h5ad", type=Path, default=DEFAULT_DC)
    parser.add_argument("--tls-root", type=Path, default=DEFAULT_TLS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--include-halo100",
        action="store_true",
        help="Keep TLS halo100 members as well as TLS-core cells.",
    )
    parser.add_argument(
        "--exclude-lymph-nodes",
        action="store_true",
        help="Drop cancer_type == Lymph_nodes.",
    )
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--n-pcs", type=int, default=40)
    parser.add_argument("--n-neighbors", type=int, default=15)
    parser.add_argument("--leiden-resolution", type=float, default=0.5)
    parser.add_argument(
        "--leiden-resolutions",
        nargs="+",
        type=float,
        default=None,
        help="Run Leiden at each resolution, e.g. 0.8 1 1.5 2.",
    )
    parser.add_argument("--harmony-theta", type=float, default=2.0)
    parser.add_argument("--harmony-max-iter", type=int, default=20)
    parser.add_argument("--batch-key", default="dataset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-harmony",
        action="store_true",
        help="Cluster/UMAP on uncorrected PCA.",
    )
    parser.add_argument(
        "--skip-dc",
        action="store_true",
        help="Cluster TLS fibroblasts only; do not load DC.",
    )
    parser.add_argument(
        "--fdc-genes",
        nargs="+",
        default=["CXCL13", "CR2", "FCER2", "TNFSF11", "CD3E", "CD19", "PDGFRA", "PDGFRB", "COL5A2", "POSTN"],
        help="Genes for the Leiden dotplot.",
    )
    return parser.parse_args()


def decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "t", "yes"})
    )


def normalize_tls_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "outside", "nan", "none"}:
        return ""
    match = re.search(r"TLS[-_ ]?(\d+)", text, flags=re.IGNORECASE)
    if match:
        return f"TLS-{int(match.group(1))}"
    return text


def is_brca_al_sample(sample: str, cancer: str) -> bool:
    sample = str(sample)
    cancer = str(cancer)
    return cancer == "BRCA" and sample.startswith("AL")


def load_tls_cells(tls_root: Path, include_halo100: bool) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    cancers = sorted(p.name for p in tls_root.iterdir() if p.is_dir())
    for cancer in cancers:
        for sample_dir in sorted((tls_root / cancer).iterdir()):
            if not sample_dir.is_dir():
                continue
            sample = sample_dir.name
            if is_brca_al_sample(sample, cancer):
                continue
            path = sample_dir / f"{sample}_tls.csv"
            if not path.is_file():
                continue
            tls = pd.read_csv(
                path,
                usecols=["cell_id", "tls_region_halo100", "in_tls_core"],
                dtype=str,
                low_memory=False,
            )
            tls["tls_id"] = tls["tls_region_halo100"].map(normalize_tls_id)
            core = as_bool(tls["in_tls_core"])
            if include_halo100:
                keep = tls["tls_id"].ne("")
            else:
                keep = core & tls["tls_id"].ne("")
            tls = tls.loc[keep, ["cell_id", "tls_id"]].copy()
            if tls.empty:
                continue
            tls["cancer_type"] = cancer
            tls["dataset"] = sample
            parts.append(tls)
            print(
                f"[TLS] {cancer}/{sample}: {len(tls)} cells "
                f"({'core+halo100' if include_halo100 else 'core'})",
                flush=True,
            )
    if not parts:
        raise RuntimeError(f"No TLS cells found under {tls_root}")
    assigned = pd.concat(parts, ignore_index=True)
    assigned["cell_id"] = assigned["cell_id"].astype(str)
    assigned = assigned.drop_duplicates("cell_id", keep="first")
    print(f"[TLS] unique TLS cells: {len(assigned)}", flush=True)
    return assigned


def read_obs_frame(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as handle:
        obs = read_elem(handle["obs"])
    obs = obs.copy()
    if "barcode" in obs.columns:
        obs.index = obs["barcode"].astype(str)
    else:
        obs.index = obs.index.astype(str)
    return obs


def subset_csr_rows(group: h5py.Group, keep_idx: np.ndarray) -> csr_matrix:
    n_vars = int(group.attrs["shape"][1])
    if keep_idx.size == 0:
        return csr_matrix((0, n_vars), dtype=np.float32)
    indptr = group["indptr"][:]
    starts = indptr[keep_idx]
    ends = indptr[keep_idx + 1]
    counts = ends - starts
    new_indptr = np.empty(keep_idx.size + 1, dtype=np.int64)
    new_indptr[0] = 0
    np.cumsum(counts, out=new_indptr[1:])
    nnz = int(new_indptr[-1])
    data = np.empty(nnz, dtype=group["data"].dtype)
    indices = np.empty(nnz, dtype=group["indices"].dtype)
    order = np.argsort(starts, kind="mergesort")
    dest = new_indptr[:-1][order]
    src_start = starts[order]
    src_end = ends[order]
    dset_data = group["data"]
    dset_indices = group["indices"]
    i = 0
    n = len(order)
    while i < n:
        run_end = i + 1
        while (
            run_end < n
            and src_start[run_end] == src_end[run_end - 1]
        ):
            run_end += 1
        left = int(src_start[i])
        right = int(src_end[run_end - 1])
        dest_left = int(dest[i])
        dest_right = dest_left + (right - left)
        data[dest_left:dest_right] = dset_data[left:right]
        indices[dest_left:dest_right] = dset_indices[left:right]
        i = run_end
    return csr_matrix((data, indices, new_indptr), shape=(keep_idx.size, n_vars))


def load_lineage_subset(
    path: Path,
    lineage: str,
    tls_ids: set[str],
    exclude_lymph_nodes: bool,
) -> ad.AnnData:
    print(f"[LOAD] {lineage}: {path}", flush=True)
    obs = read_obs_frame(path)
    sample = obs["dataset"].astype(str) if "dataset" in obs.columns else pd.Series("", index=obs.index)
    cancer = (
        obs["cancer_type"].astype(str)
        if "cancer_type" in obs.columns
        else pd.Series("", index=obs.index)
    )
    keep = np.asarray(obs.index.isin(tls_ids), dtype=bool)
    keep &= ~np.array(
        [is_brca_al_sample(s, c) for s, c in zip(sample, cancer)],
        dtype=bool,
    )
    if exclude_lymph_nodes:
        keep &= cancer.ne("Lymph_nodes").to_numpy()
    keep_idx = np.flatnonzero(keep)
    print(
        f"[LOAD] {lineage}: {keep_idx.size}/{len(obs)} TLS cells after AL filter",
        flush=True,
    )
    with h5py.File(path, "r") as handle:
        var = read_elem(handle["var"])
        x = subset_csr_rows(handle["X"], keep_idx)
        counts = None
        if "layers" in handle and "counts" in handle["layers"]:
            counts = subset_csr_rows(handle["layers"]["counts"], keep_idx)
    adata = ad.AnnData(X=x, obs=obs.iloc[keep_idx].copy(), var=var.copy())
    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)
    adata.obs["lineage"] = lineage
    if counts is not None:
        adata.layers["counts"] = counts
    return adata


def harmony_corrected(harmony_out, n_obs: int) -> np.ndarray:
    z_corr = np.asarray(harmony_out.Z_corr)
    if z_corr.shape[0] == n_obs:
        return z_corr
    if z_corr.shape[1] == n_obs:
        return z_corr.T
    raise ValueError(f"Harmony Z_corr shape {z_corr.shape} != n_obs={n_obs}")


def run_leiden(adata: ad.AnnData, resolution: float, key: str, seed: int) -> None:
    """Leiden via igraph. Old docker images (dailybox) lack GLIBC_2.28."""
    try:
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=key,
            random_state=seed,
            flavor="igraph",
            n_iterations=2,
        )
        return
    except TypeError:
        pass
    except (ImportError, OSError) as err:
        raise ImportError(
            "scanpy Leiden needs python-igraph, but this node/container cannot "
            "load it (often GLIBC_2.28 missing in dailybox). Rerun with "
            "sbatch on general-cpu, e.g. run_tls_fibroblast_harmony_umap.sh. "
            f"Original error: {err}"
        ) from err
    try:
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=key,
            random_state=seed,
        )
    except (ImportError, OSError) as err:
        raise ImportError(
            "scanpy Leiden needs python-igraph, but this node/container cannot "
            "load it (often GLIBC_2.28 missing in dailybox). Rerun with "
            "sbatch on general-cpu. "
            f"Original error: {err}"
        ) from err


def run_dimred(adata: ad.AnnData, args: argparse.Namespace) -> ad.AnnData:
    adata = adata.copy()
    if "counts" in adata.layers:
        adata.layers["input_X"] = adata.X.copy()
        adata.X = adata.layers["counts"].copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    else:
        adata.layers["input_X"] = adata.X.copy()
        print("[DIMRED] no counts layer; using existing X without re-normalization", flush=True)
    raw_src = adata.copy()
    exclude = [str(g) for g in (getattr(args, "exclude_pca_genes", None) or [])]
    if exclude:
        drop = [g for g in exclude if g in adata.var_names]
        keep = [g for g in adata.var_names if g not in set(drop)]
        if drop:
            print(
                f"[DIMRED] exclude {len(drop)} genes before HVG/PCA "
                f"(kept {len(keep)}/{raw_src.n_vars})",
                flush=True,
            )
        if len(keep) < 50:
            raise RuntimeError(f"too few genes left after exclude_pca_genes: {len(keep)}")
        adata = adata[:, keep].copy()
    pca_genes = getattr(args, "pca_genes", None)
    if pca_genes:
        found = [g for g in pca_genes if g in adata.var_names]
        missing = [g for g in pca_genes if g not in adata.var_names]
        if missing:
            print(f"[DIMRED] pca_genes missing: {missing}", flush=True)
        if len(found) < 10:
            raise RuntimeError(f"too few pca_genes present: {found}")
        print(f"[DIMRED] PCA on {len(found)} state genes (not HVG)", flush=True)
        adata = adata[:, found].copy()
        n_hvg = len(found)
    else:
        n_hvg = min(args.n_hvg, adata.n_vars - 1, max(adata.n_obs - 1, 2))
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat")
        adata = adata[:, adata.var["highly_variable"]].copy()
    adata.raw = raw_src
    sc.pp.scale(adata, max_value=10)
    n_pcs = min(args.n_pcs, adata.n_obs - 1, adata.n_vars)
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack", random_state=args.seed)
    use_rep = "X_pca"
    if not args.skip_harmony:
        import harmonypy

        n_batches = int(adata.obs[args.batch_key].astype(str).nunique())
        print(
            f"[Harmony] batch_key={args.batch_key} n_batches={n_batches}",
            flush=True,
        )
        if n_batches >= 2:
            adata.obsm["X_pca_uncorrected"] = adata.obsm["X_pca"].copy()
            harmony_out = harmonypy.run_harmony(
                adata.obsm["X_pca"].astype(np.float64),
                adata.obs,
                args.batch_key,
                theta=args.harmony_theta,
                max_iter_harmony=args.harmony_max_iter,
                random_state=args.seed,
            )
            adata.obsm["X_pca_harmony"] = harmony_corrected(harmony_out, adata.n_obs)
            use_rep = "X_pca_harmony"
        else:
            print("[Harmony] skipped: fewer than 2 batches", flush=True)
    sc.pp.neighbors(
        adata,
        n_neighbors=min(args.n_neighbors, adata.n_obs - 1),
        use_rep=use_rep,
        random_state=args.seed,
    )
    sc.tl.umap(adata, random_state=args.seed)
    resolutions = args.leiden_resolutions or [args.leiden_resolution]
    leiden_keys = []
    for res in resolutions:
        key = f"leiden_res{res:g}" if args.leiden_resolutions else "leiden"
        run_leiden(adata, resolution=res, key=key, seed=args.seed)
        adata.obs[key] = adata.obs[key].astype(str)
        leiden_keys.append(key)
        print(f"[Leiden] {key}: {adata.obs[key].nunique()} clusters", flush=True)
    adata.obs["leiden"] = adata.obs[leiden_keys[0]]
    adata.uns["leiden_keys"] = leiden_keys
    adata.uns["dimred"] = {
        "use_rep": use_rep,
        "n_hvg": int(n_hvg),
        "n_pcs": int(n_pcs),
        "batch_key": args.batch_key,
        "leiden_resolution": args.leiden_resolution,
        "leiden_resolutions": [float(r) for r in resolutions],
        "harmony": use_rep == "X_pca_harmony",
        "tls_scope": (
            "core+halo100" if getattr(args, "include_halo100", True) else "core"
        ),
    }
    return adata


def _set_dotplot_colorbar_ticks(fig, vmin: float, vmax: float) -> None:
    ticks = np.linspace(vmin, vmax, int(round(vmax - vmin)) + 1)
    for ax in fig.axes:
        title = ax.get_title() or ""
        if ax.get_label() == "<colorbar>" or "Mean expression" in title:
            ax.set_xlim(vmin, vmax)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:g}" for t in ticks])
            return


def save_gene_dotplot(
    adata: ad.AnnData,
    genes: list[str],
    out_dir: Path,
    groupby: str,
    stem: str,
    title: str,
    standard_scale: str | None = "var",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_title: str | None = None,
) -> None:
    src = adata.raw.to_adata() if adata.raw is not None else adata
    src.obs = adata.obs.copy()
    found = [g for g in genes if g in src.var_names]
    missing = [g for g in genes if g not in src.var_names]
    if missing:
        print(f"[WARN] missing genes: {missing}", flush=True)
    if not found:
        print("[WARN] no requested genes found; skip dotplot", flush=True)
        return
    kwargs = {}
    if vmin is not None:
        kwargs["vmin"] = vmin
    if vmax is not None:
        kwargs["vmax"] = vmax
    if colorbar_title is not None:
        kwargs["colorbar_title"] = colorbar_title
    sc.pl.dotplot(
        src,
        var_names=found,
        groupby=groupby,
        standard_scale=standard_scale,
        dendrogram=False,
        title=title,
        show=False,
        **kwargs,
    )
    fig = plt.gcf()
    if vmin is not None and vmax is not None:
        _set_dotplot_colorbar_ticks(fig, vmin, vmax)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=200)
    plt.close("all")

    x = src[:, found].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=float)
    groups = src.obs[groupby].astype(str)
    rows = []
    for gene_i, gene in enumerate(found):
        for group in sorted(groups.unique(), key=lambda v: (len(v), v)):
            mask = groups.eq(group).to_numpy()
            vals = x[mask, gene_i]
            n = int(mask.sum())
            rows.append(
                {
                    groupby: group,
                    "gene": gene,
                    "n_cells": n,
                    "pct_expressed": 100.0 * float(np.mean(vals > 0)) if n else np.nan,
                    "mean_expr": float(np.mean(vals)) if n else np.nan,
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / f"{stem}.tsv", sep="\t", index=False)
    print(f"[DOTPLOT] {stem}.pdf + png  genes={found}", flush=True)


def gene_values(adata: ad.AnnData, gene: str):
    if adata.raw is not None:
        if gene not in adata.raw.var_names:
            return None
        x = adata.raw[:, gene].X
    else:
        if gene not in adata.var_names:
            return None
        x = adata[:, gene].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=float).ravel()


def filter_by_cluster(
    adata: ad.AnnData,
    cluster_key: str,
    identity_genes: list[str],
    contaminant_genes: list[str],
    min_identity_frac: float = 0.5,
    max_contaminant_frac: float = 0.4,
) -> tuple[ad.AnnData, pd.DataFrame, pd.DataFrame]:
    """Drop whole Leiden clusters, not single cells with 1 leaked count.

    A cluster is removed if identity+ fraction is below min_identity_frac,
    or any one contaminant gene is expressed in at least max_contaminant_frac
    of its cells.
    """
    if cluster_key not in adata.obs.columns:
        raise KeyError(f"{cluster_key} not in obs; have {list(adata.obs.columns)}")
    groups = adata.obs[cluster_key].astype(str)
    ident = np.zeros(adata.n_obs, dtype=bool)
    used_ident = []
    for gene in identity_genes:
        vals = gene_values(adata, gene)
        if vals is None:
            continue
        ident = ident | (vals > 0)
        used_ident.append(gene)
    ident_frac = pd.Series(ident, index=adata.obs_names).groupby(groups).mean()
    n_per = groups.value_counts()

    gene_fracs: dict[str, pd.Series] = {}
    used_cont = []
    for gene in contaminant_genes:
        vals = gene_values(adata, gene)
        if vals is None:
            continue
        gene_fracs[gene] = pd.Series(vals > 0, index=adata.obs_names).groupby(groups).mean()
        used_cont.append(gene)

    rows = []
    drop_clusters = set()
    for cl in ident_frac.index:
        reasons = []
        if ident_frac[cl] < min_identity_frac:
            reasons.append(f"identity<{min_identity_frac:g}")
        worst_gene = ""
        worst_frac = 0.0
        for gene, frac in gene_fracs.items():
            f = float(frac.get(cl, 0.0))
            if f > worst_frac:
                worst_frac = f
                worst_gene = gene
            if f >= max_contaminant_frac:
                reasons.append(f"{gene}>={max_contaminant_frac:g}")
        if reasons:
            drop_clusters.add(str(cl))
        rows.append(
            {
                "cluster": cl,
                "n_cells": int(n_per.get(cl, 0)),
                "identity_genes": ",".join(used_ident),
                "identity_frac": float(ident_frac[cl]),
                "worst_contaminant": worst_gene,
                "worst_contaminant_frac": worst_frac,
                "drop": bool(reasons),
                "drop_reason": ";".join(reasons),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["drop", "cluster"], ascending=[False, True])
    drop = groups.isin(drop_clusters)
    kept = adata[~drop].copy()
    dropped = adata.obs.loc[drop].copy()
    dropped["drop_cluster"] = groups.loc[drop].to_numpy()
    print(
        f"[FILTER] cluster {cluster_key}: keep {kept.n_obs}/{adata.n_obs} cells; "
        f"drop clusters {sorted(drop_clusters, key=lambda x: (len(x), x)) or 'none'} "
        f"(identity {used_ident} <{min_identity_frac:g} or any contaminant "
        f">={max_contaminant_frac:g})",
        flush=True,
    )
    return kept, dropped, summary


def save_figure(fig, out_dir: Path, filename: str, dpi: int = 200) -> None:
    path = out_dir / filename
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def save_umap(
    adata: ad.AnnData,
    out_dir: Path,
    color: str,
    filename: str,
    size: float | None = None,
) -> None:
    if color not in adata.obs.columns:
        return
    fig = sc.pl.umap(
        adata,
        color=color,
        legend_loc="on data" if color.startswith("leiden") or color == "leiden" else "right margin",
        frameon=False,
        show=False,
        return_fig=True,
        size=size,
    )
    if fig is None:
        fig = plt.gcf()
    save_figure(fig, out_dir, filename)
    plt.close("all")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = str(args.out_dir)
    sc.settings.verbosity = 2

    tls = load_tls_cells(args.tls_root, args.include_halo100)
    tls_ids = set(tls["cell_id"])
    tls_meta = tls.set_index("cell_id")[["tls_id"]]

    fibro = load_lineage_subset(
        args.fibroblast_h5ad,
        "Fibroblast",
        tls_ids,
        args.exclude_lymph_nodes,
    )
    if args.skip_dc:
        adata = fibro
        stem = "TLS_fibroblast"
    else:
        dc = load_lineage_subset(
            args.dc_h5ad,
            "DC",
            tls_ids,
            args.exclude_lymph_nodes,
        )
        adata = ad.concat(
            [fibro, dc],
            join="inner",
            merge="same",
            label="source_h5ad",
            keys=["Fibroblast.slim", "DC.slim"],
            index_unique=None,
        )
        stem = "TLS_fibroblast_DC"
    overlap = adata.obs_names.duplicated()
    if overlap.any():
        print(f"[WARN] dropping {int(overlap.sum())} duplicated barcodes", flush=True)
        adata = adata[~overlap].copy()
    adata.obs["tls_id"] = adata.obs_names.map(tls_meta["tls_id"])
    adata.obs["sample"] = adata.obs["dataset"].astype(str)

    inventory = (
        adata.obs.groupby(["cancer_type", "dataset", "lineage"], observed=True)
        .size()
        .reset_index(name="n_cells")
        .sort_values(["cancer_type", "dataset", "lineage"])
    )
    inventory.to_csv(args.out_dir / "cell_inventory.tsv", sep="\t", index=False)
    print(inventory.groupby("lineage")["n_cells"].sum().to_string())
    print(f"[CONCAT] {adata.n_obs} cells x {adata.n_vars} genes", flush=True)
    if adata.n_obs < 20:
        raise RuntimeError("Too few TLS cells to cluster")

    adata.write_h5ad(args.out_dir / f"{stem}_rawsubset.h5ad")
    analyzed = run_dimred(adata, args)
    analyzed.write_h5ad(args.out_dir / f"{stem}_harmony_leiden.h5ad")

    leiden_keys = analyzed.uns.get("leiden_keys", ["leiden"])
    count_parts = []
    for key in leiden_keys:
        part = (
            analyzed.obs.groupby([key, "lineage", "cancer_type"], observed=True)
            .size()
            .reset_index(name="n_cells")
            .rename(columns={key: "leiden"})
        )
        part.insert(0, "leiden_key", key)
        count_parts.append(part)
        save_umap(analyzed, args.out_dir, key, f"umap_{key}.pdf")
        save_gene_dotplot(
            analyzed,
            args.fdc_genes,
            args.out_dir,
            key,
            f"dotplot_CXCL13_CR2_FCER2_CD3E_CD19_PTPRC_{key}",
            f"Fibroblast {key}: CXCL13 CR2 FCER2 CD3E CD19 PTPRC",
        )
    pd.concat(count_parts, ignore_index=True).to_csv(
        args.out_dir / "cluster_counts.tsv", sep="\t", index=False
    )
    meta_cols = ["dataset", "cancer_type", "lineage", "tls_id", "leiden"] + [
        k for k in leiden_keys if k != "leiden"
    ]
    analyzed.obs[meta_cols].to_csv(args.out_dir / "cell_metadata.tsv", sep="\t")

    save_umap(analyzed, args.out_dir, "lineage", "umap_lineage.pdf")
    save_umap(analyzed, args.out_dir, "cancer_type", "umap_cancer_type.pdf")
    save_umap(analyzed, args.out_dir, "dataset", "umap_dataset.pdf")
    print(f"Wrote outputs to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
