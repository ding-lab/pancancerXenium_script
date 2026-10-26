#!/usr/bin/env python3
"""Run Hotspot on tumor cells from every non-lymph-node Xenium sample.

Cell types are always taken from ``<CANCER>_cell_annotations.csv`` rather than
from h5ad.obs. Genes are restricted to ``shared.genepanel.tsv`` before Hotspot.
The task-list mode only inspects paths; it does not open h5ad files.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import hotspot
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


ROOT = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/01_cell_typing"
)
ANALYSIS_DIR = ROOT / "01_analysis" / "00_tumor_gene_module"
PANEL_FILE = ROOT / "shared.genepanel.tsv"
OUT_ROOT = ANALYSIS_DIR / "hotspot_tumor_all_cancers_shared_panel"
EXCLUDED_DIRECTORIES = {"Lymph_nodes", "01_analysis"}

# Override default <CANCER>_cell_annotations.csv for specific cancer types.
ANNOTATION_OVERRIDES: dict[str, Path] = {
    "PDAC": ROOT / "PDAC" / "PDAC_cell_annotations.updated.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cancer")
    parser.add_argument("--sample")
    parser.add_argument(
        "--annotation-path",
        type=Path,
        help="Override the default <CANCER>_cell_annotations.csv path.",
    )
    parser.add_argument(
        "--write-task-list",
        type=Path,
        help="Write all runnable cancer/sample pairs and exit without opening h5ad.",
    )
    parser.add_argument(
        "--write-pdac-task-list",
        type=Path,
        help="Write PDAC-only task list using the updated annotation file.",
    )
    parser.add_argument("--min-tumor-cells", type=int, default=100)
    parser.add_argument("--min-cells-per-gene", type=int, default=10)
    parser.add_argument("--tumor-regex", default="Tumor")
    parser.add_argument("--neighbors", type=int, default=30)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    return parser.parse_args()


def resolve_annotation_path(cancer: str, override: Path | None = None) -> Path:
    if override is not None:
        return override
    if cancer in ANNOTATION_OVERRIDES:
        return ANNOTATION_OVERRIDES[cancer]
    return ROOT / cancer / f"{cancer}_cell_annotations.csv"


def discover_tasks(annotation_overrides: dict[str, Path] | None = None) -> list[tuple[str, str]]:
    overrides = annotation_overrides or ANNOTATION_OVERRIDES
    tasks: list[tuple[str, str]] = []
    for cancer_dir in sorted(ROOT.iterdir()):
        cancer = cancer_dir.name
        if not cancer_dir.is_dir() or cancer in EXCLUDED_DIRECTORIES:
            continue
        annotation = overrides.get(cancer, cancer_dir / f"{cancer}_cell_annotations.csv")
        h5ad_dir = cancer_dir / "adata_by_singlesample_allprobe"
        if not annotation.is_file() or not h5ad_dir.is_dir():
            continue
        for h5ad_path in sorted(h5ad_dir.glob("*.h5ad")):
            if h5ad_path.is_file():
                tasks.append((cancer, h5ad_path.stem))
    return tasks


def discover_pdac_tasks() -> list[tuple[str, str]]:
    annotation = resolve_annotation_path("PDAC")
    h5ad_dir = ROOT / "PDAC" / "adata_by_singlesample_allprobe"
    if not annotation.is_file():
        raise FileNotFoundError(f"Updated PDAC annotation not found: {annotation}")
    if not h5ad_dir.is_dir():
        raise FileNotFoundError(f"PDAC h5ad directory not found: {h5ad_dir}")
    return [("PDAC", h5ad_path.stem) for h5ad_path in sorted(h5ad_dir.glob("*.h5ad"))]


def write_task_list(path: Path) -> None:
    tasks = discover_tasks()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{cancer}\t{sample}\n" for cancer, sample in tasks))
    print(f"Wrote {len(tasks)} tasks to {path}")
    print("No h5ad file was opened and no Hotspot analysis was run.")


def write_pdac_task_list(path: Path) -> None:
    tasks = discover_pdac_tasks()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{cancer}\t{sample}\n" for cancer, sample in tasks))
    print(f"Wrote {len(tasks)} PDAC tasks to {path}")
    print(f"Annotation file: {resolve_annotation_path('PDAC')}")
    print("No h5ad file was opened and no Hotspot analysis was run.")


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
    raise ValueError(
        f"Unsupported annotation format in {path}. Columns: {columns}"
    )


def load_sample_annotation(
    path: Path,
    sample: str,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Stream one sample from either the LUNG or standard annotation format."""
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
    selected = []
    suffix = f"_{sample}"
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype=str,
        chunksize=chunksize,
        keep_default_na=False,
    ):
        if schema["sample"] is not None:
            mask = chunk[schema["sample"]].eq(sample)
        else:
            mask = chunk[schema["barcode"]].str.endswith(suffix, na=False)
        if mask.any():
            selected.append(chunk.loc[mask].copy())
    if not selected:
        raise RuntimeError(f"No annotation rows found for sample {sample} in {path}")

    annotation = pd.concat(selected, ignore_index=True)
    annotation = annotation.rename(
        columns={
            schema["barcode"]: "annotation_barcode",
            schema["cell_type"]: "cell_type",
        }
    )
    if schema["cell_id"] is None:
        annotation["cell_id"] = annotation["annotation_barcode"].str.removesuffix(
            suffix
        )
    else:
        annotation = annotation.rename(columns={schema["cell_id"]: "cell_id"})
    return annotation[["annotation_barcode", "cell_id", "cell_type"]].drop_duplicates(
        subset=["annotation_barcode"]
    )


def attach_annotation(
    adata: sc.AnnData,
    annotation: pd.DataFrame,
) -> tuple[sc.AnnData, str]:
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
    adata.obs["cell_type_external"] = labels
    return adata, match_key


def panel_var_mask(adata: sc.AnnData, panel: set[str]) -> np.ndarray:
    candidates: list[tuple[int, str | None, np.ndarray]] = []
    direct = np.asarray(pd.Index(adata.var_names.astype(str)).isin(panel))
    candidates.append((int(direct.sum()), None, direct))
    for column in ("gene_name", "gene_symbol", "feature_name"):
        if column not in adata.var:
            continue
        values = adata.var[column].astype(str)
        mask = values.isin(panel).to_numpy()
        candidates.append((int(mask.sum()), column, mask))
    n_matches, column, best_mask = max(candidates, key=lambda item: item[0])
    if n_matches == 0:
        raise RuntimeError("No shared-panel genes matched h5ad.var.")
    if column is not None:
        values = adata.var[column].astype(str)
        if values[best_mask].duplicated().any():
            raise ValueError(f"Duplicate panel symbols in adata.var[{column!r}]")
        adata.var_names = pd.Index(values)
    return best_mask


def matrix_for_layer(adata: sc.AnnData, layer_key: str | None):
    return adata.layers[layer_key] if layer_key is not None else adata.X


def remove_zero_variance_genes(
    adata: sc.AnnData,
    layer_key: str | None,
) -> sc.AnnData:
    matrix = matrix_for_layer(adata, layer_key)
    if sparse.issparse(matrix):
        mean = np.asarray(matrix.mean(axis=0)).ravel()
        mean_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel()
        variance = mean_sq - mean**2
    else:
        variance = np.var(np.asarray(matrix), axis=0)
    return adata[:, variance > 0].copy()


def calculate_total_counts(
    adata: sc.AnnData,
    layer_key: str | None,
) -> None:
    matrix = matrix_for_layer(adata, layer_key)
    adata.obs["total_counts"] = np.asarray(matrix.sum(axis=1)).ravel()


def run_hotspot(
    adata_sub: sc.AnnData,
    cancer: str,
    sample: str,
    out_root: Path,
    min_cells_per_gene: int,
    neighbors: int,
    jobs: int,
    metadata: dict[str, object],
) -> None:
    out_dir = out_root / cancer / sample
    out_dir.mkdir(parents=True, exist_ok=True)

    sc.pp.filter_genes(adata_sub, min_cells=min_cells_per_gene)
    layer_key = "counts" if "counts" in adata_sub.layers else None
    adata_sub = remove_zero_variance_genes(adata_sub, layer_key)
    if adata_sub.n_vars < 2:
        raise RuntimeError(f"Only {adata_sub.n_vars} variable panel genes remain.")
    calculate_total_counts(adata_sub, layer_key)

    n_comps = min(50, adata_sub.n_obs - 1, adata_sub.n_vars - 1)
    if n_comps < 2:
        raise RuntimeError("Too few tumor cells or genes for PCA.")
    sc.tl.pca(adata_sub, n_comps=n_comps)

    hs = hotspot.Hotspot(
        adata_sub,
        layer_key=layer_key,
        model="danb",
        latent_obsm_key="X_pca",
        umi_counts_obs_key="total_counts",
    )
    n_neighbors = min(neighbors, adata_sub.n_obs - 1)
    hs.create_knn_graph(weighted_graph=False, n_neighbors=n_neighbors)

    hs_results = hs.compute_autocorrelations(jobs=jobs)
    hs_results.to_csv(out_dir / f"{sample}_hotspot_autocorrelations.tsv", sep="\t")

    sig_genes = hs_results.loc[hs_results["FDR"] < 0.05].index.tolist()[:1500]
    metadata.update(
        {
            "n_panel_genes_after_filtering": int(adata_sub.n_vars),
            "n_significant_hotspot_genes": len(sig_genes),
            "layer_used": layer_key or "X",
            "latent_space": "X_pca",
            "n_neighbors": int(n_neighbors),
        }
    )
    if len(sig_genes) >= 15:
        hs.compute_local_correlations(sig_genes, jobs=jobs)
        modules = hs.create_modules(
            min_gene_threshold=15,
            core_only=True,
            fdr_threshold=0.05,
        )
        modules.to_csv(out_dir / f"{sample}_gene_modules.tsv", sep="\t")
        module_scores = hs.calculate_module_scores()
        module_scores.to_csv(
            out_dir / f"{sample}_cell_module_scores.tsv",
            sep="\t",
        )
        metadata["module_analysis_run"] = True
    else:
        metadata["module_analysis_run"] = False

    metadata["status"] = "completed"
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Completed {cancer}/{sample}")


def run_sample(args: argparse.Namespace) -> None:
    if not args.cancer or not args.sample:
        raise ValueError("--cancer and --sample are required for analysis.")
    if args.cancer == "Lymph_nodes":
        raise ValueError("Lymph_nodes is explicitly excluded.")

    cancer_dir = ROOT / args.cancer
    h5ad_path = (
        cancer_dir / "adata_by_singlesample_allprobe" / f"{args.sample}.h5ad"
    )
    annotation_path = resolve_annotation_path(args.cancer, args.annotation_path)
    if not h5ad_path.is_file():
        raise FileNotFoundError(h5ad_path)
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)

    panel = {
        line.strip().split("\t")[0]
        for line in PANEL_FILE.read_text().splitlines()
        if line.strip()
    }
    annotation = load_sample_annotation(annotation_path, args.sample)
    adata = sc.read_h5ad(h5ad_path)
    adata, match_key = attach_annotation(adata, annotation)
    n_annotated = int(adata.obs["cell_type_external"].notna().sum())
    tumor_mask = adata.obs["cell_type_external"].astype(str).str.contains(
        args.tumor_regex,
        case=False,
        na=False,
        regex=True,
    )
    n_tumor = int(tumor_mask.sum())
    if n_tumor < args.min_tumor_cells:
        out_dir = args.out_root / args.cancer / args.sample
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "status": "skipped_insufficient_tumor_cells",
                    "cancer": args.cancer,
                    "sample": args.sample,
                    "h5ad_path": str(h5ad_path),
                    "annotation_path": str(annotation_path),
                    "annotation_match_key": match_key,
                    "n_h5ad_cells": int(adata.n_obs),
                    "n_cells_matched_to_external_annotation": n_annotated,
                    "n_tumor_cells": n_tumor,
                    "min_tumor_cells": args.min_tumor_cells,
                    **(
                        {"annotation_version": "updated"}
                        if args.cancer in ANNOTATION_OVERRIDES
                        and args.annotation_path is None
                        else {}
                    ),
                },
                indent=2,
            )
        )
        print(
            f"Skipped {args.cancer}/{args.sample}: {n_tumor} tumor cells "
            f"< {args.min_tumor_cells}."
        )
        return

    adata_sub = adata[tumor_mask].copy()
    del adata
    gc.collect()
    var_mask = panel_var_mask(adata_sub, panel)
    adata_sub = adata_sub[:, var_mask].copy()

    metadata = {
        "cancer": args.cancer,
        "sample": args.sample,
        "h5ad_path": str(h5ad_path),
        "annotation_path": str(annotation_path),
        "annotation_match_key": match_key,
        "n_h5ad_cells": int(len(tumor_mask)),
        "n_cells_matched_to_external_annotation": n_annotated,
        "n_tumor_cells": n_tumor,
        "tumor_regex": args.tumor_regex,
        "n_shared_panel_genes_before_expression_filter": int(adata_sub.n_vars),
    }
    if args.cancer in ANNOTATION_OVERRIDES and args.annotation_path is None:
        metadata["annotation_version"] = "updated"
    run_hotspot(
        adata_sub,
        args.cancer,
        args.sample,
        args.out_root,
        args.min_cells_per_gene,
        args.neighbors,
        args.jobs,
        metadata,
    )


def main() -> None:
    args = parse_args()
    if args.write_task_list is not None:
        write_task_list(args.write_task_list)
        return
    if args.write_pdac_task_list is not None:
        write_pdac_task_list(args.write_pdac_task_list)
        return
    run_sample(args)


if __name__ == "__main__":
    main()
