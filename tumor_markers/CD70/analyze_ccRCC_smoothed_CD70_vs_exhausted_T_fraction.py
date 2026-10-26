#!/usr/bin/env python3

### CD8+ Exhausted T label in this script is effectively noncycling exhausted T.

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import anndata as ad
import decoupler as dc
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


RCC_DIR = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/RCC"
)
ANNOTATION_DIR = (
    RCC_DIR
    / "analysis/ccRCC_CD27_CD70_and_Tcell_subtyping/Tcell_full_harmony/"
    "celltype_annotation_original_res1"
)
DEFAULT_RCC_H5AD = RCC_DIR / "annotated.Yuwei.slim.h5ad"
DEFAULT_TCELL_ANNOTATION = (
    ANNOTATION_DIR / "ccRCC_Tcell_celltype_annotation.tsv.gz"
)
DEFAULT_OUTPUT = (
    ANNOTATION_DIR / "CD70_smoothed_tumor_region_exhausted_T"
)
TARGET_SUBTYPES = {
    "exhausted": {
        "label": "CD8+ Exhausted T",
        "display": "CD8+ Exhausted T",
        "slug": "exhausted_T",
    },
    "proliferative_exhausted": {
        "label": "Proliferative Exhausted T",
        "display": "Proliferative Exhausted T",
        "slug": "proliferative_exhausted_T",
    },
}
CD8_TYPES = {
    "CD8+ T",
    "CD8+ Exhausted T",
    "Proliferative Exhausted T",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rcc-h5ad", type=Path, default=DEFAULT_RCC_H5AD)
    parser.add_argument(
        "--tcell-annotation",
        type=Path,
        default=DEFAULT_TCELL_ANNOTATION,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--target-subtype",
        choices=sorted(TARGET_SUBTYPES),
        default="exhausted",
        help="CD8 subtype used as the numerator; denominator is all CD8 T cells.",
    )
    parser.add_argument("--smooth-k", type=int, default=50)
    parser.add_argument(
        "--smoothing-method",
        choices=["fixed_knn", "sklearn_knn", "decoupler"],
        default="fixed_knn",
        help=(
            "fixed_knn: scipy cKDTree mean of k neighbours. "
            "sklearn_knn: NearestNeighbors kneighbors_graph + l1 mean. "
            "These match when pairwise distances are unique."
        ),
    )
    parser.add_argument("--bandwidth", type=float, default=50)
    parser.add_argument("--cutoff", type=float, default=0.1)
    parser.add_argument("--max-neighbors", type=int, default=100)
    parser.add_argument("--maximum-tumor-distance-um", type=float, default=50)
    parser.add_argument(
        "--split-method",
        choices=["quartiles", "median"],
        default="quartiles",
        help=(
            "Compare top/bottom quartiles (default), or split smoothed "
            "tumor CD70 at each dataset's median."
        ),
    )
    parser.add_argument("--low-quantile", type=float, default=0.25)
    parser.add_argument("--high-quantile", type=float, default=0.75)
    parser.add_argument(
        "--min-cd8-per-region",
        "--min-tcells-per-region",
        dest="min_cd8_per_region",
        type=int,
        default=10,
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def vector(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray().ravel()
    return np.asarray(matrix).ravel()


def smooth_tumor_cd70(
    coordinates: np.ndarray,
    expression: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int]:
    if args.smoothing_method == "decoupler":
        tumor = ad.AnnData(
            X=expression[:, None],
            obs=pd.DataFrame(
                index=[f"tumor_{i}" for i in range(len(expression))]
            ),
            var=pd.DataFrame(index=["CD70"]),
        )
        tumor.obsm["spatial"] = coordinates
        effective_max_neighbors = min(args.max_neighbors, len(tumor))
        dc.pp.knn(
            tumor,
            key="spatial",
            bw=args.bandwidth,
            max_nn=effective_max_neighbors,
            cutoff=args.cutoff,
        )
        smoothed = np.asarray(
            tumor.obsp["spatial_connectivities"].dot(tumor.X)
        ).ravel()
        return smoothed, effective_max_neighbors

    k = args.smooth_k
    effective_k = min(k, len(coordinates))
    if args.smoothing_method == "sklearn_knn":
        adjacency = normalize(
            NearestNeighbors(n_neighbors=effective_k)
            .fit(coordinates)
            .kneighbors_graph(coordinates),
            norm="l1",
            axis=1,
        )
        smoothed = np.asarray(adjacency.dot(np.asarray(expression)[:, None])).ravel()
        return smoothed, effective_k
    tree = cKDTree(coordinates)
    _, positions = tree.query(coordinates, k=effective_k)
    if effective_k == 1:
        positions = positions[:, None]
    return np.mean(expression[positions], axis=1), effective_k


def process_dataset(
    backed: sc.AnnData,
    dataset: str,
    tcell_annotation: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict]:
    dataset_mask = backed.obs["dataset"].astype(str).eq(dataset).to_numpy()
    if not dataset_mask.any():
        raise ValueError(f"{dataset}: absent from RCC h5ad")
    sample = backed[dataset_mask, ["CD70"]].to_memory()
    try:
        required = {"cell_type", "xcoord", "ycoord"}
        if not required.issubset(sample.obs):
            raise KeyError(f"{dataset}: missing obs columns {required}")
        if "counts" not in sample.layers:
            raise KeyError("layers['counts'] is required")
        total_column = next(
            (
                column
                for column in ("total_counts", "n_counts")
                if column in sample.obs
            ),
            None,
        )
        if total_column is None:
            raise KeyError("Neither total_counts nor n_counts is present")

        raw_cd70 = vector(sample.layers["counts"][:, 0]).astype(float)
        totals = sample.obs[total_column].to_numpy(dtype=float)
        log_cp10k = np.zeros_like(raw_cd70)
        valid_totals = totals > 0
        log_cp10k[valid_totals] = np.log1p(
            raw_cd70[valid_totals] / totals[valid_totals] * 1e4
        )
        coordinates = sample.obs[["xcoord", "ycoord"]].to_numpy(dtype=float)
        finite = np.isfinite(coordinates).all(axis=1)
        tumor = (
            sample.obs["cell_type"]
            .astype(str)
            .eq("Tumor_ccRCC")
            .to_numpy()
            & finite
        )
        if not tumor.any():
            raise ValueError(f"{dataset}: no Tumor_ccRCC cells")
        tumor_coordinates = coordinates[tumor]
        tumor_expression = log_cp10k[tumor]
        smoothed, effective_k = smooth_tumor_cd70(
            tumor_coordinates,
            tumor_expression,
            args,
        )
        if float(np.max(smoothed)) <= float(np.min(smoothed)):
            raise ValueError(
                f"{dataset}: smoothed CD70 has insufficient variation"
            )
        if args.split_method == "median":
            low_cutoff = high_cutoff = float(np.median(smoothed))
        else:
            low_cutoff = float(np.quantile(smoothed, args.low_quantile))
            high_cutoff = float(np.quantile(smoothed, args.high_quantile))
            if not high_cutoff > low_cutoff:
                raise ValueError(
                    f"{dataset}: smoothed CD70 has insufficient variation "
                    f"(Qlow={low_cutoff}, Qhigh={high_cutoff})"
                )
        score_mean = float(np.mean(smoothed))
        score_sd = float(np.std(smoothed, ddof=0))

        tcells = tcell_annotation.loc[
            tcell_annotation["dataset"].eq(dataset)
            & tcell_annotation["ccRCC_Tcell_type"].isin(CD8_TYPES)
        ].copy()
        if tcells.empty:
            raise ValueError(f"{dataset}: no annotated CD8 T cells")
        t_coordinates = tcells[["xcoord", "ycoord"]].to_numpy(dtype=float)
        valid_t = np.isfinite(t_coordinates).all(axis=1)
        tcells = tcells.loc[valid_t].copy()
        t_coordinates = t_coordinates[valid_t]

        tree = cKDTree(tumor_coordinates)
        distances, nearest_positions = tree.query(t_coordinates, k=1)
        local_scores = smoothed[nearest_positions]
        tcells["nearest_tumor_distance_um"] = distances
        tcells["local_smoothed_CD70_log1pCP10K"] = local_scores
        tcells["local_smoothed_CD70_zscore"] = (
            (local_scores - score_mean) / score_sd
            if score_sd > 0
            else 0.0
        )
        if args.split_method == "median":
            tcells["CD70_tumor_region"] = np.where(
                local_scores > high_cutoff,
                "CD70-High",
                "CD70-Low",
            )
        else:
            tcells["CD70_tumor_region"] = np.select(
                [
                    local_scores <= low_cutoff,
                    local_scores >= high_cutoff,
                ],
                ["CD70-Low", "CD70-High"],
                default="Middle",
            )
        tcells = tcells.loc[
            tcells["nearest_tumor_distance_um"]
            <= args.maximum_tumor_distance_um
        ].copy()
        tcells["is_target_T"] = (
            tcells["ccRCC_Tcell_type"].eq(args.target_label).astype(int)
        )
        summary = {
            "dataset": dataset,
            "n_tumor_cells": int(tumor.sum()),
            "n_CD70_positive_tumor_cells_raw": int(
                np.sum(tumor_expression > 0)
            ),
            "fraction_CD70_positive_tumor_cells_raw": float(
                np.mean(tumor_expression > 0)
            ),
            "split_method": args.split_method,
            "smoothing_method": args.smoothing_method,
            "smooth_k": effective_k,
            "bandwidth": (
                args.bandwidth
                if args.smoothing_method == "decoupler"
                else np.nan
            ),
            "cutoff": (
                args.cutoff
                if args.smoothing_method == "decoupler"
                else np.nan
            ),
            "smoothed_CD70_low_cutoff": low_cutoff,
            "smoothed_CD70_high_cutoff": high_cutoff,
            "smoothed_CD70_mean": score_mean,
            "smoothed_CD70_sd": score_sd,
            "n_CD8_cells_within_distance": len(tcells),
            "n_CD8_cells_CD70_Low_region": int(
                tcells["CD70_tumor_region"].eq("CD70-Low").sum()
            ),
            "n_CD8_cells_CD70_High_region": int(
                tcells["CD70_tumor_region"].eq("CD70-High").sum()
            ),
        }
        print(
            f"{dataset}: tumor={int(tumor.sum()):,}; "
            f"CD8<=50um={len(tcells):,}; "
            f"Low={summary['n_CD8_cells_CD70_Low_region']:,}; "
            f"High={summary['n_CD8_cells_CD70_High_region']:,}",
            flush=True,
        )
        return tcells, summary
    finally:
        del sample
        gc.collect()


def calculate_dataset_fractions(
    cells: pd.DataFrame,
    min_cells: int,
) -> pd.DataFrame:
    extremes = cells.loc[
        cells["CD70_tumor_region"].isin(["CD70-Low", "CD70-High"])
    ]
    fractions = (
        extremes.groupby(["dataset", "CD70_tumor_region"], observed=True)
        .agg(
            n_all_CD8=("is_target_T", "size"),
            n_target_CD8=("is_target_T", "sum"),
        )
        .reset_index()
    )
    fractions["target_CD8_fraction"] = (
        fractions["n_target_CD8"] / fractions["n_all_CD8"]
    )
    fractions["target_CD8_percentage"] = (
        100 * fractions["target_CD8_fraction"]
    )
    fractions["passes_min_cells"] = fractions["n_all_CD8"] >= min_cells
    return fractions


def paired_test(fractions: pd.DataFrame) -> pd.DataFrame:
    eligible = fractions.loc[fractions["passes_min_cells"]]
    pivot = eligible.pivot(
        index="dataset",
        columns="CD70_tumor_region",
        values="target_CD8_fraction",
    ).dropna(subset=["CD70-Low", "CD70-High"])
    if len(pivot) < 2:
        raise ValueError("Fewer than two paired datasets pass cell thresholds")
    differences = pivot["CD70-High"] - pivot["CD70-Low"]
    try:
        statistic, p_value = wilcoxon(
            pivot["CD70-High"],
            pivot["CD70-Low"],
            alternative="two-sided",
        )
    except ValueError:
        statistic, p_value = np.nan, np.nan
    return pd.DataFrame(
        [
            {
                "comparison": "CD70-High minus CD70-Low",
                "n_paired_datasets": len(pivot),
                "mean_high_fraction": pivot["CD70-High"].mean(),
                "mean_low_fraction": pivot["CD70-Low"].mean(),
                "mean_paired_difference": differences.mean(),
                "median_paired_difference": differences.median(),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_value": p_value,
            }
        ]
    )


def plot_paired_fractions(
    fractions: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    target_display: str,
    target_slug: str,
) -> None:
    eligible = fractions.loc[fractions["passes_min_cells"]]
    pivot = eligible.pivot(
        index="dataset",
        columns="CD70_tumor_region",
        values="target_CD8_percentage",
    ).dropna(subset=["CD70-Low", "CD70-High"])
    values = [
        pivot["CD70-Low"].to_numpy(),
        pivot["CD70-High"].to_numpy(),
    ]
    fig, axis = plt.subplots(figsize=(3.5, 4.8))
    boxes = axis.boxplot(
        values,
        positions=[0, 1],
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black"},
    )
    boxes["boxes"][0].set_facecolor("#4C78A8")
    boxes["boxes"][1].set_facecolor("#D62728")
    for box in boxes["boxes"]:
        box.set_alpha(0.45)
    for _, row in pivot.iterrows():
        axis.plot(
            [0, 1],
            [row["CD70-Low"], row["CD70-High"]],
            color="#999999",
            linewidth=0.8,
            alpha=0.7,
        )
        axis.scatter(
            [0, 1],
            [row["CD70-Low"], row["CD70-High"]],
            color=["#4C78A8", "#D62728"],
            s=25,
            zorder=3,
        )
    result = test.iloc[0]
    axis.set_title(
        f"{target_display} fraction\n"
        f"Paired Wilcoxon P={result['wilcoxon_p_value']:.2g}",
        fontsize=11,
    )
    axis.set_xticks([0, 1])
    axis.set_xticklabels(["CD70-Low", "CD70-High"])
    axis.set_ylabel(f"{target_display} among all CD8 T cells (%)")
    axis.set_xlim(-0.45, 1.45)
    axis.set_ylim(bottom=0)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    base = output_dir / f"{target_slug}_fraction_CD70_high_vs_low_tumor_region"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    target = TARGET_SUBTYPES[args.target_subtype]
    args.target_label = target["label"]
    target_display = target["display"]
    target_slug = target["slug"]
    if (
        args.split_method == "quartiles"
        and not 0 <= args.low_quantile < args.high_quantile <= 1
    ):
        raise ValueError("Require 0 <= low quantile < high quantile <= 1")
    if args.smooth_k < 1:
        raise ValueError("--smooth-k must be positive")
    if args.bandwidth <= 0:
        raise ValueError("--bandwidth must be positive")
    if not 0 <= args.cutoff < 1:
        raise ValueError("--cutoff must be in [0, 1)")
    if args.max_neighbors < 1:
        raise ValueError("--max-neighbors must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    annotation = pd.read_csv(args.tcell_annotation, sep="\t", dtype=str)
    required_annotation = {
        "dataset",
        "ccRCC_Tcell_type",
        "xcoord",
        "ycoord",
    }
    if not required_annotation.issubset(annotation):
        raise KeyError(
            f"Missing annotation columns: {required_annotation - set(annotation)}"
        )
    subtype_counts = annotation["ccRCC_Tcell_type"].value_counts()
    if args.target_label not in subtype_counts:
        raise ValueError(
            f"Target subtype {args.target_label!r} is absent from "
            f"{args.tcell_annotation}. Available subtypes: "
            f"{sorted(subtype_counts.index.tolist())}"
        )
    annotation[["xcoord", "ycoord"]] = annotation[
        ["xcoord", "ycoord"]
    ].astype(float)

    backed = sc.read_h5ad(args.rcc_h5ad, backed="r")
    try:
        if "CD70" not in backed.var_names:
            raise KeyError("CD70 is absent from RCC h5ad")
        frames = []
        summaries = []
        for dataset in sorted(annotation["dataset"].unique()):
            try:
                frame, summary = process_dataset(
                    backed,
                    dataset,
                    annotation,
                    args,
                )
            except ValueError as error:
                if "insufficient variation" not in str(error):
                    raise
                print(f"Skipping {dataset}: {error}", flush=True)
                continue
            frames.append(frame)
            summaries.append(summary)
    finally:
        backed.file.close()

    if not frames:
        raise ValueError("No datasets had sufficient smoothed CD70 variation")
    cells = pd.concat(frames, ignore_index=True)
    cells.to_csv(
        args.output_dir
        / f"Tcells_local_smoothed_tumor_CD70_and_{target_slug}.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(summaries).to_csv(
        args.output_dir / "dataset_CD70_smoothing_summary.tsv",
        sep="\t",
        index=False,
    )
    fractions = calculate_dataset_fractions(
        cells,
        args.min_cd8_per_region,
    )
    fractions.to_csv(
        args.output_dir
        / f"dataset_{target_slug}_fraction_CD70_high_low.tsv",
        sep="\t",
        index=False,
    )
    test = paired_test(fractions)
    test.to_csv(
        args.output_dir
        / f"paired_wilcoxon_CD70_high_low_{target_slug}.tsv",
        sep="\t",
        index=False,
    )
    plot_paired_fractions(
        fractions,
        test,
        args.output_dir,
        args.dpi,
        target_display,
        target_slug,
    )
    print(test.to_string(index=False), flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
