#!/usr/bin/env python3
"""Analyze exhausted/proliferative CD8 fractions and CD27 expression."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


BASE_DIR = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/RCC/analysis/"
    "ccRCC_CD27_CD70_and_Tcell_subtyping/Tcell_full_harmony/"
    "celltype_annotation_original_res1_cluster12_proliferative_exhausted"
)
DEFAULT_DISTANCE_INPUT = (
    BASE_DIR
    / "Tcell_subtype_vs_tumor_distance/"
    "ccRCC_Tcells_nearest_tumor_distance.tsv.gz"
)
DEFAULT_ANNOTATION = BASE_DIR / "ccRCC_Tcell_celltype_annotation.tsv"
DEFAULT_H5AD = BASE_DIR / "ccRCC_Tcell_celltype_annotated.h5ad"
DEFAULT_OUTPUT = (
    BASE_DIR
    / "Exhausted_and_proliferative_exhausted_among_CD8_vs_tumor_distance"
)

CD8_TYPES = [
    "CD8+ T",
    "CD8+ Exhausted T",
    "Proliferative Exhausted T",
]
OUTCOMES = {
    "Cytotoxic CD8+ T": ["CD8+ T"],
    "CD8+ Exhausted T": ["CD8+ Exhausted T"],
    "Proliferative Exhausted T": ["Proliferative Exhausted T"],
    "Combined exhausted CD8": [
        "CD8+ Exhausted T",
        "Proliferative Exhausted T",
    ],
}
DISTANCE_LABELS = ["0–10", "10–20", "20–30", "30–40", "40–50"]
OUTCOME_COLORS = {
    "Cytotoxic CD8+ T": "#1F77B4",
    "CD8+ Exhausted T": "#8C564B",
    "Proliferative Exhausted T": "#9467BD",
    "Combined exhausted CD8": "#D62728",
}
CELLTYPE_ORDER = [
    "Naive T",
    "Treg",
    "CD8+ T",
    "CD8+ Exhausted T",
    "Proliferative Exhausted T",
    "IFN-T",
    "Other T",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--distance-input",
        type=Path,
        default=DEFAULT_DISTANCE_INPUT,
    )
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-cd8-per-dataset-bin", type=int, default=10)
    parser.add_argument("--figure-width", type=float, default=7)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def dense_vector(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray().ravel()
    return np.asarray(matrix).ravel()


def calculate_distance_fractions(
    cells: pd.DataFrame,
    min_cd8: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (dataset, distance_bin), frame in cells.groupby(
        ["dataset", "distance_bin"],
        observed=True,
    ):
        for outcome, positive_types in OUTCOMES.items():
            n_positive = int(
                frame["ccRCC_Tcell_type"].isin(positive_types).sum()
            )
            rows.append(
                {
                    "dataset": dataset,
                    "distance_bin": distance_bin,
                    "outcome": outcome,
                    "n_all_CD8": len(frame),
                    "n_positive": n_positive,
                    "percentage_among_all_CD8": 100 * n_positive / len(frame),
                    "passes_min_CD8": len(frame) >= min_cd8,
                }
            )
    counts = pd.DataFrame(rows)
    counts["distance_bin"] = pd.Categorical(
        counts["distance_bin"],
        categories=DISTANCE_LABELS,
        ordered=True,
    )
    summary = (
        counts.loc[counts["passes_min_CD8"]]
        .groupby(["distance_bin", "outcome"], observed=True)
        .agg(
            mean_percentage=("percentage_among_all_CD8", "mean"),
            sem_percentage=("percentage_among_all_CD8", "sem"),
            median_percentage=("percentage_among_all_CD8", "median"),
            n_datasets=("dataset", "nunique"),
            total_CD8_cells=("n_all_CD8", "sum"),
        )
        .reset_index()
    )
    return counts, summary


def plot_distance_fractions(
    counts: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
    figure_width: float = 7,
) -> None:
    compact = figure_width <= 4.5
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(figure_width, 7.6),
        squeeze=False,
    )
    rng = np.random.default_rng(1)
    for axis, outcome in zip(axes.flat, OUTCOMES):
        eligible = counts.loc[
            counts["passes_min_CD8"] & counts["outcome"].eq(outcome)
        ]
        outcome_summary = (
            summary.loc[summary["outcome"].eq(outcome)]
            .set_index("distance_bin")
            .reindex(DISTANCE_LABELS)
        )
        for position, distance_bin in enumerate(DISTANCE_LABELS):
            values = eligible.loc[
                eligible["distance_bin"].astype(str).eq(distance_bin),
                "percentage_among_all_CD8",
            ].to_numpy()
            axis.scatter(
                position + rng.normal(0, 0.045, len(values)),
                values,
                s=18,
                color=OUTCOME_COLORS[outcome],
                alpha=0.4,
                edgecolors="none",
            )
        axis.errorbar(
            np.arange(len(DISTANCE_LABELS)),
            outcome_summary["mean_percentage"],
            yerr=outcome_summary["sem_percentage"],
            color=OUTCOME_COLORS[outcome],
            marker="o",
            linewidth=2,
            capsize=3,
        )
        axis.set_title(outcome, fontsize=7.5 if compact else 10)
        axis.set_xticks(range(len(DISTANCE_LABELS)))
        axis.set_xticklabels(DISTANCE_LABELS, rotation=35, ha="right")
        axis.tick_params(labelsize=7 if compact else None)
        axis.set_xlabel(
            "" if compact else "Nearest Tumor_ccRCC distance (µm)"
        )
        axis.set_ylabel(
            "" if compact else "Fraction among all CD8 T cells (%)"
        )
        if outcome == "Cytotoxic CD8+ T":
            # Coordinate zoom only; underlying values are not filtered.
            axis.set_ylim(bottom=60)
        else:
            axis.set_ylim(bottom=0)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes.flat[len(OUTCOMES) :]:
        axis.set_visible(False)
    if compact:
        fig.supxlabel("Nearest Tumor_ccRCC distance (µm)", fontsize=9)
        fig.supylabel("Fraction among all CD8 T cells (%)", fontsize=9)
        fig.tight_layout(rect=(0.04, 0.04, 1, 1), w_pad=0.6)
    else:
        fig.tight_layout()
    base = output_dir / "exhausted_CD8_fractions_vs_tumor_distance"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_stacked_cd8_composition(
    summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    states = [
        "Cytotoxic CD8+ T",
        "CD8+ Exhausted T",
        "Proliferative Exhausted T",
    ]
    composition = (
        summary.loc[summary["outcome"].isin(states)]
        .pivot(
            index="distance_bin",
            columns="outcome",
            values="mean_percentage",
        )
        .reindex(index=DISTANCE_LABELS, columns=states)
        .fillna(0)
    )
    # Ensure each dataset-balanced bar displays exactly 100%.
    composition = composition.div(composition.sum(axis=1), axis=0) * 100

    fig, axis = plt.subplots(figsize=(5.2, 4.8))
    bottom = np.zeros(len(composition))
    for state in states:
        values = composition[state].to_numpy()
        bars = axis.bar(
            DISTANCE_LABELS,
            values,
            bottom=bottom,
            width=0.72,
            color=OUTCOME_COLORS[state],
            label=state,
        )
        for bar, value, base in zip(bars, values, bottom):
            if value >= 4:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    base + value / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white",
                    fontweight="bold",
                )
        bottom += values
    axis.set_ylim(0, 100)
    axis.set_ylabel("CD8 T-cell state composition (%)")
    axis.set_xlabel("Nearest Tumor_ccRCC distance (µm)")
    axis.tick_params(axis="x", rotation=35)
    axis.legend(
        frameon=False,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    base = (
        output_dir
        / "CD8_state_composition_stacked_percentage_by_tumor_distance"
    )
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def count_by_sample(
    annotation: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    selected = annotation.loc[
        annotation["ccRCC_Tcell_type"].isin(
            ["CD8+ Exhausted T", "Proliferative Exhausted T"]
        )
    ]
    counts = (
        selected.groupby(["dataset", "ccRCC_Tcell_type"])
        .size()
        .unstack(fill_value=0)
        .reindex(
            columns=["CD8+ Exhausted T", "Proliferative Exhausted T"],
            fill_value=0,
        )
    )
    total_t = annotation.groupby("dataset").size().rename("n_all_T_cells")
    total_cd8 = (
        annotation.loc[annotation["ccRCC_Tcell_type"].isin(CD8_TYPES)]
        .groupby("dataset")
        .size()
        .rename("n_all_CD8")
    )
    result = counts.join(total_t).join(total_cd8).fillna(0)
    result["combined_exhausted_count"] = (
        result["CD8+ Exhausted T"]
        + result["Proliferative Exhausted T"]
    )
    result["exhausted_pct_all_T"] = (
        100 * result["CD8+ Exhausted T"] / result["n_all_T_cells"]
    )
    result["proliferative_exhausted_pct_all_T"] = (
        100
        * result["Proliferative Exhausted T"]
        / result["n_all_T_cells"]
    )
    result["combined_exhausted_pct_all_CD8"] = (
        100 * result["combined_exhausted_count"] / result["n_all_CD8"]
    )
    result = result.sort_values(
        "combined_exhausted_count",
        ascending=False,
    )
    result.to_csv(
        output_dir / "exhausted_T_counts_and_fractions_by_sample.tsv",
        sep="\t",
    )

    plot_data = result.iloc[::-1]
    fig, axis = plt.subplots(
        figsize=(6.2, max(4.5, 0.42 * len(plot_data)))
    )
    axis.barh(
        plot_data.index,
        plot_data["CD8+ Exhausted T"],
        color="#8C564B",
        label="CD8+ Exhausted T",
    )
    axis.barh(
        plot_data.index,
        plot_data["Proliferative Exhausted T"],
        left=plot_data["CD8+ Exhausted T"],
        color="#9467BD",
        label="Proliferative Exhausted T",
    )
    axis.set_xlabel("Number of cells")
    axis.legend(frameon=False)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    base = output_dir / "exhausted_T_cell_counts_by_sample"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def analyze_cd27(
    h5ad_path: Path,
    output_dir: Path,
) -> None:
    backed = sc.read_h5ad(h5ad_path, backed="r")
    try:
        if "CD27" not in backed.var_names:
            raise KeyError("CD27 is absent from h5ad")
        adata = backed[:, ["CD27"]].to_memory()
    finally:
        backed.file.close()
    try:
        observed_types = [
            subtype
            for subtype in CELLTYPE_ORDER
            if subtype in set(adata.obs["ccRCC_Tcell_type"].astype(str))
        ]
        adata.obs["ccRCC_Tcell_type"] = pd.Categorical(
            adata.obs["ccRCC_Tcell_type"].astype(str),
            categories=observed_types,
            ordered=True,
        )
        expression = dense_vector(adata.X).astype(float)
        rows = []
        for subtype in observed_types:
            mask = (
                adata.obs["ccRCC_Tcell_type"]
                .astype(str)
                .eq(subtype)
                .to_numpy()
            )
            values = expression[mask]
            rows.append(
                {
                    "ccRCC_Tcell_type": subtype,
                    "n_cells": len(values),
                    "mean_CD27_expression": float(np.mean(values)),
                    "median_CD27_expression": float(np.median(values)),
                    "fraction_CD27_positive": float(np.mean(values > 0)),
                }
            )
        pd.DataFrame(rows).to_csv(
            output_dir / "CD27_expression_by_Tcell_subtype.tsv",
            sep="\t",
            index=False,
        )

        dataset_rows = []
        for (dataset, subtype), indices in adata.obs.groupby(
            ["dataset", "ccRCC_Tcell_type"],
            observed=True,
        ).indices.items():
            values = expression[np.asarray(indices)]
            dataset_rows.append(
                {
                    "dataset": dataset,
                    "ccRCC_Tcell_type": subtype,
                    "n_cells": len(values),
                    "mean_CD27_expression": float(np.mean(values)),
                    "fraction_CD27_positive": float(np.mean(values > 0)),
                }
            )
        pd.DataFrame(dataset_rows).to_csv(
            output_dir / "CD27_expression_by_dataset_Tcell_subtype.tsv",
            sep="\t",
            index=False,
        )

        for standardized in (False, True):
            dotplot = sc.pl.dotplot(
                adata,
                var_names=["CD27"],
                groupby="ccRCC_Tcell_type",
                categories_order=observed_types,
                standard_scale="var" if standardized else None,
                use_raw=False,
                dendrogram=False,
                show=False,
                return_fig=True,
            )
            suffix = "standardized_by_var" if standardized else "unscaled"
            dotplot.savefig(
                output_dir
                / f"dotplot_CD27_by_Tcell_subtype_{suffix}.pdf"
            )
            plt.close("all")
    finally:
        del adata


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    distances = pd.read_csv(args.distance_input, sep="\t")
    distances = distances.loc[
        distances["ccRCC_Tcell_type"].isin(CD8_TYPES)
        & distances["distance_bin"].isin(DISTANCE_LABELS)
    ].copy()
    if distances.empty:
        raise ValueError("No CD8 cells within 0-50 um were found")
    distances["distance_bin"] = pd.Categorical(
        distances["distance_bin"],
        categories=DISTANCE_LABELS,
        ordered=True,
    )
    counts, summary = calculate_distance_fractions(
        distances,
        args.min_cd8_per_dataset_bin,
    )
    counts.to_csv(
        args.output_dir
        / "exhausted_fractions_by_dataset_distance_bin.tsv",
        sep="\t",
        index=False,
    )
    summary.to_csv(
        args.output_dir
        / "exhausted_fractions_distance_bin_dataset_balanced_summary.tsv",
        sep="\t",
        index=False,
    )
    plot_distance_fractions(
        counts,
        summary,
        args.output_dir,
        args.dpi,
        args.figure_width,
    )
    plot_stacked_cd8_composition(summary, args.output_dir, args.dpi)

    annotation = pd.read_csv(args.annotation, sep="\t")
    count_by_sample(annotation, args.output_dir, args.dpi)
    analyze_cd27(args.h5ad, args.output_dir)
    print(summary.to_string(index=False), flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
