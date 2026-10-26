#!/usr/bin/env python3
"""Test exhausted-CD8 fraction among all CD8 T cells vs tumor distance."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANNOTATION_DIR = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/RCC/analysis/"
    "ccRCC_CD27_CD70_and_Tcell_subtyping/Tcell_full_harmony/"
    "celltype_annotation_original_res1"
)
DEFAULT_INPUT = (
    ANNOTATION_DIR
    / "Tcell_subtype_vs_tumor_distance/"
    "ccRCC_Tcells_nearest_tumor_distance.tsv.gz"
)
DEFAULT_OUTPUT = (
    ANNOTATION_DIR
    / "Exhausted_fraction_among_CD8_vs_tumor_distance"
)
CD8_TYPES = ["CD8+ T", "CD8+ Exhausted T"]
EXHAUSTED_TYPE = "CD8+ Exhausted T"
DISTANCE_LABELS = ["0–10", "10–20", "20–30", "30–40", "40–50"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-cd8-per-dataset-bin", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def calculate_fractions(
    cells: pd.DataFrame,
    min_cd8: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = (
        cells.groupby(["dataset", "distance_bin"], observed=True)
        .agg(
            n_all_CD8=("is_exhausted_CD8", "size"),
            n_exhausted_CD8=("is_exhausted_CD8", "sum"),
        )
        .reset_index()
    )
    counts["exhausted_fraction_among_CD8"] = (
        counts["n_exhausted_CD8"] / counts["n_all_CD8"]
    )
    counts["exhausted_percentage_among_CD8"] = (
        100 * counts["exhausted_fraction_among_CD8"]
    )
    counts["passes_min_CD8"] = counts["n_all_CD8"] >= min_cd8
    summary = (
        counts.loc[counts["passes_min_CD8"]]
        .groupby("distance_bin", observed=True)
        .agg(
            mean_percentage=("exhausted_percentage_among_CD8", "mean"),
            sem_percentage=("exhausted_percentage_among_CD8", "sem"),
            median_percentage=("exhausted_percentage_among_CD8", "median"),
            n_datasets=("dataset", "nunique"),
            total_CD8_cells=("n_all_CD8", "sum"),
        )
        .reset_index()
    )
    return counts, summary


def plot_fraction(
    counts: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
    dpi: int,
) -> None:
    eligible = counts.loc[counts["passes_min_CD8"]]
    summary = summary.set_index("distance_bin").reindex(DISTANCE_LABELS)
    rng = np.random.default_rng(1)
    fig, axis = plt.subplots(figsize=(4.5, 4.8))
    for position, distance_bin in enumerate(DISTANCE_LABELS):
        values = eligible.loc[
            eligible["distance_bin"].astype(str).eq(distance_bin),
            "exhausted_percentage_among_CD8",
        ].to_numpy()
        axis.scatter(
            position + rng.normal(0, 0.045, len(values)),
            values,
            s=22,
            color="#8C564B",
            alpha=0.45,
            edgecolors="none",
        )
    axis.errorbar(
        np.arange(len(DISTANCE_LABELS)),
        summary["mean_percentage"],
        yerr=summary["sem_percentage"],
        color="#8C564B",
        marker="o",
        linewidth=2,
        capsize=3,
    )
    axis.set_title("Exhausted CD8 among all CD8 T cells", fontsize=11)
    axis.set_xticks(range(len(DISTANCE_LABELS)))
    axis.set_xticklabels(DISTANCE_LABELS, rotation=35, ha="right")
    axis.set_xlabel("Nearest Tumor_ccRCC distance (µm)")
    axis.set_ylabel("CD8+ Exhausted T / all CD8 T cells (%)")
    axis.set_ylim(bottom=0)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    base = (
        output_dir
        / "exhausted_CD8_fraction_among_all_CD8_vs_tumor_distance"
    )
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(args.input, sep="\t")
    required = {
        "dataset",
        "ccRCC_Tcell_type",
        "distance_bin",
        "nearest_Tumor_ccRCC_distance_um",
    }
    if not required.issubset(cells):
        raise KeyError(f"Missing input columns: {required - set(cells)}")
    cells = cells.loc[
        cells["ccRCC_Tcell_type"].isin(CD8_TYPES)
        & cells["distance_bin"].isin(DISTANCE_LABELS)
    ].copy()
    if cells.empty:
        raise ValueError("No CD8 T cells within 0-50 um were found")
    cells["distance_bin"] = pd.Categorical(
        cells["distance_bin"],
        categories=DISTANCE_LABELS,
        ordered=True,
    )
    cells["is_exhausted_CD8"] = (
        cells["ccRCC_Tcell_type"].eq(EXHAUSTED_TYPE).astype(int)
    )
    cells.to_csv(
        args.output_dir / "CD8_cells_within_50um.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )

    counts, summary = calculate_fractions(
        cells,
        args.min_cd8_per_dataset_bin,
    )
    counts.to_csv(
        args.output_dir
        / "exhausted_CD8_fraction_by_dataset_distance_bin.tsv",
        sep="\t",
        index=False,
    )
    summary.to_csv(
        args.output_dir
        / "exhausted_CD8_fraction_distance_bin_dataset_balanced_summary.tsv",
        sep="\t",
        index=False,
    )
    plot_fraction(counts, summary, args.output_dir, args.dpi)
    print(summary.to_string(index=False), flush=True)
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
