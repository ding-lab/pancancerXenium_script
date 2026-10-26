#!/usr/bin/env python3
"""Compare immune composition across distances from ANXA8- and MUC5B-high tumor cells.

Two complementary analyses are exported:

1. Cumulative neighborhoods: all non-tumor cells within <=25, <=50, ...,
   <=150 micrometers of each marker-high tumor population. A cell can
   contribute to both populations if the two neighborhoods overlap.
2. Exclusive annuli: each non-tumor cell is assigned to whichever marker-high
   tumor population is closer, then binned into 0-25, 25-50, ..., 125-150
   micrometer annuli. This avoids double counting between marker populations.

Fractions are reported using both all nearby non-tumor cells and the combined
DC + macrophage + NK/T population as denominators.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from deseq2_anxa8_vs_muc5b_positive_exclusive import read_h5ad_vector


ROOT = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/01_analysis/00_tumor_gene_module/plot/"
    "tumor_cancersea_pearson_results"
)
RESULT_ROOT = ROOT / "ANXA8_vs_MUC5B_Niche_Results_k10"
DEFAULT_H5AD = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/PDAC/annotated.Yuwei.updated.h5ad"
)
DEFAULT_OUTPUT = RESULT_ROOT / "distance_gradient_celltype_composition"
RADII_UM = np.asarray([25, 50, 75, 100, 125, 150], dtype=float)
MARKER_GROUPS = ("MUC5B_High", "ANXA8_High")
TARGET_CELL_TYPES = ("DC", "Macrophage", "NK_T")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument(
        "--anxa8-cells",
        type=Path,
        default=RESULT_ROOT / "ANXA8_High_cells.csv",
    )
    parser.add_argument(
        "--muc5b-cells",
        type=Path,
        default=RESULT_ROOT / "MUC5B_High_cells.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-column", default="dataset")
    parser.add_argument("--cell-type-column", default="cell_type")
    parser.add_argument(
        "--spatial-key",
        default="spatial",
        help="Coordinate key in adata.obsm; X_spatial is used as fallback",
    )
    parser.add_argument(
        "--coordinate-scale-to-um",
        type=float,
        default=1.0,
        help="Multiply coordinates by this value to obtain micrometers",
    )
    parser.add_argument(
        "--min-denominator-cells",
        type=int,
        default=20,
        help="Minimum denominator required to report a fraction",
    )
    return parser.parse_args()


def normalize_label(value: object) -> str:
    label = str(value).strip().casefold()
    return re.sub(r"[^a-z0-9]+", "_", label).strip("_")


def broad_cell_class(value: object) -> str:
    """Map refined annotation labels to DC, macrophage, NK/T, or Other."""
    label = normalize_label(value)
    tokens = set(label.split("_"))

    dc_exact = {
        "dc",
        "cdc1",
        "cdc2",
        "mregdc",
        "pdc",
        "dendritic",
        "dendriticcell",
        "dendriticcells",
    }
    if (
        label in dc_exact
        or bool(tokens & {"cdc1", "cdc2", "mregdc", "pdc"})
        or "dendritic" in label
        or re.search(r"(^|_)dc($|_)", label)
    ):
        return "DC"

    if (
        "macrophage" in label
        or label in {"mac", "macs"}
        or re.search(r"(^|_)mac($|_)", label)
        or ("mono" in label and "mac" in label)
    ):
        return "Macrophage"

    nk_t_exact = {
        "t",
        "tcell",
        "tcells",
        "t_cell",
        "t_cells",
        "nk",
        "nkt",
        "nk_t",
        "nk_t_cell",
        "nk_t_cells",
        "natural_killer",
        "natural_killer_cell",
        "natural_killer_cells",
    }
    nk_t_tokens = {
        "cd4",
        "cd8",
        "treg",
        "tregs",
        "nkt",
        "nk",
        "tcell",
        "tcells",
    }
    if (
        label in nk_t_exact
        or bool(tokens & nk_t_tokens)
        or label.startswith(("cd4_t", "cd8_t", "t_cell", "nk_t"))
        or "natural_killer" in label
    ):
        return "NK_T"

    return "Other"


def load_cell_ids(path: Path) -> set[str]:
    table = pd.read_csv(path, dtype=str)
    if "cell_id" not in table.columns:
        raise KeyError(f"{path} does not contain a cell_id column")
    return set(table["cell_id"].dropna().astype(str))


def safe_fraction(numerator: int, denominator: int, minimum: int) -> float:
    if denominator < minimum:
        return np.nan
    return numerator / denominator


def summarize_region(
    obs: pd.DataFrame,
    sample: str,
    marker_group: str,
    distance_type: str,
    radius_um: float,
    inner_radius_um: float,
    mask: np.ndarray,
    minimum: int,
) -> list[dict[str, object]]:
    region = obs.loc[mask]
    n_all_nontumor = len(region)
    selected = region["broad_cell_class"].isin(TARGET_CELL_TYPES)
    n_selected_immune = int(selected.sum())
    rows = []
    for cell_type in TARGET_CELL_TYPES:
        numerator = int(region["broad_cell_class"].eq(cell_type).sum())
        rows.append(
            {
                "sample_id": sample,
                "marker_group": marker_group,
                "distance_type": distance_type,
                "inner_radius_um": inner_radius_um,
                "outer_radius_um": radius_um,
                "cell_type": cell_type,
                "n_cell_type": numerator,
                "n_all_nontumor": n_all_nontumor,
                "n_selected_immune": n_selected_immune,
                "fraction_all_nontumor": safe_fraction(
                    numerator,
                    n_all_nontumor,
                    minimum,
                ),
                "fraction_selected_immune": safe_fraction(
                    numerator,
                    n_selected_immune,
                    minimum,
                ),
            }
        )
    return rows


def paired_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [
        "distance_type",
        "inner_radius_um",
        "outer_radius_um",
        "cell_type",
        "fraction_definition",
    ]
    for keys, group in metrics.groupby(group_columns, dropna=False):
        wide = group.pivot_table(
            index="sample_id",
            columns="marker_group",
            values="fraction",
            aggfunc="first",
        )
        if not set(MARKER_GROUPS).issubset(wide.columns):
            continue
        wide = wide.dropna(subset=list(MARKER_GROUPS))
        differences = (
            wide["ANXA8_High"] - wide["MUC5B_High"]
        ).to_numpy()
        if len(wide) < 3:
            statistic, pvalue = np.nan, np.nan
        elif np.allclose(differences, 0):
            statistic, pvalue = 0.0, 1.0
        else:
            statistic, pvalue = wilcoxon(
                wide["ANXA8_High"],
                wide["MUC5B_High"],
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            )
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "n_paired_samples": len(wide),
                "mean_ANXA8_High": wide["ANXA8_High"].mean(),
                "mean_MUC5B_High": wide["MUC5B_High"].mean(),
                "mean_delta_ANXA8_minus_MUC5B": differences.mean(),
                "wilcoxon_statistic": statistic,
                "wilcoxon_pvalue": pvalue,
            }
        )

    result = pd.DataFrame(rows)
    result["wilcoxon_fdr_bh"] = np.nan
    if result.empty:
        return result

    fdr_groups = ["cell_type", "distance_type", "fraction_definition"]
    for _, idx in result.groupby(fdr_groups, dropna=False).groups.items():
        valid = result.loc[idx, "wilcoxon_pvalue"].notna()
        valid_idx = result.loc[idx].index[valid]
        if len(valid_idx) == 0:
            continue
        result.loc[valid_idx, "wilcoxon_fdr_bh"] = multipletests(
            result.loc[valid_idx, "wilcoxon_pvalue"],
            method="fdr_bh",
        )[1]
    return result


def plot_gradient(
    metrics: pd.DataFrame,
    distance_type: str,
    fraction_definition: str,
    output_file: Path,
) -> None:
    data = metrics.loc[
        metrics["distance_type"].eq(distance_type)
        & metrics["fraction_definition"].eq(fraction_definition)
    ].copy()
    data["marker_group_label"] = data["marker_group"].replace(
        {
            "ANXA8_High": "Near ANXA8-high tumor",
            "MUC5B_High": "Near MUC5B-high tumor",
        }
    )

    colors = {
        "Near ANXA8-high tumor": "#D62728",
        "Near MUC5B-high tumor": "#1F77B4",
    }
    g = sns.relplot(
        data=data,
        x="outer_radius_um",
        y="fraction",
        hue="marker_group_label",
        col="cell_type",
        col_order=list(TARGET_CELL_TYPES),
        kind="line",
        estimator="mean",
        errorbar="se",
        marker="o",
        palette=colors,
        height=4.2,
        aspect=1.0,
        facet_kws={"sharey": False},
    )
    for axis, cell_type in zip(g.axes.flat, TARGET_CELL_TYPES):
        sample_data = data.loc[data["cell_type"].eq(cell_type)]
        sns.lineplot(
            data=sample_data,
            x="outer_radius_um",
            y="fraction",
            units="sample_id",
            hue="marker_group_label",
            estimator=None,
            palette=colors,
            alpha=0.18,
            linewidth=0.7,
            legend=False,
            ax=axis,
        )
        axis.set_title(cell_type)
        axis.set_xticks(RADII_UM)
        axis.set_xlabel(
            "Outer radius (µm)"
            if distance_type == "exclusive_annulus"
            else "Cumulative radius (µm)"
        )
        axis.set_ylabel("Cell fraction")
        axis.grid(axis="y", alpha=0.25)

    title_scope = (
        "exclusive nearest-group annuli"
        if distance_type == "exclusive_annulus"
        else "overlapping cumulative neighborhoods"
    )
    denominator = (
        "all non-tumor cells"
        if fraction_definition == "fraction_all_nontumor"
        else "DC + macrophage + NK/T cells"
    )
    g.fig.suptitle(
        f"Immune distance gradients: {title_scope}\n"
        f"Fraction denominator: {denominator}",
        y=1.06,
    )
    g.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(g.fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    anxa8_ids = load_cell_ids(args.anxa8_cells)
    muc5b_ids = load_cell_ids(args.muc5b_cells)

    print(f"Reading annotations and coordinates from {args.h5ad}")
    with h5py.File(args.h5ad, "r") as handle:
        obs_group = handle["obs"]
        for column in (args.sample_column, args.cell_type_column):
            if column not in obs_group:
                raise KeyError(f"adata.obs is missing {column!r}")

        index_key = obs_group.attrs["_index"]
        if isinstance(index_key, bytes):
            index_key = index_key.decode("utf-8")

        obs = pd.DataFrame(
            {
                args.sample_column: read_h5ad_vector(
                    obs_group, args.sample_column
                ),
                args.cell_type_column: read_h5ad_vector(
                    obs_group, args.cell_type_column
                ),
            },
            index=read_h5ad_vector(obs_group, index_key),
        )
        obs.index = obs.index.astype(str)

        spatial_key = args.spatial_key
        if spatial_key not in handle["obsm"]:
            if "X_spatial" in handle["obsm"]:
                spatial_key = "X_spatial"
            else:
                raise KeyError(
                    f"Neither {args.spatial_key!r} nor 'X_spatial' "
                    "exists in adata.obsm"
                )
        coords = (
            np.asarray(handle["obsm"][spatial_key][:], dtype=float)
            * args.coordinate_scale_to_um
        )

    obs["broad_cell_class"] = obs[args.cell_type_column].map(
        broad_cell_class
    )
    obs["is_tumor"] = obs[args.cell_type_column].astype(str).eq("Tumor")

    if not (
        obs.index.isin(anxa8_ids).any()
        or obs.index.isin(muc5b_ids).any()
    ):
        raise ValueError(
            "No marker-high cell IDs matched the h5ad observation names"
        )
    samples = sorted(
        set(obs.loc[obs.index.isin(anxa8_ids), args.sample_column].astype(str))
        | set(obs.loc[obs.index.isin(muc5b_ids), args.sample_column].astype(str))
    )

    mapping = (
        obs.groupby([args.cell_type_column, "broad_cell_class"], dropna=False)
        .size()
        .reset_index(name="n_cells")
        .sort_values(["broad_cell_class", "n_cells"], ascending=[True, False])
    )
    mapping.to_csv(
        args.output_dir / "cell_type_to_broad_class_mapping.tsv",
        sep="\t",
        index=False,
    )

    metric_rows = []
    inventory_rows = []
    for sample in samples:
        sample_mask = obs[args.sample_column].astype(str).eq(sample).to_numpy()
        sample_obs = obs.loc[sample_mask].copy()
        sample_coords = coords[sample_mask]
        sample_ids = sample_obs.index

        anxa8_seed_mask = sample_ids.isin(anxa8_ids)
        muc5b_seed_mask = sample_ids.isin(muc5b_ids)
        candidate_mask = ~sample_obs["is_tumor"].to_numpy()
        inventory_rows.append(
            {
                "sample_id": sample,
                "n_cells": len(sample_obs),
                "n_non_tumor_candidates": int(candidate_mask.sum()),
                "n_ANXA8_High_tumor_seeds": int(anxa8_seed_mask.sum()),
                "n_MUC5B_High_tumor_seeds": int(muc5b_seed_mask.sum()),
            }
        )
        if (
            anxa8_seed_mask.sum() == 0
            or muc5b_seed_mask.sum() == 0
            or candidate_mask.sum() == 0
        ):
            print(f"[skip] {sample}: missing seeds or non-tumor candidates")
            continue

        candidate_obs = sample_obs.loc[candidate_mask].copy()
        candidate_coords = sample_coords[candidate_mask]
        distances = {}
        for marker_group, seed_mask in (
            ("ANXA8_High", anxa8_seed_mask),
            ("MUC5B_High", muc5b_seed_mask),
        ):
            tree = cKDTree(sample_coords[seed_mask])
            distances[marker_group] = tree.query(
                candidate_coords,
                k=1,
                workers=-1,
            )[0]

            for radius in RADII_UM:
                metric_rows.extend(
                    summarize_region(
                        candidate_obs,
                        sample,
                        marker_group,
                        "cumulative",
                        radius,
                        0.0,
                        distances[marker_group] <= radius,
                        args.min_denominator_cells,
                    )
                )

        distance_matrix = np.column_stack(
            [distances["MUC5B_High"], distances["ANXA8_High"]]
        )
        nearest_index = np.argmin(distance_matrix, axis=1)
        nearest_distance = np.min(distance_matrix, axis=1)
        ties = np.isclose(
            distance_matrix[:, 0],
            distance_matrix[:, 1],
        )
        for marker_index, marker_group in enumerate(MARKER_GROUPS):
            assigned = (nearest_index == marker_index) & ~ties
            inner_radius = 0.0
            for outer_radius in RADII_UM:
                annulus_mask = (
                    assigned
                    & (nearest_distance > inner_radius)
                    & (nearest_distance <= outer_radius)
                )
                metric_rows.extend(
                    summarize_region(
                        candidate_obs,
                        sample,
                        marker_group,
                        "exclusive_annulus",
                        outer_radius,
                        inner_radius,
                        annulus_mask,
                        args.min_denominator_cells,
                    )
                )
                inner_radius = outer_radius

    metrics = pd.DataFrame(metric_rows)
    if metrics.empty:
        raise RuntimeError("No distance-gradient metrics were generated")
    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(
        args.output_dir / "sample_seed_and_candidate_inventory.tsv",
        sep="\t",
        index=False,
    )
    metrics.to_csv(
        args.output_dir / "distance_gradient_counts_and_fractions.tsv",
        sep="\t",
        index=False,
    )

    long_metrics = metrics.melt(
        id_vars=[
            "sample_id",
            "marker_group",
            "distance_type",
            "inner_radius_um",
            "outer_radius_um",
            "cell_type",
            "n_cell_type",
            "n_all_nontumor",
            "n_selected_immune",
        ],
        value_vars=[
            "fraction_all_nontumor",
            "fraction_selected_immune",
        ],
        var_name="fraction_definition",
        value_name="fraction",
    )
    tests = paired_tests(long_metrics)
    tests.to_csv(
        args.output_dir / "distance_gradient_paired_wilcoxon.tsv",
        sep="\t",
        index=False,
    )

    for distance_type in ("cumulative", "exclusive_annulus"):
        for fraction_definition in (
            "fraction_all_nontumor",
            "fraction_selected_immune",
        ):
            plot_gradient(
                long_metrics,
                distance_type,
                fraction_definition,
                args.output_dir
                / (
                    f"{distance_type}_{fraction_definition}_"
                    "DC_Macrophage_NKT_facets.pdf"
                ),
            )

    pd.DataFrame(
        {
            "parameter": [
                "h5ad",
                "spatial_key",
                "coordinate_scale_to_um",
                "radii_um",
                "candidate_cells",
                "cumulative_overlap",
                "exclusive_annulus_assignment",
                "minimum_denominator_cells",
            ],
            "value": [
                str(args.h5ad),
                spatial_key,
                args.coordinate_scale_to_um,
                ",".join(str(int(radius)) for radius in RADII_UM),
                "all cells whose cell_type does not contain Tumor",
                "allowed",
                "assigned to nearest marker-high tumor group; ties excluded",
                args.min_denominator_cells,
            ],
        }
    ).to_csv(
        args.output_dir / "analysis_parameters.tsv",
        sep="\t",
        index=False,
    )
    print(f"Results written to {args.output_dir}")


if __name__ == "__main__":
    main()
