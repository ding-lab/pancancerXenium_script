#!/usr/bin/env python3
"""Quantify overlap between ANXA8>0 and MUC5B>0 PDAC tumor cells."""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import sparse
from scipy.stats import fisher_exact


DEFAULT_H5AD = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/PDAC/annotated.Yuwei.updated.h5ad"
)
OUTPUT_DIR = Path(__file__).resolve().parent / (
    "ANXA8_MUC5B_positive_overlap"
)

TARGET_SAMPLES = [
    "C3N-02295-14Fp1U1",
    "C3L-03395-13Fp1U1",
    "HT227P1-S1H1L1U2",
    "HT1046P1-S1H1Fp1U1",
    "HT910P1-S1H1Fp1U1",
    "HT700P1-S1H1Fp1U1",
    "HT535P1-S1H1Fp1U1",
    "HT943P1-S1H2Fp1U1",
    "HT873P1-S1H1Fp1U2",
    "C3L-03394-13Fp1U1",
    "HT608P1-S1H2Fp1U1",
    "C3N-02589-11Fp1U1",
    "HT124P1-S1H3A1Fp1U1",
    "SP041P1-A20-S1Fp1U3_2",
    "C3L-02116-11Fp1U1",
    "SP090P1-A3-S1Fp1U1",
]


def to_dense(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def summarize(
    sample: str,
    anxa8_positive: np.ndarray,
    muc5b_positive: np.ndarray,
) -> dict[str, object]:
    both = int(np.sum(anxa8_positive & muc5b_positive))
    anxa8_only = int(np.sum(anxa8_positive & ~muc5b_positive))
    muc5b_only = int(np.sum(~anxa8_positive & muc5b_positive))
    neither = int(np.sum(~anxa8_positive & ~muc5b_positive))
    total = both + anxa8_only + muc5b_only + neither
    n_anxa8 = both + anxa8_only
    n_muc5b = both + muc5b_only
    union = both + anxa8_only + muc5b_only
    expected_both = n_anxa8 * n_muc5b / total if total else np.nan
    contingency = [[both, anxa8_only], [muc5b_only, neither]]
    odds_ratio, p_two_sided = fisher_exact(
        contingency,
        alternative="two-sided",
    )
    _, p_depletion = fisher_exact(contingency, alternative="less")

    return {
        "sample_id": sample,
        "n_tumor_cells": total,
        "n_ANXA8_positive": n_anxa8,
        "n_MUC5B_positive": n_muc5b,
        "n_double_positive": both,
        "n_ANXA8_only": anxa8_only,
        "n_MUC5B_only": muc5b_only,
        "n_neither": neither,
        "ANXA8_positive_fraction": n_anxa8 / total,
        "MUC5B_positive_fraction": n_muc5b / total,
        "double_positive_fraction_all_tumor": both / total,
        "double_positive_fraction_of_ANXA8_positive": (
            both / n_anxa8 if n_anxa8 else np.nan
        ),
        "double_positive_fraction_of_MUC5B_positive": (
            both / n_muc5b if n_muc5b else np.nan
        ),
        "double_positive_jaccard_of_positive_union": (
            both / union if union else np.nan
        ),
        "exclusive_fraction_of_positive_union": (
            (anxa8_only + muc5b_only) / union if union else np.nan
        ),
        "expected_double_positive_under_independence": expected_both,
        "observed_expected_double_positive_ratio": (
            both / expected_both if expected_both > 0 else np.nan
        ),
        "fisher_odds_ratio": odds_ratio,
        "fisher_two_sided_pvalue": p_two_sided,
        "fisher_depletion_pvalue": p_depletion,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "ANXA8_MUC5B_positive_overlap",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h5ad_path = args.h5ad
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    backed = ad.read_h5ad(h5ad_path, backed="r")
    try:
        samples = backed.obs["dataset"].astype(str)
        tumor = backed.obs["cell_type"].astype(str).eq("Tumor")
        mask = np.asarray(samples.isin(TARGET_SAMPLES) & tumor)
        adata = backed[mask, ["ANXA8", "MUC5B"]].to_memory()
    finally:
        backed.file.close()

    expression = to_dense(adata.X)
    anxa8_positive = expression[:, 0] > 0
    muc5b_positive = expression[:, 1] > 0
    sample_values = adata.obs["dataset"].astype(str).to_numpy()

    rows = [
        summarize("Pooled", anxa8_positive, muc5b_positive)
    ]
    for sample in TARGET_SAMPLES:
        sample_mask = sample_values == sample
        rows.append(
            summarize(
                sample,
                anxa8_positive[sample_mask],
                muc5b_positive[sample_mask],
            )
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(
        output_dir / "ANXA8_MUC5B_positive_overlap_by_sample.tsv",
        sep="\t",
        index=False,
    )

    sample_summary = summary.loc[summary["sample_id"].ne("Pooled")].copy()
    plot_data = sample_summary[
        [
            "sample_id",
            "n_ANXA8_only",
            "n_double_positive",
            "n_MUC5B_only",
        ]
    ].melt(
        id_vars="sample_id",
        var_name="positive_group",
        value_name="n_cells",
    )
    plot_data["positive_group"] = plot_data["positive_group"].replace(
        {
            "n_ANXA8_only": "ANXA8 positive only",
            "n_double_positive": "Double positive",
            "n_MUC5B_only": "MUC5B positive only",
        }
    )
    plot_data["fraction_of_positive_union"] = plot_data.groupby(
        "sample_id"
    )["n_cells"].transform(lambda values: values / values.sum())

    plt.figure(figsize=(11, 5))
    sns.barplot(
        data=plot_data,
        x="sample_id",
        y="fraction_of_positive_union",
        hue="positive_group",
        palette={
            "ANXA8 positive only": "#D62728",
            "Double positive": "#9467BD",
            "MUC5B positive only": "#1F77B4",
        },
    )
    plt.xticks(rotation=35, ha="right")
    plt.xlabel("")
    plt.ylabel("Fraction of ANXA8/MUC5B-positive union")
    plt.title("Overlap of ANXA8>0 and MUC5B>0 PDAC tumor cells")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(
        output_dir / "ANXA8_MUC5B_positive_overlap_by_sample.pdf",
        bbox_inches="tight",
    )
    plt.close()
    print(summary.to_string(index=False))
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()
