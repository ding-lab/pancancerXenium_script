#!/usr/bin/env python3
"""Compare CancerSEA scores in ANXA8-only and MUC5B-only PDAC tumor cells."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from deseq2_anxa8_vs_muc5b_positive_exclusive import (
    DEFAULT_H5AD,
    TARGET_SAMPLES,
    read_h5ad_vector,
)


ROOT = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/01_analysis/00_tumor_gene_module/plot/"
    "tumor_cancersea_pearson_results"
)
H5AD_PATH = Path(DEFAULT_H5AD)
if not H5AD_PATH.is_file():
    fallback = H5AD_PATH.with_name("annotated.Yuwei.h5ad")
    if not fallback.is_file():
        raise FileNotFoundError(
            f"Neither {H5AD_PATH} nor {fallback} exists"
        )
    H5AD_PATH = fallback
CANCERSEA_PATH = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "00_makeh5ad/01_x5k/00_integration/03_scoring/Output/"
    "PDAC_cancerSEA_AUC_Summary.csv"
)
OUTPUT_DIR = ROOT / "ANXA8_vs_MUC5B_positive_CancerSEA_scores"
MATRIX_CHUNK_SIZE = 20_000
SCORE_CHUNK_SIZE = 250_000
GROUP_ORDER = ("MUC5B_only", "ANXA8_only")


def identify_exclusive_positive_cells() -> pd.DataFrame:
    with h5py.File(H5AD_PATH, "r") as handle:
        obs_group = handle["obs"]
        index_key = obs_group.attrs["_index"]
        if isinstance(index_key, bytes):
            index_key = index_key.decode("utf-8")
        cell_ids = read_h5ad_vector(obs_group, index_key)
        sample_ids = read_h5ad_vector(obs_group, "dataset")
        cell_types = read_h5ad_vector(obs_group, "cell_type")
        genes = read_h5ad_vector(handle["var"], "_index")

        gene_lookup = {gene: index for index, gene in enumerate(genes)}
        missing = {"ANXA8", "MUC5B"} - set(gene_lookup)
        if missing:
            raise ValueError(f"Genes absent from H5AD: {sorted(missing)}")

        target_set = set(TARGET_SAMPLES)
        is_target = np.array(
            [sample in target_set for sample in sample_ids]
        )
        tumor_mask = (cell_types.astype(str) == "Tumor") & is_target
        counts = handle["layers"]["counts"]
        indptr = counts["indptr"][:]
        data_source = counts["data"]
        index_source = counts["indices"]
        shape = tuple(int(value) for value in counts.attrs["shape"])
        marker_indices = [
            gene_lookup["ANXA8"],
            gene_lookup["MUC5B"],
        ]

        selected_rows = []
        for start in range(0, shape[0], MATRIX_CHUNK_SIZE):
            stop = min(start + MATRIX_CHUNK_SIZE, shape[0])
            local_tumor = np.flatnonzero(tumor_mask[start:stop])
            if local_tumor.size == 0:
                continue

            data_start = int(indptr[start])
            data_stop = int(indptr[stop])
            chunk = sparse.csr_matrix(
                (
                    data_source[data_start:data_stop],
                    index_source[data_start:data_stop],
                    indptr[start : stop + 1] - data_start,
                ),
                shape=(stop - start, shape[1]),
            )
            marker_counts = chunk[local_tumor][:, marker_indices].toarray()
            anxa8_positive = marker_counts[:, 0] > 0
            muc5b_positive = marker_counts[:, 1] > 0
            anxa8_only = anxa8_positive & ~muc5b_positive
            muc5b_only = muc5b_positive & ~anxa8_positive

            for group, mask in (
                ("ANXA8_only", anxa8_only),
                ("MUC5B_only", muc5b_only),
            ):
                if not np.any(mask):
                    continue
                global_rows = start + local_tumor[mask]
                selected_rows.append(
                    pd.DataFrame(
                        {
                            "cell_id": cell_ids[global_rows],
                            "sample_id": sample_ids[global_rows],
                            "group": group,
                        }
                    )
                )

    if not selected_rows:
        raise ValueError(
            "No exclusive-positive Tumor cells in the DESeq2 sample list"
        )
    groups = pd.concat(selected_rows, ignore_index=True)
    groups["cell_id"] = groups["cell_id"].astype(str)
    groups["sample_id"] = groups["sample_id"].astype(str)
    print(
        f"Using h5ad {H5AD_PATH} and {len(TARGET_SAMPLES)} DESeq2 samples; "
        f"found exclusive-positive cells in "
        f"{groups['sample_id'].nunique()} samples."
    )
    return groups


def load_cancersea_scores(groups: pd.DataFrame) -> pd.DataFrame:
    group_map = groups.set_index("cell_id")[["sample_id", "group"]]
    selected_ids = set(group_map.index)
    retained = []
    score_columns = None

    for chunk in pd.read_csv(
        CANCERSEA_PATH,
        index_col=0,
        chunksize=SCORE_CHUNK_SIZE,
    ):
        chunk.index = chunk.index.astype(str)
        if score_columns is None:
            score_columns = [
                column
                for column in chunk.columns
                if column.startswith("CSEA_")
            ]
            if not score_columns:
                raise ValueError("No CSEA_ columns found in score file")
        matched_ids = chunk.index.intersection(selected_ids)
        if len(matched_ids) == 0:
            continue
        selected = chunk.loc[matched_ids, score_columns].copy()
        selected = selected.join(group_map, how="inner")
        retained.append(selected)

    if not retained:
        raise ValueError("No exclusive-positive cells matched CancerSEA scores")
    result = pd.concat(retained)
    result.index.name = "cell_id"
    return result


def summarize_and_test(cell_scores: pd.DataFrame) -> None:
    score_columns = [
        column
        for column in cell_scores.columns
        if column.startswith("CSEA_")
    ]
    sample_means = (
        cell_scores.groupby(["sample_id", "group"])[score_columns]
        .mean()
        .reset_index()
    )
    cell_counts = (
        cell_scores.groupby(["sample_id", "group"])
        .size()
        .rename("n_cells")
        .reset_index()
    )
    sample_means = sample_means.merge(
        cell_counts,
        on=["sample_id", "group"],
        how="left",
    )

    paired_counts = sample_means.groupby("sample_id")["group"].nunique()
    paired_samples = paired_counts[paired_counts == 2].index
    sample_means = sample_means.loc[
        sample_means["sample_id"].isin(paired_samples)
    ].copy()
    sample_means.to_csv(
        OUTPUT_DIR / "CancerSEA_scores_by_sample_and_group.tsv",
        sep="\t",
        index=False,
    )

    test_rows = []
    delta_rows = []
    for score in score_columns:
        wide = sample_means.pivot(
            index="sample_id",
            columns="group",
            values=score,
        ).dropna(subset=list(GROUP_ORDER))
        difference = wide["ANXA8_only"] - wide["MUC5B_only"]
        statistic, pvalue = wilcoxon(
            wide["ANXA8_only"],
            wide["MUC5B_only"],
            alternative="two-sided",
            method="auto",
        )
        test_rows.append(
            {
                "cancersea_score": score,
                "n_paired_samples": len(wide),
                "mean_ANXA8_only": wide["ANXA8_only"].mean(),
                "mean_MUC5B_only": wide["MUC5B_only"].mean(),
                "mean_delta_ANXA8_minus_MUC5B": difference.mean(),
                "median_delta_ANXA8_minus_MUC5B": difference.median(),
                "n_samples_ANXA8_greater": int((difference > 0).sum()),
                "wilcoxon_statistic": statistic,
                "wilcoxon_pvalue": pvalue,
            }
        )
        for sample_id, delta in difference.items():
            delta_rows.append(
                {
                    "sample_id": sample_id,
                    "cancersea_score": score,
                    "delta_ANXA8_minus_MUC5B": delta,
                }
            )

    tests = pd.DataFrame(test_rows)
    tests["wilcoxon_fdr_bh"] = multipletests(
        tests["wilcoxon_pvalue"],
        method="fdr_bh",
    )[1]
    tests = tests.sort_values("wilcoxon_fdr_bh")
    tests.to_csv(
        OUTPUT_DIR / "CancerSEA_paired_wilcoxon.tsv",
        sep="\t",
        index=False,
    )
    pd.DataFrame(delta_rows).to_csv(
        OUTPUT_DIR / "CancerSEA_sample_deltas.tsv",
        sep="\t",
        index=False,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Identifying ANXA8-only and MUC5B-only tumor cells...")
    groups = identify_exclusive_positive_cells()
    groups.to_csv(
        OUTPUT_DIR / "exclusive_positive_tumor_cell_groups.tsv",
        sep="\t",
        index=False,
    )
    print(groups["group"].value_counts())

    print("Loading matched CancerSEA scores...")
    cell_scores = load_cancersea_scores(groups)
    cell_scores.reset_index().to_csv(
        OUTPUT_DIR / "CancerSEA_scores_exclusive_positive_cells.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    summarize_and_test(cell_scores)
    print(f"Results written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
