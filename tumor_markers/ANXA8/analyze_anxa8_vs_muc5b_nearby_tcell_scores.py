#!/usr/bin/env python3


from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from analyze_anxa8_nearby_tcell_scores import (
    DEFAULT_H5AD,
    DEFAULT_SCORE_FILE,
    SCORE_COLUMNS,
    TARGET_SAMPLES,
    load_obs_and_coords,
    load_tcell_scores,
)
from deseq2_anxa8_vs_muc5b_positive_exclusive import read_h5ad_vector


DEFAULT_OUTPUT = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/01_analysis/00_tumor_gene_module/plot/"
    "tumor_cancersea_pearson_results/ANXA8_vs_MUC5B_nearby_tcell_scores"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--score-file", type=Path, default=DEFAULT_SCORE_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", nargs="+", default=None)
    parser.add_argument("--sample-column", default="dataset")
    parser.add_argument("--cell-type-column", default="cell_type")
    parser.add_argument("--t-cell-label", default="T_cell")
    parser.add_argument("--radius-um", type=float, default=100.0)
    parser.add_argument("--score-chunk-size", type=int, default=500_000)
    parser.add_argument("--count-chunk-size", type=int, default=10_000)
    return parser.parse_args()


def find_exclusive_positive_tumor_cells(
    h5ad_path: Path,
    obs: pd.DataFrame,
    sample_column: str,
    cell_type_column: str,
    target_samples: set[str],
    chunk_size: int,
) -> tuple[set[str], set[str], set[str]]:
    with h5py.File(h5ad_path, "r") as handle:
        genes = read_h5ad_vector(handle["var"], "_index")
        gene_lookup = {gene: i for i, gene in enumerate(genes)}
        missing = {"ANXA8", "MUC5B"} - set(gene_lookup)
        if missing:
            raise ValueError(f"Genes absent from H5AD: {sorted(missing)}")
        anxa8_index = gene_lookup["ANXA8"]
        muc5b_index = gene_lookup["MUC5B"]
        counts_group = handle["layers"]["counts"]
        shape = tuple(int(value) for value in counts_group.attrs["shape"])
        indptr = counts_group["indptr"][:]
        data_source = counts_group["data"]
        index_source = counts_group["indices"]

        sample_ids = obs[sample_column].astype(str).to_numpy()
        cell_types = obs[cell_type_column].astype(str).to_numpy()
        barcodes = obs.index.to_numpy()
        eligible = np.array(
            [
                sample in target_samples and cell_type == "Tumor"
                for sample, cell_type in zip(sample_ids, cell_types)
            ]
        )

        anxa8_only: list[str] = []
        muc5b_only: list[str] = []
        both: list[str] = []
        for start in range(0, shape[0], chunk_size):
            stop = min(start + chunk_size, shape[0])
            local = np.flatnonzero(eligible[start:stop])
            if local.size == 0:
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
            selected = chunk[local]
            selected_ids = barcodes[start:stop][local]
            anxa8_pos = selected[:, anxa8_index].toarray().ravel() > 0
            muc5b_pos = selected[:, muc5b_index].toarray().ravel() > 0
            anxa8_only.extend(selected_ids[anxa8_pos & ~muc5b_pos])
            muc5b_only.extend(selected_ids[muc5b_pos & ~anxa8_pos])
            both.extend(selected_ids[anxa8_pos & muc5b_pos])
            print(f"Scanned tumor rows {start:,}-{stop:,}")

    return set(anxa8_only), set(muc5b_only), set(both)


def nearest_distance(seed_coords: np.ndarray, query_coords: np.ndarray) -> np.ndarray:
    if len(seed_coords) == 0:
        return np.full(len(query_coords), np.inf)
    return cKDTree(seed_coords).query(query_coords, k=1, workers=-1)[0]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    target_samples = list(args.samples) if args.samples else list(TARGET_SAMPLES)
    target_set = set(target_samples)
    print(f"Samples: {len(target_samples)}")
    print(f"Output : {output_dir}")

    obs, coords = load_obs_and_coords(
        args.h5ad, args.sample_column, args.cell_type_column
    )
    sample_mask = obs[args.sample_column].astype(str).isin(target_set)
    tcell_mask = obs[args.cell_type_column].astype(str).eq(args.t_cell_label)
    tcell_ids = set(obs.index[sample_mask & tcell_mask])
    print(f"T_cell barcodes in target samples: {len(tcell_ids):,}")

    print("Calling exclusive ANXA8-only / MUC5B-only tumor cells ...")
    anxa8_ids, muc5b_ids, both_ids = find_exclusive_positive_tumor_cells(
        args.h5ad,
        obs,
        args.sample_column,
        args.cell_type_column,
        target_set,
        args.count_chunk_size,
    )
    print(f"ANXA8-only tumor: {len(anxa8_ids):,}")
    print(f"MUC5B-only tumor: {len(muc5b_ids):,}")
    print(f"Double-positive excluded from seeds: {len(both_ids):,}")
    pd.Series(sorted(anxa8_ids), name="cell_id").to_csv(
        output_dir / "ANXA8_only_tumor_cells.csv", index=False
    )
    pd.Series(sorted(muc5b_ids), name="cell_id").to_csv(
        output_dir / "MUC5B_only_tumor_cells.csv", index=False
    )

    print("Loading T-cell scores ...")
    scores = load_tcell_scores(args.score_file, tcell_ids, args.score_chunk_size)
    locations = obs.index.get_indexer(scores.index)
    matched = locations >= 0
    scores = scores.loc[matched].copy()
    locations = locations[matched]
    scores["sample_id"] = (
        obs.iloc[locations][args.sample_column].astype(str).to_numpy()
    )
    scores["x"] = coords[locations, 0]
    scores["y"] = coords[locations, 1]
    print(f"Matched scored T cells: {len(scores):,}")

    rows = []
    inventory = []
    for sample in target_samples:
        sample_obs_mask = obs[args.sample_column].astype(str).eq(sample).to_numpy()
        sample_obs = obs.loc[sample_obs_mask]
        sample_coords = coords[sample_obs_mask]
        sample_scores = scores.loc[scores["sample_id"].eq(sample)].copy()
        anxa8_mask = sample_obs.index.isin(anxa8_ids)
        muc5b_mask = sample_obs.index.isin(muc5b_ids)
        inventory.append(
            {
                "sample_id": sample,
                "n_ANXA8_only_tumor_seeds": int(anxa8_mask.sum()),
                "n_MUC5B_only_tumor_seeds": int(muc5b_mask.sum()),
                "n_scored_T_cells": len(sample_scores),
            }
        )
        if anxa8_mask.sum() == 0 or muc5b_mask.sum() == 0 or sample_scores.empty:
            print(f"[skip] {sample}: missing exclusive seeds or scored T cells")
            continue

        query = sample_scores[["x", "y"]].to_numpy()
        dist_anxa8 = nearest_distance(sample_coords[anxa8_mask], query)
        dist_muc5b = nearest_distance(sample_coords[muc5b_mask], query)
        near_anxa8 = (dist_anxa8 <= args.radius_um) & (dist_anxa8 < dist_muc5b)
        near_muc5b = (dist_muc5b <= args.radius_um) & (dist_muc5b < dist_anxa8)

        for group, mask in (
            ("near_ANXA8", near_anxa8),
            ("near_MUC5B", near_muc5b),
        ):
            selected = sample_scores.loc[mask]
            if selected.empty:
                continue
            rows.append(
                {
                    "sample_id": sample,
                    "group": group,
                    "n_T_cells": len(selected),
                    **{score: float(selected[score].mean()) for score in SCORE_COLUMNS},
                }
            )

    pd.DataFrame(inventory).to_csv(
        output_dir / "sample_inventory.tsv", sep="\t", index=False
    )

    summary = pd.DataFrame(rows)
    paired = summary.groupby("sample_id")["group"].nunique()
    paired_samples = paired[paired == 2].index
    summary = summary.loc[summary["sample_id"].isin(paired_samples)].copy()
    summary.to_csv(
        output_dir / "ANXA8_vs_MUC5B_nearby_tcell_scores_by_sample.tsv",
        sep="\t",
        index=False,
    )
    print(f"Paired samples: {len(paired_samples)}")

    test_rows = []
    for score_type in SCORE_COLUMNS:
        wide = summary.pivot(
            index="sample_id", columns="group", values=score_type
        ).dropna()
        if len(wide) < 3:
            continue
        difference = wide["near_ANXA8"] - wide["near_MUC5B"]
        statistic, pvalue = wilcoxon(
            wide["near_ANXA8"],
            wide["near_MUC5B"],
            alternative="two-sided",
            method="auto",
        )
        test_rows.append(
            {
                "score_type": score_type,
                "n_paired_samples": len(wide),
                "mean_near_ANXA8": wide["near_ANXA8"].mean(),
                "mean_near_MUC5B": wide["near_MUC5B"].mean(),
                "mean_delta_ANXA8_minus_MUC5B": difference.mean(),
                "n_samples_ANXA8_greater": int((difference > 0).sum()),
                "wilcoxon_statistic": statistic,
                "wilcoxon_pvalue": pvalue,
            }
        )

    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        tests["wilcoxon_fdr_bh"] = multipletests(
            tests["wilcoxon_pvalue"], method="fdr_bh"
        )[1]
    tests.to_csv(
        output_dir / "ANXA8_vs_MUC5B_nearby_tcell_paired_wilcoxon.tsv",
        sep="\t",
        index=False,
    )
    print(tests.to_string(index=False))
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
