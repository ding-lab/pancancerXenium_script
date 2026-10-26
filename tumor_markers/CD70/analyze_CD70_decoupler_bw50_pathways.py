#!/usr/bin/env python3
"""Compare RCC CancerSEA/MSigDB scores in CD70-low/high tumor regions."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import anndata as ad
import decoupler as dc
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


ROOT = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k"
)
RCC_DIR = ROOT / "01_cell_typing/RCC"
WORK = (
    ROOT
    / "01_cell_typing/01_analysis/00_tumor_gene_module/targetgene/"
    "RCC_CD70/CD70_decoupler_bw50_Q25_Q75_pathways"
)
DEFAULT_H5AD = RCC_DIR / "annotated.Yuwei.slim.h5ad"
DEFAULT_CANCERSEA = (
    ROOT
    / "00_makeh5ad/01_x5k/00_integration/03_scoring/Output/"
    "RCC_cancerSEA_AUC_Summary.csv"
)
DEFAULT_MSIGDB = (
    ROOT
    / "00_makeh5ad/01_x5k/00_integration/03_scoring/Msigdb_output/"
    "RCC_AllCells_MSigDB_AUC.csv"
)
SAMPLES = [
    "C3L-00813-11U1",
    "C3L-02841-11U1",
    "HT1012N1-S1H1Fp1U1",
    "HT1056N1-S1H1Fp1U1",
    "HT941N1-S1H1Fp1U2",
    "HT955N1-S2H1Fp1U1",
    "HT958N1-S1H1Fp1U1",
    "HT963N1-S1H1Fp1U1",
    "HT970N1-S1H1Fp1U1",
]
LOW = "CD70_low_window"
HIGH = "CD70_high_window"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--cancersea", type=Path, default=DEFAULT_CANCERSEA)
    parser.add_argument("--msigdb", type=Path, default=DEFAULT_MSIGDB)
    parser.add_argument("--output-dir", type=Path, default=WORK)
    parser.add_argument(
        "--analysis-label",
        default="CD70_decoupler_bw50",
        help="Label embedded in output filenames.",
    )
    parser.add_argument(
        "--smoothing-method",
        choices=["decoupler", "fixed_knn"],
        default="decoupler",
    )
    parser.add_argument("--knn", type=int, default=20)
    parser.add_argument("--bandwidth", type=float, default=50)
    parser.add_argument("--cutoff", type=float, default=0.1)
    parser.add_argument("--max-neighbors", type=int, default=100)
    parser.add_argument("--low-quantile", type=float, default=0.25)
    parser.add_argument("--high-quantile", type=float, default=0.75)
    parser.add_argument("--chunk-size", type=int, default=200_000)
    return parser.parse_args()


def vector(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray().ravel()
    return np.asarray(matrix).ravel()


def prepare_assignments(
    h5ad_path: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    backed = sc.read_h5ad(h5ad_path, backed="r")
    assignment_frames = []
    summary_rows = []
    try:
        if "CD70" not in backed.var_names:
            raise KeyError("CD70 is absent from the RCC H5AD")
        for sample_name in SAMPLES:
            sample_mask = (
                backed.obs["dataset"]
                .astype(str)
                .eq(sample_name)
                .to_numpy()
            )
            if not sample_mask.any():
                raise ValueError(f"Sample absent from H5AD: {sample_name}")
            sample = backed[sample_mask, ["CD70"]].to_memory()
            try:
                required = {"cell_type", "xcoord", "ycoord"}
                if not required.issubset(sample.obs):
                    raise KeyError(
                        f"{sample_name}: missing obs columns "
                        f"{required - set(sample.obs)}"
                    )
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
                    raise KeyError(
                        "Neither total_counts nor n_counts is present"
                    )
                coordinates = sample.obs[
                    ["xcoord", "ycoord"]
                ].to_numpy(dtype=float)
                finite = np.isfinite(coordinates).all(axis=1)
                tumor = (
                    sample.obs["cell_type"]
                    .astype(str)
                    .eq("Tumor_ccRCC")
                    .to_numpy()
                    & finite
                )
                if not tumor.any():
                    raise ValueError(
                        f"{sample_name}: no Tumor_ccRCC cells"
                    )
                raw = vector(sample.layers["counts"][:, 0]).astype(float)
                totals = sample.obs[total_column].to_numpy(dtype=float)
                expression = np.zeros(len(sample), dtype=float)
                valid_total = totals > 0
                expression[valid_total] = np.log1p(
                    raw[valid_total] / totals[valid_total] * 1e4
                )
                tumor_coordinates = coordinates[tumor]
                tumor_expression = expression[tumor]
                if args.smoothing_method == "decoupler":
                    tumor_adata = ad.AnnData(
                        X=tumor_expression[:, None],
                        obs=pd.DataFrame(
                            index=sample.obs.index[tumor].astype(str)
                        ),
                        var=pd.DataFrame(index=["CD70"]),
                    )
                    tumor_adata.obsm["spatial"] = tumor_coordinates
                    dc.pp.knn(
                        tumor_adata,
                        key="spatial",
                        bw=args.bandwidth,
                        max_nn=min(
                            args.max_neighbors,
                            len(tumor_adata),
                        ),
                        cutoff=args.cutoff,
                    )
                    smoothed = np.asarray(
                        tumor_adata.obsp[
                            "spatial_connectivities"
                        ].dot(tumor_adata.X)
                    ).ravel()
                else:
                    neighbor_count = min(args.knn, len(tumor_expression))
                    _, indices = cKDTree(tumor_coordinates).query(
                        tumor_coordinates,
                        k=neighbor_count,
                    )
                    if neighbor_count == 1:
                        indices = indices[:, None]
                    smoothed = tumor_expression[indices].mean(axis=1)
                low_cutoff = float(
                    np.quantile(smoothed, args.low_quantile)
                )
                high_cutoff = float(
                    np.quantile(smoothed, args.high_quantile)
                )
                if high_cutoff <= low_cutoff:
                    raise ValueError(
                        f"{sample_name}: insufficient smoothed variation"
                    )
                windows = np.select(
                    [
                        smoothed <= low_cutoff,
                        smoothed >= high_cutoff,
                    ],
                    [LOW, HIGH],
                    default="Middle",
                )
                cell_ids = (
                    sample.obs["cell_id"].astype(str).to_numpy()
                    if "cell_id" in sample.obs
                    else sample.obs.index.astype(str).to_numpy()
                )
                frame = pd.DataFrame(
                    {
                        "cell_id": cell_ids[tumor],
                        "sample_id": sample_name,
                        "window": windows,
                        "CD70_log1pCP10K": tumor_expression,
                        "CD70_smoothed": smoothed,
                    }
                )
                frame = frame.loc[frame["window"].isin([LOW, HIGH])]
                assignment_frames.append(frame)
                summary_rows.append(
                    {
                        "sample_id": sample_name,
                        "n_tumor": int(tumor.sum()),
                        "n_low": int((windows == LOW).sum()),
                        "n_high": int((windows == HIGH).sum()),
                        "low_cutoff": low_cutoff,
                        "high_cutoff": high_cutoff,
                        "smoothing_method": args.smoothing_method,
                        "knn": (
                            args.knn
                            if args.smoothing_method == "fixed_knn"
                            else np.nan
                        ),
                        "bandwidth": args.bandwidth,
                        "cutoff": args.cutoff,
                    }
                )
                print(
                    f"{sample_name}: tumor={int(tumor.sum()):,}; "
                    f"low={int((windows == LOW).sum()):,}; "
                    f"high={int((windows == HIGH).sum()):,}",
                    flush=True,
                )
            finally:
                del sample
                gc.collect()
    finally:
        backed.file.close()
    return (
        pd.concat(assignment_frames, ignore_index=True),
        pd.DataFrame(summary_rows),
    )


def detect_column(columns: list[str], candidates: list[str]) -> str | None:
    lookup = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def assignment_keys(assignments: pd.DataFrame) -> pd.DataFrame:
    keys = assignments[["cell_id", "sample_id", "window"]].copy()
    keys["cell_id_local"] = keys["cell_id"].str.split("_", n=1).str[0]
    keys = keys.drop_duplicates(
        ["sample_id", "cell_id_local"],
        keep=False,
    )
    return keys[["sample_id", "cell_id_local", "window"]]


def aggregate_scores(
    score_path: Path,
    pathway_prefix: str,
    assignments: pd.DataFrame,
    chunk_size: int,
) -> pd.DataFrame:
    header = pd.read_csv(score_path, nrows=0)
    columns = list(header.columns)
    id_column = detect_column(
        columns,
        ["cell_id", "barcode", "cell", "Unnamed: 0"],
    )
    if id_column is None:
        id_column = columns[0]
    sample_column = detect_column(
        columns,
        ["sample_id", "dataset", "sample"],
    )
    pathway_columns = [
        column
        for column in columns
        if str(column).startswith(pathway_prefix)
    ]
    if not pathway_columns:
        raise ValueError(
            f"No {pathway_prefix!r} columns found in {score_path}"
        )
    usecols = [id_column, *pathway_columns]
    if sample_column is not None and sample_column not in usecols:
        usecols.append(sample_column)

    partial = []
    matched_cells = 0
    for chunk in pd.read_csv(
        score_path,
        usecols=usecols,
        chunksize=chunk_size,
        low_memory=False,
    ):
        score_ids = chunk[id_column].astype(str)
        split_ids = score_ids.str.rsplit("_", n=1, expand=True)
        chunk["cell_id_local"] = split_ids[0]
        if sample_column is not None:
            chunk["sample_id"] = chunk[sample_column].astype(str)
        elif split_ids.shape[1] == 2:
            chunk["sample_id"] = split_ids[1]
        else:
            raise ValueError(
                f"{score_path}: cannot infer sample from {id_column}"
            )
        merged = chunk.merge(
            assignments,
            on=["sample_id", "cell_id_local"],
            how="inner",
            validate="many_to_one",
        )
        if merged.empty:
            continue
        matched_cells += len(merged)
        long = merged.melt(
            id_vars=["sample_id", "window"],
            value_vars=pathway_columns,
            var_name="pathway",
            value_name="score",
        )
        long["score"] = pd.to_numeric(long["score"], errors="coerce")
        partial.append(
            long.dropna(subset=["score"])
            .groupby(
                ["sample_id", "window", "pathway"],
                observed=True,
            )
            .agg(score_sum=("score", "sum"), n_cells=("score", "size"))
            .reset_index()
        )
    if not partial:
        raise ValueError(
            f"No score rows from {score_path} matched assignments"
        )
    totals = (
        pd.concat(partial, ignore_index=True)
        .groupby(["sample_id", "window", "pathway"], observed=True)
        .agg(score_sum=("score_sum", "sum"), n_cells=("n_cells", "sum"))
        .reset_index()
    )
    totals["mean_score"] = totals["score_sum"] / totals["n_cells"]
    totals.attrs["matched_cells"] = matched_cells
    return totals


def paired_tests(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pathway, frame in scores.groupby("pathway", observed=True):
        pivot = frame.pivot(
            index="sample_id",
            columns="window",
            values="mean_score",
        ).dropna(subset=[LOW, HIGH])
        difference = pivot[HIGH] - pivot[LOW]
        if len(pivot) >= 2 and np.any(difference != 0):
            test = wilcoxon(
                pivot[HIGH],
                pivot[LOW],
                alternative="two-sided",
            )
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
        else:
            statistic = np.nan
            p_value = np.nan
        rows.append(
            {
                "pathway": pathway,
                "n_paired_samples": len(pivot),
                "mean_low": pivot[LOW].mean(),
                "mean_high": pivot[HIGH].mean(),
                "mean_paired_difference_high_minus_low": (
                    difference.mean()
                ),
                "median_paired_difference_high_minus_low": (
                    difference.median()
                ),
                "n_samples_higher_in_high": int((difference > 0).sum()),
                "wilcoxon_statistic": statistic,
                "wilcoxon_p_value": p_value,
            }
        )
    results = pd.DataFrame(rows)
    results["wilcoxon_fdr_bh"] = np.nan
    tested = results["wilcoxon_p_value"].notna()
    if tested.any():
        results.loc[tested, "wilcoxon_fdr_bh"] = multipletests(
            results.loc[tested, "wilcoxon_p_value"],
            method="fdr_bh",
        )[1]
    return results.sort_values(
        ["wilcoxon_fdr_bh", "wilcoxon_p_value"],
        na_position="last",
    )


def save_score_analysis(
    name: str,
    scores: pd.DataFrame,
    output_dir: Path,
    analysis_label: str,
) -> None:
    scores.to_csv(
        output_dir
        / f"{name}_scores_by_sample_{analysis_label}_window.tsv",
        sep="\t",
        index=False,
    )
    paired_tests(scores).to_csv(
        output_dir
        / f"{name}_paired_{analysis_label}_window_tests.tsv",
        sep="\t",
        index=False,
    )
    print(
        f"{name}: matched={scores.attrs.get('matched_cells', 'NA')}; "
        f"pathways={scores['pathway'].nunique()}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if args.bandwidth <= 0 or args.max_neighbors < 1 or args.knn < 1:
        raise ValueError("Bandwidth and max-neighbors must be positive")
    if not 0 <= args.cutoff < 1:
        raise ValueError("--cutoff must be in [0, 1)")
    if not 0 <= args.low_quantile < args.high_quantile <= 1:
        raise ValueError("Require 0 <= low quantile < high quantile <= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    assignments, summary = prepare_assignments(args.h5ad, args)
    assignments.to_csv(
        args.output_dir
        / f"{args.analysis_label}_Q25_Q75_tumor_assignments.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    summary.to_csv(
        args.output_dir / f"{args.analysis_label}_window_summary.tsv",
        sep="\t",
        index=False,
    )
    keys = assignment_keys(assignments)
    cancersea = aggregate_scores(
        args.cancersea,
        "CSEA_",
        keys,
        args.chunk_size,
    )
    msigdb = aggregate_scores(
        args.msigdb,
        "MSig_",
        keys,
        args.chunk_size,
    )
    save_score_analysis(
        "CancerSEA",
        cancersea,
        args.output_dir,
        args.analysis_label,
    )
    save_score_analysis(
        "MSigDB",
        msigdb,
        args.output_dir,
        args.analysis_label,
    )
    print(f"Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
