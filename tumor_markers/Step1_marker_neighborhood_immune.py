#!/usr/bin/env python3
"""Associate dotplot tumor markers with immune composition near tumor cells.

Two complementary tests per cancer, marker gene and immune group:

1. Sample level: Spearman correlation across samples between the mean tumor
   cell marker expression (log1p CP10K) and the fraction of each immune group
   among TME cells within RADIUS um of any tumor cell.
2. Cell level: within each sample, Spearman correlation between tumor-cell
   marker expression and the immune-group fraction among TME cells within
   RADIUS um of that tumor cell; per-sample rhos are summarized with a
   Wilcoxon signed-rank test against zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

ROOT = Path("/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/01_cell_typing")
MARKER_DIR = ROOT / "01_analysis/05_markerexp"
GENE_LIST = MARKER_DIR / "final_cancer_type_gene_list_long.tsv"
OUT_DIR = MARKER_DIR / "marker_neighborhood_immune"

RARE_RCC_DATASETS = {     # here we only focus on ccRCC.
    "HT934N1-S2H1Fp1U1",
    "HT870N1-S1H1Fp1U2",
    "C3N-03021-11U1",
    "HT870N1-S1H1Fp1U3",
    "HT588N1-H1Fp1U1",
    "HT902N1-S1H1Fp1U2",
    "C3N-01175-11U1",
    "C3L-01469-11U1",
    "HT1094N-S1H1Fp1U1",
}
EXCLUDED_DATASETS = {    # they have no definitive tumor cells. 
    "SP074P1-A3-S1Fp1U1",
    "SP076P1-A7-S1Fp1U18",
    "SP089P1-B3-S1Fp1U1",
}

FINE_IMMUNE_GROUPS = {
    "T_cell": ["NK_T", "T_cell"],
    "B_cell": ["B_cell"],
    "Plasma": ["Plasma"],
    "Macrophage_Monocyte": ["Macrophage", "Macrophage_Monocyte", "Kupffer"],
    "Microglia": ["Microglia"],
    "Neutrophil_Granulocyte": ["Neutrophil", "Granulocyte"],
    "Mast": ["Mast"],
    "cDC": ["cDC", "cDC1", "cDC2", "DC", "Langerhans_cell"],
    "mregDC": ["mregDC"],
    "pDC": ["pDC"],
}
BROAD_IMMUNE_GROUPS = {
    "B_cell": FINE_IMMUNE_GROUPS["B_cell"],
    "Plasma": FINE_IMMUNE_GROUPS["Plasma"],
    "T_cell": FINE_IMMUNE_GROUPS["T_cell"],
    "Macrophage": FINE_IMMUNE_GROUPS["Macrophage_Monocyte"] + FINE_IMMUNE_GROUPS["Microglia"],
    "Mast": FINE_IMMUNE_GROUPS["Mast"],
    "DC": FINE_IMMUNE_GROUPS["cDC"] + FINE_IMMUNE_GROUPS["mregDC"] + FINE_IMMUNE_GROUPS["pDC"],
}
GROUPINGS = {"fine": FINE_IMMUNE_GROUPS, "broad": BROAD_IMMUNE_GROUPS}
ALL_IMMUNE = "All_immune"
NON_TME_LABELS = {"Necrosis", "Necrotic", "Unknown"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cancer", required=True)
    parser.add_argument("--grouping", choices=sorted(GROUPINGS), default="broad")
    parser.add_argument("--radius-um", type=float, default=50.0)
    parser.add_argument("--min-tumor-cells", type=int, default=500)
    parser.add_argument("--min-nearby-tme-cells", type=int, default=100)
    parser.add_argument("--min-cell-tme-neighbors", type=int, default=5)
    parser.add_argument(
        "--max-cells-per-sample",
        type=int,
        default=0,
        help="Randomly subsample tumor cells for the cell-level test (0 = use all cells)",
    )
    parser.add_argument("--out-subdir", default=None)
    parser.add_argument("--min-samples", type=int, default=6)
    parser.add_argument("--row-chunk", type=int, default=400_000)
    parser.add_argument(
        "--knn",
        type=int,
        default=0,
        help="Cell-level test: pool counts over the k nearest tumor cells (0 = single cell)",
    )
    return parser.parse_args()


def h5ad_path(cancer: str) -> Path:
    if cancer == "BRCA":
        return ROOT / "BRCA/annotated.Yuwei.new.h5ad"
    return ROOT / cancer / "annotated.Yuwei.h5ad"


def decode(values) -> np.ndarray:
    return np.asarray(
        [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values],
        dtype=object,
    )


def read_categorical(group: h5py.Group, key: str) -> pd.Categorical:
    node = group[key]
    if isinstance(node, h5py.Dataset):
        return pd.Categorical(decode(node[:]))
    return pd.Categorical.from_codes(node["codes"][:], decode(node["categories"][:]))


def load_genes(cancer: str) -> list[str]:
    genes = pd.read_csv(GENE_LIST, sep="\t")
    key = "RCC" if cancer == "RCC" else cancer
    selected = genes.loc[genes["cancer_type"] == key].sort_values("rank")["gene"]
    if selected.empty:
        raise SystemExit(f"No dotplot genes for {cancer}")
    return selected.tolist()


def extract_tumor_expression(
    handle: h5py.File,
    tumor_rows: np.ndarray,
    gene_idx: np.ndarray,
    total_counts: np.ndarray,
    row_chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw counts for the requested genes and total counts per row."""
    matrix = handle["layers"]["counts"]
    n_rows = int(matrix.attrs["shape"][0])
    indptr = matrix["indptr"][:]
    row_to_tumor = np.full(n_rows, -1, dtype=np.int64)
    row_to_tumor[tumor_rows] = np.arange(len(tumor_rows))
    gene_to_col = np.full(int(matrix.attrs["shape"][1]), -1, dtype=np.int64)
    gene_to_col[gene_idx] = np.arange(len(gene_idx))
    counts = np.zeros((len(tumor_rows), len(gene_idx)), dtype=np.float32)

    for start in range(0, n_rows, row_chunk):
        stop = min(start + row_chunk, n_rows)
        if not (row_to_tumor[start:stop] >= 0).any():
            continue
        lo, hi = int(indptr[start]), int(indptr[stop])
        indices = matrix["indices"][lo:hi]
        hit = np.flatnonzero(gene_to_col[indices] >= 0)
        if not hit.size:
            continue
        data = matrix["data"][lo:hi][hit]
        rows = np.searchsorted(indptr, lo + hit, side="right") - 1
        tumor_idx = row_to_tumor[rows]
        keep = tumor_idx >= 0
        counts[tumor_idx[keep], gene_to_col[indices[hit][keep]]] = data[keep]
        print(f"  rows {start:,}-{stop:,} scanned", flush=True)

    return counts, total_counts[tumor_rows].astype(np.float32)


def log1p_cp10k(counts: np.ndarray, totals: np.ndarray) -> np.ndarray:
    scale = np.divide(1e4, totals, out=np.zeros_like(totals), where=totals > 0)
    return np.log1p(counts * scale[:, None])


def bh(pvalues: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=pvalues.index)
    valid = pvalues.notna()
    if valid.any():
        out[valid] = multipletests(pvalues[valid], method="fdr_bh")[1]
    return out


def main() -> None:
    args = parse_args()
    cancer = args.cancer
    immune_map = GROUPINGS[args.grouping]
    subdir = "" if args.grouping == "fine" else args.grouping
    if args.knn > 0:
        subdir = f"{subdir}_knn{args.knn}".lstrip("_")
    if args.out_subdir:
        subdir = args.out_subdir
    out_dir = OUT_DIR / subdir / cancer
    out_dir.mkdir(parents=True, exist_ok=True)
    genes = load_genes(cancer)
    tumor_label = "Tumor_ccRCC" if cancer == "RCC" else "Tumor"

    with h5py.File(h5ad_path(cancer), "r") as handle:
        obs = handle["obs"]
        cell_type = read_categorical(obs, "cell_type")
        dataset = read_categorical(obs, "dataset")
        total_counts = obs["total_counts"][:]
        coords = np.asarray(handle["obsm"]["spatial"][:], dtype=float)
        var_names = decode(handle["var"]["_index"][:])

        allowed = ~np.isin(np.asarray(dataset), list(EXCLUDED_DATASETS))
        if cancer == "RCC":
            allowed &= ~np.isin(np.asarray(dataset), list(RARE_RCC_DATASETS))
        finite = np.isfinite(coords).all(axis=1)
        labels = np.asarray(cell_type)
        is_tumor = (labels == tumor_label) & allowed & finite
        is_tme = (
            ~pd.Series(labels).str.contains("Tumor", regex=False).to_numpy()
            & ~np.isin(labels, list(NON_TME_LABELS))
            & allowed
            & finite
        )
        group_of = np.full(len(labels), "", dtype=object)
        for group, members in immune_map.items():
            group_of[np.isin(labels, members)] = group
        immune_groups = [
            g for g in immune_map if (group_of[is_tme] == g).sum() > 0
        ]

        present = [g for g in genes if g in set(var_names)]
        missing = sorted(set(genes) - set(present))
        if missing:
            print(f"Genes missing from panel: {missing}", flush=True)
        gene_idx = np.array([int(np.flatnonzero(var_names == g)[0]) for g in present])

        sample_ids = np.asarray(dataset)
        tumor_counts = pd.Series(sample_ids[is_tumor]).value_counts()
        samples = sorted(tumor_counts.index[tumor_counts >= args.min_tumor_cells])
        tumor_rows = np.flatnonzero(is_tumor & np.isin(sample_ids, samples))
        print(
            f"{cancer}: {len(samples)} samples, {len(tumor_rows):,} tumor cells, "
            f"genes={present}",
            flush=True,
        )
        raw_counts, raw_totals = extract_tumor_expression(
            handle, tumor_rows, gene_idx, total_counts, args.row_chunk
        )
    expr = log1p_cp10k(raw_counts, raw_totals)

    tumor_sample = sample_ids[tumor_rows]
    sample_expr_rows, sample_frac_rows, cell_rows, inventory = [], [], [], []
    columns = immune_groups + [ALL_IMMUNE]

    for sample in samples:
        t_mask = tumor_sample == sample
        t_rows = tumor_rows[t_mask]
        t_expr = expr[t_mask]
        tme_rows = np.flatnonzero(is_tme & (sample_ids == sample))
        info = {
            "sample_id": sample,
            "n_tumor_cells": len(t_rows),
            "n_candidate_TME_cells": len(tme_rows),
        }
        if not len(tme_rows):
            info["status"] = "no_TME_cells"
            inventory.append(info)
            continue

        dist = cKDTree(coords[t_rows]).query(coords[tme_rows], k=1, workers=-1)[0]
        near = tme_rows[dist <= args.radius_um]
        info["n_TME_cells_within_radius"] = len(near)
        if len(near) < args.min_nearby_tme_cells:
            info["status"] = "too_few_nearby_TME_cells"
            inventory.append(info)
            continue
        info["status"] = "eligible"
        inventory.append(info)

        near_groups = group_of[near]
        frac = {g: float((near_groups == g).mean()) for g in immune_groups}
        frac[ALL_IMMUNE] = float((near_groups != "").mean())
        sample_frac_rows.append({"sample_id": sample, **frac})

        for j, gene in enumerate(present):
            values = t_expr[:, j]
            sample_expr_rows.append(
                {
                    "sample_id": sample,
                    "gene": gene,
                    "mean_log1p_cp10k": float(values.mean()),
                    "pct_positive": float((values > 0).mean() * 100),
                }
            )

        # Cell level: local immune fraction among TME neighbors of each tumor cell.
        rng = np.random.default_rng(7)
        pick = np.arange(len(t_rows))
        if 0 < args.max_cells_per_sample < len(pick):
            pick = np.sort(rng.choice(pick, args.max_cells_per_sample, replace=False))
        query = coords[t_rows[pick]]
        if args.knn > 0:
            k = min(args.knn, len(t_rows))
            neighbors = cKDTree(coords[t_rows]).query(query, k=k, workers=-1)[1]
            neighbors = neighbors.reshape(len(pick), k)
            cell_expr = log1p_cp10k(
                raw_counts[t_mask][neighbors].sum(axis=1),
                raw_totals[t_mask][neighbors].sum(axis=1),
            )
        else:
            cell_expr = t_expr[pick]
        n_tme = cKDTree(coords[tme_rows]).query_ball_point(
            query, r=args.radius_um, return_length=True, workers=-1
        )
        local = {}
        tme_groups = group_of[tme_rows]
        for g in immune_groups:
            g_rows = tme_rows[tme_groups == g]
            local[g] = (
                cKDTree(coords[g_rows]).query_ball_point(
                    query, r=args.radius_um, return_length=True, workers=-1
                )
                if len(g_rows)
                else np.zeros(len(pick), dtype=int)
            )
        local[ALL_IMMUNE] = np.sum([local[g] for g in immune_groups], axis=0)
        ok = n_tme >= args.min_cell_tme_neighbors
        if ok.sum() < 50:
            continue
        for j, gene in enumerate(present):
            x = cell_expr[:, j][ok]
            if np.ptp(x) == 0:
                continue
            for g in columns:
                y = local[g][ok] / n_tme[ok]
                if np.ptp(y) == 0:
                    continue
                rho = spearmanr(x, y).statistic
                cell_rows.append(
                    {
                        "sample_id": sample,
                        "gene": gene,
                        "immune_group": g,
                        "n_tumor_cells_tested": int(ok.sum()),
                        "rho": rho,
                    }
                )

    inventory = pd.DataFrame(inventory)
    sample_expr = pd.DataFrame(sample_expr_rows)
    sample_frac = pd.DataFrame(sample_frac_rows)
    cell_level = pd.DataFrame(cell_rows)
    inventory.to_csv(out_dir / "sample_inventory.tsv", sep="\t", index=False)
    sample_expr.to_csv(out_dir / "sample_tumor_marker_expression.tsv", sep="\t", index=False)
    sample_frac.to_csv(
        out_dir / f"sample_immune_fraction_within_{int(args.radius_um)}um_of_tumor.tsv",
        sep="\t",
        index=False,
    )
    cell_level.to_csv(out_dir / "cell_level_per_sample_spearman.tsv", sep="\t", index=False)

    merged = sample_expr.merge(sample_frac, on="sample_id")
    rows = []
    for gene, frame in merged.groupby("gene", sort=False):
        for g in columns:
            row = {"cancer": cancer, "gene": gene, "immune_group": g, "n_samples": len(frame)}
            for metric in ("mean_log1p_cp10k", "pct_positive"):
                rho = p = np.nan
                if len(frame) >= args.min_samples and frame[g].nunique() > 1:
                    result = spearmanr(frame[metric], frame[g])
                    rho, p = result.statistic, result.pvalue
                row[f"rho_{metric}"] = rho
                row[f"p_{metric}"] = p
            rows.append(row)
    sample_level = pd.DataFrame(rows)
    for metric in ("mean_log1p_cp10k", "pct_positive"):
        sample_level[f"fdr_{metric}"] = bh(sample_level[f"p_{metric}"])
    sample_level.to_csv(out_dir / "sample_level_spearman.tsv", sep="\t", index=False)

    summary_rows = []
    if not cell_level.empty:
        for (gene, g), frame in cell_level.groupby(["gene", "immune_group"], sort=False):
            rhos = frame["rho"].dropna().to_numpy()
            p = np.nan
            if len(rhos) >= args.min_samples and np.any(rhos != 0):
                p = wilcoxon(rhos).pvalue
            summary_rows.append(
                {
                    "cancer": cancer,
                    "gene": gene,
                    "immune_group": g,
                    "n_samples": len(rhos),
                    "median_rho": float(np.median(rhos)) if len(rhos) else np.nan,
                    "frac_samples_positive_rho": float((rhos > 0).mean()) if len(rhos) else np.nan,
                    "wilcoxon_p": p,
                }
            )
    cell_summary = pd.DataFrame(summary_rows)
    if not cell_summary.empty:
        cell_summary["wilcoxon_fdr"] = bh(cell_summary["wilcoxon_p"])
    cell_summary.to_csv(out_dir / "cell_level_summary.tsv", sep="\t", index=False)
    print(f"Done: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
