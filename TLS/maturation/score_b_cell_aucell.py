#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import decoupler as dc
import pandas as pd

INPUT_H5AD = Path(
    "/rdcw/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "00_makeh5ad/01_x5k/00_integration/00_finalslim_subsets/B.slim.h5ad"
)
DEFAULT_OUT_DIR = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/01_analysis/03_TLS/b_cell_aucell"
)

GCB_MARKERS = [
    "BCL6", "MEF2B", "S1PR2", "RGS13", "GCSAM",
    "AICDA", "MKI67", "FAS", "CD83",
]
AUC_TOP_FRAC = 0.05
MIN_N = 5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-h5ad", type=Path, default=INPUT_H5AD)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--auc-top-frac", type=float, default=AUC_TOP_FRAC)
    p.add_argument("--min-n", type=int, default=MIN_N)
    return p.parse_args()


def marker_network(var_names, min_n: int) -> pd.DataFrame:
    present = [g for g in GCB_MARKERS if g in set(map(str, var_names))]
    if len(present) < min_n:
        missing = [g for g in GCB_MARKERS if g not in present]
        raise RuntimeError(f"GCB markers present={present}, missing={missing}")
    return pd.DataFrame({"source": "GCB", "target": present})


def run_aucell(adata: ad.AnnData, network: pd.DataFrame, auc_top_frac: float, min_n: int):
    n_up = max(1, int(adata.n_vars * auc_top_frac))
    dc.run_aucell(
        adata, network,
        source="source", target="target",
        n_up=n_up, min_n=min_n, seed=42,
        verbose=False, use_raw=False,
    )
    return adata.obsm["aucell_estimate"].copy()


def score_table(adata: ad.AnnData, scores: pd.DataFrame) -> pd.DataFrame:
    barcodes = pd.Series(adata.obs_names.astype(str))
    if "cell_id" in adata.obs.columns:
        cell_id = adata.obs["cell_id"].astype(str).to_numpy()
    else:
        cell_id = barcodes.str.rsplit("_", n=1).str[0].to_numpy()
    if "sample" in adata.obs.columns:
        sample = adata.obs["sample"].astype(str).to_numpy()
    else:
        sample = barcodes.str.rsplit("_", n=1).str[1].to_numpy()
    return pd.DataFrame({
        "sample": sample,
        "cell_id": cell_id,
        "GCB_AUCell": scores["GCB"].to_numpy() if "GCB" in scores.columns else scores.iloc[:, 0].to_numpy(),
    })


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(args.input_h5ad)
    net = marker_network(adata.var_names, args.min_n)
    scores = run_aucell(adata, net, args.auc_top_frac, args.min_n)
    scores.columns = [c if str(c).endswith("_AUCell") else f"{c}_AUCell" for c in scores.columns]
    out = score_table(adata, scores.rename(columns={"GCB_AUCell": "GCB"}))
    out.to_csv(args.out_dir / "b_cells_GCB_AUCell_scores.tsv.gz", sep="\t", index=False, compression="gzip")
    print(f"Wrote {len(out)} B-cell GCB AUCell scores")


if __name__ == "__main__":
    main()
