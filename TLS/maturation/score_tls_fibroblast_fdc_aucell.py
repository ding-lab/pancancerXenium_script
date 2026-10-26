#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import decoupler as dc
import pandas as pd

ROOT = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/01_analysis/03_TLS"
)
INPUT_H5AD = ROOT / "tls_stromal_fdc_harmony/TLS_stromal_rawsubset.h5ad"
ANNO_TSV = (
    ROOT / "tls_stromal_fdc_harmony/filtered_drop242526/annotated_res0.8/cell_metadata.tsv"
)
DEFAULT_OUT = ROOT / "tls_fibroblast_fdc_aucell"

FDC_MARKERS = [
    "CXCL13", "FDCSP", "CLU", "CR1", "CR2",
    "FCER2", "MFGE8", "CXCL12", "VCAM1",
]
KEEP_TYPES = {
    "CXCL13+ TRC", "FDC", "MRC", "Fibroblast", "CXCL9+ activated TRC",
}
MIN_N = 3
AUC_TOP_FRAC = 0.05


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input-h5ad", type=Path, default=INPUT_H5AD)
    p.add_argument("--anno-tsv", type=Path, default=ANNO_TSV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--min-n", type=int, default=MIN_N)
    p.add_argument("--auc-top-frac", type=float, default=AUC_TOP_FRAC)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(args.input_h5ad)
    anno = pd.read_csv(
        args.anno_tsv, sep="\t",
        usecols=["barcode", "cancer_type", "sample", "tls_id", "cell_type"],
        dtype=str,
    )
    keep = anno.loc[anno["cell_type"].isin(KEEP_TYPES)]
    adata = adata[adata.obs_names.astype(str).isin(set(keep["barcode"]))].copy()
    meta = keep.set_index("barcode").reindex(adata.obs_names.astype(str))
    adata.obs["sample"] = meta["sample"].to_numpy()
    adata.obs["tls_id"] = meta["tls_id"].to_numpy()
    adata.obs["cell_type"] = meta["cell_type"].to_numpy()
    adata.obs["cancer_type"] = meta["cancer_type"].to_numpy()

    present = [g for g in FDC_MARKERS if g in set(map(str, adata.var_names))]
    if len(present) < args.min_n:
        raise RuntimeError(f"FDC markers in panel: {present}")
    net = pd.DataFrame({"source": "FDC", "target": present})

    n_up = max(1, int(adata.n_vars * args.auc_top_frac))
    dc.run_aucell(
        adata, net,
        source="source", target="target",
        n_up=n_up, min_n=args.min_n, seed=42,
        verbose=False, use_raw=False,
    )
    scores = adata.obsm["aucell_estimate"]
    fdc = scores["FDC"].to_numpy() if "FDC" in scores.columns else scores.iloc[:, 0].to_numpy()

    barcodes = pd.Series(adata.obs_names.astype(str))
    out = pd.DataFrame({
        "barcode": barcodes.to_numpy(),
        "sample": adata.obs["sample"].astype(str).to_numpy(),
        "cell_id": barcodes.str.rsplit("_", n=1).str[0].to_numpy(),
        "cancer_type": adata.obs["cancer_type"].astype(str).to_numpy(),
        "tls_id": adata.obs["tls_id"].astype(str).to_numpy(),
        "cell_type": adata.obs["cell_type"].astype(str).to_numpy(),
        "FDC_AUCell": fdc,
    })
    out.to_csv(args.out_dir / "tls_fibroblasts_FDC_AUCell_scores.tsv", sep="\t", index=False)
    print(f"Wrote {len(out)} fibroblast FDC AUCell scores")


if __name__ == "__main__":
    main()
