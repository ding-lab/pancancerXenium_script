#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/01_analysis/03_TLS"
)
DEFAULT_B_META = ROOT / "tls_bcell_harmony/filtered/cell_metadata_filtered.tsv"
DEFAULT_B_SUB = ROOT / "tls_bcell_harmony/filtered/gcb_vs_otherB/cell_B_subtype.tsv"
DEFAULT_STROMAL = (
    ROOT / "tls_stromal_fdc_harmony/filtered_drop242526/annotated_res0.8/cell_metadata.tsv"
)
DEFAULT_OUT = ROOT / "tls_subtype_maturation_hierarchical"

OTHER_FIBRO = {"Fibroblast", "CXCL9+ activated TRC"}
KIND_MAP = {
    "CXCL13+ TRC": "n_CXCL13_TRC",
    "MRC": "n_MRC",
    "FDC": "n_FDC",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--b-meta", type=Path, default=DEFAULT_B_META)
    p.add_argument("--b-subtype", type=Path, default=DEFAULT_B_SUB)
    p.add_argument("--stromal-meta", type=Path, default=DEFAULT_STROMAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    b = pd.read_csv(args.b_meta, sep="\t", usecols=["barcode", "cancer_type", "sample", "tls_id"], dtype=str)
    b_sub = pd.read_csv(args.b_subtype, sep="\t", usecols=["barcode", "B_subtype"], dtype=str)
    b = b.merge(b_sub, on="barcode", how="left")
    if b["B_subtype"].isna().any():
        raise ValueError("B cells missing GCB/otherB labels")

    keys = ["cancer_type", "sample", "tls_id"]
    b_tls = (
        b.groupby(keys, observed=True)
        .agg(
            n_B_scored=("barcode", "size"),
            n_GCB=("B_subtype", lambda x: int((x == "GCB").sum())),
        )
        .reset_index()
    )

    stromal = pd.read_csv(
        args.stromal_meta, sep="\t",
        usecols=["barcode", "cancer_type", "sample", "tls_id", "cell_type"],
        dtype=str,
    )
    stromal["kind"] = stromal["cell_type"].map(KIND_MAP)
    stromal.loc[stromal["cell_type"].isin(OTHER_FIBRO), "kind"] = "n_otherFibroblast"
    stromal = stromal.dropna(subset=["kind"])
    s_tls = (
        stromal.groupby(keys + ["kind"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in ("n_CXCL13_TRC", "n_MRC", "n_FDC", "n_otherFibroblast"):
        if col not in s_tls.columns:
            s_tls[col] = 0

    tls = b_tls.merge(s_tls, on=keys, how="outer")
    for col in ("n_B_scored", "n_GCB", "n_CXCL13_TRC", "n_MRC", "n_FDC", "n_otherFibroblast"):
        tls[col] = tls[col].fillna(0).astype(int)
    tls = tls.rename(columns={"tls_id": "tls_region"})
    tls = tls[
        ["cancer_type", "sample", "tls_region",
         "n_CXCL13_TRC", "n_MRC", "n_FDC", "n_otherFibroblast",
         "n_B_scored", "n_GCB"]
    ].sort_values(["cancer_type", "sample", "tls_region"])

    out = args.out_dir / "tls_stromal_gcb_program_metrics.tsv"
    tls.to_csv(out, sep="\t", index=False)
    print(f"Wrote {out} ({len(tls)} TLS)")


if __name__ == "__main__":
    main()
