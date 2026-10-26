#!/usr/bin/env python3
"""Keep V21 samples with n_near >= min_near."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


V21 = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/01_analysis/04_schwann/0_crcV21/boundary_CRC"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        default=V21 / "crc_v21_enteric_ensgt5_min10_csea",
    )
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--min-near", type=int, default=20)
    args = parser.parse_args()
    src = Path(args.src)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    qc = pd.read_csv(src / "sample_group_cell_counts.tsv", sep="\t")
    qc["keep"] = qc["n_near"] >= args.min_near
    qc["drop_reason"] = [
        "" if keep else f"near_tumor<{args.min_near}" for keep in qc["keep"]
    ]
    qc.to_csv(dest / "sample_group_cell_counts.tsv", sep="\t", index=False)
    keep = set(qc.loc[qc["keep"], "sample"].astype(str))

    assigned = pd.read_csv(src / "tumor_cell_group_assignments.tsv", sep="\t")
    assigned = assigned.loc[assigned["sample"].astype(str).isin(keep)].copy()
    assigned.to_csv(dest / "tumor_cell_group_assignments.tsv", sep="\t", index=False)
    counts = (
        assigned.groupby(["patient_id", "comparison_group"])
        .size()
        .unstack(fill_value=0)
    )
    counts.to_csv(dest / "patient_group_cell_counts.tsv", sep="\t")

    for name in (
        "v21_tumoral_nerves_used.tsv",
        "v21_tumoral_nerve_cells.tsv",
        "v21_near_tumor.tsv",
        "v21_per_nerve_yuwei_tumor_le50.tsv",
        "v20_tumoral_nerves_used.tsv",
        "v20_tumoral_nerve_cells.tsv",
        "v20_near_tumor.tsv",
    ):
        src_f = src / name
        if src_f.exists():
            shutil.copy2(src_f, dest / name)

    cache_src = src / "cache"
    cache_dest = dest / "cache"
    cache_dest.mkdir(exist_ok=True)
    if cache_src.exists():
        for f in cache_src.iterdir():
            dest_f = cache_dest / f.name
            if dest_f.exists() or dest_f.is_symlink():
                continue
            dest_f.symlink_to(f.resolve())

    print(
        f"{dest}: kept {len(keep)} / {len(qc)} samples; "
        f"assigned cells {len(assigned):,}"
    )


if __name__ == "__main__":
    main()
