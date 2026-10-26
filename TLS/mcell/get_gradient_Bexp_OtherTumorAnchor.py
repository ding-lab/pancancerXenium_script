"""Matched control gradient: TLS B cells measured from otherTumor anchors.

The eligible B cells and TLS are read from an Mcell-anchored raw table. This
keeps exactly the same TLS cohort and B-cell targets, while replacing the
within-TLS Mcell anchor with every otherTumor cell assigned to that TLS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

TLS_ROOT = Path(
    "/storage1/fs1/dinglab/Active/Projects/pdc/projects/general/x1000/"
    "tls/tls_define/gen12/outputs/tls/LUNG"
)
H5AD = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/LUNG/annotated.Yuwei.new.h5ad"
)
PERCENTILE_LABELS = [
    "0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
    "50-60%", "60-70%", "70-80%", "80-90%", "90-100%",
]
METADATA = {
    "cell_id", "SampleID", "TLS_ID", "tls", "cell_type", "x", "y",
    "dist_to_mcell", "n_anchor_mcell", "TLS_Percentile", "Norm_Bin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--annotations-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tls-root", type=Path, default=TLS_ROOT)
    parser.add_argument("--h5ad", type=Path, default=H5AD)
    parser.add_argument("--halo", default="tls_region_halo100")
    return parser.parse_args()


def coordinate_table(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        index_key = obs.attrs["_index"]
        cell_ids = np.array([x.decode() for x in obs[index_key][:]])
        spatial = handle["obsm"]["spatial"][:, :2]
    return pd.DataFrame(
        {
            "cell_id": cell_ids,
            "x_anchor": spatial[:, 0].astype("float32"),
            "y_anchor": spatial[:, 1].astype("float32"),
        }
    )


def full_ids(annotation_path: Path, sample: str) -> set[str]:
    annotation = pd.read_csv(
        annotation_path, usecols=["cell_id", "group"], dtype=str
    )
    selected = annotation.loc[annotation["group"].eq("otherTumor"), "cell_id"]
    tag = f"_{sample}"
    return {x if x.endswith(tag) else f"{x}{tag}" for x in selected}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(
        args.source_dir / "LUNG_Bcell_McellAnchored_Raw.csv", low_memory=False
    )
    source = source.drop(
        columns=["dist_to_mcell", "n_anchor_mcell", "TLS_Percentile", "Norm_Bin"],
        errors="ignore",
    )
    eligible = set(source["TLS_ID"].unique())
    coordinates = coordinate_table(args.h5ad)

    frames: list[pd.DataFrame] = []
    for sample, targets_sample in source.groupby("SampleID", sort=True):
        annotation_path = (
            args.annotations_dir / f"{sample}_Mcell_tumor_groups.csv"
        )
        tls_path = args.tls_root / sample / f"{sample}_tls.csv"
        if not annotation_path.is_file() or not tls_path.is_file():
            print(f"[SKIP] {sample}: missing annotation or TLS table")
            continue

        ids = full_ids(annotation_path, sample)
        tls = pd.read_csv(tls_path, usecols=["cell_id", args.halo], dtype=str)
        tls = tls.rename(columns={args.halo: "tls"})
        anchors = (
            tls.loc[tls["cell_id"].isin(ids) & tls["tls"].notna()]
            .merge(coordinates, on="cell_id", how="inner")
        )

        kept = 0
        for tls_id, targets in targets_sample.groupby("TLS_ID", sort=True):
            if tls_id not in eligible:
                continue
            tls_label = targets["tls"].iloc[0]
            anchor_group = anchors.loc[anchors["tls"].eq(tls_label)]
            if anchor_group.empty:
                print(f"  [SKIP] {tls_id}: no within-TLS otherTumor anchor")
                continue
            tree = KDTree(anchor_group[["x_anchor", "y_anchor"]].to_numpy())
            distance, _ = tree.query(targets[["x", "y"]].to_numpy())
            part = targets.copy()
            part["dist_to_anchor"] = distance
            part["n_anchor_otherTumor"] = len(anchor_group)
            frames.append(part)
            kept += 1
        print(f"{sample}: {kept} matched TLS")

    if not frames:
        raise SystemExit("No matched TLS produced")
    final = pd.concat(frames, ignore_index=True)
    final["TLS_Percentile"] = final.groupby("TLS_ID")["dist_to_anchor"].rank(
        method="average", pct=True
    )
    final["Norm_Bin"] = pd.cut(
        final["TLS_Percentile"],
        bins=np.linspace(0, 1, 11),
        labels=PERCENTILE_LABELS,
        include_lowest=True,
    )

    agg_cols = [
        column
        for column in final.columns
        if column not in METADATA
        and column not in {"dist_to_anchor", "n_anchor_otherTumor"}
        and pd.api.types.is_numeric_dtype(final[column])
    ]
    group = ["TLS_ID", "Norm_Bin"]
    tls_mean = final.groupby(group, observed=True)[agg_cols].mean().reset_index()
    global_mean = (
        tls_mean.groupby("Norm_Bin", observed=True)[agg_cols].mean().reset_index()
    )
    pooled_mean = (
        final.groupby("Norm_Bin", observed=True)[agg_cols].mean().reset_index()
    )
    sample_mean = (
        final.groupby(["SampleID", "Norm_Bin"], observed=True)[agg_cols]
        .mean()
        .reset_index()
    )

    final.to_csv(
        args.out_dir / "LUNG_Bcell_OtherTumorAnchored_Raw.csv", index=False
    )
    global_mean.to_csv(
        args.out_dir / "LUNG_Global_TLSNormalized_Average.csv", index=False
    )
    pooled_mean.to_csv(
        args.out_dir / "LUNG_Global_CellPooled_Average.csv", index=False
    )
    sample_mean.to_csv(
        args.out_dir / "LUNG_Sample_TLSNormalized_Average.csv", index=False
    )
    print(
        f"{len(final)} B cells, {final['TLS_ID'].nunique()} matched TLS, "
        f"{final['SampleID'].nunique()} samples"
    )
    print(f"Outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
