#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.stats import wilcoxon

TLS_ROOT = Path(
    "/storage1/fs1/dinglab/Active/Projects/pdc/projects/general/x1000/"
    "tls/tls_define/gen12/outputs/tls/LUNG"
)
H5AD = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/LUNG/annotated.Yuwei.new.h5ad"
)
GROUPS = ("Mcell-tumor", "otherTumor")
TARGETS = ("B_cell", "NK_T") # NK_T (NK or T) is effectively T cell.
RADII = (25, 50, 100)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, required=True)
    p.add_argument("--annotations-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tls-root", type=Path, default=TLS_ROOT)
    p.add_argument("--h5ad", type=Path, default=H5AD)
    p.add_argument("--halo", default="tls_region_halo100")
    return p.parse_args()


def coordinates(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        ids = np.array([x.decode() for x in obs[obs.attrs["_index"]][:]])
        xy = handle["obsm"]["spatial"][:, :2]
    return pd.DataFrame({"cell_id": ids, "x": xy[:, 0].astype("float32"), "y": xy[:, 1].astype("float32")})


def annotation_groups(path: Path, sample: str) -> pd.DataFrame:
    table = pd.read_csv(path, usecols=["cell_id", "group"], dtype=str)
    tag = f"_{sample}"
    table["cell_id"] = table["cell_id"].where(table["cell_id"].str.endswith(tag), table["cell_id"] + tag)
    return table.loc[table["group"].isin(GROUPS)]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    coordinate = coordinates(args.h5ad)
    eligible = pd.read_csv(
        args.source_dir / "LUNG_Bcell_McellAnchored_Raw.csv",
        usecols=["SampleID", "TLS_ID", "tls"],
    ).drop_duplicates("TLS_ID")

    rows = []
    for sample, sample_tls in eligible.groupby("SampleID", sort=True):
        annotation = annotation_groups(args.annotations_dir / f"{sample}_Mcell_tumor_groups.csv", sample)
        tls = pd.read_csv(
            args.tls_root / sample / f"{sample}_tls.csv",
            usecols=["cell_id", args.halo, "in_tls_core", "cell_type"],
            dtype=str,
        ).rename(columns={args.halo: "tls"})
        tls = tls.merge(coordinate, on="cell_id", how="inner").merge(annotation, on="cell_id", how="left")

        for item in sample_tls.itertuples(index=False):
            region = tls.loc[tls["tls"].eq(item.tls)]
            target_xy = {
                target: region.loc[region["cell_type"].eq(target), ["x", "y"]].to_numpy()
                for target in TARGETS
            }
            if any(len(target_xy[t]) == 0 for t in TARGETS):
                continue
            trees = {target: KDTree(target_xy[target]) for target in TARGETS}
            for group in GROUPS:
                anchors = region.loc[region["group"].eq(group)]
                if anchors.empty:
                    continue
                xy = anchors[["x", "y"]].to_numpy()
                data = pd.DataFrame({
                    "sample": sample,
                    "TLS_ID": item.TLS_ID,
                    "tls": item.tls,
                    "anchor_cell_id": anchors["cell_id"].to_numpy(),
                    "anchor_group": group,
                })
                for target in TARGETS:
                    data[f"nearest_{target}_um"] = trees[target].query(xy)[0]
                    for radius in RADII:
                        counts = trees[target].query_ball_point(xy, radius, return_length=True)
                        data[f"{target}_count_r{radius}"] = counts
                        data[f"{target}_density_r{radius}_per_mm2"] = counts / (np.pi * radius ** 2) * 1e6
                rows.append(data)

    anchors = pd.concat(rows, ignore_index=True)
    anchors.to_csv(args.out_dir / "anchor_level_B_NKT_proximity_density.tsv", sep="\t", index=False)

    metric_cols = [c for c in anchors.columns if c.startswith("nearest_") or "_density_" in c]
    per_tls = anchors.groupby(["sample", "TLS_ID", "anchor_group"])[metric_cols].median().reset_index()
    per_tls.to_csv(args.out_dir / "per_tls_median_B_NKT_proximity_density.tsv", sep="\t", index=False)

    paired = per_tls.pivot(index=["sample", "TLS_ID"], columns="anchor_group", values=metric_cols)
    stats = []
    for metric in metric_cols:
        subset = paired[metric].dropna()
        mcell = subset["Mcell-tumor"].to_numpy()
        other = subset["otherTumor"].to_numpy()
        delta = mcell - other
        p = 1.0 if np.allclose(delta, 0) else float(wilcoxon(delta).pvalue)
        stats.append({
            "metric": metric,
            "n_tls": len(subset),
            "median_Mcell_tumor": np.median(mcell),
            "median_otherTumor": np.median(other),
            "median_paired_delta_Mcell_minus_other": np.median(delta),
            "paired_wilcoxon_p": p,
        })
    pd.DataFrame(stats).to_csv(
        args.out_dir / "paired_B_NKT_proximity_density_statistics.tsv", sep="\t", index=False
    )
    print(f"Wrote {args.out_dir} ({len(per_tls)} TLS-group rows)")


if __name__ == "__main__":
    main()
