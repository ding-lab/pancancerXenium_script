#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon

SCORE_CSV = Path(
    "/rdcw/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "00_makeh5ad/01_x5k/00_integration/00_immune/Macrophage/"
    "Macrophage_score_allcell.csv"
)
H5AD_DIR = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/PRAD/adata_by_sample"
)
OUT = Path(__file__).resolve().parent
RADIUS = 50.0
MIN_PER_GROUP = 20


def coords(adata):
    if {"xcoord", "ycoord"}.issubset(adata.obs.columns):
        return adata.obs[["xcoord", "ycoord"]].to_numpy(float)
    key = "spatial" if "spatial" in adata.obsm else "X_spatial"
    return np.asarray(adata.obsm[key], dtype=float)[:, :2]


def types(adata):
    col = next(c for c in ("cell_type_refine", "cell_type") if c in adata.obs)
    return adata.obs[col].astype(str).to_numpy()


def cell_ids(adata, sample):
    raw = adata.obs["cell_id"].astype(str) if "cell_id" in adata.obs else adata.obs_names.astype(str)
    suf = f"_{sample}"
    return np.array([x[: -len(suf)] if x.endswith(suf) else x for x in raw])


def load_scores():
    header = pd.read_csv(SCORE_CSV, nrows=0)
    score_cols = [c for c in header.columns if c not in {"barcode", "cancer_type"}]
    parts = []
    for chunk in pd.read_csv(SCORE_CSV, usecols=["barcode", "cancer_type", *score_cols],
                             chunksize=250000):
        chunk = chunk.loc[chunk["cancer_type"].eq("PRAD")]
        if chunk.empty:
            continue
        bc = chunk["barcode"].astype(str).str.split("_", n=1, expand=True)
        chunk = chunk.assign(cell_id=bc[0].str.lower(), sample=bc[1])
        parts.append(chunk[["sample", "cell_id", *score_cols]])
    scores = pd.concat(parts, ignore_index=True)
    for c in score_cols:
        scores[c] = pd.to_numeric(scores[c], errors="coerce")
    return scores, score_cols


def one_sample(sample, path, scores, score_cols):
    adata = sc.read_h5ad(path)
    xy = coords(adata)
    lab = types(adata)
    ids = np.char.lower(cell_ids(adata, sample).astype(str))
    ok = np.isfinite(xy).all(1)
    fib, mac = (lab == "Fibroblast") & ok, (lab == "Macrophage") & ok
    d, _ = cKDTree(xy[fib]).query(xy[mac], k=1)
    out = pd.DataFrame({
        "sample": sample,
        "cell_id": ids[mac],
        "near_fib": d <= RADIUS,
    })
    hit = scores.set_index("cell_id").reindex(out["cell_id"])
    for c in score_cols:
        out[c] = hit[c].to_numpy(dtype=float)
    return out


def main():
    scores, score_cols = load_scores()
    cells = []
    for path in sorted(H5AD_DIR.glob("*.h5ad")):
        sample = path.stem
        sub = scores.loc[scores["sample"].eq(sample)]
        if sub.empty:
            continue
        cells.append(one_sample(sample, path, sub, score_cols))
    cells = pd.concat(cells, ignore_index=True)
    cells["group"] = np.where(cells["near_fib"], "near_fibroblast", "other")

    pb = (
        cells.groupby(["sample", "group"], observed=True)[score_cols]
        .mean()
        .join(cells.groupby(["sample", "group"]).size().rename("n"))
        .reset_index()
    )
    pb = pb.loc[pb["n"] >= MIN_PER_GROUP]
    both = pb.groupby("sample").filter(lambda g: set(g["group"]) == {"near_fibroblast", "other"})

    rows = []
    for c in score_cols:
        wide = both.pivot(index="sample", columns="group", values=c).dropna()
        if len(wide) < 3:
            continue
        stat = wilcoxon(wide["near_fibroblast"], wide["other"], alternative="two-sided")
        rows.append({
            "score": c,
            "n_samples": len(wide),
            "median_near": float(wide["near_fibroblast"].median()),
            "median_other": float(wide["other"].median()),
            "W": float(stat.statistic),
            "p": float(stat.pvalue),
        })
    OUT.mkdir(parents=True, exist_ok=True)
    cells.to_csv(OUT / "PRAD_macrophage_fibroblast_distance.csv", index=False)
    both.to_csv(OUT / "PRAD_macrophage_near_fibroblast_pseudobulk.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "PRAD_macrophage_near_fibroblast_wilcoxon.csv", index=False)


if __name__ == "__main__":
    main()
