#!/usr/bin/env python3
from __future__ import annotations

import glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError

H5AD = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "00_makeh5ad/01_x5k/00_integration/"
    "concated_withLymphNodes.annotated.Yuwei.filtered.h5ad"
)
BOUNDARY_ROOT = Path(
    "/storage1/fs1/dinglab/Active/Projects/pdc/projects/general/x1000/"
    "boundary/gen11/5k/outputs"
)
OUT = Path(__file__).resolve().parent

TARGET = ("Fibroblast", "Macrophage", "NK_T")
NKT = {"NK_T", "T_cell"}
DIST = (-50.0, 0.0)
N_PERM = 1000
MIN_CELLS = 30
MIN_EACH = 10
MIN_TUMOR = 1000
RNG = np.random.default_rng(1)


def _as_str(x):
    if isinstance(x, (bytes, np.bytes_)):
        return x.decode("utf-8")
    return str(x)


def _read_obs(node):
    if isinstance(node, h5py.Dataset):
        v = node[()]
        if getattr(v, "dtype", None) is not None and v.dtype.kind in {"S", "O", "U"}:
            return np.array([_as_str(i) for i in np.asarray(v)], dtype=object)
        return v
    cats = np.array([_as_str(i) for i in node["categories"][()]], dtype=object)
    return pd.Categorical.from_codes(np.asarray(node["codes"][()]), cats)


def load_obs_spatial(path):
    want = ("cancer_type", "cancer_subtype", "cell_type_refine", "cell_type",
            "dataset", "sample", "sample_id")
    with h5py.File(path, "r") as f:
        obs = f["obs"]
        key = _as_str(obs.attrs.get("_index", "_index"))
        index = pd.Index([_as_str(i) for i in obs[key][()]], name=key)
        cols = {c: _read_obs(obs[c]) for c in want if c in obs}
        node = f["obsm"]["spatial"]
        if isinstance(node, h5py.Dataset):
            xy = np.asarray(node[()], dtype=float)
        else:
            from anndata.io import read_elem
            xy = np.asarray(read_elem(node), dtype=float)
    df = pd.DataFrame(cols, index=index.astype(str))
    if len(df) != len(xy):
        raise ValueError("obs and spatial length differ")
    return df, xy


def cancer_label(x):
    x = str(x)
    if x in {"BRCA_Basal", "BRCA_basal", "BRCA_Luminal", "BRCA_luminal"}:
        return "BRCA"
    return x


def n_triads(labels, simplices):
    v = labels[simplices]
    hit = np.ones(len(v), dtype=bool)
    for t in TARGET:
        hit &= (v == t).any(axis=1)
    return int(hit.sum())


def dtp_one(xy, labels):
    simplices = Delaunay(xy).simplices
    obs = n_triads(labels, simplices)
    if obs == 0:
        return None
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        null[i] = n_triads(RNG.permutation(labels), simplices)
    mu, sd = float(null.mean()), float(null.std())
    if sd == 0:
        return None
    z = (obs - mu) / sd
    p = float((null >= obs).mean() if z > 0 else (null <= obs).mean())
    return obs, mu, z, p


def boundary_barcodes():
    keep = set()
    for f in glob.glob(str(BOUNDARY_ROOT / "**" / "*_boundary_distance.csv"), recursive=True):
        sample = Path(f).name.replace("_boundary_distance.csv", "")
        cancer = Path(f).parent.parent.name
        if sample.startswith("AL") or "rcc" in cancer.lower():
            continue
        df = pd.read_csv(f, usecols=["cell_id", "tumor_region", "dist_iface_um", "cell_type"])
        n = df.loc[df["cell_type"] == "Tumor", "tumor_region"].value_counts()
        regions = n.index if cancer.upper() == "PDAC" else n.index[n >= MIN_TUMOR]
        layer = df["tumor_region"].isin(regions) & df["dist_iface_um"].between(*DIST)
        keep.update((df.loc[layer, "cell_id"].astype(str) + "_" + sample).tolist())
    return keep


def main():
    obs, xy = load_obs_spatial(H5AD)
    cancer_col = next(c for c in ("cancer_type", "cancer_subtype") if c in obs)
    type_col = next(c for c in ("cell_type_refine", "cell_type") if c in obs)
    sample_col = next(c for c in ("dataset", "sample", "sample_id") if c in obs)

    cancer = obs[cancer_col].map(cancer_label)
    drop = (
        cancer.str.contains("rcc|lymph", case=False, na=False)
        | obs[sample_col].astype(str).str.startswith("AL")
    )
    obs, xy = obs.loc[~drop].copy(), xy[~drop.to_numpy()]
    obs["_cancer"] = cancer.loc[obs.index]

    in_b = obs.index.isin(boundary_barcodes())
    obs, xy = obs.loc[in_b], xy[in_b]

    rows = []
    sample_ids = obs[sample_col].astype(str).to_numpy()
    for sample in np.unique(sample_ids):
        mask = sample_ids == sample
        if mask.sum() < MIN_CELLS:
            continue
        sub = obs.loc[mask]
        ca = cancer_label(sub["_cancer"].iloc[0])
        types = np.where(np.isin(sub[type_col].astype(str), list(NKT)), "NK_T",
                         sub[type_col].astype(str))
        counts = {t: int((types == t).sum()) for t in TARGET}
        if min(counts.values()) < MIN_EACH:
            continue
        try:
            out = dtp_one(xy[mask], types)
        except QhullError:
            continue
        if out is None:
            continue
        o, mu, z, p = out
        rows.append({
            "dataset": sample,
            "cancer_type": ca,
            **{f"n_{t}": counts[t] for t in TARGET},
            "Observed_Triangles": o,
            "Expected_Triangles": mu,
            "Z_Score": z,
            "P_value": p,
        })
        print(f"{sample} {ca} Z={z:.2f} p={p:.3g}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "FMNKT_3way_Interaction_Permutation_Results.csv", index=False)
    summary = (
        res.groupby("cancer_type")
        .agg(
            n=("Z_Score", "size"),
            median_z=("Z_Score", "median"),
            mean_z=("Z_Score", "mean"),
            frac_z_ge_1p96=("Z_Score", lambda s: float((s >= 1.96).mean())),
            frac_z_le_m1p96=("Z_Score", lambda s: float((s <= -1.96).mean())),
            frac_p_lt_0p05=("P_value", lambda s: float((s < 0.05).mean())),
        )
        .reset_index()
        .sort_values("median_z", ascending=False)
    )
    summary.to_csv(OUT / "FMNKT_3way_Interaction_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
