#!/usr/bin/env python3
from __future__ import annotations

import glob
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

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

T_KEYS = {"nk_t", "t_cell", "tcell"}
LUNG_DROP = {"C3L-01455", "C3N-01411", "C3N-04162"}
DIST = (-50.0, 0.0)
MIN_TUMOR = 1000
MIN_T = 20
RADIUS = 50.0
MIN_CLUMP = 8
N_PERM = 200
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
            "dataset", "sample", "sample_id", "barcode")
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
    return pd.DataFrame(cols, index=index.astype(str)), xy


def cancer_label(x):
    x = str(x)
    if x in {"BRCA_Basal", "BRCA_basal", "BRCA_Luminal", "BRCA_luminal"}:
        return "BRCA"
    return x


def drop_sample(s):
    s = str(s)
    return s.startswith("AL") or any(s == u or s.startswith(u + "-") for u in LUNG_DROP)


def is_t(x):
    return str(x).strip().lower().replace(" ", "_").replace("-", "_") in T_KEYS


def neighbors_50(xy):
    n = np.asarray(cKDTree(xy).query_ball_point(xy, RADIUS, return_length=True), float)
    return n - 1.0


def median_nn(xy):
    return float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))


def frac_in_clumps(xy):
    n = len(xy)
    if n < MIN_CLUMP:
        return 0.0
    nbrs = cKDTree(xy).query_ball_tree(cKDTree(xy), RADIUS)
    seen = np.zeros(n, dtype=bool)
    in_big = 0
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        size = 0
        while stack:
            j = stack.pop()
            size += 1
            for k in nbrs[j]:
                if not seen[k]:
                    seen[k] = True
                    stack.append(k)
        if size >= MIN_CLUMP:
            in_big += size
    return in_big / n


def metrics(xy):
    return {
        "mean_t_neighbors_50um": float(neighbors_50(xy).mean()),
        "median_nn_um": median_nn(xy),
        "frac_in_cluster50": frac_in_clumps(xy),
    }


def zscore(obs, null):
    sd = float(np.std(null, ddof=1))
    return np.nan if sd <= 0 else float((obs - null.mean()) / sd)


def one_sample(xy, is_t_mask):
    n_t = int(is_t_mask.sum())
    if n_t < MIN_T or len(xy) < n_t:
        return None
    obs = metrics(xy[is_t_mask])
    null = {k: np.empty(N_PERM) for k in obs}
    for i in range(N_PERM):
        take = RNG.choice(len(xy), size=n_t, replace=False)
        m = metrics(xy[take])
        for k in obs:
            null[k][i] = m[k]
    out = {
        "n_boundary": len(xy),
        "n_t": n_t,
        "t_fraction": n_t / len(xy),
        **obs,
    }
    for k, v in obs.items():
        out[f"z_{k}"] = zscore(v, null[k])
    return out


def boundary_barcodes():
    keep = set()
    for f in glob.glob(str(BOUNDARY_ROOT / "**" / "*_boundary_distance.csv"), recursive=True):
        sample = Path(f).name.replace("_boundary_distance.csv", "")
        cancer = Path(f).parent.parent.name
        if drop_sample(sample) or "rcc" in cancer.lower():
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
    bc = obs["barcode"].astype(str) if "barcode" in obs.columns else obs.index.astype(str)
    layer = boundary_barcodes()
    keep = obs.index.isin(layer) | bc.isin(layer)
    obs, xy = obs.loc[keep].copy(), xy[np.asarray(keep)]
    obs["sample"] = obs[sample_col].astype(str)
    obs["cancer"] = obs[cancer_col].map(cancer_label)
    obs["is_t"] = obs[type_col].map(is_t)
    ok = (
        ~obs["sample"].map(drop_sample)
        & ~obs["cancer"].str.contains("rcc|lymph", case=False, na=False)
        & np.isfinite(xy).all(1)
    )
    obs, xy = obs.loc[ok].copy(), xy[ok.to_numpy()]

    rows = []
    samples = obs["sample"].to_numpy()
    for sample in np.unique(samples):
        mask = samples == sample
        mets = one_sample(xy[mask], obs.loc[mask, "is_t"].to_numpy())
        if mets is None:
            continue
        mets["sample"] = sample
        mets["cancer"] = cancer_label(obs.loc[mask, "cancer"].iloc[0])
        rows.append(mets)
        print(f"{sample} {mets['cancer']} nT={mets['n_t']} "
              f"nb50={mets['mean_t_neighbors_50um']:.1f}", flush=True)

    dt = pd.DataFrame(rows)
    dt["nb_over_frac"] = dt["mean_t_neighbors_50um"] / dt["t_fraction"]
    b0, b1 = np.polyfit(dt["t_fraction"], dt["mean_t_neighbors_50um"], 1)
    dt["nb_resid"] = dt["mean_t_neighbors_50um"] - (b0 + b1 * dt["t_fraction"])
    dt.to_csv(OUT / "boundary_tcell_density_by_sample.tsv", sep="\t", index=False)
    (
        dt.groupby("cancer", observed=True)
        .agg(
            n_samples=("sample", "nunique"),
            median_neighbors50=("mean_t_neighbors_50um", "median"),
            median_z_neighbors50=("z_mean_t_neighbors_50um", "median"),
            median_nn=("median_nn_um", "median"),
        )
        .reset_index()
        .to_csv(OUT / "boundary_tcell_density_by_cancer.tsv", sep="\t", index=False)
    )


if __name__ == "__main__":
    main()
