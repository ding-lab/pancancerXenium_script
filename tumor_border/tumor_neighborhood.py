#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

H5AD = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "00_makeh5ad/01_x5k/00_integration/"
    "concated_withLymphNodes.annotated.Yuwei.filtered.h5ad"
)
OUT = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "00_makeh5ad/01_x5k/00_integration/00_tumor_niche"
)
RADIUS = 50.0
EXCLUDE = {"Unknown", "Others", "Exclude"}
BRAIN_TO_PARENCHYMA = {"Astrocyte"}
DROP_RARE = {"Skeletal_muscle", "Adipocyte"}


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
    want = ("cell_type", "cell_type_refine", "dataset", "cancer_type")
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


def niche_label(refine, raw):
    lab = np.asarray(refine, dtype=object)
    raw = np.asarray(raw, dtype=object)
    lab = np.where(np.isin(raw, list(BRAIN_TO_PARENCHYMA)), "NonMalignant_Parenchymal", lab)
    lab = np.where(np.isin(raw, list(DROP_RARE)), "Exclude", lab)
    return lab


def one_section(xy, labels, is_tumor, types):
    keep = ~np.isin(labels, list(EXCLUDE))
    if not keep.any() or not is_tumor.any():
        return None
    nn = NearestNeighbors(radius=RADIUS, algorithm="kd_tree")
    nn.fit(xy[keep])
    ids = pd.Categorical(labels[keep], categories=types).codes
    hits = nn.radius_neighbors(xy[is_tumor], return_distance=False)
    counts = np.zeros((len(hits), len(types)), dtype=np.int32)
    for i, nbr in enumerate(hits):
        if len(nbr):
            counts[i] = np.bincount(ids[nbr], minlength=len(types))
    return counts


def main():
    obs, xy = load_obs_spatial(H5AD)
    lab = niche_label(obs["cell_type_refine"].astype(str), obs["cell_type"].astype(str))
    types = np.sort(np.unique(lab[~np.isin(lab, list(EXCLUDE))]))
    samples = obs["dataset"].astype(str).to_numpy()
    is_tumor = lab == "Tumor"

    mats, bcs = [], []
    for sample in np.unique(samples):
        m = samples == sample
        out = one_section(xy[m], lab[m], is_tumor[m], types)
        if out is None:
            continue
        mats.append(out)
        bcs.append(obs.index.to_numpy()[m][is_tumor[m]])
        print(sample, out.shape[0], flush=True)

    counts = pd.DataFrame(np.vstack(mats), index=np.concatenate(bcs), columns=types)
    counts["total_neighbor_count"] = counts[types].sum(axis=1)
    cancer = obs["cancer_type"].astype(str)
    counts["cancer_type"] = cancer.reindex(counts.index).to_numpy()
    frac = counts[types].div(counts["total_neighbor_count"], axis=0).fillna(0)
    frac["total_neighbor_count"] = counts["total_neighbor_count"]
    frac["cancer_type"] = counts["cancer_type"]

    OUT.mkdir(parents=True, exist_ok=True)
    counts.to_csv(OUT / "tumor_niche_r50_count_df_includeNecrosis.tsv", sep="\t")
    frac.to_csv(OUT / "tumor_niche_r50_df_includeNecrosis.tsv", sep="\t")


if __name__ == "__main__":
    main()
