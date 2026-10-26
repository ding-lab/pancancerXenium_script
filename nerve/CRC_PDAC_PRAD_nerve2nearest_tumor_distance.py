#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


SCH = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/01_analysis/04_schwann"
)
OUT = SCH / "0_crcV21" / "boundary_CRC" / "nearest_tumor_distance"
CRC_INDEX = SCH / "0_crcV21" / "boundary_CRC" / "nerve_index.tsv"
OTHER_INDEX = (
    SCH
    / "0_crcV21"
    / "boundary_CRC"
    / "enteric_peripheral_analysis"
    / "tables"
    / "v21_crc_pdac_prad_nerve_index.tsv"
)
CRC_MEMBER = SCH / "0_crcV21" / "boundary_CRC" / "nerve_cell_membership.tsv"
CRC_META = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/CRC/CRC_metadata.csv"
)
CRC_ANNOT = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/CRC/CRC_cell_annotations.csv"
)
V21_DATA = Path(
    "/rdcw/fs1/dinglab/Active/Projects/pdc/projects/general/x1000/"
    "schwann/spatial_define/v21_v11-glialens-sox10uchl1/"
    "with_expression/outputs"
)
H5AD = {
    "PDAC": Path(
        "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
        "01_cell_typing/PDAC/annotated.Yuwei.updated.h5ad"
    ),
    "PRAD": Path(
        "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
        "01_cell_typing/PRAD/adata_slim.h5ad"
    ),
}
TUMOR = "Tumor"
SCHWANN = "Schwann"


def log(message: str) -> None:
    print(message, flush=True)


def decode_maybe_bytes(values) -> np.ndarray:
    return np.array(
        [v.decode() if isinstance(v, bytes) else str(v) for v in values],
        dtype=object,
    )


def decode_obs_column_subset(
    obs: h5py.Group, name: str, row_idx: np.ndarray
) -> np.ndarray:
    obj = obs[name]
    if isinstance(obj, h5py.Group) and "categories" in obj and "codes" in obj:
        cats = decode_maybe_bytes(obj["categories"][()])
        codes = np.asarray(obj["codes"][()])[row_idx]
        out = np.empty(codes.shape[0], dtype=object)
        valid = codes >= 0
        out[valid] = cats[codes[valid]]
        out[~valid] = ""
        return out
    values = obj[()]
    return decode_maybe_bytes(values[row_idx] if hasattr(values, "__getitem__") else values)


def normalize_nerve(value: object) -> str | None:
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "na"}:
        return None
    if text.lower().startswith("schwann-"):
        text = text.split("-", 1)[1]
    return f"Schwann-{int(float(text))}"


def bool_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1"})


def summarize_nerve(distances: np.ndarray) -> dict[str, float | int]:
    finite = distances[np.isfinite(distances)]
    if finite.size == 0:
        return {
            "n_query_cells": 0,
            "min_nearest_tumor_um": np.nan,
            "median_nearest_tumor_um": np.nan,
            "mean_nearest_tumor_um": np.nan,
        }
    return {
        "n_query_cells": int(finite.size),
        "min_nearest_tumor_um": float(np.min(finite)),
        "median_nearest_tumor_um": float(np.median(finite)),
        "mean_nearest_tumor_um": float(np.mean(finite)),
    }


def query_nearest(query_xy: np.ndarray, tumor_xy: np.ndarray) -> np.ndarray:
    if query_xy.size == 0 or tumor_xy.size == 0:
        return np.array([], dtype=float)
    tree = cKDTree(tumor_xy)
    distances, _ = tree.query(query_xy, k=1)
    return np.asarray(distances, dtype=float)


def load_crc_coords(samples: set[str]) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        CRC_META,
        usecols=["cell_id", "dataset", "xcoord", "ycoord"],
        dtype={"dataset": "string", "cell_id": "string"},
        chunksize=500_000,
    ):
        hit = chunk.loc[chunk["dataset"].astype(str).isin(samples)]
        if len(hit):
            chunks.append(hit)
            log(f"  CRC meta kept {len(hit):,}")
    if not chunks:
        raise RuntimeError("No CRC_metadata rows for tumoral-nerve samples")
    coords = pd.concat(chunks, ignore_index=True)
    coords["sample_id"] = coords["dataset"].astype(str)
    coords["cell_id"] = coords["cell_id"].astype(str)
    coords["x"] = pd.to_numeric(coords["xcoord"], errors="coerce")
    coords["y"] = pd.to_numeric(coords["ycoord"], errors="coerce")
    return coords.dropna(subset=["x", "y"])[["sample_id", "cell_id", "x", "y"]]


def load_crc_tumor(samples: set[str]) -> pd.DataFrame:
    tumor = pd.read_csv(
        CRC_ANNOT, usecols=["barcode", "cell_id", "cell_type"], dtype=str
    )
    tumor = tumor.loc[tumor["cell_type"].eq(TUMOR)].copy()
    tumor["cell_id"] = tumor["cell_id"].astype(str)
    tumor["barcode"] = tumor["barcode"].astype(str)
    tumor["sample_id"] = [
        barcode[len(cell_id) + 1 :] if barcode.startswith(cell_id + "_") else ""
        for barcode, cell_id in zip(tumor["barcode"], tumor["cell_id"])
    ]
    tumor = tumor.loc[tumor["sample_id"].isin(samples)].copy()
    log(f"CRC Yuwei Tumor in tumoral-nerve samples: {len(tumor):,}")
    return tumor[["sample_id", "cell_id"]]


def load_h5ad_obs(cancer: str, samples: set[str]) -> pd.DataFrame:
    path = H5AD[cancer]
    log(f"Loading {cancer} obs from {path}")
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        sample_col = "dataset" if "dataset" in obs else "sample"
        sample_obj = obs[sample_col]
        if isinstance(sample_obj, h5py.Group) and "categories" in sample_obj:
            sample_cats = decode_maybe_bytes(sample_obj["categories"][()])
            sample_codes = np.asarray(sample_obj["codes"][()])
            keep_codes = {
                i for i, name in enumerate(sample_cats) if str(name) in samples
            }
            row_idx = np.flatnonzero(np.isin(sample_codes, list(keep_codes)))
            sample = sample_cats[sample_codes[row_idx]]
        else:
            sample_all = decode_maybe_bytes(sample_obj[()])
            row_idx = np.flatnonzero(np.isin(sample_all, list(samples)))
            sample = sample_all[row_idx]
        cell_type = decode_obs_column_subset(obs, "cell_type", row_idx)
        if "cell_id" in obs:
            cell_id = decode_obs_column_subset(obs, "cell_id", row_idx)
        else:
            index_name = "_index" if "_index" in obs else "barcode"
            index = decode_obs_column_subset(obs, index_name, row_idx)
            cell_id = np.array(
                [str(x).rsplit("_", 1)[0] if "_" in str(x) else str(x) for x in index],
                dtype=object,
            )
        spatial = np.asarray(handle["obsm"]["spatial"][row_idx], dtype=float)
    out = pd.DataFrame(
        {
            "cell_id": np.asarray(cell_id, dtype=str),
            "sample_id": np.asarray(sample, dtype=str),
            "cell_type": np.asarray(cell_type, dtype=str),
            "x": spatial[:, 0],
            "y": spatial[:, 1],
        }
    )
    log(
        f"{cancer} cells {len(out):,} Tumor {int(out['cell_type'].eq(TUMOR).sum()):,}"
    )
    return out


def crc_distances(tumoral: pd.DataFrame) -> pd.DataFrame:
    samples = set(tumoral["sample_id"].astype(str))
    coords = load_crc_coords(samples)
    tumor = load_crc_tumor(samples).merge(
        coords, on=["sample_id", "cell_id"], how="inner"
    )
    member = pd.read_csv(CRC_MEMBER, sep="\t", dtype=str)
    member["in_tumor"] = bool_col(member["in_tumor"])
    query = member.loc[
        member["nerve_key"].isin(set(tumoral["nerve_key"].astype(str)))
        & (bool_col(member["is_glial"]) | bool_col(member["is_ens_neuron"]))
    ].copy()
    query = query.merge(coords, on=["sample_id", "cell_id"], how="left")
    rows = []
    for sample, nerves in tumoral.groupby("sample_id", observed=True):
        tumor_xy = tumor.loc[
            tumor["sample_id"].eq(sample), ["x", "y"]
        ].to_numpy(dtype=float)
        for _, nerve in nerves.iterrows():
            q = query.loc[
                query["nerve_key"].eq(nerve["nerve_key"]), ["x", "y"]
            ].dropna()
            summary = summarize_nerve(
                query_nearest(q.to_numpy(dtype=float), tumor_xy)
            )
            rows.append(
                {
                    "cancer_type": "CRC",
                    "sample_id": sample,
                    "nerve_id": nerve["nerve_id"],
                    "nerve_key": nerve["nerve_key"],
                    "query_cells": "Glial_or_ENS_neuron",
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def other_distances(tumoral: pd.DataFrame, cancer: str) -> pd.DataFrame:
    samples = set(tumoral["sample_id"].astype(str))
    obs = load_h5ad_obs(cancer, samples)
    data_dir = V21_DATA / cancer / "data"
    rows = []
    for sample, nerves in tumoral.groupby("sample_id", observed=True):
        path = data_dir / f"{sample}_cells.csv"
        if not path.exists():
            log(f"missing cells.csv: {cancer} {sample}")
            continue
        cells = pd.read_csv(
            path, usecols=["cell_id", "schwann_cluster", "cell_type"], dtype=str
        )
        cells["nerve_id"] = cells["schwann_cluster"].map(normalize_nerve)
        sample_obs = obs.loc[obs["sample_id"].eq(sample)].drop_duplicates("cell_id")
        if sample_obs.empty:
            log(f"{cancer} {sample} absent from h5ad")
            continue
        tumor_xy = sample_obs.loc[
            sample_obs["cell_type"].eq(TUMOR), ["x", "y"]
        ].to_numpy(dtype=float)
        xy = sample_obs.set_index("cell_id")[["x", "y"]]
        keep = set(nerves["nerve_id"].astype(str))
        for nerve_id in keep:
            schwann_ids = cells.loc[
                cells["nerve_id"].eq(nerve_id)
                & cells["cell_type"].eq(SCHWANN),
                "cell_id",
            ].astype(str)
            q = xy.reindex(schwann_ids).dropna()
            summary = summarize_nerve(
                query_nearest(q.to_numpy(dtype=float), tumor_xy)
            )
            nerve_key = nerves.loc[
                nerves["nerve_id"].astype(str).eq(nerve_id), "nerve_key"
            ]
            rows.append(
                {
                    "cancer_type": cancer,
                    "sample_id": sample,
                    "nerve_id": nerve_id,
                    "nerve_key": nerve_key.iloc[0] if len(nerve_key) else "",
                    "query_cells": "Schwann",
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def pairwise_stats(table: pd.DataFrame, metric: str) -> pd.DataFrame:
    from itertools import combinations
    from scipy.stats import mannwhitneyu

    rows = []
    for a, b in combinations(["CRC", "PDAC", "PRAD"], 2):
        x = table.loc[table["cancer_type"].eq(a), metric].to_numpy(dtype=float)
        y = table.loc[table["cancer_type"].eq(b), metric].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        y = y[np.isfinite(y)]
        if len(x) < 2 or len(y) < 2:
            p = np.nan
        else:
            p = mannwhitneyu(x, y, alternative="two-sided").pvalue
        rows.append(
            {
                "metric": metric,
                "group_a": a,
                "group_b": b,
                "n_a": int(len(x)),
                "n_b": int(len(y)),
                "median_a": float(np.median(x)) if len(x) else np.nan,
                "median_b": float(np.median(y)) if len(y) else np.nan,
                "wilcox_p_raw": float(p) if np.isfinite(p) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def load_analysis_index() -> pd.DataFrame:
    crc = pd.read_csv(CRC_INDEX, sep="\t")
    crc["cancer_type"] = "CRC"
    crc["n_ens_neuron"] = pd.to_numeric(
        crc["n_ens_neuron"], errors="coerce"
    ).fillna(0)
    crc["nerve_identity"] = np.where(
        crc["n_ens_neuron"].eq(0), "Peripheral", "Enteric"
    )
    other = pd.read_csv(OTHER_INDEX, sep="\t")
    other = other.loc[other["cancer_type"].isin(["PDAC", "PRAD"])].copy()
    other["nerve_identity"] = "Schwann"
    other["n_ens_neuron"] = np.nan
    columns = [
        "cancer_type",
        "sample_id",
        "nerve_id",
        "n_schwann",
        "in_tumor",
        "tumor_status_method",
        "nerve_identity",
        "n_ens_neuron",
    ]
    return pd.concat([crc[columns], other[columns]], ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    index = load_analysis_index()
    index["in_tumor"] = bool_col(index["in_tumor"])
    tumoral = index.loc[index["in_tumor"]].copy()
    log(
        "tumoral nerves "
        + str(tumoral.groupby("cancer_type").size().to_dict())
    )

    pieces = [
        crc_distances(tumoral.loc[tumoral["cancer_type"].eq("CRC")]),
        other_distances(tumoral.loc[tumoral["cancer_type"].eq("PDAC")], "PDAC"),
        other_distances(tumoral.loc[tumoral["cancer_type"].eq("PRAD")], "PRAD"),
    ]
    result = pd.concat(pieces, ignore_index=True)
    result = result.merge(
        tumoral[
            [
                "cancer_type",
                "sample_id",
                "nerve_id",
                "n_schwann",
                "tumor_status_method",
                "nerve_identity",
                "n_ens_neuron",
            ]
        ],
        on=["cancer_type", "sample_id", "nerve_id"],
        how="left",
    )
    result.to_csv(OUT / "tumoral_nerve_nearest_tumor_distance.tsv", sep="\t", index=False)

    summary = (
        result.groupby("cancer_type", as_index=False)
        .agg(
            n_nerves=("nerve_id", "nunique"),
            n_with_distance=("min_nearest_tumor_um", lambda x: int(np.isfinite(x).sum())),
            median_min_um=("min_nearest_tumor_um", "median"),
            median_median_um=("median_nearest_tumor_um", "median"),
            mean_min_um=("min_nearest_tumor_um", "mean"),
            mean_median_um=("median_nearest_tumor_um", "mean"),
        )
    )
    stats = pd.concat(
        [
            pairwise_stats(result, "min_nearest_tumor_um"),
            pairwise_stats(result, "median_nearest_tumor_um"),
        ],
        ignore_index=True,
    )
    summary.to_csv(OUT / "nearest_tumor_distance_summary.tsv", sep="\t", index=False)
    stats.to_csv(OUT / "nearest_tumor_distance_pairwise.tsv", sep="\t", index=False)
    log("=== per-cancer ===")
    log(summary.to_string(index=False))
    log("=== pairwise ===")
    log(stats.to_string(index=False))
    log(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
