#!/usr/bin/env python3

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr


SCH = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/01_analysis/04_schwann"
)
BCRC = SCH / "0_crcV21" / "boundary_CRC"
INDEX = (
    BCRC
    / "enteric_peripheral_analysis"
    / "tables"
    / "v21_crc_pdac_prad_nerve_index.tsv"
)
REPAIR = (
    BCRC
    / "enteric_peripheral_analysis"
    / "tables"
    / "v21_crc_pdac_prad_repair_scores.tsv"
)
CRC_MEMBER = BCRC / "nerve_cell_membership.tsv"
REPAIR_OUT = BCRC / "repair_vs_pdgfa"
H5AD = {
    "CRC": Path(
        "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
        "01_cell_typing/CRC/adata_slim.h5ad"
    ),
    "PDAC": Path(
        "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
        "01_cell_typing/PDAC/annotated.Yuwei.h5ad"
    ),
    "PRAD": Path(
        "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
        "01_cell_typing/PRAD/adata_slim.h5ad"
    ),
}
V21_ROOTS = (
    Path(
        "/storage1/fs1/dinglab/Active/Projects/pdc/projects/general/x1000/"
        "schwann/spatial_define/v21_v11-glialens-sox10uchl1/"
        "with_expression/outputs"
    ),
    Path(
        "/rdcw/fs1/dinglab/Active/Projects/pdc/projects/general/x1000/"
        "schwann/spatial_define/v21_v11-glialens-sox10uchl1/"
        "with_expression/outputs"
    ),
)
CANCERS = ("CRC", "PDAC", "PRAD")
MIN_COR = 3


def log(message: str) -> None:
    print(message, flush=True)


def bool_col(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1"})


def normalize_nerve(value: object) -> str | None:
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "na"}:
        return None
    if text.lower().startswith("schwann-"):
        text = text.split("-", 1)[1]
    return f"Schwann-{int(float(text))}"


def decode(values) -> np.ndarray:
    return np.asarray(
        [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in values],
        dtype=object,
    )


def decode_obs_column(obs: h5py.Group, name: str, row_idx: np.ndarray | None = None) -> np.ndarray:
    obj = obs[name]
    if isinstance(obj, h5py.Group) and "categories" in obj and "codes" in obj:
        cats = decode(obj["categories"][()])
        codes = np.asarray(obj["codes"][()])
        if row_idx is not None:
            codes = codes[row_idx]
        out = np.empty(codes.shape[0], dtype=object)
        valid = codes >= 0
        out[valid] = cats[codes[valid]]
        out[~valid] = ""
        return out
    values = obj[()]
    if row_idx is not None:
        values = values[row_idx]
    return decode(values)


def subset_csr_rows(group, keep_idx: np.ndarray):
    n_vars = int(np.asarray(group.attrs["shape"]).ravel()[-1])
    if keep_idx.size == 0:
        return sparse.csr_matrix((0, n_vars), dtype=np.float32)
    indptr = group["indptr"][:]
    starts = indptr[keep_idx]
    ends = indptr[keep_idx + 1]
    counts = ends - starts
    new_indptr = np.empty(keep_idx.size + 1, dtype=np.int64)
    new_indptr[0] = 0
    np.cumsum(counts, out=new_indptr[1:])
    nnz = int(new_indptr[-1])
    data = np.empty(nnz, dtype=group["data"].dtype)
    indices = np.empty(nnz, dtype=group["indices"].dtype)
    order = np.argsort(starts, kind="mergesort")
    dest = new_indptr[:-1][order]
    src_start = starts[order]
    src_end = ends[order]
    i = 0
    n = len(order)
    while i < n:
        run_end = i + 1
        while run_end < n and src_start[run_end] == src_end[run_end - 1]:
            run_end += 1
        left = int(src_start[i])
        right = int(src_end[run_end - 1])
        dest_left = int(dest[i])
        dest_right = dest_left + (right - left)
        data[dest_left:dest_right] = group["data"][left:right]
        indices[dest_left:dest_right] = group["indices"][left:right]
        i = run_end
    return sparse.csr_matrix((data, indices, new_indptr), shape=(keep_idx.size, n_vars))


def load_index() -> pd.DataFrame:
    df = pd.read_csv(INDEX, sep="\t")
    df["in_tumor"] = bool_col(df["in_tumor"])
    df["cancer_type"] = df["cancer_type"].astype(str)
    df["sample_id"] = df["sample_id"].astype(str)
    df["nerve_id"] = df["nerve_id"].astype(str)
    df["nerve_key"] = df["nerve_key"].astype(str)
    return df


def v21_data_dir(cancer: str) -> Path:
    for root in V21_ROOTS:
        path = root / cancer / "data"
        if path.is_dir():
            log(f"V21 {cancer} data dir: {path}")
            return path
    raise FileNotFoundError(f"No V21 data dir for {cancer}")


def load_other_v21_cells(cancer: str, samples: set[str]) -> pd.DataFrame:
    data_dir = v21_data_dir(cancer)
    parts = []
    for sample in sorted(samples):
        path = data_dir / f"{sample}_cells.csv"
        if not path.exists():
            log(f"missing cells.csv: {cancer} {sample} ({path})")
            continue
        cells = pd.read_csv(
            path, usecols=["cell_id", "schwann_cluster", "cell_type"], dtype=str
        )
        cells["nerve_id"] = cells["schwann_cluster"].map(normalize_nerve)
        cells = cells.loc[cells["nerve_id"].notna()].copy()
        if cells.empty:
            continue
        cells["cancer_type"] = cancer
        cells["sample_id"] = sample
        parts.append(cells[["cancer_type", "sample_id", "nerve_id", "cell_id", "cell_type"]])
    if not parts:
        return pd.DataFrame(
            columns=["cancer_type", "sample_id", "nerve_id", "cell_id", "cell_type"]
        )
    return pd.concat(parts, ignore_index=True)


def load_identity_for_repair(index: pd.DataFrame) -> pd.DataFrame:
    parts = []
    crc_index = index.loc[index["cancer_type"].eq("CRC")]
    member = pd.read_csv(CRC_MEMBER, sep="\t", dtype=str)
    member = member.loc[member["nerve_key"].isin(set(crc_index["nerve_key"]))]
    crc = member.loc[bool_col(member["is_glial"]) | bool_col(member["is_ens_neuron"])].copy()
    crc["cancer_type"] = "CRC"
    crc["query_cells"] = "Glial_or_ENS_neuron"
    parts.append(crc[["cancer_type", "sample_id", "nerve_id", "cell_id", "query_cells"]])

    for cancer in ("PDAC", "PRAD"):
        samples = set(index.loc[index["cancer_type"].eq(cancer), "sample_id"])
        keep = set(
            zip(
                index.loc[index["cancer_type"].eq(cancer), "sample_id"],
                index.loc[index["cancer_type"].eq(cancer), "nerve_id"],
            )
        )
        v21 = load_other_v21_cells(cancer, samples)
        keep_mask = np.array(
            [(s, n) in keep for s, n in zip(v21["sample_id"], v21["nerve_id"])],
            dtype=bool,
        )
        v21 = v21.loc[v21["cell_type"].eq("Schwann") & keep_mask].copy()
        v21["query_cells"] = "Schwann"
        parts.append(v21[["cancer_type", "sample_id", "nerve_id", "cell_id", "query_cells"]])
    return pd.concat(parts, ignore_index=True)


def extract_gene_means(cells: pd.DataFrame, cancer: str, gene: str = "PDGFA") -> pd.DataFrame:
    path = H5AD[cancer]
    sub = cells.loc[cells["cancer_type"].eq(cancer)].copy()
    if sub.empty:
        return pd.DataFrame()
    samples = set(sub["sample_id"])
    log(f"Extract {gene} from {path} for {len(sub):,} {cancer} cells")
    with h5py.File(path, "r") as handle:
        obs = handle["obs"]
        sample_col = "dataset" if "dataset" in obs else "sample"
        dataset = decode_obs_column(obs, sample_col)
        if "cell_id" in obs:
            cell_id = decode_obs_column(obs, "cell_id")
        else:
            index_name = "_index" if "_index" in obs else "barcode"
            index = decode_obs_column(obs, index_name)
            cell_id = np.array(
                [str(x).rsplit("_", 1)[0] if "_" in str(x) else str(x) for x in index],
                dtype=object,
            )
        keep = np.isin(np.asarray(dataset, dtype=str), list(samples))
        row_idx = np.flatnonzero(keep)
        var = handle["var"]
        var_names = decode(var["_index"][()] if "_index" in var else var[list(var.keys())[0]][()])
        if gene not in set(var_names):
            raise KeyError(f"{gene} missing from {path}")
        gene_i = int(np.where(var_names == gene)[0][0])
        key_to_pos = {
            (str(s), str(c)): int(i)
            for i, s, c in zip(row_idx, dataset[row_idx], cell_id[row_idx])
        }
        sub["obs_pos"] = [
            key_to_pos.get((s, c), np.nan)
            for s, c in zip(sub["sample_id"], sub["cell_id"])
        ]
        matched = sub.loc[sub["obs_pos"].notna()].copy()
        matched["obs_pos"] = matched["obs_pos"].astype(int)
        log(f"{cancer} matched h5ad {len(matched):,} / {len(sub):,}")
        xgrp = handle["X"]
        rows = []
        for (sample, nerve), grp in matched.groupby(["sample_id", "nerve_id"], sort=True):
            pos = np.sort(grp["obs_pos"].to_numpy(dtype=int))
            if isinstance(xgrp, h5py.Group) and "indptr" in xgrp:
                mat = subset_csr_rows(xgrp, pos)[:, [gene_i]]
                mean_val = float(np.asarray(mat.mean()).ravel()[0])
            else:
                mean_val = float(np.asarray(xgrp[pos][:, gene_i], dtype=float).mean())
            rows.append(
                {
                    "cancer_type": cancer,
                    "sample_id": sample,
                    "nerve_id": nerve,
                    "n_cells_expr": int(pos.size),
                    "mean_PDGFA": mean_val,
                    "query_cells": str(grp["query_cells"].iloc[0]),
                }
            )
    return pd.DataFrame(rows)


def safe_cor(x: np.ndarray, y: np.ndarray, method: str) -> tuple[float, float, int]:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    n = int(x.size)
    if n < MIN_COR or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return (np.nan, np.nan, n)
    if method == "pearson":
        r, p = pearsonr(x, y)
    else:
        r, p = spearmanr(x, y)
    return (float(r), float(p), n)


def main() -> None:
    REPAIR_OUT.mkdir(parents=True, exist_ok=True)
    index = load_index()
    identity = load_identity_for_repair(index)
    means = pd.concat(
        [extract_gene_means(identity, cancer) for cancer in CANCERS],
        ignore_index=True,
    )
    means.to_csv(REPAIR_OUT / "nerve_pdgfa_means.tsv", sep="\t", index=False)
    repair = pd.read_csv(REPAIR, sep="\t")
    repair["cancer_type"] = repair["cancer_type"].astype(str)
    repair["sample_id"] = repair["sample_id"].astype(str)
    repair["nerve_id"] = repair["nerve_id"].astype(str)
    repair["Repair_SC"] = pd.to_numeric(repair["Repair_SC"], errors="coerce")
    joined = repair.merge(
        means,
        on=["cancer_type", "sample_id", "nerve_id"],
        how="inner",
    )
    joined = joined.loc[joined["Repair_SC"].notna() & joined["mean_PDGFA"].notna()].copy()
    joined.to_csv(REPAIR_OUT / "repair_vs_pdgfa_joined.tsv", sep="\t", index=False)
    log(f"Repair vs PDGFA joined nerves {len(joined)}")
    print(joined.groupby("cancer_type").size().to_string(), flush=True)

    rows = []
    for cancer, grp in joined.groupby("cancer_type", sort=False):
        r_p, p_p, n = safe_cor(
            grp["Repair_SC"].to_numpy(float), grp["mean_PDGFA"].to_numpy(float), "pearson"
        )
        r_s, p_s, _ = safe_cor(
            grp["Repair_SC"].to_numpy(float), grp["mean_PDGFA"].to_numpy(float), "spearman"
        )
        rows.append(
            {
                "cancer_type": cancer,
                "query_cells": str(grp["query_cells"].iloc[0]) if len(grp) else "",
                "gene": "PDGFA",
                "x": "Repair_SC",
                "y": "mean_PDGFA",
                "n_clusters": n,
                "pearson_r": r_p,
                "pearson_p": p_p,
                "spearman_r": r_s,
                "spearman_p": p_s,
            }
        )
    stats = pd.DataFrame(rows)
    stats.to_csv(REPAIR_OUT / "repair_vs_pdgfa_correlation.tsv", sep="\t", index=False)
    log(stats.to_string(index=False))


if __name__ == "__main__":
    main()
