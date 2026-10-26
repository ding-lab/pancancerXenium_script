#!/usr/bin/env python3
"""Export one LUNG h5ad per sample for tumoral-TLS CellChat.

Tumoral TLS are Intratumoral or Boundary (renamed Peritumoral) in the current
LUNG boundary-based location table. For each sample, all cells assigned to the
core or cumulative halo100 of those TLS are exported together. Raw counts,
spatial coordinates, current annotation.csv cell types, TLS ID/location, and
core-versus-halo scope are retained.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]
DEFAULT_H5AD = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/LUNG/annotated.Yuwei.new.h5ad"
)
DEFAULT_ANNOTATION = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/LUNG/annotation.csv"
)
DEFAULT_TLS_ROOT = Path(
    "/storage1/fs1/dinglab/Active/Projects/pdc/projects/general/x1000/"
    "tls/tls_define/gen12/outputs/tls/LUNG"
)
DEFAULT_LOCATIONS = BASE / "tls_spatial_info/gen11_TLS_locations_assign_v3.tsv"
DEFAULT_OUT = ROOT / "h5ad_by_sample"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--tls-root", type=Path, default=DEFAULT_TLS_ROOT)
    parser.add_argument("--locations", type=Path, default=DEFAULT_LOCATIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in values
        ],
        dtype=object,
    )


def as_bool(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values.fillna(False)
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "t", "yes"})
    )


def is_outside(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin(
        {"outside", "", "nan", "none"}
    )


def load_tumoral_tls(locations_path: Path) -> pd.DataFrame:
    locations = pd.read_csv(
        locations_path,
        sep="\t",
        usecols=["cancer_type", "sample", "structure_id", "location"],
        dtype=str,
    )
    locations = locations.loc[
        locations["cancer_type"].eq("LUNG")
        & locations["location"].isin(["Intratumoral", "Boundary"])
    ].copy()
    locations["spatial_location"] = locations["location"].replace(
        {"Boundary": "Peritumoral"}
    )
    locations = locations.rename(columns={"structure_id": "tls_id"})
    return locations[
        ["sample", "tls_id", "spatial_location"]
    ].drop_duplicates()


def load_memberships(
    tumoral_tls: pd.DataFrame,
    tls_root: Path,
) -> pd.DataFrame:
    parts = []
    for sample, sample_tls in tumoral_tls.groupby("sample", sort=True):
        path = tls_root / sample / f"{sample}_tls.csv"
        if not path.is_file():
            print(f"[WARN] Missing TLS table: {path}")
            continue
        cells = pd.read_csv(
            path,
            usecols=["cell_id", "tls_region_halo100", "in_tls_core"],
            dtype=str,
            low_memory=False,
        )
        cells = cells.rename(columns={"tls_region_halo100": "tls_id"})
        cells = cells.loc[~is_outside(cells["tls_id"])].copy()
        cells = cells.merge(
            sample_tls[["tls_id", "spatial_location"]],
            on="tls_id",
            how="inner",
        )
        if cells.empty:
            continue
        cells["sample"] = sample
        cells["in_tls_core"] = as_bool(cells["in_tls_core"])
        cells["tls_scope"] = np.where(
            cells["in_tls_core"], "TLS_core", "TLS_halo100_only"
        )
        parts.append(
            cells[
                [
                    "cell_id",
                    "sample",
                    "tls_id",
                    "spatial_location",
                    "in_tls_core",
                    "tls_scope",
                ]
            ]
        )
    memberships = pd.concat(parts, ignore_index=True)
    duplicated = memberships["cell_id"].duplicated(keep=False)
    if duplicated.any():
        memberships = memberships.sort_values(
            ["cell_id", "in_tls_core"], ascending=[True, False]
        ).drop_duplicates("cell_id")
    return memberships


def attach_current_annotations(
    memberships: pd.DataFrame,
    annotation_path: Path,
) -> pd.DataFrame:
    wanted = set(memberships["cell_id"])
    parts = []
    for chunk in pd.read_csv(
        annotation_path,
        usecols=["index", "dataset", "cell_type"],
        dtype=str,
        chunksize=500_000,
        low_memory=False,
    ):
        chunk = chunk.loc[chunk["index"].isin(wanted)]
        if not chunk.empty:
            parts.append(chunk)
    annotations = pd.concat(parts, ignore_index=True).rename(
        columns={"index": "cell_id", "dataset": "annotation_sample"}
    )
    annotations = annotations.drop_duplicates("cell_id")
    result = memberships.merge(annotations, on="cell_id", how="left")
    mismatch = (
        result["annotation_sample"].notna()
        & result["annotation_sample"].ne(result["sample"])
    )
    if mismatch.any():
        raise ValueError(
            f"{mismatch.sum()} cell IDs have inconsistent sample annotations"
        )
    missing = result["cell_type"].isna()
    if missing.any():
        print(
            f"[WARN] Dropping {missing.sum()} cells absent from annotation.csv"
        )
        result = result.loc[~missing].copy()
    return result.drop(columns=["annotation_sample"])


def locate_h5ad_rows(
    handle: h5py.File,
    memberships: pd.DataFrame,
) -> pd.DataFrame:
    obs = handle["obs"]
    index_key = obs.attrs["_index"]
    index = obs[index_key]
    wanted = set(memberships["cell_id"])
    row_lookup: dict[str, int] = {}
    chunk_size = 250_000
    for start in range(0, len(index), chunk_size):
        stop = min(start + chunk_size, len(index))
        cell_ids = decode(index[start:stop])
        for offset, cell_id in enumerate(cell_ids):
            if cell_id in wanted:
                row_lookup[cell_id] = start + offset
    result = memberships.copy()
    result["h5ad_row"] = result["cell_id"].map(row_lookup)
    missing = result["h5ad_row"].isna()
    if missing.any():
        print(f"[WARN] Dropping {missing.sum()} cells absent from source h5ad")
        result = result.loc[~missing].copy()
    result["h5ad_row"] = result["h5ad_row"].astype(np.int64)
    return result


def take_rows(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows)
    values = dataset[rows[order]]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return values[inverse]


def take_obs_values(
    element: h5py.Dataset | h5py.Group,
    rows: np.ndarray,
) -> np.ndarray:
    """Read a dense/string or categorical h5ad obs column."""
    if isinstance(element, h5py.Group):
        if "categories" not in element or "codes" not in element:
            raise ValueError("Unsupported h5ad obs group encoding")
        categories = decode(element["categories"][:])
        codes = take_rows(element["codes"], rows).astype(int)
        values = np.full(len(codes), pd.NA, dtype=object)
        valid = codes >= 0
        values[valid] = categories[codes[valid]]
        return values
    values = take_rows(element, rows)
    if values.dtype.kind in {"O", "S", "U"}:
        return decode(values)
    return values


def extract_csr_rows(
    matrix: h5py.Group,
    rows: np.ndarray,
) -> sparse.csr_matrix:
    shape = tuple(matrix.attrs["shape"])
    source_indptr = matrix["indptr"][:]
    data_parts = []
    indices_parts = []
    output_indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    for output_row, source_row in enumerate(rows):
        start = int(source_indptr[source_row])
        stop = int(source_indptr[source_row + 1])
        data_parts.append(matrix["data"][start:stop])
        indices_parts.append(matrix["indices"][start:stop])
        output_indptr[output_row + 1] = (
            output_indptr[output_row] + stop - start
        )
    return sparse.csr_matrix(
        (
            np.concatenate(data_parts),
            np.concatenate(indices_parts),
            output_indptr,
        ),
        shape=(len(rows), shape[1]),
    )


def export_sample(
    handle: h5py.File,
    sample_cells: pd.DataFrame,
    var_names: np.ndarray,
    out_path: Path,
) -> None:
    rows = sample_cells["h5ad_row"].to_numpy(np.int64)
    counts_group = handle["layers"]["counts"]
    counts = extract_csr_rows(counts_group, rows)
    spatial = take_rows(handle["obsm"]["spatial"], rows).astype(np.float32)
    cell_type_mcell = take_obs_values(
        handle["obs"]["cell_type_mcell"],
        rows,
    )

    obs = sample_cells.set_index("cell_id")[
        [
            "sample",
            "cell_type",
            "tls_id",
            "spatial_location",
            "in_tls_core",
            "tls_scope",
        ]
    ].copy()
    obs["cell_type_mcell"] = cell_type_mcell
    obs.index.name = "cell_id"
    for column in (
        "sample",
        "cell_type",
        "cell_type_mcell",
        "tls_id",
        "spatial_location",
        "tls_scope",
    ):
        obs[column] = obs[column].astype("category")

    output = ad.AnnData(
        X=counts.copy(),
        obs=obs,
        var=pd.DataFrame(index=pd.Index(var_names, name="gene")),
        obsm={"spatial": spatial},
    )
    output.layers["counts"] = counts
    output.uns["selection"] = {
        "cancer_type": "LUNG",
        "tls_locations": ["Intratumoral", "Peritumoral"],
        "included_scopes": ["TLS_core", "TLS_halo100_only"],
        "halo_um": 100,
        "core_included": True,
    }
    output.write_h5ad(out_path, compression="gzip")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tumoral_tls = load_tumoral_tls(args.locations)
    memberships = load_memberships(tumoral_tls, args.tls_root)
    memberships = attach_current_annotations(memberships, args.annotation)

    manifest_rows = []
    with h5py.File(args.h5ad, "r") as handle:
        var = handle["var"]
        var_names = decode(var[var.attrs["_index"]][:])
        memberships = locate_h5ad_rows(handle, memberships)

        for sample, sample_cells in memberships.groupby("sample", sort=True):
            out_path = args.out_dir / f"{sample}.h5ad"
            n_cells = len(sample_cells)
            status = "written"
            if n_cells < args.min_cells:
                status = f"skipped_lt_{args.min_cells}_cells"
            elif out_path.is_file() and not args.overwrite:
                status = "skipped_exists"
            else:
                export_sample(handle, sample_cells, var_names, out_path)
            manifest_rows.append(
                {
                    "sample": sample,
                    "n_cells": n_cells,
                    "n_tls": sample_cells["tls_id"].nunique(),
                    "n_tls_core_cells": int(sample_cells["in_tls_core"].sum()),
                    "n_halo100_only_cells": int(
                        (~sample_cells["in_tls_core"]).sum()
                    ),
                    "cell_type_counts": ";".join(
                        f"{name}:{count}"
                        for name, count in sample_cells["cell_type"]
                        .value_counts()
                        .sort_index()
                        .items()
                    ),
                    "h5ad_path": str(out_path),
                    "status": status,
                }
            )
            print(f"{sample}: {status}; cells={n_cells}")

    memberships.drop(columns=["h5ad_row"]).to_csv(
        args.out_dir / "selected_cell_membership.tsv.gz",
        sep="\t",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(manifest_rows).to_csv(
        args.out_dir / "sample_manifest.tsv",
        sep="\t",
        index=False,
    )
    tumoral_tls.to_csv(
        args.out_dir / "selected_tumoral_TLS.tsv",
        sep="\t",
        index=False,
    )
    print(f"Wrote outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
