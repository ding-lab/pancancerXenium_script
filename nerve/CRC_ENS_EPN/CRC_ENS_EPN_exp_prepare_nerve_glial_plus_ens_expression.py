#!/usr/bin/env python3

from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd


SCH = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/01_analysis/04_schwann"
)
BASE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else (
    SCH / "0_crcV21" / "boundary_CRC"
)
MEMBERSHIP = BASE / "nerve_cell_membership.tsv"
INDEX = BASE / "nerve_index.tsv"
OUT = BASE / "analysis_tables"
H5AD = SCH.parents[1] / "CRC" / "annotated.Yuwei.h5ad"
GENES = (
    "SYP",
    "FGF1",
    "NRG1",
    "SNAP25",
    "STX1A",
    "VIP",
    "RET",
    "PHOX2B",
    "SOX10",
    "UCHL1",
)
TARGET_SUM = 10_000.0


def decode(values) -> np.ndarray:
    return np.asarray(
        [
            value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in values
        ],
        dtype=object,
    )


def locate_rows(wanted: set[str]) -> pd.DataFrame:
    rows: list[int] = []
    barcodes: list[str] = []
    with h5py.File(H5AD, "r") as handle:
        dataset = handle["obs"]["barcode"]
        chunk_size = 500_000
        for start in range(0, len(dataset), chunk_size):
            stop = min(start + chunk_size, len(dataset))
            values = decode(dataset[start:stop])
            matches = np.fromiter(
                (
                    offset
                    for offset, barcode in enumerate(values)
                    if barcode in wanted
                ),
                dtype=np.int64,
            )
            rows.extend((start + matches).tolist())
            barcodes.extend(values[matches].tolist())
            print(f"barcode scan {stop:,}/{len(dataset):,}", flush=True)
    selected = pd.DataFrame({"row_id": rows, "barcode": barcodes})
    if len(selected) != len(wanted):
        raise RuntimeError(
            f"Matched {len(selected):,}/{len(wanted):,} membership barcodes"
        )
    return selected.sort_values("row_id").reset_index(drop=True)


def extract_expression(selected: pd.DataFrame) -> pd.DataFrame:
    rows = selected["row_id"].to_numpy(np.int64)
    with h5py.File(H5AD, "r") as handle:
        genes = decode(handle["var"]["_index"][()])
        lookup = {gene: index for index, gene in enumerate(genes)}
        missing = [gene for gene in GENES if gene not in lookup]
        if missing:
            raise RuntimeError(f"Missing genes: {missing}")
        feature_to_column = {
            lookup[gene]: column for column, gene in enumerate(GENES)
        }
        matrix = handle["layers"]["counts"]
        indptr = matrix["indptr"]
        indices = matrix["indices"]
        data = matrix["data"]
        library = np.asarray(handle["obs"]["n_counts"][rows], dtype=float)
        expression = np.zeros((len(rows), len(GENES)), dtype=np.float32)
        for output_row, row in enumerate(rows):
            left, right = int(indptr[row]), int(indptr[row + 1])
            one_indices = np.asarray(indices[left:right], dtype=np.int64)
            one_values = np.asarray(data[left:right], dtype=float)
            for feature, value in zip(one_indices, one_values):
                column = feature_to_column.get(int(feature))
                if column is not None:
                    expression[output_row, column] = value
        valid = library > 0
        expression[valid] *= (TARGET_SUM / library[valid])[:, None]
        np.log1p(expression, out=expression)
    result = selected[["barcode"]].copy()
    for column, gene in enumerate(GENES):
        result[gene] = expression[:, column]
    return result


def aggregate(cells: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return cells.groupby(keys, observed=True)[list(GENES)].mean().reset_index()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    membership = pd.read_csv(MEMBERSHIP, sep="\t")
    index = pd.read_csv(INDEX, sep="\t")
    metadata = index[
        [
            "nerve_key",
            "sample_id",
            "nerve_id",
            "region",
            "in_tumor",
            "nerve_class",
            "n_cells",
            "n_glial",
            "n_ens_neuron",
            "peripheral_candidate",
        ]
    ].copy()

    locations = locate_rows(set(membership["barcode"]))
    expression = extract_expression(locations)
    cells = membership[["nerve_key", "barcode", "cell_type"]].merge(
        expression,
        on="barcode",
        how="inner",
        validate="one_to_one",
    )

    all_expression = aggregate(cells, ["nerve_key"]).merge(
        metadata,
        on="nerve_key",
        how="left",
        validate="one_to_one",
    )
    all_expression.to_csv(OUT / "nerve_all_cell_expression.tsv", sep="\t", index=False)

    neural_cells = cells.loc[cells["cell_type"].isin(["Glial", "ENS_neuron"])].copy()
    neural_expression = aggregate(neural_cells, ["nerve_key"]).merge(
        metadata,
        on="nerve_key",
        how="left",
        validate="one_to_one",
    )
    neural_expression["n_glial_plus_ens"] = (
        neural_expression["n_glial"] + neural_expression["n_ens_neuron"]
    )
    neural_path = OUT / "nerve_glial_plus_ens_expression.tsv"
    neural_expression.to_csv(neural_path, sep="\t", index=False)

    by_type = aggregate(neural_cells, ["nerve_key", "cell_type"]).merge(
        metadata,
        on="nerve_key",
        how="left",
        validate="many_to_one",
    )
    type_counts = (
        neural_cells.groupby(["nerve_key", "cell_type"], observed=True)["barcode"]
        .nunique()
        .rename("n_expression_cells")
        .reset_index()
    )
    by_type = by_type.merge(
        type_counts,
        on=["nerve_key", "cell_type"],
        how="left",
        validate="one_to_one",
    )
    by_type.to_csv(OUT / "nerve_expression_by_cell_type.tsv", sep="\t", index=False)
    print(
        f"nerves={len(index):,}; cells={len(cells):,}; "
        f"Glial+ENS={len(neural_cells):,}",
        flush=True,
    )
    print(f"DONE {neural_path}", flush=True)


if __name__ == "__main__":
    main()
