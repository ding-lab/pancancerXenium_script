#!/usr/bin/env python3

from pathlib import Path

import pandas as pd


SCH = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/01_analysis/04_schwann"
)
BCRC = SCH / "0_crcV21" / "boundary_CRC"
TABLES = BCRC / "enteric_peripheral_analysis" / "tables"
CRC_INDEX = BCRC / "nerve_index.tsv"
CRC_SCORES = BCRC / "repair_genecorr" / "gn_nerve_mean_scores.tsv"
OTHER_INDEX = TABLES / "v21_crc_pdac_prad_nerve_index.tsv"
V21 = Path(
    "/rdcw/fs1/dinglab/Active/Projects/pdc/projects/general/x1000/"
    "schwann/spatial_define/v21_v11-glialens-sox10uchl1"
)
SCORE_PATH = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "00_makeh5ad/01_x5k/00_integration/schwann/SC_score_allcell.csv"
)
MIN_SCORE_CELLS = 5


def log(message: str) -> None:
    print(message, flush=True)


def normalize_nerve(value: object) -> str | None:
    text = str(value).strip()
    if not text or text.lower() in {"none", "nan", "na"}:
        return None
    if text.lower().startswith("schwann-"):
        text = text.split("-", 1)[1]
    return f"Schwann-{int(float(text))}"


def score_pdac_prad_repair(index: pd.DataFrame) -> pd.DataFrame:
    eligible = index.loc[index["n_schwann"].ge(MIN_SCORE_CELLS)].copy()
    eligible_keys = set(
        zip(eligible["cancer_type"], eligible["sample_id"], eligible["nerve_id"])
    )
    scores = pd.read_csv(SCORE_PATH)
    scores["cancer_type"] = scores["cancer_type"].astype(str)
    scores = scores.loc[scores["cancer_type"].isin(["PDAC", "PRAD"])].copy()
    split = scores["barcode"].str.rsplit("_", n=1, expand=True)
    scores["cell_id"] = split[0]
    sample_cancer = split[1].str.rsplit("-", n=1, expand=True)
    scores["sample_id"] = sample_cancer[0]
    pieces: list[pd.DataFrame] = []
    for cancer in ("PDAC", "PRAD"):
        data_dir = V21 / "with_expression/outputs" / cancer / "data"
        cancer_samples = sorted(
            eligible.loc[eligible["cancer_type"].eq(cancer), "sample_id"].unique()
        )
        for sample in cancer_samples:
            path = data_dir / f"{sample}_cells.csv"
            if not path.exists():
                log(f"missing cells.csv for repair: {cancer} {sample}")
                continue
            cells = pd.read_csv(
                path,
                usecols=["cell_id", "schwann_cluster", "cell_type"],
                dtype=str,
            )
            cells["nerve_id"] = cells["schwann_cluster"].map(normalize_nerve)
            cells = cells.loc[
                cells["nerve_id"].notna() & cells["cell_type"].eq("Schwann")
            ].copy()
            if cells.empty:
                continue
            cells["cancer_type"] = cancer
            cells["sample_id"] = sample
            keep = [
                (cancer, sample, nerve) in eligible_keys
                for nerve in cells["nerve_id"]
            ]
            cells = cells.loc[keep].copy()
            if cells.empty:
                continue
            pieces.append(cells)
    if not pieces:
        return pd.DataFrame()
    membership = pd.concat(pieces, ignore_index=True)
    merged = membership.merge(
        scores[
            [
                "cell_id",
                "sample_id",
                "cancer_type",
                "Myelinating_SC",
                "Non_myelinating_SC",
                "Precursor_SC",
                "Repair_SC",
            ]
        ],
        on=["cell_id", "sample_id", "cancer_type"],
        how="left",
    )
    scored = merged.groupby(
        ["cancer_type", "sample_id", "nerve_id"], as_index=False
    ).agg(
        n_schwann_scored=("Repair_SC", lambda x: int(x.notna().sum())),
        Myelinating_SC=("Myelinating_SC", "mean"),
        Non_myelinating_SC=("Non_myelinating_SC", "mean"),
        Precursor_SC=("Precursor_SC", "mean"),
        Repair_SC=("Repair_SC", "mean"),
    )
    return scored.loc[scored["n_schwann_scored"].ge(MIN_SCORE_CELLS)].copy()


def crc_glial_repair() -> pd.DataFrame:
    index = pd.read_csv(CRC_INDEX, sep="\t")
    scores = pd.read_csv(CRC_SCORES, sep="\t")
    scores = scores.drop(columns=["in_tumor", "region"], errors="ignore").merge(
        index[["nerve_key", "sample_id", "nerve_id", "region"]],
        on="nerve_key",
        how="left",
        suffixes=("", "_idx"),
    )
    scores = scores.loc[
        scores["Repair_SC"].notna()
        & scores["n_glial_scored"].ge(MIN_SCORE_CELLS)
        & scores["region"].notna()
    ].copy()
    scores["cancer_type"] = "CRC"
    scores["nerve_group"] = scores["region"].str.lower()
    scores["source"] = "v21_glial_ge5"
    return scores[
        [
            "cancer_type",
            "sample_id",
            "nerve_id",
            "nerve_group",
            "Repair_SC",
            "source",
            "n_glial_scored",
        ]
    ].rename(columns={"n_glial_scored": "n_cells_scored"})


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    other = pd.read_csv(OTHER_INDEX, sep="\t")
    other = other.loc[other["cancer_type"].isin(["PDAC", "PRAD"])].copy()
    other["n_schwann"] = pd.to_numeric(other["n_schwann"], errors="coerce").fillna(0)
    other_repair = score_pdac_prad_repair(other)
    other_repair = other_repair.merge(
        other[["cancer_type", "sample_id", "nerve_id", "region"]],
        on=["cancer_type", "sample_id", "nerve_id"],
        how="left",
    )
    other_repair["nerve_group"] = other_repair["region"].str.lower()
    other_repair["source"] = "v21_schwann_ge5"
    other_repair = other_repair[
        [
            "cancer_type",
            "sample_id",
            "nerve_id",
            "nerve_group",
            "Repair_SC",
            "source",
            "n_schwann_scored",
        ]
    ].rename(columns={"n_schwann_scored": "n_cells_scored"})
    repair = pd.concat([crc_glial_repair(), other_repair], ignore_index=True)
    out = TABLES / "v21_crc_pdac_prad_repair_scores.tsv"
    repair.to_csv(out, sep="\t", index=False)
    log(
        repair.groupby(["cancer_type", "nerve_group"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )
    log(f"DONE {out}")


if __name__ == "__main__":
    main()
