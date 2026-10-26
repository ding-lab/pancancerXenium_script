#!/usr/bin/env python3
"""Per-TLS T-cell composition in TLS core only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(
    0,
    str(
        Path(
            "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
            "01_cell_typing/01_analysis/03_TLS"
        )
    ),
)
from tls_fibroblast_dc_harmony_umap import (
    DEFAULT_TLS_ROOT,
    as_bool,
    is_brca_al_sample,
    normalize_tls_id,
)

SCRIPT_DIR = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/01_analysis/03_TLS"
)
DEFAULT_SCORES = (
    SCRIPT_DIR / "00_TLS_new/tls_maturation_gc_fdc_score/tls_maturation_scores.tsv"
)
DEFAULT_T_META = (
    SCRIPT_DIR / "tls_tcell_harmony_halo100/annotated/cell_metadata.tsv"
)
DEFAULT_OUT = (
    SCRIPT_DIR / "tls_tcell_harmony_halo100/annotated/composition_core_halo"
)
T_TYPES = ["Naive/TCM", "CXCL13+_T", "Treg", "Th17", "other_CD4+_T", "CD8+_T"]
COMPARTMENTS = ["core"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--t-meta", type=Path, default=DEFAULT_T_META)
    parser.add_argument("--tls-root", type=Path, default=DEFAULT_TLS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--t-types",
        nargs="+",
        default=None,
        help="T types to count. Default: the original six types.",
    )
    return parser.parse_args()


def load_in_tls_core(tls_root: Path, pairs: pd.DataFrame, keep_ids: set[str]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    missing = []
    for cancer, sample in pairs.itertuples(index=False):
        cancer = str(cancer)
        sample = str(sample)
        if is_brca_al_sample(sample, cancer):
            continue
        path = tls_root / cancer / sample / f"{sample}_tls.csv"
        if not path.is_file():
            missing.append(f"{cancer}/{sample}")
            continue
        tls = pd.read_csv(
            path,
            usecols=["cell_id", "tls_region_halo100", "in_tls_core"],
            dtype=str,
            low_memory=False,
        )
        tls["cell_id"] = tls["cell_id"].astype(str)
        tls = tls.loc[tls["cell_id"].isin(keep_ids)].copy()
        if tls.empty:
            continue
        tls["tls_id_csv"] = tls["tls_region_halo100"].map(normalize_tls_id)
        tls["in_tls_core"] = as_bool(tls["in_tls_core"])
        tls = tls.loc[tls["tls_id_csv"].ne(""), ["cell_id", "tls_id_csv", "in_tls_core"]]
        tls["cancer_type"] = cancer
        tls["dataset"] = sample
        parts.append(tls)
        print(f"[TLS] {cancer}/{sample}: {len(tls)} annotated T cells", flush=True)
    if missing:
        print(f"[WARN] missing TLS csv: {len(missing)}", flush=True)
        for item in missing[:20]:
            print(f"  {item}", flush=True)
    if not parts:
        raise RuntimeError(f"No TLS membership found under {tls_root}")
    assigned = pd.concat(parts, ignore_index=True)
    assigned = assigned.drop_duplicates("cell_id", keep="first")
    print(f"[TLS] unique matched T cells: {len(assigned)}", flush=True)
    return assigned


def add_loc_group(dat: pd.DataFrame) -> pd.DataFrame:
    dat = dat.copy()
    dat["loc_group"] = pd.NA
    not_ln = dat["Analysis_Group"].ne("Lymph_nodes")
    dat.loc[dat["location"].eq("Normal") & not_ln, "loc_group"] = "Normal"
    dat.loc[
        dat["location"].isin(["Intratumoral", "Peritumoral"]) & not_ln,
        "loc_group",
    ] = "Tumoral"
    return dat


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t = pd.read_csv(args.t_meta, sep="\t", dtype=str)
    if "barcode" not in t.columns:
        raise KeyError("barcode missing from T-cell metadata")
    t["barcode"] = t["barcode"].astype(str)
    types = list(args.t_types) if args.t_types else list(T_TYPES)
    t = t.loc[t["cell_type"].isin(types)].copy()
    print(f"[T] annotated T cells: {len(t)}", flush=True)
    print(t["cell_type"].value_counts().to_string(), flush=True)

    pairs = t[["cancer_type", "dataset"]].drop_duplicates()
    membership = load_in_tls_core(args.tls_root, pairs, set(t["barcode"]))
    t = t.merge(membership[["cell_id", "tls_id_csv", "in_tls_core"]], left_on="barcode", right_on="cell_id", how="left")
    unmatched = t["in_tls_core"].isna().sum()
    print(f"[JOIN] unmatched barcodes: {int(unmatched)} / {len(t)}", flush=True)
    if "tls_id" in t.columns:
        mismatch = (
            t["tls_id"].notna()
            & t["tls_id_csv"].notna()
            & t["tls_id"].astype(str).ne(t["tls_id_csv"].astype(str))
        )
        print(f"[JOIN] tls_id mismatches: {int(mismatch.sum())}", flush=True)

    t = t.loc[t["in_tls_core"].eq(True)].copy()
    t["compartment"] = "core"
    t["tls_region"] = t["tls_id_csv"].where(t["tls_id_csv"].notna(), t.get("tls_id"))
    t["sample"] = t["dataset"].astype(str)

    cell_out = t[
        [
            "barcode",
            "dataset",
            "cancer_type",
            "cell_type",
            "tls_region",
            "compartment",
            "in_tls_core",
        ]
    ].copy()
    cell_out.to_csv(args.out_dir / "cell_metadata_with_compartment.tsv", sep="\t", index=False)

    cell_n = (
        t.groupby(["compartment", "cell_type"], observed=True)
        .size()
        .reset_index(name="n_cells")
        .sort_values(["compartment", "cell_type"])
    )
    cell_n.to_csv(args.out_dir / "cell_type_counts_by_compartment.tsv", sep="\t", index=False)
    print(cell_n.to_string(index=False), flush=True)

    t_tls = (
        t.groupby(["cancer_type", "sample", "tls_region", "compartment", "cell_type"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in types:
        if col not in t_tls.columns:
            t_tls[col] = 0
    t_tls["n_T"] = t_tls[types].sum(axis=1)

    scores = pd.read_csv(args.scores, sep="\t")
    base = scores.copy()
    base["key"] = 1
    comps = pd.DataFrame({"compartment": COMPARTMENTS, "key": 1})
    dat = base.merge(comps, on="key").drop(columns="key")
    keys = ["cancer_type", "sample", "tls_region", "compartment"]
    dat = dat.merge(t_tls[keys + types + ["n_T"]], on=keys, how="left")
    for col in types + ["n_T"]:
        dat[col] = dat[col].fillna(0).astype(int)
    for col in types:
        dat[f"frac_{col}"] = dat[col] / dat["n_T"].where(dat["n_T"] > 0)
    dat = add_loc_group(dat)
    dat["compartment"] = pd.Categorical(dat["compartment"], categories=COMPARTMENTS)

    out = args.out_dir / "tls_tcell_composition_core_halo.tsv"
    dat.to_csv(out, sep="\t", index=False)
    print(dat.groupby(["compartment", "loc_group"], dropna=False).size().to_string(), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
