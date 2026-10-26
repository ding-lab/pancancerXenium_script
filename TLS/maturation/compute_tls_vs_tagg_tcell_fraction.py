#!/usr/bin/env python3
"""TLS-core vs T-aggregate T-cell fractions, Normal vs Tumoral.

TLS subtypes come from plus_tagg CXCL13-merged composition (official TLS catalog).
Tagg subtypes come from the same annotation on tagg_core_only cells.

Also writes structure-level T / all-cells: official TLS-core frac_T, and
Tagg n_T / location n_cells.

Does not overwrite tcell_composition_core_halo or tls_broad_composition tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/01_analysis/03_TLS"
)
NEW = ROOT / "00_TLS_new"
T_TYPES = [
    "Naive/TCM",
    "Tfh",
    "Treg",
    "Th17",
    "IFN_response_T",
    "other_CD4+_T",
    "CD8+_T",
]
DEFAULT_TLS = NEW / "tcell_composition_core_halo_plus_tagg_cxcl13_merged/tls_tcell_composition_core_halo.tsv"
DEFAULT_T_META = (
    ROOT / "tls_tcell_harmony_halo100_plus_tagg/annotated_res1/"
    "cell_metadata_cxcl13_merged_otherCD4.tsv"
)
DEFAULT_LOC = NEW / "tls_spatial_info/gen11_TLS_Taggregate_locations_all.tsv"
DEFAULT_BROAD = NEW / "tls_broad_composition/tls_broad_composition.tsv"
DEFAULT_OUT = NEW / "tcell_fraction_tls_vs_tagg"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tls-comp", type=Path, default=DEFAULT_TLS)
    p.add_argument("--t-meta", type=Path, default=DEFAULT_T_META)
    p.add_argument("--locations", type=Path, default=DEFAULT_LOC)
    p.add_argument("--broad", type=Path, default=DEFAULT_BROAD)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def loc_group(location: pd.Series, analysis_group: pd.Series) -> pd.Series:
    loc = location.replace({"Boundary": "Peritumoral"})
    out = pd.Series(pd.NA, index=location.index, dtype="object")
    not_ln = analysis_group.astype(str).ne("Lymph_nodes")
    out.loc[loc.eq("Normal") & not_ln] = "Normal"
    out.loc[loc.isin(["Intratumoral", "Peritumoral"]) & not_ln] = "Tumoral"
    return out


def add_fracs(dat: pd.DataFrame) -> pd.DataFrame:
    for col in T_TYPES:
        if col not in dat.columns:
            dat[col] = 0
    dat["n_T"] = dat[T_TYPES].sum(axis=1)
    for col in T_TYPES:
        dat[f"frac_{col}"] = dat[col] / dat["n_T"].where(dat["n_T"] > 0)
    return dat


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    tls = pd.read_csv(args.tls_comp, sep="\t")
    tls = tls.loc[tls["compartment"].astype(str).eq("core")].copy()
    tls["structure_type"] = "TLS"
    tls["structure_id"] = tls["tls_region"].astype(str)
    tls = add_fracs(tls)
    if "loc_group" not in tls.columns:
        tls["loc_group"] = loc_group(tls["location"], tls["Analysis_Group"])
    print(f"[TLS] core rows {len(tls)}  loc\n{tls['loc_group'].value_counts(dropna=False).to_string()}", flush=True)

    meta = pd.read_csv(args.t_meta, sep="\t", dtype=str)
    tagg_cells = meta.loc[meta["tagg_id"].fillna("").ne("")].copy()
    tagg_cells = tagg_cells.loc[~tagg_cells["cancer_type"].str.upper().isin(["LN", "LYMPH_NODES"])]
    tagg_cells = tagg_cells.loc[
        ~(tagg_cells["cancer_type"].eq("BRCA") & tagg_cells["dataset"].str.startswith("AL"))
    ]
    print(f"[TAGG cells] {len(tagg_cells)}", flush=True)
    print(tagg_cells["cell_type"].value_counts().to_string(), flush=True)

    loc = pd.read_csv(args.locations, sep="\t")
    loc_t = loc.loc[loc["structure_type"].astype(str).eq("T_aggregate")].copy()
    loc_t["structure_id"] = loc_t["structure_id"].astype(str)
    loc_t["sample"] = loc_t["sample"].astype(str)
    loc_t["cancer_type"] = loc_t["cancer_type"].astype(str)
    loc_t["location"] = loc_t["location"].replace({"Boundary": "Peritumoral"})
    loc_t["loc_group"] = loc_group(loc_t["location"], loc_t["Analysis_Group"])

    counts = (
        tagg_cells.groupby(["cancer_type", "dataset", "tagg_id", "cell_type"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"dataset": "sample", "tagg_id": "structure_id"})
    )
    tagg = loc_t.merge(counts, on=["cancer_type", "sample", "structure_id"], how="left")
    tagg = add_fracs(tagg)
    tagg["structure_type"] = "T_aggregate"
    unmatched = int((tagg["n_T"] == 0).sum())
    print(f"[TAGG] location rows {len(tagg)}  n_T==0 {unmatched}", flush=True)
    print(tagg["loc_group"].value_counts(dropna=False).to_string(), flush=True)

    keep = [
        "structure_type",
        "cancer_type",
        "sample",
        "structure_id",
        "location",
        "loc_group",
        "Analysis_Group",
        "n_T",
        *T_TYPES,
        *[f"frac_{c}" for c in T_TYPES],
    ]
    if "n_cells" in tagg.columns:
        keep.append("n_cells")
    if "kmeans_stage" in tls.columns:
        tls_keep = keep + ["kmeans_stage"]
    else:
        tls_keep = keep
    tls_out = tls[[c for c in tls_keep if c in tls.columns]].copy()
    tagg_out = tagg[[c for c in keep if c in tagg.columns]].copy()
    out = pd.concat([tls_out, tagg_out], ignore_index=True)
    out.to_csv(args.out_dir / "tls_vs_tagg_tcell_composition.tsv", sep="\t", index=False)

    broad = pd.read_csv(args.broad, sep="\t")
    tls_frac = broad[["cancer_type", "sample", "tls_id", "loc_group", "frac_T", "T", "n_core"]].copy()
    tls_frac = tls_frac.rename(columns={"tls_id": "structure_id", "T": "n_T", "n_core": "n_cells"})
    tls_frac["structure_type"] = "TLS"
    tagg_frac = tagg[["cancer_type", "sample", "structure_id", "loc_group", "n_T", "n_cells"]].copy()
    tagg_frac["n_cells"] = pd.to_numeric(tagg_frac["n_cells"], errors="coerce")
    tagg_frac["frac_T"] = tagg_frac["n_T"] / tagg_frac["n_cells"].where(tagg_frac["n_cells"] > 0)
    tagg_frac["structure_type"] = "T_aggregate"
    frac = pd.concat([tls_frac, tagg_frac], ignore_index=True)
    frac.to_csv(args.out_dir / "tls_vs_tagg_T_fraction.tsv", sep="\t", index=False)
    print(frac.groupby(["structure_type", "loc_group"], dropna=False).size().to_string(), flush=True)
    print(f"Wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
