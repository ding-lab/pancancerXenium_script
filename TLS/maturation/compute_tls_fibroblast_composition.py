#!/usr/bin/env python3
"""Per-TLS fibroblast subtype composition for official GC x FDC stages.

Uses tls_stromal_fdc_harmony annotated TLS-core fibroblasts:
  FDC, CXCL13+ TRC, CXCL9+ activated TRC, MRC, Fibroblast

Fractions are among those annotated fibroblasts. Lymph nodes are excluded
from loc_group (Normal / Tumoral). Official stage is kmeans_stage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

NEW = Path(__file__).resolve().parent
ROOT = NEW.parent
DEFAULT_SCORES = NEW / "tls_maturation_gc_fdc_score/tls_maturation_scores.tsv"
DEFAULT_STROMAL = (
    ROOT
    / "tls_stromal_fdc_harmony/filtered_drop242526/annotated_res0.8/cell_metadata.tsv"
)
DEFAULT_OUT = NEW / "fibroblast_composition"

FIBRO_TYPES = [
    "FDC",
    "CXCL13+ TRC",
    "CXCL9+ activated TRC",
    "MRC",
    "Fibroblast",
]
COUNT_COLS = {
    "FDC": "n_FDC_anno",
    "CXCL13+ TRC": "n_CXCL13_TRC_anno",
    "CXCL9+ activated TRC": "n_CXCL9_TRC",
    "MRC": "n_MRC_anno",
    "Fibroblast": "n_generic_Fibroblast",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    p.add_argument("--stromal-meta", type=Path, default=DEFAULT_STROMAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


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

    stromal = pd.read_csv(args.stromal_meta, sep="\t", dtype=str)
    stromal = stromal.loc[stromal["cell_type"].isin(FIBRO_TYPES)].copy()
    if "sample" not in stromal.columns or stromal["sample"].astype(str).eq("").all():
        stromal["sample"] = stromal["dataset"].astype(str)
    else:
        stromal["sample"] = stromal["sample"].astype(str)
    stromal["tls_region"] = stromal["tls_id"].astype(str)
    print(f"[FIBRO] annotated TLS-core fibroblasts: {len(stromal)}", flush=True)
    print(stromal["cell_type"].value_counts().to_string(), flush=True)

    s_tls = (
        stromal.groupby(
            ["cancer_type", "sample", "tls_region", "cell_type"], observed=True
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    for col in FIBRO_TYPES:
        if col not in s_tls.columns:
            s_tls[col] = 0
    rename = {src: COUNT_COLS[src] for src in FIBRO_TYPES}
    s_tls = s_tls.rename(columns=rename)
    count_names = list(COUNT_COLS.values())
    s_tls["n_fibroblast_anno"] = s_tls[count_names].sum(axis=1)

    scores = pd.read_csv(args.scores, sep="\t")
    scores["location"] = scores["location"].replace({"Boundary": "Peritumoral"})
    keys = ["cancer_type", "sample", "tls_region"]
    dat = scores.merge(s_tls[keys + count_names + ["n_fibroblast_anno"]], on=keys, how="left")
    for col in count_names + ["n_fibroblast_anno"]:
        dat[col] = dat[col].fillna(0).astype(int)

    n = dat["n_fibroblast_anno"]
    for src, col in COUNT_COLS.items():
        dat[f"frac_{src}"] = dat[col] / n.where(n > 0)
    dat["frac_FDC_anno"] = dat["frac_FDC"]
    dat["frac_CXCL13_TRC"] = dat["frac_CXCL13+ TRC"]
    dat["frac_CXCL9_TRC"] = dat["frac_CXCL9+ activated TRC"]
    dat["frac_MRC_anno"] = dat["frac_MRC"]
    dat["frac_generic_Fibroblast"] = dat["frac_Fibroblast"]

    # Official 4-type fractions from the maturation score table (other includes CXCL9+ TRC).
    n4 = (
        pd.to_numeric(dat["n_CXCL13_TRC"], errors="coerce").fillna(0)
        + pd.to_numeric(dat["n_FDC"], errors="coerce").fillna(0)
        + pd.to_numeric(dat["n_MRC"], errors="coerce").fillna(0)
        + pd.to_numeric(dat["n_otherFibroblast"], errors="coerce").fillna(0)
    )
    dat["n_fibroblast_score"] = n4.astype(int)
    dat["frac_FDC_score"] = pd.to_numeric(dat["n_FDC"], errors="coerce") / n4.where(n4 > 0)
    dat["frac_CXCL13_TRC_score"] = (
        pd.to_numeric(dat["n_CXCL13_TRC"], errors="coerce") / n4.where(n4 > 0)
    )
    dat["frac_MRC_score"] = pd.to_numeric(dat["n_MRC"], errors="coerce") / n4.where(n4 > 0)
    dat["frac_otherFibroblast"] = (
        pd.to_numeric(dat["n_otherFibroblast"], errors="coerce") / n4.where(n4 > 0)
    )
    dat = add_loc_group(dat)

    out = args.out_dir / "tls_fibroblast_composition.tsv"
    dat.to_csv(out, sep="\t", index=False)
    print(dat["loc_group"].value_counts(dropna=False).to_string(), flush=True)
    keep = dat["loc_group"].isin(["Normal", "Tumoral"])
    print(
        dat.loc[keep]
        .groupby(["loc_group", "kmeans_stage"], dropna=False)[count_names + ["n_fibroblast_anno"]]
        .sum()
        .to_string(),
        flush=True,
    )
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
