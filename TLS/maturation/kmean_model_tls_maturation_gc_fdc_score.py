#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/01_analysis/03_TLS"
)
NEW = ROOT / "00_TLS_new"
DEFAULT_METRICS = ROOT / "tls_subtype_maturation_hierarchical/tls_stromal_gcb_program_metrics.tsv"
DEFAULT_LOCATION = NEW / "tls_spatial_info/gen11_TLS_Taggregate_locations_all.tsv"
DEFAULT_GCB = NEW / "tls_maturation_withmass/gcb_fraction_vs_score.tsv"
DEFAULT_FDC = ROOT / "tls_fibroblast_fdc_aucell/tls_fibroblasts_FDC_AUCell_scores.tsv"
DEFAULT_OUT = NEW / "tls_maturation_gc_fdc_score"

REFERENCE = "Lymph_nodes"
MIN_B = 10
GEOM_FLOOR = 1e-3
STAGE_LEVELS = ("early", "immature", "mature")
TUMORAL = {"Intratumoral", "Peritumoral"}
FDC_TYPES = {"CXCL13+ TRC", "FDC"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    p.add_argument("--location-tsv", type=Path, default=DEFAULT_LOCATION)
    p.add_argument("--gcb-score-tsv", type=Path, default=DEFAULT_GCB)
    p.add_argument("--fdc-score-tsv", type=Path, default=DEFAULT_FDC)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def midrank_percentile(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=float)
    ref = ref[np.isfinite(ref)]
    x = np.asarray(query, dtype=float)
    out = np.full(x.shape, np.nan)
    for i, v in enumerate(x):
        if np.isfinite(v):
            out[i] = (np.sum(ref < v) + 0.5 * np.sum(ref == v)) / ref.size
    return out


def geom_mean(block: np.ndarray) -> np.ndarray:
    ok = np.isfinite(block).all(axis=1)
    out = np.full(block.shape[0], np.nan)
    out[ok] = np.exp(np.mean(np.log(np.clip(block[ok], GEOM_FLOOR, None)), axis=1))
    return out


def name_kmeans(labels: np.ndarray, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = sorted(
        ((k, float(np.nanmean(index[labels == k]))) for k in sorted(set(labels))),
        key=lambda z: z[1],
    )
    remap = {old: i + 1 for i, (old, _) in enumerate(means)}
    ranked = np.array([remap[int(v)] for v in labels], dtype=int)
    named = np.array([STAGE_LEVELS[i - 1] for i in ranked])
    return ranked, named


def aggregate_fdc_scores(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, sep="\t")
    raw["FDC_AUCell"] = pd.to_numeric(raw["FDC_AUCell"], errors="coerce")
    raw = raw.loc[raw["FDC_AUCell"].notna()]
    keys = ["cancer_type", "sample", "tls_id"]
    all_fibro = (
        raw.groupby(keys, observed=True)
        .agg(n_fibro_scored=("FDC_AUCell", "size"), FDC_mean_all_fibro=("FDC_AUCell", "mean"))
        .reset_index()
    )
    among = (
        raw.loc[raw["cell_type"].isin(FDC_TYPES)]
        .groupby(keys, observed=True)
        .agg(n_TRC_FDC_scored=("FDC_AUCell", "size"), FDC_mean_among_TRC_FDC=("FDC_AUCell", "mean"))
        .reset_index()
    )
    out = all_fibro.merge(among, on=keys, how="left")
    out["n_TRC_FDC_scored"] = out["n_TRC_FDC_scored"].fillna(0).astype(int)
    return out.rename(columns={"tls_id": "tls_region"})


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dat = pd.read_csv(args.metrics, sep="\t")
    dat = dat.loc[~(dat["cancer_type"].eq("BRCA") & dat["sample"].astype(str).str.startswith("AL"))]
    dat["Analysis_Group"] = dat["cancer_type"].astype(str)

    n_b = pd.to_numeric(dat["n_B_scored"], errors="coerce").to_numpy(float)
    n_gcb = pd.to_numeric(dat["n_GCB"], errors="coerce").to_numpy(float)
    n_fdc = pd.to_numeric(dat["n_FDC"], errors="coerce").to_numpy(float)
    n_trc = pd.to_numeric(dat["n_CXCL13_TRC"], errors="coerce").to_numpy(float)
    n_mrc = pd.to_numeric(dat["n_MRC"], errors="coerce").to_numpy(float)
    n_other = pd.to_numeric(dat["n_otherFibroblast"], errors="coerce").to_numpy(float)
    n_fibro = n_trc + n_fdc + n_mrc + n_other
    n_follicular = n_trc + n_fdc

    dat["gc_fraction"] = np.where(n_b > 0, n_gcb / n_b, np.nan)
    dat["fdc_fraction"] = np.divide(n_follicular, n_fibro, out=np.zeros_like(n_follicular), where=n_fibro > 0)
    used = dat.loc[n_b >= MIN_B].copy()

    gcb = pd.read_csv(
        args.gcb_score_tsv, sep="\t",
        usecols=["cancer_type", "sample", "tls_region", "GCB_mean", "GCB_mean_among_GCB"],
    )
    used = used.merge(gcb, on=["cancer_type", "sample", "tls_region"], how="left")
    used["gcb_score"] = pd.to_numeric(used["GCB_mean"], errors="coerce").fillna(0.0)

    used = used.merge(aggregate_fdc_scores(args.fdc_score_tsv), on=["cancer_type", "sample", "tls_region"], how="left")
    used["fdc_score"] = pd.to_numeric(used["FDC_mean_all_fibro"], errors="coerce").fillna(0.0)

    loc = pd.read_csv(args.location_tsv, sep="\t", dtype=str, keep_default_na=False)
    loc = loc.loc[loc["structure_type"].eq("TLS")].copy()
    loc["location"] = loc["location"].replace({"Boundary": "Peritumoral"})
    loc = loc.rename(columns={"structure_id": "tls_region"})[
        ["cancer_type", "sample", "tls_region", "location"]
    ].drop_duplicates()
    used = used.merge(loc, on=["cancer_type", "sample", "tls_region"], how="left")

    ln = used["cancer_type"].eq(REFERENCE)
    if not ln.any():
        raise RuntimeError("No lymph-node TLS for percentile reference")

    used["gc_frac_index"] = midrank_percentile(used.loc[ln, "gc_fraction"].to_numpy(float), used["gc_fraction"].to_numpy(float))
    used["gcb_score_index"] = midrank_percentile(used.loc[ln, "gcb_score"].to_numpy(float), used["gcb_score"].to_numpy(float))
    used["fdc_frac_index"] = midrank_percentile(used.loc[ln, "fdc_fraction"].to_numpy(float), used["fdc_fraction"].to_numpy(float))
    used["fdc_score_index"] = midrank_percentile(used.loc[ln, "fdc_score"].to_numpy(float), used["fdc_score"].to_numpy(float))
    used["gc_maturity"] = geom_mean(used[["gc_frac_index", "gcb_score_index"]].to_numpy(float))
    used["fdc_maturity"] = geom_mean(used[["fdc_frac_index", "fdc_score_index"]].to_numpy(float))
    used["maturation_index_geom"] = geom_mean(used[["gc_maturity", "fdc_maturity"]].to_numpy(float))

    raw_lab = KMeans(n_clusters=3, random_state=args.seed, n_init=20).fit_predict(
        used[["gc_maturity", "fdc_maturity"]].to_numpy(float)
    )
    used["kmeans_cluster"], used["kmeans_stage"] = name_kmeans(
        raw_lab, used["maturation_index_geom"].to_numpy(float)
    )

    used.to_csv(args.out_dir / "tls_maturation_scores.tsv", sep="\t", index=False)
    used.loc[used["location"].isin(TUMORAL) & used["Analysis_Group"].eq("LUNG")].to_csv(
        args.out_dir / "lung_tumoral_tls.tsv", sep="\t", index=False
    )
    print(f"LN reference TLS: {int(ln.sum())}; modelled TLS: {len(used)}")
    print(used["kmeans_stage"].value_counts().reindex(list(STAGE_LEVELS)).to_string())


if __name__ == "__main__":
    main()
