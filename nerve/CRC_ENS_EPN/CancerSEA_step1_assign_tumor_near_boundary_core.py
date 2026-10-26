#!/usr/bin/env python3
"""Assign CRC V21 Tumor cells to Near tumoral-nerve / Boundary / Core.

Near: Yuwei Tumor <=50 um from any cell in a V21 tumoral nerve
      (in_tumor on 0_crcV21/nerve_index.tsv), or Tumor inside that nerve.
Boundary: Tumor side, -50 <= dist_iface_um <= 0 (gen10).
Core: Tumor side, dist_iface_um < -50.
Priority: Near > Boundary > Core.

Samples with fewer than --min-near Near Tumor cells are dropped.
Use --exclude-peripheral to keep only enteric tumoral nerves (n_ens_neuron > 0).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from scipy.spatial import cKDTree


SCH = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/01_analysis/04_schwann"
)
V21 = SCH / "0_crcV21" / "boundary_CRC" 
INDEX = V21 / "nerve_index.tsv"
MEMBER = V21 / "nerve_cell_membership.tsv"
META = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/CRC/CRC_metadata.csv"
)
ANNOT = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/CRC/CRC_cell_annotations.csv"
)
BOUNDARY_ROOT = Path(
    "/storage1/fs1/dinglab/Active/Projects/pdc/projects/general/x1000/"
    "boundary/gen10/5k/outputs/CRC"
)
DEFAULT_OUT = V21 / "crc_v21_enteric_ensgt5_min10_csea"
NEAR_UM = 50.0
BOUNDARY_DEPTH = 50.0
GROUP_NEAR = "Near_Tumoral_Nerve_50um"
GROUP_BOUNDARY = "Boundary_Tumor"
GROUP_CORE = "Tumor_Core"


def log(message: str) -> None:
    print(message, flush=True)


def load_coords(samples: set[str]) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        META,
        usecols=["cell_id", "dataset", "xcoord", "ycoord"],
        dtype={"dataset": "string", "cell_id": "string"},
        chunksize=500_000,
    ):
        hit = chunk.loc[chunk["dataset"].astype(str).isin(samples)]
        if len(hit):
            chunks.append(hit)
            log(f"  meta kept {len(hit):,}")
    if not chunks:
        raise RuntimeError("No CRC_metadata rows for tumoral-nerve samples")
    coords = pd.concat(chunks, ignore_index=True)
    coords["sample"] = coords["dataset"].astype(str)
    coords["cell_id"] = coords["cell_id"].astype(str)
    coords["barcode"] = coords["cell_id"] + "_" + coords["sample"]
    coords["xcoord"] = pd.to_numeric(coords["xcoord"], errors="coerce")
    coords["ycoord"] = pd.to_numeric(coords["ycoord"], errors="coerce")
    return coords.dropna(subset=["xcoord", "ycoord"])


def load_yuwei_tumor(samples: set[str]) -> pd.DataFrame:
    tumor = pd.read_csv(
        ANNOT,
        usecols=["barcode", "cell_id", "cell_type"],
        dtype=str,
    )
    tumor = tumor.loc[tumor["cell_type"].eq("Tumor")].copy()
    tumor["cell_id"] = tumor["cell_id"].astype(str)
    tumor["barcode"] = tumor["barcode"].astype(str)
    tumor["sample"] = [
        barcode[len(cell_id) + 1 :] if barcode.startswith(cell_id + "_") else ""
        for barcode, cell_id in zip(tumor["barcode"], tumor["cell_id"])
    ]
    tumor = tumor.loc[tumor["sample"].isin(samples)].copy()
    log(f"Yuwei Tumor in tumoral-nerve samples: {len(tumor):,}")
    return tumor[["sample", "cell_id", "barcode"]]


def load_zone(sample: str, zone: str) -> set[str]:
    path = BOUNDARY_ROOT / sample / f"{sample}_boundary_distance.csv"
    if not path.is_file():
        log(f"missing boundary {path}")
        return set()
    table = pd.read_csv(path, usecols=["cell_id", "side", "dist_iface_um", "cell_type"])
    table["cell_id"] = table["cell_id"].astype(str)
    table["dist_iface_um"] = pd.to_numeric(table["dist_iface_um"], errors="coerce")
    table = table.loc[
        table["cell_type"].astype(str).eq("Tumor")
        & table["side"].astype(str).eq("Tumor")
        & table["dist_iface_um"].notna()
    ]
    if zone == "boundary":
        table = table.loc[
            table["dist_iface_um"].le(0) & table["dist_iface_um"].ge(-BOUNDARY_DEPTH)
        ]
    else:
        table = table.loc[table["dist_iface_um"].lt(-BOUNDARY_DEPTH)]
    return set(table["cell_id"].astype(str) + "_" + sample)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--min-near", type=int, default=10)
    parser.add_argument(
        "--exclude-peripheral",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only enteric tumoral nerves (n_ens_neuron > 0).",
    )
    parser.add_argument(
        "--min-glial",
        type=int,
        default=0,
        help="Keep tumoral nerves with n_glial >= this value (0 = no glial filter).",
    )
    parser.add_argument(
        "--min-neural",
        type=int,
        default=0,
        help="Keep nerves with n_glial + n_ens_neuron >= this value.",
    )
    parser.add_argument(
        "--min-ens",
        type=int,
        default=5,
        help="Keep nerves with n_ens_neuron > this value (strict).",
    )
    parser.add_argument(
        "--min-yuwei-tumor-near",
        type=int,
        default=0,
        help=(
            "Re-evaluate each nerve with Yuwei Tumor <=50 um. "
            "Keep a nerve only if that count is > this value, and only those "
            "nearby Tumor cells enter sample Near scoring. 0 = use in_tumor."
        ),
    )
    parser.add_argument(
        "--nerve-key-file",
        type=Path,
        default=None,
        help=(
            "TSV with nerve_key column. Use these nerves and ignore in_tumor. "
            "Other filters (--exclude-peripheral, --min-ens, etc.) still apply."
        ),
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=V21,
        help="Directory with nerve_index.tsv and nerve_cell_membership.tsv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.out_dir
    min_near = args.min_near
    out.mkdir(parents=True, exist_ok=True)
    index = pd.read_csv(args.base_dir / "nerve_index.tsv", sep="\t")
    index["in_tumor"] = index["in_tumor"].astype(str).str.lower().isin({"true", "1"})
    index["n_ens_neuron"] = pd.to_numeric(index["n_ens_neuron"], errors="coerce").fillna(0)
    index["n_glial"] = pd.to_numeric(index["n_glial"], errors="coerce").fillna(0)
    if args.nerve_key_file is not None:
        keys = pd.read_csv(args.nerve_key_file, sep="\t")
        if "nerve_key" not in keys.columns:
            raise SystemExit("--nerve-key-file must contain a nerve_key column")
        keep_keys = set(keys["nerve_key"].astype(str))
        tumoral = index.loc[index["nerve_key"].astype(str).isin(keep_keys)].copy()
        log(
            f"nerve-key-file: {len(tumoral)} / {len(keep_keys)} keys "
            f"from {args.nerve_key_file} (ignore in_tumor)"
        )
    elif args.min_yuwei_tumor_near > 0:
        tumoral = index.copy()
        log("Yuwei per-nerve tumor gate: start from all nerves, ignore in_tumor")
    else:
        tumoral = index.loc[index["in_tumor"]].copy()
    if args.exclude_peripheral:
        before = len(tumoral)
        tumoral = tumoral.loc[tumoral["n_ens_neuron"].gt(0)].copy()
        log(
            f"exclude peripheral: {before} tumoral -> {len(tumoral)} enteric tumoral"
        )
    if args.min_glial > 0:
        before = len(tumoral)
        tumoral = tumoral.loc[tumoral["n_glial"].ge(args.min_glial)].copy()
        log(
            f"min glial>={args.min_glial}: {before} -> {len(tumoral)} nerves"
        )
    if args.min_neural > 0:
        before = len(tumoral)
        n_neural = tumoral["n_glial"] + tumoral["n_ens_neuron"]
        tumoral = tumoral.loc[n_neural.ge(args.min_neural)].copy()
        log(
            f"min glial+ens>={args.min_neural}: {before} -> {len(tumoral)} nerves"
        )
    if args.min_ens > 0:
        before = len(tumoral)
        tumoral = tumoral.loc[tumoral["n_ens_neuron"].gt(args.min_ens)].copy()
        log(
            f"min ens>{args.min_ens}: {before} -> {len(tumoral)} nerves"
        )
    samples = sorted(tumoral["sample_id"].unique())
    log(
        f"V21 nerves used {len(tumoral)} in {len(samples)} samples; "
        f"min_near={min_near}; min_glial={args.min_glial}; "
        f"min_neural={args.min_neural}; min_ens={args.min_ens}"
    )
    tumoral.to_csv(out / "v21_tumoral_nerves_used.tsv", sep="\t", index=False)

    member = pd.read_csv(args.base_dir / "nerve_cell_membership.tsv", sep="\t", dtype=str)
    keep_keys = set(tumoral["nerve_key"].astype(str))
    nerve_cells = member.loc[
        member["nerve_key"].astype(str).isin(keep_keys),
        ["sample_id", "cell_id", "barcode", "nerve_key"],
    ]
    nerve_cells.to_csv(out / "v21_tumoral_nerve_cells.tsv", sep="\t", index=False)

    coords = load_coords(set(samples))
    tumor = load_yuwei_tumor(set(samples))
    tumor = tumor.merge(
        coords[["barcode", "xcoord", "ycoord"]],
        on="barcode",
        how="inner",
    )
    nerve_xy = nerve_cells.merge(
        coords[["barcode", "xcoord", "ycoord"]],
        on="barcode",
        how="inner",
    )
    log(f"nerve cells with coords {len(nerve_xy):,} / {len(nerve_cells):,}")
    log(f"tumor cells with coords {len(tumor):,}")

    near_rows: list[pd.DataFrame] = []
    nerve_tumor_rows: list[dict[str, object]] = []
    if args.min_yuwei_tumor_near > 0:
        keep_keys = set()
        for sample in samples:
            nxy = nerve_xy.loc[nerve_xy["sample_id"].eq(sample)]
            txy = tumor.loc[tumor["sample"].eq(sample)]
            if nxy.empty or txy.empty:
                log(f"{sample}: nerve {len(nxy)} tumor {len(txy)} — skip near")
                continue
            tree = cKDTree(nxy[["xcoord", "ycoord"]].to_numpy())
            hits = tree.query_ball_point(
                txy[["xcoord", "ycoord"]].to_numpy(), r=NEAR_UM
            )
            nerve_key_arr = nxy["nerve_key"].astype(str).to_numpy()
            t_barcodes = txy["barcode"].astype(str).to_numpy()
            t_cells = txy["cell_id"].astype(str).to_numpy()
            tumors_by_nerve: dict[str, set[int]] = {}
            for i, neigh in enumerate(hits):
                for j in neigh:
                    tumors_by_nerve.setdefault(str(nerve_key_arr[j]), set()).add(i)
            sample_near: set[int] = set()
            for key, idxs in tumors_by_nerve.items():
                n_tum = len(idxs)
                ok = n_tum > args.min_yuwei_tumor_near
                nerve_tumor_rows.append(
                    {
                        "sample": sample,
                        "nerve_key": key,
                        "n_yuwei_tumor_le50": n_tum,
                        "keep_nerve": ok,
                    }
                )
                if ok:
                    keep_keys.add(key)
                    sample_near.update(idxs)
            if sample_near:
                near = txy.iloc[sorted(sample_near)].copy()
                near_rows.append(near[["sample", "cell_id", "barcode"]])
            log(
                f"{sample}: nerves with Yuwei Tumor>{args.min_yuwei_tumor_near} "
                f"{sum(1 for k,i in tumors_by_nerve.items() if len(i) > args.min_yuwei_tumor_near)}"
                f"/{len(tumors_by_nerve)}; near tumor {len(sample_near):,}"
            )
        nerve_qc = pd.DataFrame(nerve_tumor_rows)
        if len(nerve_qc):
            nerve_qc = nerve_qc.merge(
                tumoral[
                    [
                        "nerve_key",
                        "n_glial",
                        "n_ens_neuron",
                        "n_tumor_le50_legacy",
                        "in_tumor",
                        "region",
                        "nerve_class",
                    ]
                ],
                on="nerve_key",
                how="left",
            )
        nerve_qc.to_csv(out / "v21_per_nerve_yuwei_tumor_le50.tsv", sep="\t", index=False)
        tumoral = tumoral.loc[tumoral["nerve_key"].astype(str).isin(keep_keys)].copy()
        samples = sorted(tumoral["sample_id"].unique())
        log(
            f"Yuwei Tumor>{args.min_yuwei_tumor_near}: kept {len(tumoral)} nerves "
            f"in {len(samples)} samples"
        )
        tumoral.to_csv(out / "v21_tumoral_nerves_used.tsv", sep="\t", index=False)
    else:
        for sample in samples:
            nxy = nerve_xy.loc[nerve_xy["sample_id"].eq(sample)]
            txy = tumor.loc[tumor["sample"].eq(sample)]
            if nxy.empty or txy.empty:
                log(f"{sample}: nerve {len(nxy)} tumor {len(txy)} — skip near")
                continue
            dist, _ = cKDTree(nxy[["xcoord", "ycoord"]].to_numpy()).query(
                txy[["xcoord", "ycoord"]].to_numpy(), k=1
            )
            keep = dist <= NEAR_UM
            near = txy.loc[keep].copy()
            near["distance_um"] = dist[keep]
            near_rows.append(near[["sample", "cell_id", "barcode", "distance_um"]])
            log(f"{sample}: near tumor {int(keep.sum()):,}")
    near = pd.concat(near_rows, ignore_index=True) if near_rows else pd.DataFrame()
    near.to_csv(out / "v21_near_tumor.tsv", sep="\t", index=False)

    assigned_rows: list[dict[str, str]] = []
    qc_rows: list[dict[str, object]] = []
    for sample in samples:
        near_set = set(near.loc[near["sample"].eq(sample), "barcode"]) if len(near) else set()
        bound_set = load_zone(sample, "boundary")
        core_set = load_zone(sample, "core")
        sample_tumor = set(tumor.loc[tumor["sample"].eq(sample), "barcode"])
        n_near = n_bound = n_core = 0
        for barcode in sample_tumor:
            if barcode in near_set:
                group = GROUP_NEAR
                n_near += 1
            elif barcode in bound_set:
                group = GROUP_BOUNDARY
                n_bound += 1
            elif barcode in core_set:
                group = GROUP_CORE
                n_core += 1
            else:
                continue
            assigned_rows.append(
                {
                    "cancer_type": "CRC",
                    "sample": sample,
                    "patient_id": sample,
                    "barcode": barcode,
                    "comparison_group": group,
                }
            )
        keep = n_near >= min_near
        qc_rows.append(
            {
                "sample": sample,
                "n_near": n_near,
                "n_boundary": n_bound,
                "n_core": n_core,
                "keep": keep,
                "drop_reason": "" if keep else f"near_tumor<{min_near}",
            }
        )
        log(
            f"{sample} assigned near={n_near} boundary={n_bound} core={n_core} "
            f"{'KEEP' if keep else 'DROP'}"
        )

    qc = pd.DataFrame(qc_rows)
    qc.to_csv(out / "sample_group_cell_counts.tsv", sep="\t", index=False)
    keep_samples = set(qc.loc[qc["keep"], "sample"])
    assigned = pd.DataFrame(assigned_rows)
    assigned = assigned.loc[assigned["sample"].isin(keep_samples)].copy()
    assigned.to_csv(out / "tumor_cell_group_assignments.tsv", sep="\t", index=False)
    counts = (
        assigned.groupby(["patient_id", "comparison_group"])
        .size()
        .unstack(fill_value=0)
    )
    counts.to_csv(out / "patient_group_cell_counts.tsv", sep="\t")
    log(
        f"kept samples {len(keep_samples)} / {len(samples)}; "
        f"assigned cells {len(assigned):,}"
    )
    log(f"Wrote {out}")


if __name__ == "__main__":
    main()
