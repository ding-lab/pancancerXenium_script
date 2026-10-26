#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

CANCER_TYPE = "LUNG"
ANNOTATION_DIR = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/LUNG/xenium_browser_Mcell_tumor_annotations_new"
)
TLS_ROOT = Path(
    "/storage1/fs1/dinglab/Active/Projects/pdc/projects/general/x1000/"
    "tls/tls_define/gen12/outputs/tls/LUNG"
)
H5AD = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/LUNG/annotated.Yuwei.new.h5ad"
)
OUT_DIR = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/01_analysis/03_TLS/lung/B_subtype/Spatial_Gradient_McellAnchor_percentile"
)

PERCENTILE_LABELS = [
    "0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
    "50-60%", "60-70%", "70-80%", "80-90%", "90-100%",
]
BCELL_GENESETS = {
    "Pan_B": ["CD79A", "CD79B", "MS4A1", "CD22", "BANK1", "PAX5", "BLK", "CD37", "CD19"],
    "Naive_B": ["TCL1A", "IGHD", "FCER2", "MS4A1", "IL4R", "SELL"],
    "Follicular_B": ["CXCL13", "CXCR5", "CR2", "PAX5", "POU2AF1", "BANK1", "IKZF3", "MEF2B", "CD22"],
    "GC_B": ["BCL6", "AICDA", "MKI67"],
    "Memory_B": ["CD27", "TNFRSF13C", "BACH2", "MS4A1", "PAX5", "CD44", "SELPLG", "ITGA4"],
    "Activated_B": ["CD83", "CD86", "CD40", "HLA-DRA", "HLA-DPA1", "TNFRSF13B", "CCR7"],
    "Atypical_B": ["FCRL5", "ITGAX", "CXCR3", "FCGR2B", "FCRL4", "CD86", "KLF6", "TNFAIP3"],
    "Regulatory_B": ["IL10", "EBI3", "LGALS9", "CD1D", "ENTPD1", "PDCD1LG2", "TGFB1", "IL10RA"],
    "IFN_Response_B": ["STAT1", "MX1", "IFI44L", "OAS1", "ISG15", "IFITM3", "GBP1", "TAP1"],
    "TNF_NFKB_B": ["TNFAIP3", "NFKBIA", "RELB", "BIRC3", "TRAF1", "TRAF5", "JUNB", "FOS"],
    "TLS_Supportive_B": ["CXCL13", "CXCR5", "CD40", "BCL6", "LTB", "TNFRSF13B", "HLA-DRA", "CD83"],
    "CCR6_inter": ["CCR6"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--annotations-dir", type=Path, default=ANNOTATION_DIR)
    p.add_argument("--tls-root", type=Path, default=TLS_ROOT)
    p.add_argument("--h5ad", type=Path, default=H5AD)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--min-anchors", type=int, default=3)
    p.add_argument("--min-b-cells", type=int, default=10)
    p.add_argument("--min-t-cells", type=int, default=5)
    p.add_argument("--halo", default="tls_region_halo100")
    return p.parse_args()


class SpatialExpression:
    def __init__(self, path: Path, genes: list[str]) -> None:
        self.handle = h5py.File(path, "r")
        obs = self.handle["obs"]
        self.index_key = obs.attrs["_index"]
        categories = [x.decode() for x in obs["dataset"]["categories"][:]]
        codes = obs["dataset"]["codes"][:]
        self.rows_by_sample = {
            sample: np.flatnonzero(codes == code)
            for code, sample in enumerate(categories)
        }
        var_names = np.array([x.decode() for x in self.handle["var"][self.handle["var"].attrs["_index"]][:]])
        lookup = {name: i for i, name in enumerate(var_names)}
        self.genes = [g for g in genes if g in lookup]
        self.gene_columns = np.array([lookup[g] for g in self.genes])
        self.matrix = self.handle["layers"]["counts"]
        self.indptr = self.matrix["indptr"][:]

    def sample_cells(self, sample: str) -> pd.DataFrame:
        rows = self.rows_by_sample[sample]
        block = slice(rows[0], rows[-1] + 1)
        offset = rows - rows[0]
        cell_ids = self.handle["obs"][self.index_key][block][offset]
        spatial = self.handle["obsm"]["spatial"][block, :2][offset]
        return pd.DataFrame({
            "cell_id": [x.decode() for x in cell_ids],
            "row": rows,
            "x": spatial[:, 0].astype("float32"),
            "y": spatial[:, 1].astype("float32"),
        })

    def expression(self, rows: np.ndarray) -> np.ndarray:
        data = self.matrix["data"]
        indices = self.matrix["indices"]
        wanted = {column: i for i, column in enumerate(self.gene_columns)}
        out = np.zeros((rows.size, self.gene_columns.size), dtype="float32")
        for position, row in enumerate(rows):
            start, end = self.indptr[row], self.indptr[row + 1]
            if start == end:
                continue
            row_indices = indices[start:end]
            row_values = data[start:end].astype("float64")
            total = row_values.sum()
            if total <= 0:
                continue
            scaled = np.log1p(row_values * (1e4 / total))
            for column, value in zip(row_indices, scaled):
                target = wanted.get(int(column))
                if target is not None:
                    out[position, target] = value
        return out


def load_mcell_ids(annotation_dir: Path, sample: str) -> set[str]:
    path = annotation_dir / f"{sample}_Mcell_tumor_groups.csv"
    annotation = pd.read_csv(path, usecols=["cell_id", "group"], dtype=str)
    selected = annotation.loc[annotation["group"].eq("Mcell-tumor"), "cell_id"]
    tag = f"_{sample}"
    return {x if x.endswith(tag) else f"{x}{tag}" for x in selected}


def build_sample_table(sample: str, args: argparse.Namespace, cells: pd.DataFrame) -> pd.DataFrame | None:
    tls_path = args.tls_root / sample / f"{sample}_tls.csv"
    if not tls_path.is_file():
        return None
    tls = pd.read_csv(tls_path, dtype=str)
    tls["tls"] = tls[args.halo]
    tls = tls.loc[tls["tls"].ne("Outside") & tls["tls"].notna()]
    tls = tls.merge(cells, on="cell_id", how="inner")
    if tls.empty:
        return None

    anchors = tls.loc[tls["cell_id"].isin(load_mcell_ids(args.annotations_dir, sample))]
    if anchors.empty:
        return None
    core = tls.loc[tls["in_tls_core"].eq("True")]
    t_counts = core.loc[core["cell_type"].eq("NK_T"), "tls"].value_counts()
    b_cells = core.loc[core["cell_type"].eq("B_cell")]

    collected = []
    for tls_id, anchor_group in anchors.groupby("tls"):
        if len(anchor_group) < args.min_anchors:
            continue
        targets = b_cells.loc[b_cells["tls"].eq(tls_id)]
        if len(targets) < args.min_b_cells or t_counts.get(tls_id, 0) < args.min_t_cells:
            continue
        distance, _ = KDTree(anchor_group[["x", "y"]].to_numpy()).query(targets[["x", "y"]].to_numpy())
        part = targets.copy()
        part["dist_to_mcell"] = distance
        part["n_anchor_mcell"] = len(anchor_group)
        collected.append(part)
    if not collected:
        return None
    out = pd.concat(collected, ignore_index=True)
    out["SampleID"] = sample
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    requested = sorted({g for genes in BCELL_GENESETS.values() for g in genes})
    store = SpatialExpression(args.h5ad, requested)
    genesets = {name: [g for g in genes if g in store.genes] for name, genes in BCELL_GENESETS.items()}
    genesets = {name: genes for name, genes in genesets.items() if genes}
    score_cols = [f"{name}_Score" for name in genesets]

    samples = sorted(
        p.name[: -len("_Mcell_tumor_groups.csv")]
        for p in args.annotations_dir.glob("*_Mcell_tumor_groups.csv")
    )
    frames = []
    for sample in samples:
        if sample not in store.rows_by_sample:
            continue
        table = build_sample_table(sample, args, store.sample_cells(sample))
        if table is None:
            continue
        matrix = store.expression(table["row"].to_numpy())
        table = pd.concat(
            [table, pd.DataFrame(matrix, columns=store.genes, index=table.index)],
            axis=1,
        )
        frames.append(table)
    if not frames:
        raise SystemExit("No sample produced usable TLS")

    final = pd.concat(frames, ignore_index=True)
    for name, genes in genesets.items():
        final[f"{name}_Score"] = final[genes].mean(axis=1)
    final["TLS_ID"] = final["SampleID"] + "_" + final["tls"]
    final["TLS_Percentile"] = final.groupby("TLS_ID")["dist_to_mcell"].rank(method="average", pct=True)
    final["Norm_Bin"] = pd.cut(
        final["TLS_Percentile"],
        bins=np.linspace(0, 1, 11),
        labels=PERCENTILE_LABELS,
        include_lowest=True,
    )

    agg_cols = store.genes + score_cols
    pooled_mean = final.groupby("Norm_Bin", observed=True)[agg_cols].mean().reset_index()
    prefix = args.out_dir / CANCER_TYPE
    keep = [
        "cell_id", "SampleID", "TLS_ID", "tls", "cell_type", "x", "y",
        "dist_to_mcell", "n_anchor_mcell", "TLS_Percentile", "Norm_Bin",
    ] + agg_cols
    final[keep].to_csv(f"{prefix}_Bcell_McellAnchored_Raw.csv", index=False)
    pooled_mean.to_csv(f"{prefix}_Global_CellPooled_Average.csv", index=False)
    print(f"{len(final)} B cells, {final['TLS_ID'].nunique()} TLS, {final['SampleID'].nunique()} samples")


if __name__ == "__main__":
    main()
