#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gzip
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

SCHWANN_ROOT = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/"
    "01_cell_typing/01_analysis/04_schwann"
)
NERVE_ROOT = SCHWANN_ROOT / "nerve_region"
NERVE_REGION_V8 = SCHWANN_ROOT / "nerve_region"
H5AD_ROOT = Path(
    "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/01_cell_typing"
)
HALO50_CELLS = (
    NERVE_ROOT
    / "halo_tumor_count_classification"
    / "halo50"
    / "schwann_halo_classified_cells.tsv"
)
CRC_MEMBER = (
    SCHWANN_ROOT
    / "0_crcV21"
    / "boundary_CRC"
    / "nerve_cell_membership.tsv"
)
OUT_ROOT = NERVE_ROOT / "schwann_neighborhood_tme_label_permutation_50um"

H5AD_PATHS = {
    "CRC": H5AD_ROOT / "CRC" / "adata_slim.h5ad",
    "PDAC": H5AD_ROOT / "PDAC" / "annotated.Yuwei.h5ad",
    "PRAD": H5AD_ROOT / "PRAD" / "adata_slim.h5ad",
}
CANCER_TYPES_DEFAULT = ["CRC", "PDAC", "PRAD"]
TUMORAL_REGIONS = ("intratumoral", "peritumoral")
HALO_TUMORAL_CLASS = "tumor_adjacent"
SERIAL_SECTION_PATIENT_PREFIXES = ("SP048Z1", "SP093Z1")

NICHE_RADIUS_UM = 50.0
MIN_ANCHOR_SCHWANN = 2
MIN_CONDITIONAL_TME_CELLS = 3
N_PERMUTATIONS = 1000
MIN_ANALYSIS_UNITS = 3
RANDOM_SEED = 20260715

EXCLUDED_TME_TYPES = {"Tumor", "Schwann", "Necrosis"}
TUMOR_ALIASES = {
    "tumor",
    "malignant",
    "cancer",
    "cancer_cell",
    "tumor_cell",
}
SCHWANN_ALIASES = {"schwann", "schwann_cell", "schwann_cells"}
NECROSIS_ALIASES = {"necrosis", "necrotic", "necrotic_cell"}
PERICYTE_VSMC_ALIASES = {
    "pericyte",
    "pericytes",
    "vsmc",
    "vsmcs",
    "vascular_smooth_muscle",
    "vascular_smooth_muscle_cell",
    "vascular_smooth_muscle_cells",
    "pericyte_vsmc",
    "pericyte/vsmc",
}
NK_T_ALIASES = {
    "nk",
    "nk_cell",
    "nk_cells",
    "natural_killer",
    "natural_killer_cell",
    "natural_killer_cells",
    "t_cell",
    "t_cells",
    "tcell",
    "tcells",
    "t_lymphocyte",
    "t_lymphocytes",
    "nk_t",
    "nk_t_cell",
    "nkt",
}
DC_ALIASES = {
    "dc",
    "cdc1",
    "cdc1_cell",
    "cdc1_cells",
    "cdc2",
    "cdc2_cell",
    "cdc2_cells",
    "pdc",
    "pdc_cell",
    "pdc_cells",
    "mregdc",
    "mregdc_cell",
    "mregdc_cells",
    "mreg_dc",
    "mreg_dc_cell",
    "mreg_dc_cells",
}
TME_TYPE_ALIASES = {"CAF": "Fibroblast"}
CANCER_SPECIFIC_TME_TYPES = {
    "Acinar",
    "Basal",
    "Club",
    "Duct_like",
    "Dysplasia",
    "Intestinal_epithelium",
    "IPMN",
    "Islet",
    "Luminal",
    "Neuroendocrine",
    "SMC",
    "Smooth_muscle",
    "Squamous_epithelium",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Label-permutation test of conditional TME composition near "
            "tumoral Schwann cells."
        )
    )
    parser.add_argument(
        "--cancer-types",
        nargs="+",
        default=CANCER_TYPES_DEFAULT,
        help="Cancer types to analyze (default: CRC PDAC PRAD).",
    )
    parser.add_argument(
        "--niche-radius-um",
        type=float,
        default=NICHE_RADIUS_UM,
        help="Neighborhood radius around tumoral Schwann (default: 50).",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=N_PERMUTATIONS,
        help="Number of label permutations (default: 1000).",
    )
    parser.add_argument(
        "--min-anchor-schwann",
        type=int,
        default=MIN_ANCHOR_SCHWANN,
        help="Minimum tumoral Schwann cells per sample (default: 2).",
    )
    parser.add_argument(
        "--min-conditional-tme-cells",
        type=int,
        default=MIN_CONDITIONAL_TME_CELLS,
        help="Minimum TME denominator for composition (default: 3).",
    )
    parser.add_argument(
        "--min-analysis-units",
        type=int,
        default=MIN_ANALYSIS_UNITS,
        help="Minimum analysis units for cancer-level p-values (default: 3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed.",
    )
    parser.add_argument(
        "--write-null-analysis-units",
        action="store_true",
        help="Write null_analysis_unit_metrics.tsv.gz (large).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_ROOT,
        help="Output directory.",
    )
    return parser.parse_args()


def infer_analysis_unit(sample: str) -> str:
    sample = str(sample).strip()
    for prefix in SERIAL_SECTION_PATIENT_PREFIXES:
        if sample == prefix or sample.startswith(prefix + "-"):
            return prefix
    return sample


def normalized_labels(values: pd.Series) -> pd.Series:
    return (
        values.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"[\s-]+", "_", regex=True)
    )


def resolve_cell_types(obs: pd.DataFrame) -> pd.Series:
    candidates = (
        "cell_type_refined",
        "cell_type_refine",
        "cell_type",
        "Initial_cell_type",
        "cell_type_backup",
    )
    available = [column for column in candidates if column in obs.columns]
    if not available:
        raise KeyError("No cell-type annotation column found.")

    resolved = pd.Series(pd.NA, index=obs.index, dtype="string")
    for column in available:
        values = obs[column].astype("string").str.strip()
        invalid = values.isna() | normalized_labels(values).isin(
            ["", "nan", "na", "none", "unknown"]
        )
        resolved = resolved.fillna(values.mask(invalid))
    resolved = resolved.fillna("Unknown").astype(str)
    normalized = normalized_labels(resolved)
    resolved.loc[normalized.isin(TUMOR_ALIASES)] = "Tumor"
    resolved.loc[normalized.isin(SCHWANN_ALIASES)] = "Schwann"
    resolved.loc[normalized.isin(NECROSIS_ALIASES)] = "Necrosis"

    if "cell_type" in obs.columns:
        broad = normalized_labels(obs["cell_type"])
        resolved.loc[broad.isin(TUMOR_ALIASES)] = "Tumor"
        resolved.loc[broad.isin(SCHWANN_ALIASES)] = "Schwann"
        resolved.loc[broad.isin(NECROSIS_ALIASES)] = "Necrosis"

    normalized = normalized_labels(resolved)
    resolved.loc[normalized.isin(PERICYTE_VSMC_ALIASES)] = "Pericyte/vSMC"
    resolved.loc[normalized.isin(NK_T_ALIASES)] = "NK_T"
    resolved.loc[normalized.isin(DC_ALIASES)] = "DC"

    for raw, mapped in TME_TYPE_ALIASES.items():
        resolved.loc[resolved.astype(str) == raw] = mapped
    return resolved


def parse_obs(obs: pd.DataFrame) -> pd.DataFrame:
    result = obs.copy()
    result["obs_index"] = result.index.astype(str)
    split = result["obs_index"].str.split("_", n=1, expand=True)
    result["cell_id"] = split[0].astype(str)
    if "dataset" in result.columns:
        sample = result["dataset"].astype("string")
        invalid = sample.isna() | sample.str.lower().isin(
            ["", "nan", "na", "none"]
        )
        if split.shape[1] > 1:
            sample.loc[invalid] = split.loc[invalid, 1].astype("string")
        result["sample"] = sample.astype(str)
    else:
        if split.shape[1] < 2:
            raise ValueError("Expected obs index format {cell_id}_{sample}")
        result["sample"] = split[1].astype(str)
    result["analysis_unit"] = result["sample"].map(infer_analysis_unit)
    return result


def load_tumoral_schwann_anchors(cancer_type: str) -> pd.DataFrame:
    """Tumoral anchors: CRC = boundary Glial or ENS_neuron; PDAC/PRAD = halo50 Schwann."""
    if cancer_type == "CHOL":
        raise ValueError("CHOL is excluded from this analysis.")
    if cancer_type == "CRC":
        if not CRC_MEMBER.is_file():
            raise FileNotFoundError(f"Missing CRC membership: {CRC_MEMBER}")
        table = pd.read_csv(
            CRC_MEMBER,
            sep="\t",
            usecols=[
                "sample_id",
                "cell_id",
                "in_tumor",
                "is_glial",
                "is_ens_neuron",
            ],
            dtype=str,
        )
        flag = table["in_tumor"].str.lower().isin(["true", "1"])
        identity = table["is_glial"].str.lower().isin(["true", "1"]) | table[
            "is_ens_neuron"
        ].str.lower().isin(["true", "1"])
        anchors = (
            table.loc[flag & identity, ["sample_id", "cell_id"]]
            .drop_duplicates()
            .rename(columns={"sample_id": "sample"})
        )
        print(
            f"  [CRC] boundary tumoral Glial/ENS anchors: {len(anchors)} cells, "
            f"{anchors['sample'].nunique()} samples",
            flush=True,
        )
        return anchors.reset_index(drop=True)

    if not HALO50_CELLS.is_file():
        raise FileNotFoundError(f"Missing halo50 cells: {HALO50_CELLS}")
    table = pd.read_csv(
        HALO50_CELLS,
        sep="\t",
        usecols=["cancer_type", "sample", "halo_class", "cell_id", "cell_type"],
        dtype=str,
    )
    anchors = table.loc[
        table["cancer_type"].eq(cancer_type)
        & table["halo_class"].eq(HALO_TUMORAL_CLASS)
        & table["cell_type"].eq("Schwann"),
        ["sample", "cell_id"],
    ].drop_duplicates(["sample", "cell_id"])
    if anchors.empty:
        raise RuntimeError(
            f"No halo50 tumor_adjacent Schwann cells for {cancer_type}."
        )
    print(
        f"  [{cancer_type}] halo50 tumor_adjacent anchors: {len(anchors)} cells, "
        f"{anchors['sample'].nunique()} samples",
        flush=True,
    )
    return anchors.reset_index(drop=True)


def union_neighborhood_indices(
    tissue_tree: cKDTree,
    anchor_points: np.ndarray,
    radius_um: float,
) -> np.ndarray:
    neighborhoods = tissue_tree.query_ball_point(anchor_points, r=radius_um)
    if len(neighborhoods) == 0:
        return np.array([], dtype=int)
    nonempty = [
        np.asarray(indices, dtype=int)
        for indices in neighborhoods
        if len(indices) > 0
    ]
    if not nonempty:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(nonempty))


def conditional_tme_composition(
    neighbor_indices: np.ndarray,
    cell_types: np.ndarray,
    tme_cell_types: list[str],
    min_conditional_tme_cells: int,
) -> dict[str, float]:
    local_types = cell_types[neighbor_indices]
    counts = pd.Series(local_types).value_counts()
    tme_counts = counts[~counts.index.isin(EXCLUDED_TME_TYPES)]
    total_tme = int(tme_counts.sum())
    metrics: dict[str, float] = {}
    for cell_type in tme_cell_types:
        metrics[cell_type] = (
            float(tme_counts.get(cell_type, 0) / total_tme)
            if total_tme >= min_conditional_tme_cells
            else np.nan
        )
    return metrics


def empirical_statistics(
    observed: float,
    null_values: np.ndarray,
) -> dict[str, float]:
    null_values = np.asarray(null_values, dtype=float)
    null_values = null_values[np.isfinite(null_values)]
    if not np.isfinite(observed) or len(null_values) == 0:
        return {
            "null_mean": np.nan,
            "null_sd": np.nan,
            "observed_minus_null": np.nan,
            "z_score": np.nan,
            "empirical_p_enriched": np.nan,
            "empirical_p_depleted": np.nan,
            "empirical_p_directional": np.nan,
            "empirical_p_two_sided": np.nan,
            "n_valid_permutations": len(null_values),
        }
    null_mean = float(np.mean(null_values))
    null_sd = float(np.std(null_values, ddof=1)) if len(null_values) > 1 else np.nan
    effect = float(observed - null_mean)
    p_enriched = (1 + np.sum(null_values >= observed)) / (len(null_values) + 1)
    p_depleted = (1 + np.sum(null_values <= observed)) / (len(null_values) + 1)
    centered_null = np.abs(null_values - null_mean)
    p_two_sided = (
        1 + np.sum(centered_null >= abs(observed - null_mean))
    ) / (len(null_values) + 1)
    p_directional = p_enriched if effect > 0 else p_depleted
    return {
        "null_mean": null_mean,
        "null_sd": null_sd,
        "observed_minus_null": effect,
        "z_score": effect / null_sd if null_sd and null_sd > 0 else np.nan,
        "empirical_p_enriched": float(p_enriched),
        "empirical_p_depleted": float(p_depleted),
        "empirical_p_directional": float(p_directional),
        "empirical_p_two_sided": float(p_two_sided),
        "n_valid_permutations": len(null_values),
    }


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    values = pvalues.to_numpy(dtype=float)
    output = np.full(values.shape, np.nan)
    valid = np.where(np.isfinite(values))[0]
    if len(valid) == 0:
        return pd.Series(output, index=pvalues.index)
    order = np.argsort(values[valid])
    ranked = values[valid][order]
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty(n)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    output[valid] = restored
    return pd.Series(output, index=pvalues.index)


def analyze_sample(
    sample: str,
    cancer_type: str,
    sample_obs: pd.DataFrame,
    sample_coords: np.ndarray,
    cell_types: np.ndarray,
    anchor_cell_ids: set[str],
    tme_cell_types: list[str],
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    analysis_unit = infer_analysis_unit(sample)
    cell_ids = sample_obs["cell_id"].astype(str).to_numpy()
    anchor_mask = np.isin(cell_ids, list(anchor_cell_ids))
    n_anchors = int(anchor_mask.sum())
    qc = {
        "cancer_type": cancer_type,
        "sample": sample,
        "analysis_unit": analysis_unit,
        "n_tumoral_schwann": n_anchors,
        "n_cells_total": len(cell_ids),
        "skip_reason": "",
    }
    if n_anchors < args.min_anchor_schwann:
        qc["skip_reason"] = "insufficient_tumoral_schwann"
        return pd.DataFrame(), pd.DataFrame(), qc

    tissue_tree = cKDTree(sample_coords)
    anchor_points = sample_coords[anchor_mask]
    neighbor_idx = union_neighborhood_indices(
        tissue_tree, anchor_points, args.niche_radius_um
    )
    qc["n_neighbor_cells"] = int(len(neighbor_idx))
    if len(neighbor_idx) == 0:
        qc["skip_reason"] = "empty_neighborhood"
        return pd.DataFrame(), pd.DataFrame(), qc

    observed_metrics = conditional_tme_composition(
        neighbor_idx,
        cell_types,
        tme_cell_types,
        args.min_conditional_tme_cells,
    )
    null_matrix = {
        cell_type: np.full(args.n_permutations, np.nan, dtype=np.float32)
        for cell_type in tme_cell_types
    }
    non_anchor = np.where(~anchor_mask)[0]
    base_labels = cell_types.copy()
    for perm_id in range(args.n_permutations):
        labels_perm = base_labels.copy()
        shuffled = base_labels[non_anchor].copy()
        rng.shuffle(shuffled)
        labels_perm[non_anchor] = shuffled
        metrics = conditional_tme_composition(
            neighbor_idx,
            labels_perm,
            tme_cell_types,
            args.min_conditional_tme_cells,
        )
        for cell_type in tme_cell_types:
            null_matrix[cell_type][perm_id] = metrics[cell_type]

    observed_rows: list[dict] = []
    null_rows: list[pd.DataFrame] = []
    common = {
        "cancer_type": cancer_type,
        "sample": sample,
        "analysis_unit": analysis_unit,
        "metric_type": "conditional_tme_composition",
    }
    for cell_type in tme_cell_types:
        observed_value = observed_metrics[cell_type]
        null_values = null_matrix[cell_type]
        stats = empirical_statistics(observed_value, null_values)
        observed_rows.append(
            {
                **common,
                "metric_name": cell_type,
                "observed_value": observed_value,
                **stats,
            }
        )
        null_rows.append(
            pd.DataFrame(
                {
                    **common,
                    "metric_name": cell_type,
                    "permutation_id": np.arange(args.n_permutations),
                    "null_value": null_values,
                }
            )
        )

    qc["skip_reason"] = ""
    qc["n_requested_permutations"] = args.n_permutations
    return (
        pd.DataFrame(observed_rows),
        pd.concat(null_rows, ignore_index=True),
        qc,
    )


def process_cancer(
    cancer_type: str,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    h5ad_path = H5AD_PATHS[cancer_type]
    if not h5ad_path.is_file():
        raise FileNotFoundError(f"Missing h5ad: {h5ad_path}")
    print(f"[{cancer_type}] Loading {h5ad_path}", flush=True)
    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs = parse_obs(adata.obs)
    obs["resolved_cell_type"] = resolve_cell_types(obs)
    if "spatial" not in adata.obsm:
        raise KeyError(f"{h5ad_path} missing obsm['spatial']")
    coordinates = np.asarray(adata.obsm["spatial"], dtype=float)

    anchors = load_tumoral_schwann_anchors(cancer_type)
    tme_cell_types = sorted(
        (
            set(obs["resolved_cell_type"].astype(str))
            - EXCLUDED_TME_TYPES
            - {"Unknown"}
        )
        | set(TME_TYPE_ALIASES.values())
    )

    observed_parts: list[pd.DataFrame] = []
    null_parts: list[pd.DataFrame] = []
    qc_rows: list[dict] = []

    for sample, sample_anchors in anchors.groupby("sample", sort=True):
        sample_mask = obs["sample"].astype(str).to_numpy() == str(sample)
        sample_positions = np.where(sample_mask)[0]
        if len(sample_positions) == 0:
            qc_rows.append(
                {
                    "cancer_type": cancer_type,
                    "sample": str(sample),
                    "analysis_unit": infer_analysis_unit(str(sample)),
                    "n_tumoral_schwann": 0,
                    "n_cells_total": 0,
                    "skip_reason": "sample_missing_in_h5ad",
                }
            )
            continue
        sample_obs = obs.iloc[sample_positions].reset_index(drop=True)
        sample_coords = coordinates[sample_positions]
        cell_types = sample_obs["resolved_cell_type"].astype(str).to_numpy()
        anchor_ids = set(sample_anchors["cell_id"].astype(str))
        observed, null, qc = analyze_sample(
            sample=str(sample),
            cancer_type=cancer_type,
            sample_obs=sample_obs,
            sample_coords=sample_coords,
            cell_types=cell_types,
            anchor_cell_ids=anchor_ids,
            tme_cell_types=tme_cell_types,
            args=args,
            rng=rng,
        )
        qc_rows.append(qc)
        if qc["skip_reason"]:
            print(
                f"  [{cancer_type}] {sample}: skipped ({qc['skip_reason']})",
                flush=True,
            )
            continue
        print(
            f"  [{cancer_type}] {sample}: "
            f"{qc['n_tumoral_schwann']} tumoral Schwann, "
            f"{qc['n_neighbor_cells']} neighbors, "
            f"{args.n_permutations} permutations",
            flush=True,
        )
        observed_parts.append(observed)
        null_parts.append(null)

    observed = (
        pd.concat(observed_parts, ignore_index=True)
        if observed_parts
        else pd.DataFrame()
    )
    null = (
        pd.concat(null_parts, ignore_index=True)
        if null_parts
        else pd.DataFrame()
    )
    qc = pd.DataFrame(qc_rows)
    return observed, null, qc


def aggregate_analysis_units(
    observed: pd.DataFrame,
    null: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = [
        "cancer_type",
        "analysis_unit",
        "metric_type",
        "metric_name",
    ]
    unit_observed = (
        observed.groupby(group_columns, observed=True)
        .agg(
            n_samples=("sample", "nunique"),
            observed_value=("observed_value", "mean"),
        )
        .reset_index()
    )
    unit_null = (
        null.groupby([*group_columns, "permutation_id"], observed=True)
        .agg(
            n_samples=("sample", "nunique"),
            null_value=("null_value", "mean"),
        )
        .reset_index()
    )
    null_mean = (
        unit_null.groupby(group_columns, observed=True)["null_value"]
        .mean()
        .rename("null_mean")
        .reset_index()
    )
    unit_observed = unit_observed.merge(
        null_mean, on=group_columns, how="left"
    )
    unit_observed["observed_minus_null"] = (
        unit_observed["observed_value"] - unit_observed["null_mean"]
    )
    return unit_observed, unit_null


def build_cancer_statistics(
    unit_observed: pd.DataFrame,
    unit_null: pd.DataFrame,
    min_analysis_units: int,
) -> pd.DataFrame:
    keys = ["cancer_type", "metric_type", "metric_name"]
    rows: list[dict] = []
    null_groups = {
        key: (
            group.groupby("permutation_id", observed=True)["null_value"]
            .mean()
            .to_numpy(dtype=float)
        )
        for key, group in unit_null.groupby(keys, observed=True)
    }
    for key, group in unit_observed.groupby(keys, observed=True):
        cancer_type, metric_type, metric_name = key
        n_units = group["analysis_unit"].nunique()
        observed_value = float(group["observed_value"].mean())
        null_values = null_groups.get(key, np.array([]))
        result = (
            empirical_statistics(observed_value, null_values)
            if n_units >= min_analysis_units
            else empirical_statistics(np.nan, np.array([]))
        )
        rows.append(
            {
                "cancer_type": cancer_type,
                "metric_type": metric_type,
                "metric_name": metric_name,
                "n_analysis_units": n_units,
                "n_samples": int(group["n_samples"].sum()),
                "observed_value": observed_value,
                **result,
            }
        )
    output = pd.DataFrame(rows)
    if output.empty:
        return output
    output["empirical_padj_two_sided"] = (
        output.groupby("cancer_type", group_keys=False)[
            "empirical_p_two_sided"
        ].transform(bh_adjust)
    )
    output["direction"] = np.select(
        [
            output["observed_minus_null"] > 0,
            output["observed_minus_null"] < 0,
        ],
        ["enriched_near_schwann", "depleted_near_schwann"],
        default="no_difference",
    )
    output["significant"] = (
        output["empirical_padj_two_sided"].notna()
        & (output["empirical_padj_two_sided"] < 0.05)
    )
    return output


def main() -> int:
    args = parse_args()
    if args.n_permutations < 10:
        raise ValueError("--n-permutations must be at least 10.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    if "CHOL" in args.cancer_types:
        raise ValueError("CHOL is excluded. Use CRC PDAC PRAD only.")
    print("Tumoral-Schwann neighborhood TME label-permutation test")
    print("  cancers: CRC (v8) / PDAC+PRAD (halo50); no CHOL")
    print(f"  niche radius: {args.niche_radius_um} um")
    print(f"  permutations: {args.n_permutations}")
    print("  null: shuffle non-anchor labels; anchors + coordinates fixed")
    print(f"  output: {args.out_dir}", flush=True)

    observed_parts: list[pd.DataFrame] = []
    null_parts: list[pd.DataFrame] = []
    qc_parts: list[pd.DataFrame] = []
    for cancer_type in args.cancer_types:
        observed, null, qc = process_cancer(cancer_type, args, rng)
        qc_parts.append(qc)
        if not observed.empty:
            observed_parts.append(observed)
        if not null.empty:
            null_parts.append(null)

    if not observed_parts or not null_parts:
        raise RuntimeError("No valid samples completed label permutation.")

    observed = pd.concat(observed_parts, ignore_index=True)
    null = pd.concat(null_parts, ignore_index=True)
    qc = pd.concat(qc_parts, ignore_index=True)

    observed.to_csv(
        args.out_dir / "sample_observed_effects.tsv", sep="\t", index=False
    )
    qc.to_csv(args.out_dir / "sample_qc.tsv", sep="\t", index=False)

    unit_observed, unit_null = aggregate_analysis_units(observed, null)
    unit_observed.to_csv(
        args.out_dir / "analysis_unit_observed_effects.tsv",
        sep="\t",
        index=False,
    )
    if args.write_null_analysis_units:
        null_path = args.out_dir / "null_analysis_unit_metrics.tsv.gz"
        with gzip.open(null_path, "wt") as handle:
            unit_null.to_csv(handle, sep="\t", index=False)

    cancer_stats = build_cancer_statistics(
        unit_observed, unit_null, args.min_analysis_units
    )
    cancer_stats.to_csv(
        args.out_dir / "cancer_empirical_statistics.tsv",
        sep="\t",
        index=False,
    )
    cancer_stats.loc[cancer_stats["significant"]].to_csv(
        args.out_dir / "significant_cancer_statistics.tsv",
        sep="\t",
        index=False,
    )

    print(
        f"Done. Samples kept: "
        f"{(qc['skip_reason'] == '').sum()} / {len(qc)}; "
        f"wrote {args.out_dir}",
        flush=True,
    )
    print(
        "Plot heatmap with:\n"
        "  /storage1/fs1/dinglab/Active/Projects/yuweiz/conda/envs/scenv/bin/Rscript \\\n"
        "    plot_schwann_neighborhood_tme_label_permutation_50um.R",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
