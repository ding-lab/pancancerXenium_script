# Xenium spatial analysis code

Analysis scripts for a multi-cancer Xenium cohort (5,000-gene and 477-gene panels). The code covers cohort preprocessing, tumor-border organization, cancer-associated nerves, tertiary lymphoid structures (TLS), and tumor-marker neighborhoods.

Scripts are written to run on the project filesystem. Input and output paths inside each file point at those objects (AnnData `.h5ad` files, boundary tables, nerve indexes, and TLS catalogs). They are not packaged as an installable library, and the data are not in this directory.

## Layout

```
preprocessing/     QC, normalization, HVG, Harmony, neighbors, Leiden
tumor_border/      interface distance, tumor niche, invasion score, CellChat
tumor_markers/     marker–immune neighborhoods, spatial heterogeneity, ANXA8, CD70
nerve/             nerve-to-tumor distance, repair programs, ENS vs peripheral nerve, TCGA
TLS/               TLS maturation and LUAD M-cell–tumor analyses
```

## Requirements

Python scripts expect Scanpy, AnnData, NumPy, pandas, SciPy, scikit-learn, h5py, and Harmony (`harmonypy` or `scanpy.external`). Some steps also use PyDESeq2, decoupler, or Hotspot.

R scripts expect ggplot2, data.table, and, where named in the file, Seurat, CellChat or SpatialCellChat, DESeq2, AUCell, survival, and survminer.

Run Python with the project Scanpy environment and R with the project R environment. Large objects should be run on a compute node, not a login node.

## Preprocessing

| Script | What it does |
|---|---|
| `1_preprocess_harmony_5k_hvg.py` | 5k objects. Keeps cells with more than 20 genes and more than 50 counts, stores raw counts, normalizes to library size, log1p-transforms, selects highly variable genes within `dataset`, then runs PCA, Harmony, neighbors, and Leiden. |
| `1_preprocess_harmony_477.py` | Same Harmony workflow for the 477-gene panel. The low-count cell filter in the 5k script is not applied. |

Harmony uses `dataset` as the batch key and writes `obsm['X_pca_harmony']`.

## Tumor border

Tumor side of the tissue interface is `dist_iface_um <= 0` (negative inside the tumor, 0 at the interface). Core is `dist < -50` µm and border is `[-50, 0]` µm unless a script sets another window.

| Script | What it does |
|---|---|
| `tumor_neighborhood.py` | Cell-type composition of non-tumor cells within 50 µm of tumor cells. |
| `Tcell_density.py` | T-cell density and clumps on the tumor border (`[-50, 0]` µm), with label permutation. |
| `Delaunay.py` | Delaunay contacts among fibroblasts, macrophages, and T cells on the tumor border, tested against label permutation. |
| `plot_invasion_core_vs_border_tumor.R` | Paired core-versus-border CancerSEA invasion scores on tumor cells. Regions need at least 100 tumor cells. BRCA basal and luminal labels are merged. RCC is excluded. |
| `boder_fraction2invasion_score.R` | Region-level invasion score versus 50 µm neighbor fraction and count. |
| `PRAD_macrophage_scoring.py` | PRAD macrophage program scores within 50 µm of tumor, compared by Wilcoxon test. |
| `spatial_cellchat_run.R` | Spatial CellChat on a Seurat object (`--input`, `--outdir`). |

## Tumor markers

| Script | What it does |
|---|---|
| `Step1_marker_neighborhood_immune.py` | Spearman tests of tumor-marker expression versus immune-cell fraction within a fixed radius, at sample level and at tumor-cell level. |
| `Step2_plot_marker_neighborhood_immune_heatmap.R` | Heat map of those associations. |
| `Hetexp_hotsplot_loop.py` | Hotspot spatial modules on tumor cells. Cell types come from the annotation CSV. Genes are limited to the shared panel. |
| `Hetexp_spatial_pearson.py` | Spatially smoothed Pearson correlation among heterogeneous tumor genes. |
| `ANXA8/` | PDAC ANXA8 versus MUC5B tumor cells: overlap, CancerSEA scores, nearby T cells, macrophage distance, and paired pseudobulk DESeq2 of ANXA8-only versus MUC5B-only cells. |
| `CD70/` | ccRCC CD70: T-cell Harmony reclustering, CD70-high versus CD70-low pathway scores, and exhausted or proliferative CD8 T cells versus distance to tumor. |

## Nerve

| Script | What it does |
|---|---|
| `CRC_PDAC_PRAD_nerve2nearest_tumor_distance.py` | Distance from nerve cells to the nearest tumor cell in CRC, PDAC, and PRAD. |
| `repair_score/repair_score_build_crc_pdac_prad_repair_scores.py` | Nerve repair-program scores. Nerves with fewer than five scored cells are dropped. |
| `repair_score/plot_repair_tumoral_vs_normal.R` | Tumoral versus normal nerve repair scores. |
| `plot_synaptic_trophic_correlation.R` | Correlation of synaptic and trophic nerve programs. |
| `nerve_stromal/tme_label_permutation_50um.py` | Stromal cell types within 50 µm of nerve, tested by label permutation. |
| `nerve_stromal/plot_tme_label_permutation_50um.R` | Plot of that permutation test. |
| `nerve_stromal/repair_vs_pdgfa.py` | Nerve repair score versus PDGFA. |
| `tcga_v5_min5_ggforest_and_comparison.R` | TCGA overall survival for nerves with at least five members. Cox models are drawn with survminer. PRAD is adjusted for age and Gleason score. LUAD and LUSC are adjusted for smoking history. |
| `CRC_ENS_EPN/classify_crc_enteric_peripheral.R` | Split CRC nerves into enteric and peripheral. |
| `CRC_ENS_EPN/CRC_ENS_EPN_exp_prepare_nerve_glial_plus_ens_expression.py` | Expression matrix for nerve glia plus enteric neurons. |
| `CRC_ENS_EPN/CRC_ENS_EPN_exp_plot.R` | Plot of that expression comparison. |
| `CRC_ENS_EPN/CancerSEA_step1_assign_tumor_near_boundary_core.py` | Assign CRC tumor cells to near-nerve (≤50 µm), border (`[-50, 0]`), or core (`< -50`). Near overrides border and core. |
| `CRC_ENS_EPN/CancerSEA_step2_subset_min_near.py` | Keep samples with enough near-nerve tumor cells. |
| `CRC_ENS_EPN/CancerSEA_step3_plot_all_csea_patient_means.R` | Patient-mean CancerSEA scores for those compartments. |

## TLS

### Maturation

| Script | What it does |
|---|---|
| `tls_tcell_harmony_umap.py` | TLS-core T cells: Harmony, Leiden, UMAP, and T/Tfh marker dot plots. |
| `tls_fibroblast_dc_harmony_umap.py` | TLS-core fibroblasts and dendritic cells: Harmony, Leiden, and UMAP. |
| `score_b_cell_aucell.py` | AUCell scores for B-cell programs. |
| `score_tls_fibroblast_fdc_aucell.py` | AUCell scores for the follicular dendritic-cell program in TLS fibroblasts. |
| `kmean_model_tls_maturation_gc_fdc_score.py` | k-means maturation stage (early, immature, mature) from germinal-center B cells and FDC score. Lymph nodes are the reference. TLS with fewer than 10 B cells are excluded. |
| `compute_tls_tcell_composition.py` | T-cell composition inside TLS cores. |
| `compute_tls_fibroblast_composition.py` | Fibroblast subtype fractions (FDC, CXCL13+ TRC, CXCL9+ activated TRC, MRC, fibroblast) by maturation stage. |
| `compute_tls_fdc_gcb_program_metrics.py` | Per-TLS FDC and germinal-center B-cell program metrics. |
| `compute_tls_vs_tagg_tcell_fraction.py` | T-cell fractions in TLS cores versus T-cell aggregates, split into normal and tumoral sites. |

### M-cell tumor (LUAD)

| Script | What it does |
|---|---|
| `get_gradient_Bexp_McellAnchor.py` | B-cell expression along distance from M-cell–tumor anchors inside TLS. |
| `get_gradient_Bexp_OtherTumorAnchor.py` | Same B cells and TLS, with otherTumor cells as the anchors. |
| `compare_mcell_other_tumor_local_B_T.py` | Local B and T cells around M-cell–tumor versus other tumor cells. |
| `plot_ccr6_anchor_comparison.R` | CCR6 comparison for the two anchors. |
| `DEseq2_tlsTumor_othertumor_deg.py` | PyDESeq2 on pseudobulk counts from TLS-associated tumor versus other tumor. |
| `cellchat/01_export_tumoral_tls_h5ad.py` | One h5ad per LUAD sample for cells in tumoral TLS core or the 100 µm halo. |
| `cellchat/02_convert_h5ad_to_seurat.R` | Convert those objects to Seurat. |
| `cellchat/03_run_cellchat.R` | Run CellChat. |
| `cellchat/04_generate_cellchat_commands.py` | Write one shell command per sample RDS. |
| `cellchat/05_summarize_CCL20_CCR6_to_Bcell.R` | Summarize CCL20–CCR6 signaling to B cells. |

## Notes

Paths in the scripts are absolute and point at `/storage1` or `/rdcw` project storage. Change them before running elsewhere.

Several R and Python files write next to the script or into a hard-coded results directory. Read the path block at the top of a script before launching it.

`plot_invasion_core_vs_border_tumor.R` does not apply an extra sample or region exclusion list. Other scripts still drop named samples or rare sections. Those lists are in the script, not in this README.
