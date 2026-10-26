#!/usr/bin/env Rscript

# Repeat region1000 Invasion vs neighbor fraction/count using 50 um radius matrices.
# Does not overwrite the 100 um region1000 / truecount folders.
#
#   /storage1/fs1/dinglab/Active/Projects/yuweiz/conda/envs/scenv/bin/Rscript \
#     /storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/00_makeh5ad/01_x5k/00_integration/boundary_analysis/00_new/plot_boundary_fraction_vs_invasion_region1000_r50.R

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
})

frac_path <- paste0(
  "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/",
  "00_makeh5ad/01_x5k/00_integration/00_tumor_niche/",
  "tumor_niche_r50_df_includeNecrosis.tsv"
)
count_path <- paste0(
  "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/",
  "00_makeh5ad/01_x5k/00_integration/00_tumor_niche/",
  "tumor_niche_r50_count_df_includeNecrosis.tsv"
)
score_dir <- paste0(
  "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/",
  "00_makeh5ad/01_x5k/00_integration/03_scoring/Output"
)
boundary_root <- paste0(
  "/storage1/fs1/dinglab/Active/Projects/pdc/projects/general/x1000/",
  "boundary/gen11/5k/outputs"
)
subtype_path <- paste0(
  "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/",
  "01_cell_typing/subtype.tsv"
)
out_dir <- paste0(
  "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/",
  "00_makeh5ad/01_x5k/00_integration/boundary_analysis/00_new/",
  "brca_merged/boundary_fraction_invasion_excl_region1000_r50"
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

MIN_REGION_TUMOR_CELLS <- 1000L
MIN_BOUNDARY_CELLS_PER_REGION <- 20L
MIN_PAIRED_UNITS <- 5L

ct_order <- c(
  "Fibroblast", "Macrophage", "Endothelial", "Pericyte/vSMC", "SMC",
  "NonMalignant_Parenchymal", "Plasma", "B_cell", "NK_T", "DC",
  "Neutrophil", "Granulocyte", "Necrosis", "Lymphatic_Endothelial",
  "Schwann", "Tumor"
)
ct_cols <- c(
  "B_cell", "DC", "Endothelial", "Fibroblast", "Granulocyte",
  "Lymphatic_Endothelial", "Macrophage", "NK_T", "Necrosis", "Neutrophil",
  "NonMalignant_Parenchymal", "Pericyte/vSMC", "Plasma", "SMC", "Schwann", "Tumor"
)
cancer_pref <- c("PDAC", "CRC", "PRAD", "BRCA", "CHOL", "HNSC", "LUNG", "SKCM", "GBM")

merge_brca_label <- function(x) {
  x <- as.character(x)
  x[x %in% c("BRCA_Basal", "BRCA_basal", "BRCA_Luminal", "BRCA_luminal")] <- "BRCA"
  x
}
is_rcc_label <- function(x) grepl("rcc", as.character(x), ignore.case = TRUE)
is_excluded_sample <- function(x) {
  x <- as.character(x)
  grepl("^(C3L-01455|C3N-01411|C3N-04162)(-|$)", x) | grepl("^AL", x)
}
star_from_fdr <- function(p) {
  fcase(
    p < 0.001, "***",
    p < 0.01, "**",
    p < 0.05, "*",
    default = ""
  )
}

spearman_xy <- function(x, y) {
  ok <- is.finite(x) & is.finite(y)
  n <- sum(ok)
  if (n < MIN_PAIRED_UNITS || sd(x[ok]) == 0 || sd(y[ok]) == 0) return(NULL)
  ct <- suppressWarnings(cor.test(x[ok], y[ok], method = "spearman", exact = FALSE))
  data.table(
    n_regions = n,
    spearman_rho = unname(ct$estimate),
    spearman_p = ct$p.value
  )
}

write_heatmap <- function(stats_dt, outfile, title, subtitle, fill_name) {
  ct_levels <- intersect(ct_order, unique(stats_dt$cell_type))
  cancers <- unique(as.character(stats_dt$cancer_type))
  cancers <- c(intersect(cancer_pref, cancers), setdiff(sort(cancers), cancer_pref))
  grid <- CJ(cancer_type = cancers, cell_type = ct_levels, unique = TRUE)
  plot_df <- merge(grid, stats_dt, by = c("cancer_type", "cell_type"), all.x = TRUE)
  plot_df[is.na(star), star := ""]
  plot_df[, cell_type := factor(cell_type, levels = ct_levels)]
  plot_df[, cancer_type := factor(cancer_type, levels = cancers)]
  vals <- plot_df$spearman_rho
  lim <- max(abs(vals[is.finite(vals)]), na.rm = TRUE)
  if (!is.finite(lim) || lim == 0) lim <- 0.2
  p <- ggplot(plot_df, aes(x = cell_type, y = cancer_type, fill = spearman_rho)) +
    geom_tile(color = "white", linewidth = 0.3) +
    geom_text(aes(label = star), size = 4.2, vjust = 0.75) +
    scale_fill_gradient2(
      low = "#2166AC", mid = "white", high = "#B2182B",
      midpoint = 0, limits = c(-lim, lim), na.value = "grey70",
      name = fill_name
    ) +
    labs(title = title, subtitle = subtitle, x = "Neighbor cell type", y = "Cancer type") +
    theme_bw(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1), panel.grid = element_blank())
  ggsave(
    outfile,
    p,
    width = max(8, 0.48 * length(ct_levels) + 3.2),
    height = max(4.5, 0.35 * length(cancers) + 2.6)
  )
  message("Wrote: ", outfile)
}

annotate_cancer <- function(merged) {
  if (!file.exists(subtype_path)) return(merged)
  subtype_df <- fread(subtype_path)
  setnames(subtype_df, names(subtype_df)[1], "sample_id")
  merged <- merge(
    merged,
    subtype_df[, .(sample_id, subtype)],
    by.x = "sample",
    by.y = "sample_id",
    all.x = TRUE
  )
  merged[!is.na(subtype) & subtype != "", cancer_type := merge_brca_label(subtype)]
  merged[!is_rcc_label(cancer_type)]
}

message("Loading gen11 boundary tumor cells (same rule as region1000)...")
bnd_files <- list.files(
  boundary_root,
  pattern = "_boundary_distance.csv$",
  recursive = TRUE,
  full.names = TRUE
)
bnd_list <- lapply(bnd_files, function(f) {
  samp <- gsub("_boundary_distance.csv", "", basename(f))
  cancer_boundary <- basename(dirname(dirname(f)))
  if (is_excluded_sample(samp) || is_rcc_label(cancer_boundary)) return(NULL)
  d <- fread(f, select = c("cell_id", "tumor_region", "cell_type", "dist_iface_um"))
  n_tumor <- d[cell_type == "Tumor", .N, by = tumor_region]
  if (!nrow(n_tumor)) return(NULL)
  is_pdac <- grepl("^PDAC$", cancer_boundary, ignore.case = TRUE)
  keep_regions <- if (is_pdac) n_tumor$tumor_region else n_tumor[N >= MIN_REGION_TUMOR_CELLS, tumor_region]
  if (!length(keep_regions)) return(NULL)
  d <- d[
    cell_type == "Tumor" &
      tumor_region %in% keep_regions &
      !is.na(dist_iface_um) &
      dist_iface_um >= -50 &
      dist_iface_um <= 0
  ]
  if (!nrow(d)) return(NULL)
  d[, barcode := paste0(as.character(cell_id), "_", samp)]
  d[, sample := samp]
  d <- merge(d, n_tumor[, .(tumor_region, n_tumor_in_region = N)], by = "tumor_region")
  d[, .(barcode, sample, tumor_region, n_tumor_in_region)]
})
bnd <- rbindlist(bnd_list, fill = TRUE)
message("Boundary tumor cells: ", nrow(bnd), " in ", uniqueN(bnd[, paste(sample, tumor_region)]), " regions")
fwrite(
  unique(bnd[, .(sample, tumor_region, n_tumor_in_region)]),
  file.path(out_dir, "kept_tumor_regions.tsv"),
  sep = "\t"
)

message("Loading CSEA_Invasion...")
score_files <- list.files(score_dir, pattern = "_cancerSEA_AUC_Summary.csv$", full.names = TRUE)
score_files <- score_files[!grepl("RCC", basename(score_files), ignore.case = TRUE)]
scores <- rbindlist(lapply(score_files, function(f) {
  dt <- fread(f)
  setnames(dt, 1L, "barcode")
  dt[, .(barcode = as.character(barcode), CSEA_Invasion)]
}), fill = TRUE)
scores <- scores[!is.na(CSEA_Invasion)]

read_niche <- function(path, extra_cols = character()) {
  dt <- fread(path)
  setnames(dt, 1L, "barcode")
  dt[, barcode := as.character(barcode)]
  need <- c(ct_cols, "cancer_type", extra_cols)
  missing_ct <- setdiff(need, names(dt))
  if (length(missing_ct)) stop("Missing columns in ", path, ": ", paste(missing_ct, collapse = ", "))
  dt <- dt[barcode %in% bnd$barcode, c("barcode", need), with = FALSE]
  dt[, cancer_type := merge_brca_label(cancer_type)]
  dt <- dt[!is_rcc_label(cancer_type)]
  if (!nrow(dt)) stop("No barcode overlap with ", path)
  dt
}

# Fraction heatmap
message("Loading 50 um neighborhood fractions...")
comp <- read_niche(frac_path)
merged_f <- annotate_cancer(merge(comp, bnd, by = "barcode"))
merged_f <- merge(merged_f, scores, by = "barcode")
message("Cells with r50 fraction + Invasion: ", nrow(merged_f))
if (!nrow(merged_f)) stop("No overlapping fraction cells.")

long_f <- melt(
  merged_f,
  id.vars = c("barcode", "sample", "cancer_type", "tumor_region", "n_tumor_in_region", "CSEA_Invasion"),
  measure.vars = intersect(ct_cols, names(merged_f)),
  variable.name = "cell_type",
  value.name = "fraction",
  variable.factor = FALSE
)
long_f <- long_f[!is.na(fraction) & !is.na(CSEA_Invasion)]
region_inv_f <- unique(long_f[, .(
  mean_invasion = mean(CSEA_Invasion, na.rm = TRUE),
  n_boundary_cells = uniqueN(barcode),
  n_tumor_in_region = n_tumor_in_region[1]
), by = .(cancer_type, sample, tumor_region)])
region_inv_f <- region_inv_f[n_boundary_cells >= MIN_BOUNDARY_CELLS_PER_REGION]
region_frac <- long_f[, .(mean_fraction = mean(fraction, na.rm = TRUE)),
                      by = .(cancer_type, sample, tumor_region, cell_type)]
region_pb_f <- merge(region_frac, region_inv_f, by = c("cancer_type", "sample", "tumor_region"))
fwrite(region_pb_f, file.path(out_dir, "region_mean_fraction_vs_mean_invasion.tsv"), sep = "\t")

frac_stats <- region_pb_f[, spearman_xy(mean_fraction, mean_invasion), by = .(cancer_type, cell_type)]
frac_stats[, fdr_within_cancer := p.adjust(spearman_p, method = "BH"), by = cancer_type]
frac_stats[, star := star_from_fdr(fdr_within_cancer)]
fwrite(frac_stats, file.path(out_dir, "cancer_regionlevel_fraction_vs_invasion_stats.tsv"), sep = "\t")
write_heatmap(
  frac_stats,
  file.path(out_dir, "invasion_vs_neighbor_fraction_regionlevel_spearman_heatmap.pdf"),
  "Tumor-region pseudobulk: mean Invasion vs mean neighbor fraction (r = 50 um)",
  "Same regions as region1000 (>=1000 tumor cells except PDAC); boundary -50 to 0 um; drop AL* and LUNG C3L-01455/C3N-01411/C3N-04162; * FDR < 0.05",
  "r50 fraction\nSpearman rho"
)

# Count heatmap
message("Loading 50 um neighborhood counts...")
cnt <- read_niche(count_path, extra_cols = "total_neighbor_count")
merged_c <- annotate_cancer(merge(cnt, bnd, by = "barcode"))
merged_c <- merge(merged_c, scores, by = "barcode")
message("Cells with r50 count + Invasion: ", nrow(merged_c))
if (!nrow(merged_c)) stop("No overlapping count cells.")

long_c <- melt(
  merged_c,
  id.vars = c(
    "barcode", "sample", "cancer_type", "tumor_region",
    "n_tumor_in_region", "total_neighbor_count", "CSEA_Invasion"
  ),
  measure.vars = intersect(ct_cols, names(merged_c)),
  variable.name = "cell_type",
  value.name = "n_count",
  variable.factor = FALSE
)
long_c <- long_c[!is.na(n_count) & !is.na(CSEA_Invasion)]
region_inv_c <- unique(long_c[, .(
  mean_invasion = mean(CSEA_Invasion, na.rm = TRUE),
  mean_window_n = mean(total_neighbor_count, na.rm = TRUE),
  n_boundary_cells = uniqueN(barcode),
  n_tumor_in_region = n_tumor_in_region[1]
), by = .(cancer_type, sample, tumor_region)])
region_inv_c <- region_inv_c[n_boundary_cells >= MIN_BOUNDARY_CELLS_PER_REGION]
region_cnt <- long_c[, .(mean_count = mean(n_count, na.rm = TRUE)),
                     by = .(cancer_type, sample, tumor_region, cell_type)]
region_pb_c <- merge(region_cnt, region_inv_c, by = c("cancer_type", "sample", "tumor_region"))
fwrite(region_pb_c, file.path(out_dir, "region_mean_count_vs_mean_invasion.tsv"), sep = "\t")

count_stats <- region_pb_c[, spearman_xy(mean_count, mean_invasion), by = .(cancer_type, cell_type)]
count_stats[, fdr_within_cancer := p.adjust(spearman_p, method = "BH"), by = cancer_type]
count_stats[, star := star_from_fdr(fdr_within_cancer)]
fwrite(count_stats, file.path(out_dir, "cancer_regionlevel_count_vs_invasion_stats.tsv"), sep = "\t")
write_heatmap(
  count_stats,
  file.path(out_dir, "invasion_vs_neighbor_count_regionlevel_spearman_heatmap.pdf"),
  "Tumor-region pseudobulk: mean Invasion vs mean neighbor count (r = 50 um)",
  "Real 50 um radius counts; same regions as region1000; * FDR < 0.05",
  "r50 count\nSpearman rho"
)

win_stats <- region_inv_c[, spearman_xy(mean_window_n, mean_invasion), by = cancer_type]
if (nrow(win_stats)) {
  win_stats[, cell_type := "window_n"]
  fwrite(win_stats, file.path(out_dir, "cancer_regionlevel_windowN_vs_invasion_stats.tsv"), sep = "\t")
}

cmp <- merge(
  frac_stats[, .(cancer_type, cell_type, n_regions,
                 frac_rho = spearman_rho, frac_fdr = fdr_within_cancer)],
  count_stats[, .(cancer_type, cell_type, count_rho = spearman_rho, count_fdr = fdr_within_cancer)],
  by = c("cancer_type", "cell_type"),
  all = TRUE
)
cmp[, interpretation := fcase(
  !is.na(frac_rho) & frac_rho < 0 & frac_fdr < 0.05 &
    (is.na(count_rho) | count_fdr >= 0.05 | count_rho >= 0),
  "proportion down, count not down: likely diluted by other cells increasing",
  !is.na(frac_rho) & frac_rho < 0 & frac_fdr < 0.05 &
    !is.na(count_rho) & count_rho < 0 & count_fdr < 0.05,
  "proportion and count both down: more likely real loss",
  !is.na(frac_rho) & frac_rho > 0 & frac_fdr < 0.05 &
    !is.na(count_rho) & count_rho > 0 & count_fdr < 0.05,
  "proportion and count both up: more likely real gain",
  !is.na(frac_rho) & frac_rho > 0 & frac_fdr < 0.05 &
    (is.na(count_rho) | count_fdr >= 0.05 | count_rho <= 0),
  "proportion up, count not up: relative enrichment only",
  default = ""
)]
setorder(cmp, cancer_type, cell_type)
fwrite(cmp, file.path(out_dir, "fraction_vs_count_comparison.tsv"), sep = "\t")
message("Wrote comparison TSV: ", file.path(out_dir, "fraction_vs_count_comparison.tsv"))
message("Outputs in: ", out_dir)
