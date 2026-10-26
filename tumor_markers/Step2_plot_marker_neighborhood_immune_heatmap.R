#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(data.table)
  library(ComplexHeatmap)
  library(circlize)
  library(grid)
})

args <- commandArgs(trailingOnly = TRUE)
SUBDIR <- if (length(args)) args[[1]] else "broad"
BASE_DIR <- paste0(
  "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/",
  "01_cell_typing/01_analysis/05_markerexp/marker_neighborhood_immune"
)
RESULT_DIR <- file.path(BASE_DIR, SUBDIR)
ROW_ORDER_FILE <- file.path(
  dirname(BASE_DIR),
  "cancer_specific_relative_expression_plot_row_order.tsv"
)
IMMUNE_COLUMNS <- c("B_cell", "Plasma", "T_cell", "Macrophage", "Mast", "DC")
SAMPLE_FDR <- 0.10
CELL_FDR <- 0.05
CELL_MIN_ABS_RHO <- 0.05
CELL_MIN_CONSISTENCY <- 0.75

cancer_colors <- c(
  BRCA = "#D88BB8", CHOL = "#B5A642", CRC = "#D45740", GBM = "#6B5CA5",
  HNSC = "#C96A55", LUNG = "#3F86C5", RCC = "#2F6F8F", PRAD = "#479E88",
  PDAC = "#6CB9D3"
)

res <- fread(file.path(RESULT_DIR, "marker_neighborhood_immune_all_results.tsv"))
res <- res[immune_group %in% IMMUNE_COLUMNS]

# BH is recomputed within cancer over the plotted immune groups only.
res[, fdr_sample := p.adjust(p_mean_log1p_cp10k, "BH"), by = cancer]
res[, fdr_cell := p.adjust(wilcoxon_p, "BH"), by = cancer]
res[, sample_hit := !is.na(fdr_sample) & fdr_sample < SAMPLE_FDR]
res[, consistency := fifelse(
  median_rho > 0, frac_samples_positive_rho, 1 - frac_samples_positive_rho
)]
res[, cell_hit := !is.na(fdr_cell) & fdr_cell < CELL_FDR &
  abs(median_rho) >= CELL_MIN_ABS_RHO & consistency >= CELL_MIN_CONSISTENCY]
fwrite(
  res,
  file.path(RESULT_DIR, "marker_neighborhood_immune_results_no_all_immune.tsv"),
  sep = "\t"
)

row_order <- fread(ROW_ORDER_FILE)[order(plot_row_index)]
row_order[, cancer := sub("^RCC_ccRCC$", "RCC", marker_cancer)]
row_order[, row_key := paste(cancer, gene, sep = "|")]
res[, row_key := paste(cancer, gene, sep = "|")]

to_matrix <- function(value) {
  wide <- dcast(res, row_key ~ immune_group, value.var = value)
  mat <- as.matrix(wide[, setdiff(names(wide), "row_key"), with = FALSE])
  rownames(mat) <- wide$row_key
  mat <- mat[row_order$row_key, IMMUNE_COLUMNS, drop = FALSE]
  rownames(mat) <- row_order$gene
  mat
}
sample_rho <- to_matrix("rho_mean_log1p_cp10k")
sample_hit <- to_matrix("sample_hit")
cell_rho <- to_matrix("median_rho")
cell_hit <- to_matrix("cell_hit")

cancer_split <- factor(row_order$cancer, levels = names(cancer_colors))
cancer_anno <- rowAnnotation(
  Cancer = row_order$cancer,
  col = list(Cancer = cancer_colors),
  show_annotation_name = FALSE,
  show_legend = FALSE,
  simple_anno_size = unit(3, "mm")
)

star_fun <- function(hits) {
  function(j, i, x, y, width, height, fill) {
    if (isTRUE(as.logical(hits[i, j]))) {
      grid.text("*", x, y, vjust = 0.75, gp = gpar(fontsize = 11))
    }
  }
}

make_heatmap <- function(mat, hits, limit, title, legend_title, first = FALSE) {
  Heatmap(
    mat,
    name = legend_title,
    col = colorRamp2(c(-limit, 0, limit), c("#2166AC", "white", "#B2182B")),
    na_col = "grey88",
    cluster_rows = FALSE,
    cluster_columns = FALSE,
    row_split = cancer_split,
    row_title_rot = 0,
    row_title_gp = gpar(fontsize = 9, fontface = "bold"),
    row_gap = unit(1.2, "mm"),
    left_annotation = if (first) cancer_anno else NULL,
    show_row_names = first,
    row_names_side = "left",
    row_names_gp = gpar(fontsize = 8),
    column_names_gp = gpar(fontsize = 9),
    column_names_rot = 45,
    column_title = title,
    column_title_gp = gpar(fontsize = 10),
    rect_gp = gpar(col = "white", lwd = 0.6),
    width = unit(6 * ncol(mat), "mm"),
    cell_fun = star_fun(hits),
    heatmap_legend_param = list(
      at = c(-limit, -limit / 2, 0, limit / 2, limit),
      title_gp = gpar(fontsize = 8, fontface = "bold"),
      labels_gp = gpar(fontsize = 7),
      legend_height = unit(25, "mm")
    )
  )
}

ht_sample <- make_heatmap(
  sample_rho, sample_hit, 0.8,
  "Across samples",
  "Across-sample\nSpearman rho",
  first = TRUE
)
cell_label <- if (grepl("knn", SUBDIR)) {
  paste0("Within sample\n(", toupper(sub(".*_", "", SUBDIR)), " tumor pseudobulk)")
} else {
  "Within sample\n(single tumor cell)"
}
ht_cell <- make_heatmap(
  cell_rho, cell_hit, 0.2,
  cell_label,
  "Within-sample\nmedian rho"
)

ht_list <- ht_sample + ht_cell
caption <- paste0(
  "Across samples: Spearman rho of sample mean tumor expression vs immune fraction\n",
  "among TME cells within 50 µm of tumor; * BH FDR < 0.10.\n",
  "Within sample: median per-sample Spearman rho of tumor expression vs local immune\n",
  "fraction (50 µm); * BH FDR < 0.05, |median rho| >= 0.05, same sign in >= 75% samples.\n",
  "Grey: cell type not annotated in that cancer."
)

plot_height <- 0.17 * nrow(sample_rho) + 3.8
for (ext in c("pdf", "png")) {
  out <- file.path(RESULT_DIR, paste0("marker_neighborhood_immune_heatmap_R.", ext))
  if (ext == "pdf") {
    pdf(out, width = 8, height = plot_height)
  } else {
    png(out, width = 8, height = plot_height, units = "in", res = 300)
  }
  draw(
    ht_list,
    ht_gap = unit(8, "mm"),
    column_title = "Tumor markers vs immune composition of tumor-adjacent TME (50 µm)",
    column_title_gp = gpar(fontsize = 11, fontface = "bold"),
    merge_legend = TRUE,
    padding = unit(c(24, 4, 4, 4), "mm")
  )
  grid.text(
    caption,
    x = unit(0.5, "npc"), y = unit(12, "mm"),
    gp = gpar(fontsize = 6.5)
  )
  invisible(dev.off())
}
cat("Wrote", file.path(RESULT_DIR, "marker_neighborhood_immune_heatmap_R.pdf"), "\n")
cat("Sample-level hits:", sum(res$sample_hit), " cell-level hits:", sum(res$cell_hit), "\n")
