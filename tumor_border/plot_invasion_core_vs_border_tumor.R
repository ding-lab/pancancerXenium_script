#!/usr/bin/env Rscript

# Tumor-cell CSEA Invasion, core vs border, one tumor region = one pair.
# Tumor cells only. gen11 dist_iface_um, tumor side, dist <= 0.
# Region size = all Tumor cells in the region, before the distance window.
# Keep regions with >= 100 tumor cells.
# Core: dist < -50. Border: [-50, 0].
# A region is kept when both compartments have >= 1 scored cell.
# BRCA basal/luminal are merged. RCC is removed.
# No sample or region exclusion list.
# Does not overwrite earlier ridge PDFs.
#
#   /storage1/fs1/dinglab/Active/Projects/yuweiz/conda/envs/scenv/bin/Rscript \
#     /storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/05_draft/code/boundary/plot_invasion_core_vs_border_tumor.R

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(ggridges)
})

root <- paste0(
  "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/",
  "00_makeh5ad/01_x5k/00_integration/boundary_analysis/00_new/",
  "brca_merged/boundary_tumor_states/region_as_unit/",
  "region100_core_lt50_bnd_m50_0"
)
dir.create(root, recursive = TRUE, showWarnings = FALSE)

score_dir <- paste0(
  "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/",
  "00_makeh5ad/01_x5k/00_integration/03_scoring/Output/"
)
boundary_dir <- paste0(
  "/storage1/fs1/dinglab/Active/Projects/pdc/projects/general/x1000/",
  "boundary/gen11/5k/outputs"
)
subtype_path <- paste0(
  "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/01_cell_typing/subtype.tsv"
)

cancers <- c("CHOL", "GBM", "PDAC", "CRC", "PRAD", "BRCA", "HNSC", "LUNG", "SKCM")
min_region <- 100L

merge_brca_label <- function(x) {
  x <- as.character(x)
  x[x %in% c("BRCA_Basal", "BRCA_basal", "BRCA_Luminal", "BRCA_luminal")] <- "BRCA"
  x
}
is_rcc_label <- function(x) grepl("rcc", as.character(x), ignore.case = TRUE)

subtype_df <- fread(subtype_path)
setnames(subtype_df, 1L, "xusample_id")

read_one <- function(f, samp) {
  tmp <- tempfile(fileext = ".csv")
  on.exit(unlink(tmp), add = TRUE)
  status <- system(sprintf("cut -d, -f1,2,3,6,7 '%s' > '%s'", f, tmp))
  if (status != 0) return(NULL)
  d <- fread(tmp, header = TRUE, showProgress = FALSE)
  setnames(d, c("cell_id", "tumor_region", "side", "dist_iface_um", "cell_type"))
  d[, dist_iface_um := as.numeric(dist_iface_um)]
  d <- d[cell_type == "Tumor"]
  if (!nrow(d)) return(NULL)
  d[, n_tumor := .N, by = tumor_region]
  d <- d[
    !is.na(dist_iface_um) &
      tolower(trimws(as.character(side))) == "tumor" &
      dist_iface_um <= 0
  ]
  if (!nrow(d)) return(NULL)
  d[, compartment := fifelse(dist_iface_um < -50, "Core", "Border")]
  d[, .(
    barcode = paste0(cell_id, "_", samp),
    sample_id = samp,
    tumor_region,
    n_tumor,
    compartment
  )]
}

region_rows <- list()
for (cancer in cancers) {
  message("=== ", cancer, " ===")
  files <- list.files(
    file.path(boundary_dir, cancer),
    pattern = "_boundary_distance.csv$",
    recursive = TRUE,
    full.names = TRUE
  )
  if (!length(files)) next
  score_f <- file.path(score_dir, paste0(cancer, "_cancerSEA_AUC_Summary.csv"))
  tmp_sc <- tempfile(fileext = ".csv")
  system(sprintf("cut -d, -f1,11 '%s' > '%s'", score_f, tmp_sc))
  sc <- fread(tmp_sc, header = TRUE, showProgress = FALSE)
  unlink(tmp_sc)
  setnames(sc, c("barcode", "CSEA_Invasion"))
  sc[, barcode := as.character(barcode)]
  sc <- sc[is.finite(as.numeric(CSEA_Invasion))]
  sc[, CSEA_Invasion := as.numeric(CSEA_Invasion)]
  setkey(sc, barcode)

  parts <- lapply(files, function(f) {
    samp <- gsub("_boundary_distance.csv", "", basename(f))
    bnd <- read_one(f, samp)
    if (is.null(bnd) || !nrow(bnd)) return(NULL)
    setkey(bnd, barcode)
    m <- sc[bnd, on = "barcode", nomatch = 0]
    if (!nrow(m)) return(NULL)
    m[, .(
      sum_score = sum(CSEA_Invasion),
      n = .N
    ), by = .(sample_id, tumor_region, n_tumor, compartment)]
  })
  rm(sc)
  gc()
  part <- rbindlist(parts, fill = TRUE)
  if (!nrow(part)) next
  part[, cancer_type := cancer]
  region_rows[[cancer]] <- part
  message("  region-compartment rows: ", nrow(part))
  rm(parts, part)
  gc()
}

agg <- rbindlist(region_rows, fill = TRUE)
rm(region_rows)
gc()

wide <- dcast(
  agg,
  cancer_type + sample_id + tumor_region + n_tumor ~ compartment,
  value.var = c("sum_score", "n"),
  fill = 0
)
rm(agg)
gc()
for (col in c("sum_score_Core", "sum_score_Border", "n_Core", "n_Border")) {
  if (!col %in% names(wide)) wide[, (col) := 0]
}
wide <- merge(
  wide,
  subtype_df[, .(xusample_id, subtype)],
  by.x = "sample_id",
  by.y = "xusample_id",
  all.x = TRUE
)
wide[is.na(subtype) | subtype == "", subtype := cancer_type]
wide[, subtype := merge_brca_label(subtype)]
wide <- wide[!is_rcc_label(subtype)]

band_means <- function(sum_score, n) fifelse(n > 0, sum_score / n, NA_real_)
scored <- wide[, .(
  cancer_type, sample_id, tumor_region, n_tumor, subtype,
  n_core = n_Core,
  n_border = n_Border,
  Core = band_means(sum_score_Core, n_Core),
  Border = band_means(sum_score_Border, n_Border)
)]
scored <- scored[
  n_tumor >= min_region &
    n_core >= 1 & n_border >= 1 &
    is.finite(Core) & is.finite(Border)
]

stats <- scored[, {
  p_raw <- tryCatch(
    wilcox.test(Border, Core, paired = TRUE, exact = FALSE)$p.value,
    error = function(e) NA_real_
  )
  .(
    n_pairs = .N,
    p_raw = p_raw,
    mean_delta = mean(Border - Core),
    median_core = median(Core),
    median_border = median(Border),
    n_border_gt = sum(Border > Core)
  )
}, by = subtype]
stats <- stats[n_pairs >= 3]
stats[, p_lab := fifelse(
  is.na(p_raw), "p=NA",
  fifelse(
    p_raw < 0.001,
    paste0("p=", formatC(p_raw, format = "e", digits = 1)),
    paste0("p=", sprintf("%.3f", p_raw))
  )
)]

plot_dt <- melt(
  scored[subtype %in% stats$subtype],
  id.vars = c("sample_id", "tumor_region", "subtype"),
  measure.vars = c("Core", "Border"),
  variable.name = "spatial_region",
  value.name = "mean_score"
)
setnames(plot_dt, "subtype", "cancer")
plot_dt[, spatial_region := factor(spatial_region, levels = c("Core", "Border"))]
med_rank <- plot_dt[, .(med = median(mean_score)), by = cancer][order(med), as.character(cancer)]
lab_map <- stats[, setNames(
  paste0(as.character(subtype), "\n", p_lab, "  n=", n_pairs),
  as.character(subtype)
)]
plot_dt[, ylab := factor(lab_map[as.character(cancer)], levels = lab_map[med_rank])]
plot_dt[, y_num := as.numeric(ylab)]
set.seed(1)
plot_dt[, y_dot := y_num + runif(.N, -0.12, 0.12)]
means <- plot_dt[, .(mn = mean(mean_score), y_num = y_num[1]), by = .(ylab, spatial_region)]

p <- ggplot(plot_dt, aes(x = mean_score, y = ylab, fill = spatial_region)) +
  geom_density_ridges(
    alpha = 0.32, scale = 0.9, rel_min_height = 0.01,
    color = "grey40", linewidth = 0.3
  ) +
  geom_segment(
    data = means,
    aes(x = mn, xend = mn, y = y_num, yend = y_num + 0.82, color = spatial_region),
    inherit.aes = FALSE, linetype = "dashed", linewidth = 0.55
  ) +
  geom_point(
    aes(x = mean_score, y = y_dot, color = spatial_region),
    inherit.aes = FALSE, size = 0.35, alpha = 0.15
  ) +
  scale_fill_manual(values = c(Core = "#56B4E9", Border = "#E69F00")) +
  scale_color_manual(values = c(Core = "#0072B2", Border = "#D55E00")) +
  theme_classic(base_size = 11) +
  theme(
    legend.position = "top", legend.title = element_blank(),
    axis.text.y = element_text(size = 7), plot.margin = margin(6, 6, 6, 4)
  ) +
  labs(
    title = "CSEA Invasion by tumor region",
    subtitle = "Tumor cells; region >= 100; Core dist < -50; Border [-50, 0]; raw p",
    x = "CSEA_Invasion",
    y = NULL
  )

out_pdf <- file.path(
  root,
  "PanCancer_CSEA_Invasion_Core_vs_Border_ridge_tumor_region100.pdf"
)
ggsave(out_pdf, p, width = 3.6, height = 5.6)
stats[, rank := match(as.character(subtype), med_rank)]
fwrite(
  scored[order(subtype, sample_id, tumor_region),
         .(sample_id, tumor_region, cancer = subtype, n_tumor, n_core, n_border, Core, Border)],
  file.path(root, "region_values_tumor_region100.tsv"),
  sep = "\t"
)
fwrite(
  stats[order(rank), .(
    cancer = subtype, n_pairs, p_raw, p_lab, mean_delta,
    median_core, median_border, n_border_gt, rank
  )],
  file.path(root, "stats_tumor_region100.tsv"),
  sep = "\t"
)
print(stats[order(rank), .(cancer = subtype, n_pairs, p_raw, mean_delta, n_border_gt)])
message("Wrote ", out_pdf)
