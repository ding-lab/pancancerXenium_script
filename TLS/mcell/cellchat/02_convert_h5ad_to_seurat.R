suppressPackageStartupMessages({
  library(Seurat)
  library(anndataR)
  library(Matrix)
  library(optparse)
})

full_args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", full_args, value = TRUE)
if (length(file_arg) > 0) {
  script_path <- sub("^--file=", "", file_arg[[1]])
  script_dir <- dirname(normalizePath(script_path, mustWork = TRUE))
} else {
  script_dir <- normalizePath(
    "lung/tumoral_TLS_cellchat",
    mustWork = TRUE
  )
}

option_list <- list(
  make_option(
    "--input-dir",
    dest = "input_dir",
    type = "character",
    default = file.path(script_dir, "h5ad_by_sample")
  ),
  make_option(
    "--out-dir",
    dest = "out_dir",
    type = "character",
    default = file.path(script_dir, "seurat_object")
  ),
  make_option("--overwrite", action = "store_true", default = FALSE)
)
opt <- parse_args(OptionParser(option_list = option_list))
dir.create(opt$out_dir, recursive = TRUE, showWarnings = FALSE)

# Merge only closely related/rare populations. AT1/AT2 remain explicitly
# nonmalignant; unlike the old nerve script, AT2 is never mapped to Tumor.
celltype_map <- c(
  "AT1" = "NonMalignant_Parenchymal",
  "AT2" = "NonMalignant_Parenchymal",
  "Ciliated/Club/Basal" = "NonMalignant_Parenchymal",
  "cDC1" = "DC",
  "cDC2" = "DC",
  "mregDC" = "DC",
  "pDC" = "DC",
  "Pericyte/vSMC" = "Pericyte_vSMC",
  "Lymphatic_Endothelial" = "Lymphatic_Endothelial",
  "B_cell" = "B_cell",
  "NK_T" = "NK_T",
  "Tumor" = "Tumor",
  "Endothelial" = "Endothelial",
  "Fibroblast" = "Fibroblast",
  "Macrophage" = "Macrophage",
  "Mast" = "Mast",
  "Neutrophil" = "Neutrophil",
  "Plasma" = "Plasma",
  "SMC" = "SMC",
  "Unknown" = "Unknown"
)

h5ad_paths <- list.files(
  opt$input_dir,
  pattern = "\\.h5ad$",
  full.names = TRUE
)
if (length(h5ad_paths) == 0) {
  stop("No h5ad files found in: ", opt$input_dir)
}

manifest <- list()
for (h5ad_path in sort(h5ad_paths)) {
  sample_id <- tools::file_path_sans_ext(basename(h5ad_path))
  out_rds <- file.path(opt$out_dir, paste0(sample_id, ".rds"))
  status <- "written"

  if (file.exists(out_rds) && !opt$overwrite) {
    message("Exists, skip: ", out_rds)
    status <- "skipped_exists"
  } else {
    message("Reading: ", h5ad_path)
    seu <- read_h5ad(h5ad_path, as = "Seurat")
    if (!"cell_type" %in% colnames(seu@meta.data)) {
      stop("Missing cell_type in ", h5ad_path)
    }

    original <- as.character(seu$cell_type)
    refined <- original
    matched <- refined %in% names(celltype_map)
    refined[matched] <- unname(celltype_map[refined[matched]])
    refined[is.na(refined) | refined == ""] <- "Unknown"
    seu$cell_type_refine <- factor(refined)

    if ("spatial" %in% Reductions(seu)) {
      coords <- Embeddings(seu, reduction = "spatial")
      seu$spatial_x <- coords[, 1]
      seu$spatial_y <- coords[, 2]
    } else {
      stop("Missing spatial reduction in ", h5ad_path)
    }

    saveRDS(seu, out_rds)
    message("Saved: ", out_rds)
  }

  manifest[[length(manifest) + 1]] <- data.frame(
    sample = sample_id,
    input_h5ad = h5ad_path,
    output_rds = out_rds,
    status = status
  )
}

write.table(
  do.call(rbind, manifest),
  file.path(opt$out_dir, "conversion_manifest.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
