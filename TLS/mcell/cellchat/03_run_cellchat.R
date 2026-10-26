suppressPackageStartupMessages({
  library(optparse)
  library(Seurat)
  library(CellChat)
  library(readr)
})

option_list <- list(
  make_option(c("-i", "--input"), type = "character", default = NULL),
  make_option(c("-o", "--outdir"), type = "character", default = NULL),
  make_option(
    "--min-cells",
    dest = "min_cells",
    type = "integer",
    default = 5L
  )
)
opt <- parse_args(OptionParser(option_list = option_list))
if (is.null(opt$input) || is.null(opt$outdir)) {
  stop("--input and --outdir are required")
}

sample_name <- tools::file_path_sans_ext(basename(opt$input))
sample_outdir <- file.path(opt$outdir, sample_name)
dir.create(sample_outdir, recursive = TRUE, showWarnings = FALSE)

message("Reading: ", opt$input)
seu <- readRDS(opt$input)
counts_mat <- GetAssayData(seu, assay = "RNA", layer = "counts")
rownames(counts_mat) <- gsub("_", "-", rownames(counts_mat))
meta_data <- seu@meta.data
spatial_coords <- Embeddings(seu, reduction = "spatial")

options(Seurat.object.assay.version = "v3")
seu_clean <- CreateSeuratObject(counts = counts_mat, meta.data = meta_data)
seu_clean[["spatial"]] <- CreateDimReducObject(
  embeddings = spatial_coords,
  key = "spatial_",
  assay = "RNA"
)
seu_clean <- NormalizeData(seu_clean, verbose = FALSE)

if (!"cell_type_mcell" %in% colnames(seu_clean@meta.data)) {
  stop("Missing cell_type_mcell metadata; rerun h5ad export and conversion")
}
seu_clean$cell_type_mcell <- as.character(seu_clean$cell_type_mcell)
seu_clean$cell_type_mcell <- gsub("/", "_", seu_clean$cell_type_mcell)
seu_clean$cell_type_mcell[
  is.na(seu_clean$cell_type_mcell) | seu_clean$cell_type_mcell == ""
] <- "Unknown"
seu_clean$cell_type_mcell <- factor(seu_clean$cell_type_mcell)

group_counts <- table(seu_clean$cell_type_mcell)
keep_groups <- names(group_counts[group_counts >= opt$min_cells])
seu_clean <- subset(
  seu_clean,
  cells = colnames(seu_clean)[
    seu_clean$cell_type_mcell %in% keep_groups
  ]
)
seu_clean$cell_type_mcell <- droplevels(
  factor(as.character(seu_clean$cell_type_mcell))
)
seu_clean$samples <- factor(rep(sample_name, ncol(seu_clean)))
if (length(unique(seu_clean$cell_type_mcell)) < 2) {
  stop("Fewer than two cell groups remain after min-cells filtering")
}

# This intentionally matches the nerve-region analysis: CellChat uses the
# sample-specific TLS niche subset, but communication probabilities are not
# additionally distance-weighted.
chat <- createCellChat(
  object = seu_clean,
  group.by = "cell_type_mcell",
  assay = "RNA",
  datatype = "RNA"
)
database <- subsetDB(
  CellChatDB.human,
  search = c(
    "Secreted Signaling",
    "ECM-Receptor",
    "Cell-Cell Contact"
  ),
  non_protein = FALSE
)
chat@DB <- database
chat <- subsetData(chat)
chat <- identifyOverExpressedGenes(chat)
chat <- identifyOverExpressedInteractions(chat)
chat <- computeCommunProb(
  chat,
  type = "truncatedMean",
  trim = 0,
  distance.use = FALSE,
  contact.dependent = FALSE
)

interactions <- subsetCommunication(chat)
write.table(
  interactions,
  file.path(sample_outdir, "sig_interaction.tsv"),
  sep = "\t",
  col.names = TRUE,
  row.names = FALSE,
  quote = FALSE
)
write_rds(
  chat,
  file.path(
    sample_outdir,
    paste0(sample_name, "_chat_computeCommunProb.rds")
  )
)

chat <- filterCommunication(chat, min.cells = 5)
chat <- computeCommunProbPathway(chat)
chat <- aggregateNet(chat)
chat <- netAnalysis_computeCentrality(chat, slot.name = "net")
write_rds(
  chat,
  file.path(
    sample_outdir,
    paste0(sample_name, "_Chat_final.rds")
  )
)

pathway_table <- subsetCommunication(chat, slot.name = "netP")
write.table(
  pathway_table,
  file.path(sample_outdir, "significant_pathways.tsv"),
  sep = "\t",
  col.names = TRUE,
  row.names = FALSE,
  quote = FALSE
)
write.table(
  data.frame(cell_type = names(group_counts), n_cells = as.integer(group_counts)),
  file.path(sample_outdir, "cell_type_counts.tsv"),
  sep = "\t",
  col.names = TRUE,
  row.names = FALSE,
  quote = FALSE
)
message("Completed: ", sample_name)
