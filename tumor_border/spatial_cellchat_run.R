#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(optparse)
  library(Seurat)
  library(SpatialCellChat)
})

opt <- parse_args(OptionParser(option_list = list(
  make_option(c("-i", "--input"), type = "character"),
  make_option(c("-o", "--outdir"), type = "character")
)))
if (is.null(opt$input) || is.null(opt$outdir)) {
  stop("Need --input and --outdir")
}

sample_name <- tools::file_path_sans_ext(basename(opt$input))
out <- file.path(opt$outdir, sample_name)
dir.create(out, recursive = TRUE, showWarnings = FALSE)
setEnvironment(workers = 1, future.globals.maxSize = 100000 * 1024^2, conda_env = NULL)

seu <- readRDS(opt$input)
counts <- GetAssayData(seu, assay = "RNA", layer = "counts")
rownames(counts) <- gsub("_", "-", rownames(counts))
xy <- Embeddings(seu, reduction = "spatial")

options(Seurat.object.assay.version = "v3")
obj <- CreateSeuratObject(counts = counts, meta.data = seu@meta.data)
obj[["spatial"]] <- CreateDimReducObject(embeddings = xy, key = "spatial_", assay = "RNA")
obj <- NormalizeData(obj, verbose = FALSE)

locs <- as.data.frame(Embeddings(obj, reduction = "spatial"))
locs[, 2] <- max(locs[, 2]) - locs[, 2]
tmp <- locs
locs[, 1] <- tmp[, 2]
locs[, 2] <- tmp[, 1]
locs[, 1] <- max(locs[, 1]) - locs[, 1]
colnames(locs) <- c("x", "y")
rownames(locs) <- colnames(obj)

obj$regional_cell_type <- as.factor(gsub(
  "NA", "Unknown",
  gsub("/", "_", as.character(obj$regional_cell_type))
))

chat <- createSpatialCellChat(
  object = obj,
  group.by = "regional_cell_type",
  assay = "RNA",
  datatype = "spatial",
  coordinates = locs,
  spatial.factors = data.frame(ratio = 1, tol = 5)
)
chat@DB <- subsetDB(
  CellChatDB.human,
  search = c("Secreted Signaling", "ECM-Receptor", "Cell-Cell Contact"),
  non_protein = FALSE
)
chat <- subsetData(chat)
chat <- preProcessing(chat)
chat <- identifyOverExpressedGenes(chat, selection.method = "wilcox", do.grid = FALSE)
chat <- identifyOverExpressedInteractions(chat, variable.both = FALSE)
chat <- computeCommunProb(
  chat,
  distance.use = TRUE,
  scale.distance = 6.8,
  contact.dependent = TRUE,
  interaction.range = 200,
  contact.range = 10
)
chat <- filterProbability(chat)
chat <- filterCommunication(chat, min.cells = NULL, min.links = 10, min.cells.sr = 10)
chat <- computeAvgCommunProb(
  chat,
  group.by = "regional_cell_type",
  avg.type = "avg",
  nboot = 50,
  do.permutation = TRUE
)
chat <- filterCommunication(chat, min.cells = 10, min.links = NULL, min.cells.sr = NULL)
chat <- aggregateNet(chat)

write.table(
  subsetCommunication(chat),
  file.path(out, "sig_interaction.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
readr::write_rds(chat, file.path(out, paste0(sample_name, "_Chat_computeCommunProbPathway.rds")))
