#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
base <- if (length(args) >= 1L) {
  normalizePath(args[[1L]], mustWork = TRUE)
} else {
  "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/01_cell_typing/01_analysis/04_schwann/0_crcV21/boundary_CRC"
}
out <- file.path(base, "enteric_peripheral_analysis")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

index <- read_tsv(file.path(base, "nerve_index.tsv"), show_col_types = FALSE) %>%
  mutate(
    nerve_identity = if_else(n_ens_neuron == 0, "Peripheral", "Enteric")
  )

write_tsv(index, file.path(out, "v21_nerve_index_enteric_peripheral.tsv"))

nerves <- index %>%
  transmute(
    sample_id,
    nerve_id,
    nerve_key,
    nerve_identity,
    region,
    in_tumor,
    n_cells,
    n_glial,
    n_ens_neuron
  )
write_tsv(nerves, file.path(out, "v21_crc_peripheral_vs_ens_nerves.tsv"))

by_sample <- nerves %>%
  group_by(sample_id) %>%
  summarise(
    n_nerves = n(),
    n_enteric = sum(nerve_identity == "Enteric"),
    n_peripheral = sum(nerve_identity == "Peripheral"),
    n_tumoral_enteric = sum(nerve_identity == "Enteric" & in_tumor),
    n_tumoral_peripheral = sum(nerve_identity == "Peripheral" & in_tumor),
    .groups = "drop"
  )
write_tsv(by_sample, file.path(out, "v21_crc_peripheral_vs_ens_by_sample.tsv"))

donut <- function(data, column, title, stem) {
  summary <- data %>%
    count(.data[[column]], name = "n_nerves") %>%
    mutate(
      fraction = n_nerves / sum(n_nerves),
      label = paste0(sprintf("%.0f%%", 100 * fraction), "\n(n=", n_nerves, ")")
    )
  names(summary)[1] <- "group"
  write_tsv(summary, file.path(out, paste0(stem, "_counts.tsv")))
  p <- ggplot(summary, aes(x = 2, y = fraction, fill = group)) +
    geom_col(width = 1, color = "white") +
    geom_text(aes(label = label), position = position_stack(vjust = 0.5), color = "white") +
    coord_polar(theta = "y") +
    xlim(0.5, 2.5) +
    theme_void() +
    labs(title = title, fill = NULL)
  ggsave(file.path(out, paste0(stem, ".pdf")), p, width = 5.2, height = 4.6)
}

donut(
  index,
  "nerve_identity",
  "CRC nerves: enteric vs peripheral",
  "peripheral_vs_enteric_fraction_donut"
)
