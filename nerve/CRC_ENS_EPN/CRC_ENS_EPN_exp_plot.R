#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
# Reads <base>/analysis_tables/nerve_glial_plus_ens_expression.tsv
# written by prepare_nerve_glial_plus_ens_expression.py.
base <- if (length(args) >= 1L) {
  normalizePath(args[[1L]], mustWork = TRUE)
} else {
  "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/01_cell_typing/01_analysis/04_schwann/0_crcV21/boundary_CRC"
}
out <- file.path(base, "enteric_peripheral_analysis", "plots")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

genes <- c("SYP", "SNAP25", "STX1A", "VIP", "NRG1", "FGF1")
markers <- c("PHOX2B", "RET")
min_cells <- 10L

wilcox_p <- function(x, y) {
  x <- x[is.finite(x)]
  y <- y[is.finite(y)]
  if (length(x) < 2L || length(y) < 2L) return(NA_real_)
  wilcox.test(x, y, paired = FALSE, exact = FALSE)$p.value
}

index <- read_tsv(file.path(base, "nerve_index.tsv"), show_col_types = FALSE) %>%
  transmute(
    nerve_key,
    nerve_identity = if_else(n_ens_neuron == 0, "Peripheral", "Enteric"),
    region,
    in_tumor
  )

expr <- read_tsv(
  file.path(base, "analysis_tables", "nerve_glial_plus_ens_expression.tsv"),
  show_col_types = FALSE
) %>%
  inner_join(index, by = "nerve_key")

compare_groups <- function(data, gene, group_col, a, b) {
  xa <- data[[gene]][data[[group_col]] == a]
  xb <- data[[gene]][data[[group_col]] == b]
  tibble(
    gene = gene,
    group_a = a,
    group_b = b,
    n_a = sum(is.finite(xa)),
    n_b = sum(is.finite(xb)),
    median_a = median(xa, na.rm = TRUE),
    median_b = median(xb, na.rm = TRUE),
    wilcox_p = wilcox_p(xa, xb)
  )
}

identity_long <- expr %>%
  select(nerve_key, nerve_identity, all_of(c(markers, genes))) %>%
  pivot_longer(all_of(c(markers, genes)), names_to = "gene", values_to = "expression")

identity_stats <- bind_rows(lapply(c(markers, genes), function(gene) {
  compare_groups(expr, gene, "nerve_identity", "Peripheral", "Enteric")
}))
write_tsv(identity_stats, file.path(out, "peripheral_vs_enteric_expression_stats.tsv"))
write_tsv(
  identity_stats %>% filter(gene %in% markers),
  file.path(out, "peripheral_vs_enteric_PHOX2B_RET_stats.tsv")
)

enteric <- expr %>% filter(nerve_identity == "Enteric")
region_stats <- bind_rows(lapply(genes, function(gene) {
  compare_groups(enteric, gene, "region", "Tumoral", "Normal")
}))
write_tsv(region_stats, file.path(out, "enteric_tumoral_vs_normal_expression_stats.tsv"))

corr_nerves <- enteric %>% filter(n_glial + n_ens_neuron >= min_cells)
pairs <- bind_rows(lapply(c("SYP", "SNAP25", "STX1A"), function(x_gene) {
  bind_rows(lapply(c("VIP", "NRG1", "FGF1"), function(y_gene) {
    x <- corr_nerves[[x_gene]]
    y <- corr_nerves[[y_gene]]
    ok <- is.finite(x) & is.finite(y)
    test <- cor.test(x[ok], y[ok], method = "pearson")
    tibble(
      x_gene = x_gene,
      y_gene = y_gene,
      n_nerves = sum(ok),
      pearson_r = unname(test$estimate),
      pearson_p = test$p.value
    )
  }))
}))
write_tsv(pairs, file.path(out, "enteric_synaptic_trophic_correlations_min10.tsv"))

plot_long <- identity_long %>%
  filter(gene %in% c(markers, genes)) %>%
  mutate(
    nerve_identity = factor(nerve_identity, levels = c("Peripheral", "Enteric")),
    gene = factor(gene, levels = c(markers, genes))
  )
p <- ggplot(plot_long, aes(nerve_identity, expression, fill = nerve_identity)) +
  geom_violin(scale = "width", color = NA) +
  geom_boxplot(width = 0.15, outlier.shape = NA, fill = "white") +
  facet_wrap(~gene, scales = "free_y", nrow = 2) +
  theme_classic() +
  theme(legend.position = "none") +
  labs(x = NULL, y = "Per-nerve mean, log1p CP10K")
ggsave(
  file.path(out, "peripheral_vs_enteric_expression.pdf"),
  p,
  width = 9.5,
  height = 6.2
)
