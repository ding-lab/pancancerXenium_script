#!/usr/bin/env Rscript


suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
})

BASE <- if (length(commandArgs(trailingOnly = TRUE)) >= 1L) {
  normalizePath(commandArgs(trailingOnly = TRUE)[[1L]], mustWork = TRUE)
} else {
  "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/01_cell_typing/01_analysis/04_schwann/0_crcV21/boundary_CRC"
}

PLOTS <- file.path(BASE, "enteric_peripheral_analysis", "plots")
EXPR <- file.path(BASE, "analysis_tables", "nerve_glial_plus_ens_expression.tsv")
dir.create(PLOTS, recursive = TRUE, showWarnings = FALSE)

SYNAPTIC <- c("SYP", "SNAP25", "STX1A")
TROPHIC <- c("VIP", "NRG1", "FGF1")

pooled <- read_tsv(EXPR, show_col_types = FALSE) %>%
  filter(n_ens_neuron > 0, n_glial + n_ens_neuron >= 10)

make_pair_table <- function(data) {
  bind_rows(lapply(SYNAPTIC, function(x_gene) {
    bind_rows(lapply(TROPHIC, function(y_gene) {
      tibble(
        nerve_key = data$nerve_key,
        x_gene = x_gene,
        y_gene = y_gene,
        pair = paste(x_gene, "vs", y_gene),
        x_expression = data[[.env$x_gene]],
        y_expression = data[[.env$y_gene]]
      )
    }))
  }))
}

pairs <- make_pair_table(pooled) %>%
  mutate(
    pair = factor(
      pair,
      levels = as.vector(outer(SYNAPTIC, TROPHIC, paste, sep = " vs "))
    )
  )

pair_stats <- pairs %>%
  group_by(x_gene, y_gene, pair) %>%
  summarise(
    n_nerves = n(),
    pearson_r = cor(x_expression, y_expression),
    pearson_p_raw = cor.test(
      x_expression, y_expression, method = "pearson"
    )$p.value,
    spearman_rho = cor(x_expression, y_expression, method = "spearman"),
    spearman_p_raw = cor.test(
      x_expression, y_expression, method = "spearman", exact = FALSE
    )$p.value,
    .groups = "drop"
  )

write_tsv(
  pair_stats,
  file.path(PLOTS, "enteric_synaptic_trophic_correlations_min10.tsv")
)

pair_labels <- pair_stats %>%
  mutate(
    label = paste0(
      "n=", n_nerves,
      "\nrho=", sprintf("%.2f", spearman_rho),
      "\nraw p=",
      format.pval(spearman_p_raw, digits = 2, eps = 1e-4)
    )
  ) %>%
  left_join(
    pairs %>%
      group_by(pair) %>%
      summarise(
        x = min(x_expression),
        y = max(y_expression),
        .groups = "drop"
      ),
    by = "pair"
  )

p_pairs <- ggplot(pairs, aes(x = x_expression, y = y_expression)) +
  geom_point(size = 1.5, alpha = 0.58, color = "#3C5488") +
  geom_smooth(
    method = "lm",
    formula = y ~ x,
    se = TRUE,
    color = "grey20",
    fill = "grey75",
    linewidth = 0.7
  ) +
  geom_text(
    data = pair_labels,
    aes(x = x, y = y, label = label),
    inherit.aes = FALSE,
    hjust = 0,
    vjust = 1,
    size = 2.6
  ) +
  facet_wrap(~pair, nrow = 3, scales = "free") +
  scale_x_continuous(expand = expansion(mult = c(0.05, 0.1))) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.16))) +
  theme_bw(base_size = 10) +
  labs(
    title = "CRC v21 enteric nerves: synaptic vs trophic expression",
    subtitle = "n_glial + n_ens_neuron >= 10; Glial + ENS_neuron pooled per nerve; linear fit with 95% CI",
    x = "Mean synaptic-gene log1p CP10K",
    y = "Mean trophic-gene log1p CP10K"
  ) +
  theme(strip.text = element_text(face = "bold"))

ggsave(
  file.path(PLOTS, "enteric_synaptic_trophic_correlations_min10.pdf"),
  p_pairs, width = 10, height = 9
)
ggsave(
  file.path(PLOTS, "enteric_synaptic_trophic_correlations_min10.png"),
  p_pairs, width = 10, height = 9, dpi = 300
)

message(
  "Kept ", nrow(pooled), " enteric nerves with Glial+ENS >= 10; wrote *_min10"
)
print(as.data.frame(pair_stats %>% select(pair, n_nerves, spearman_rho, spearman_p_raw)))
