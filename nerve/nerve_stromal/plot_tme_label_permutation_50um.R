#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(readr)
})


INPUT_DIR <- paste0(
  "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/",
  "01_cell_typing/01_analysis/04_schwann/nerve_region_withCHOL/",
  "schwann_neighborhood_tme_label_permutation_50um"
)
OUT_DIR <- file.path(INPUT_DIR, "heatmaps")

CANCER_TYPES <- c("CRC", "PDAC", "PRAD")
CANCER_COLORS <- c(CRC = "#D45740", PDAC = "#6CB9D3", PRAD = "#479E88")
CANCER_SPECIFIC_TME_TYPES <- c(
  "Acinar", "Basal", "Club", "Duct_like", "Dysplasia",
  "Intestinal_epithelium", "IPMN", "Islet", "Luminal",
  "Neuroendocrine", "SMC", "Smooth_muscle", "Squamous_epithelium"
)
TME_TYPE_ALIASES <- c(CAF = "Fibroblast")

harmonize_tme_type <- function(metric_name) {
  metric_name <- as.character(metric_name)
  alias_hits <- TME_TYPE_ALIASES[metric_name]
  metric_name[!is.na(alias_hits)] <- unname(alias_hits[!is.na(alias_hits)])
  metric_name
}

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

significance_star <- function(padj, p_value) {
  if (is.finite(padj)) {
    if (padj < 0.001) return("***")
    if (padj < 0.01) return("**")
    if (padj < 0.05) return("*")
    return("")
  }
  if (is.finite(p_value)) {
    if (p_value < 0.001) return("***")
    if (p_value < 0.01) return("**")
    if (p_value < 0.05) return("*")
  }
  ""
}

unit_path <- file.path(INPUT_DIR, "analysis_unit_observed_effects.tsv")
cancer_path <- file.path(INPUT_DIR, "cancer_empirical_statistics.tsv")
for (path in c(unit_path, cancer_path)) {
  if (!file.exists(path)) {
    stop(
      "Missing: ", path,
      "\nRun analyze_schwann_neighborhood_tme_label_permutation_50um.py first."
    )
  }
}

unit_observed <- read_tsv(unit_path, show_col_types = FALSE)
cancer_stats <- read_tsv(cancer_path, show_col_types = FALSE)

common_cell_types <- unit_observed %>%
  filter(metric_type == "conditional_tme_composition") %>%
  mutate(harmonized_name = harmonize_tme_type(metric_name)) %>%
  group_by(cancer_type, harmonized_name) %>%
  summarise(total_observed = sum(observed_value, na.rm = TRUE), .groups = "drop") %>%
  filter(total_observed > 0) %>%
  count(harmonized_name, name = "n_cancers") %>%
  filter(n_cancers == length(CANCER_TYPES)) %>%
  pull(harmonized_name) %>%
  setdiff(CANCER_SPECIFIC_TME_TYPES) %>%
  sort()

if (length(common_cell_types) == 0L) {
  stop("No common TME cell types found across all cancer types.")
}

plot_data <- cancer_stats %>%
  filter(
    metric_type == "conditional_tme_composition",
    cancer_type %in% CANCER_TYPES
  ) %>%
  mutate(harmonized_name = harmonize_tme_type(metric_name)) %>%
  filter(harmonized_name %in% common_cell_types) %>%
  mutate(
    cancer_type = factor(cancer_type, levels = CANCER_TYPES),
    metric_name = harmonized_name,
    star_label = mapply(
      significance_star,
      empirical_padj_two_sided,
      empirical_p_two_sided
    )
  )

row_order <- plot_data %>%
  group_by(metric_name) %>%
  summarise(max_abs = max(abs(observed_minus_null), na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(max_abs), metric_name) %>%
  pull(metric_name)

plot_data <- plot_data %>%
  mutate(metric_name = factor(metric_name, levels = rev(row_order)))

write_tsv(
  plot_data,
  file.path(OUT_DIR, "common_tme_by_cancer_heatmap_data.tsv")
)

max_abs <- max(abs(plot_data$observed_minus_null), na.rm = TRUE)
if (!is.finite(max_abs) || max_abs == 0) {
  max_abs <- 1
}

p <- ggplot(
  plot_data,
  aes(
    x = cancer_type,
    y = metric_name,
    fill = observed_minus_null
  )
) +
  geom_tile(color = "white", linewidth = 0.8) +
  geom_text(aes(label = star_label), size = 4, fontface = "bold") +
  scale_fill_gradient2(
    low = "#2166AC",
    mid = "white",
    high = "#B2182B",
    midpoint = 0,
    limits = c(-max_abs, max_abs),
    name = "Observed − null"
  ) +
  scale_x_discrete(position = "top") +
  labs(
    title = "Common TME composition near tumoral Schwann clusters",
    subtitle = paste0(
      "50 um neighborhood; CRC v8 / PDAC+PRAD halo50; ",
      "label permutation (anchors + coordinates fixed); ",
      "* padj < 0.05, ** < 0.01, *** < 0.001"
    ),
    x = "Cancer type",
    y = "Conditional TME cell type"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold"),
    panel.grid = element_blank(),
    axis.text.x = element_text(
      face = "bold",
      color = CANCER_COLORS[levels(plot_data$cancer_type)]
    ),
    axis.text.y = element_text(face = "bold", size = 9)
  )

plot_height <- max(5.5, 1.5 + 0.42 * length(common_cell_types))
ggsave(
  file.path(OUT_DIR, "common_tme_by_cancer_effects.pdf"),
  p,
  width = 5.2,
  height = plot_height
)
ggsave(
  file.path(OUT_DIR, "common_tme_by_cancer_effects.png"),
  p,
  width = 5.2,
  height = plot_height,
  dpi = 300
)

message("Wrote: ", file.path(OUT_DIR, "common_tme_by_cancer_effects.pdf"))
