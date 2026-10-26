#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[[1]] else file.path(
  "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k",
  "01_cell_typing/01_analysis/03_TLS/new_mcell"
)

bin_order <- c(
  "0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
  "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"
)
inputs <- c(
  "Mcell-tumor" = file.path(
    root,
    "Spatial_Gradient_McellAnchor_percentile",
    "LUNG_Global_CellPooled_Average.csv"
  ),
  "otherTumor" = file.path(
    root,
    "Spatial_Gradient_OtherTumorAnchor_percentile",
    "LUNG_Global_CellPooled_Average.csv"
  )
)

population_zscore <- function(values) {
  (values - mean(values)) / sqrt(mean((values - mean(values))^2))
}

plot_data <- bind_rows(lapply(names(inputs), function(group_name) {
  table <- read.csv(inputs[[group_name]], check.names = FALSE)
  table <- table[match(bin_order, table$Norm_Bin), ]
  tibble(
    anchor = group_name,
    bin = factor(table$Norm_Bin, levels = bin_order),
    bin_index = seq_along(bin_order),
    ccr6_z = population_zscore(table$CCR6)
  )
})) %>%
  mutate(anchor = factor(anchor, levels = names(inputs)))

colors <- c(
  "Mcell-tumor" = "#C61E89",
  "otherTumor" = "#2C7FBA"
)

p <- ggplot(
  plot_data,
  aes(x = bin_index, y = ccr6_z, color = anchor)
) +
  geom_hline(
    yintercept = 0,
    color = "grey70",
    linewidth = 0.35,
    linetype = "dotted"
  ) +
  geom_point(size = 2, alpha = 0.65) +
  geom_smooth(
    method = "loess",
    formula = y ~ x,
    span = 0.8,
    se = FALSE,
    linewidth = 1.1
  ) +
  scale_color_manual(values = colors) +
  scale_x_continuous(
    breaks = seq_along(bin_order),
    labels = bin_order,
    expand = expansion(mult = c(0.03, 0.03))
  ) +
  labs(
    title = "CCR6 gradient by tumor anchor",
    x = "TLS B-cell distance percentile from anchor",
    y = "CCR6 bin-level Z-score",
    color = NULL
  ) +
  theme_classic(base_size = 8) +
  theme(
    legend.position = "top",
    legend.text = element_text(size = 7),
    axis.text.x = element_text(
      angle = 60,
      hjust = 1,
      vjust = 1,
      size = 5.5
    ),
    axis.text.y = element_text(size = 7),
    axis.title = element_text(size = 7.5),
    plot.title = element_text(size = 9, face = "bold", hjust = 0.5)
  )

ggsave(
  file.path(root, "CCR6_Mcell_vs_otherTumor_anchor_gradient_R.pdf"),
  p,
  width = 2.5,
  height = 4.0,
  device = cairo_pdf
)
ggsave(
  file.path(root, "CCR6_Mcell_vs_otherTumor_anchor_gradient_R.png"),
  p,
  width = 2.5,
  height = 4.0,
  dpi = 300
)
