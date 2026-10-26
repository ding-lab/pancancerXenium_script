#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(ggplot2)
})

base <- "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/01_cell_typing/01_analysis/04_schwann/0_crcV21/boundary_CRC"
tables <- file.path(base, "enteric_peripheral_analysis", "tables")
plots <- file.path(base, "enteric_peripheral_analysis", "plots")
dir.create(plots, recursive = TRUE, showWarnings = FALSE)

cancer_types <- c("CRC", "PDAC", "PRAD")
group_levels <- c("tumoral", "normal")
group_labels <- c(normal = "Normal nerve", tumoral = "Tumoral nerve")
group_colors <- c(normal = "#1F77B4", tumoral = "#D62728")

safe_wilcox <- function(x, y) {
  x <- x[is.finite(x)]
  y <- y[is.finite(y)]
  if (length(x) < 2L || length(y) < 2L) return(NA_real_)
  wilcox.test(x, y, paired = FALSE, exact = FALSE)$p.value
}

repair <- read_tsv(
  file.path(tables, "v21_crc_pdac_prad_repair_scores.tsv"),
  show_col_types = FALSE
) %>%
  filter(is.finite(Repair_SC)) %>%
  mutate(
    cancer_type = factor(cancer_type, levels = cancer_types),
    nerve_group = factor(nerve_group, levels = group_levels)
  )

stats <- repair %>%
  group_by(cancer_type) %>%
  summarise(
    n_normal = sum(nerve_group == "normal"),
    n_tumoral = sum(nerve_group == "tumoral"),
    median_normal = median(Repair_SC[nerve_group == "normal"], na.rm = TRUE),
    median_tumoral = median(Repair_SC[nerve_group == "tumoral"], na.rm = TRUE),
    wilcox_p_raw = safe_wilcox(
      Repair_SC[nerve_group == "tumoral"],
      Repair_SC[nerve_group == "normal"]
    ),
    y = max(Repair_SC, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    p_label = paste0(
      "raw p=",
      format.pval(wilcox_p_raw, digits = 3, eps = 1e-4)
    )
  )
write_tsv(
  stats %>% select(-y, -p_label),
  file.path(plots, "glial_repair_tumoral_vs_normal_stats.tsv")
)

group_stats <- repair %>%
  group_by(cancer_type, nerve_group) %>%
  summarise(
    n_nerves = n(),
    y = max(Repair_SC, na.rm = TRUE),
    .groups = "drop"
  )

p_repair <- ggplot(
  repair,
  aes(x = nerve_group, y = Repair_SC, fill = nerve_group)
) +
  geom_violin(
    trim = TRUE,
    width = 0.78,
    alpha = 0.55,
    scale = "width",
    linewidth = 0.35
  ) +
  geom_boxplot(
    width = 0.32,
    outlier.shape = NA,
    coef = 1.5,
    alpha = 0.9,
    linewidth = 0.35,
    show.legend = FALSE
  ) +
  geom_point(
    shape = 21,
    color = "grey30",
    size = 1.6,
    alpha = 0.75,
    position = position_jitter(width = 0.06, height = 0, seed = 1)
  ) +
  geom_text(
    data = group_stats,
    aes(x = nerve_group, y = y, label = paste0("n=", n_nerves)),
    inherit.aes = FALSE,
    vjust = -0.55,
    size = 2.6,
    color = "grey20"
  ) +
  geom_text(
    data = stats,
    aes(x = 1.5, y = y, label = p_label),
    inherit.aes = FALSE,
    vjust = -2.1,
    size = 3,
    color = "grey10"
  ) +
  facet_wrap(~cancer_type, nrow = 1) +
  scale_x_discrete(labels = group_labels) +
  scale_fill_manual(values = group_colors, labels = group_labels) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.26))) +
  theme_bw(base_size = 12) +
  labs(
    x = NULL,
    y = "Per-nerve mean Repair_SC",
    fill = "Nerve group",
    title = "Repair score: tumoral vs normal nerves",
    subtitle = paste(
      "Each point = one nerve.",
      "CRC: mean Glial Repair_SC, >=5 Glial, boundary tumoral label.",
      "PDAC/PRAD: mean Schwann Repair_SC, >=5 Schwann.",
      "PDAC tumoral: Yuwei Tumor halo50 >= 5.",
      "PRAD tumoral: gen10 tumor-boundary overlap."
    )
  ) +
  theme(
    axis.text.x = element_text(angle = 20, hjust = 1),
    strip.text = element_text(face = "bold"),
    legend.position = "bottom",
    panel.spacing.x = unit(1.2, "lines")
  )
ggsave(
  file.path(plots, "glial_repair_tumoral_vs_normal.pdf"),
  p_repair, width = 8.4, height = 6.2
)
ggsave(
  file.path(plots, "glial_repair_tumoral_vs_normal.png"),
  p_repair, width = 8.4, height = 6.2, dpi = 300
)
