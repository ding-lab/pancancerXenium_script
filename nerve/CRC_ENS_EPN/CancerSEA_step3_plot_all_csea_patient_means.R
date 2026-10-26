#!/usr/bin/env Rscript

# All CancerSEA pathways for V21 ENS>5 Near>=20, plus Proliferation violin.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(data.table)
})

SCH <- "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/01_cell_typing/01_analysis/04_schwann"
source(
  file.path(SCH, "nerve_region", "compare_schwann_near50_and_repair_high_low_tumor_scores.R"),
  chdir = FALSE
)

cli_args <- commandArgs(trailingOnly = TRUE)
OUT_DIR <- if (length(cli_args) >= 1L) {
  normalizePath(cli_args[[1L]], mustWork = TRUE)
} else {
  file.path(SCH, "0_crcV21","boundary_CRC", "crc_v21_enteric_ensgt5_min10_csea")
}

ASSIGNMENT <- file.path(OUT_DIR, "tumor_cell_group_assignments.tsv")
CACHE_DIR <- file.path(OUT_DIR, "cache")
VIOLIN_DIR <- file.path(OUT_DIR, "significant_pathway_violin_heatmap", "violin", "by_pathway")
dir.create(CACHE_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(VIOLIN_DIR, recursive = TRUE, showWarnings = FALSE)

GROUP_NEAR <- "Near_Tumoral_Nerve_50um"
GROUP_BOUNDARY <- "Boundary_Tumor"
GROUP_CORE <- "Tumor_Core"
GROUP_ORDER <- c(GROUP_CORE, GROUP_NEAR, GROUP_BOUNDARY)
GROUP_LABELS <- c("Tumor core", "Near tumoral nerve <=50 um", "Boundary tumor")
GROUP_COLORS <- c(
  "Tumor core" = "#4C78A8",
  "Near tumoral nerve <=50 um" = "#D62728",
  "Boundary tumor" = "#7F7F7F"
)

EXISTING_CACHE <- file.path(
  SCH,
  "nerve_region",
  "surrounding_scores",
  "tumoral_near50_vs_other_tumor_by_patient",
  "cache",
  "CRC_all_tumor_cancersea_scores.tsv"
)
cache_dest <- file.path(CACHE_DIR, "CRC_selected_tumor_cancersea_scores.tsv")
if (!file.exists(cache_dest) && file.exists(EXISTING_CACHE)) {
  ok <- file.symlink(EXISTING_CACHE, cache_dest)
  if (!isTRUE(ok)) file.copy(EXISTING_CACHE, cache_dest)
}

header <- names(fread(cache_dest, nrows = 0, showProgress = FALSE))
keep_scores <- header[grepl("^CSEA_", header)]

tumor_groups <- fread(ASSIGNMENT, showProgress = FALSE) %>%
  as_tibble() %>%
  transmute(
    cancer_type = "CRC",
    patient_id = as.character(patient_id),
    sample = as.character(sample),
    comparison_group = as.character(comparison_group),
    barcode = as.character(barcode)
  ) %>%
  filter(!is.na(comparison_group), !is.na(barcode)) %>%
  distinct(barcode, .keep_all = TRUE) %>%
  mutate(comparison_group = factor(comparison_group, levels = GROUP_ORDER)) %>%
  filter(!is.na(comparison_group))

cancersea <- fread(
  cache_dest,
  select = c("barcode", keep_scores),
  showProgress = FALSE
) %>%
  as_tibble()
cancersea$barcode <- as.character(cancersea$barcode)

means <- compute_patient_group_means_from_scores(
  tumor_groups, cancersea, "CSEA_", "CancerSEA"
) %>%
  filter(score_name %in% keep_scores)

stats <- bind_rows(
  compute_paired_stats_by_cancer(
    make_patient_pairs(means, GROUP_NEAR, GROUP_BOUNDARY),
    GROUP_NEAR, GROUP_BOUNDARY
  ) %>%
    mutate(contrast = "Near_vs_Boundary"),
  compute_paired_stats_by_cancer(
    make_patient_pairs(means, GROUP_NEAR, GROUP_CORE),
    GROUP_NEAR, GROUP_CORE
  ) %>%
    mutate(contrast = "Near_vs_Core")
) %>%
  mutate(
    cancer_type = "CRC",
    significant_raw_p0.1 = !is.na(wilcox_p_value) & wilcox_p_value < 0.10,
    significant_raw_p0.05 = !is.na(wilcox_p_value) & wilcox_p_value < 0.05
  )

write_tsv(means, file.path(OUT_DIR, "all_csea_patient_means.tsv"))
write_tsv(stats, file.path(OUT_DIR, "all_csea_paired_stats.tsv"))
write_tsv(
  stats,
  file.path(OUT_DIR, "significant_pathway_violin_heatmap", "selected_pathways_for_plots.tsv")
)

format_p <- function(p) {
  if (length(p) == 0L || is.na(p[[1L]])) return("NA")
  sprintf("%.3g", p[[1L]])
}

plot_one <- function(score) {
  score_df <- means %>%
    filter(score_name == score, comparison_group %in% GROUP_ORDER) %>%
    mutate(
      comparison_group = factor(
        comparison_group,
        levels = GROUP_ORDER,
        labels = GROUP_LABELS
      )
    )
  row_b <- stats %>%
    filter(score_name == score, contrast == "Near_vs_Boundary") %>%
    slice_head(n = 1)
  row_c <- stats %>%
    filter(score_name == score, contrast == "Near_vs_Core") %>%
    slice_head(n = 1)
  label <- unique(score_df$score_label)
  if (length(label) == 0L || is.na(label[[1L]])) {
    label <- sub("^CSEA_", "", score)
  } else {
    label <- label[[1L]]
  }
  p <- ggplot(
    score_df,
    aes(x = comparison_group, y = mean_score, group = patient_id)
  ) +
    geom_violin(
      aes(group = comparison_group, fill = comparison_group),
      alpha = 0.28, color = NA, width = 0.8, trim = FALSE
    ) +
    geom_boxplot(
      aes(group = comparison_group),
      width = 0.18, outlier.shape = NA, alpha = 0.7,
      fill = "white", color = "grey25", linewidth = 0.45
    ) +
    geom_line(alpha = 0.5, linewidth = 0.55, color = "grey30") +
    geom_point(aes(color = comparison_group), size = 2.8, alpha = 0.95) +
    scale_fill_manual(values = GROUP_COLORS) +
    scale_color_manual(values = GROUP_COLORS) +
    theme_classic(base_size = 13) +
    labs(
      title = paste0("CRC: ", label),
      subtitle = paste0(
        "V21 ENS>5, Near>=20; Near vs Boundary p=",
        format_p(row_b$wilcox_p_value),
        " | Near vs Core p=",
        format_p(row_c$wilcox_p_value),
        " | n = ",
        if (nrow(row_b)) row_b$n_patients[[1L]] else NA
      ),
      x = NULL,
      y = "Patient mean pathway score",
      color = NULL,
      fill = NULL
    ) +
    theme(
      legend.position = "bottom",
      plot.title = element_text(face = "bold"),
      axis.text.x = element_text(angle = 20, hjust = 1)
    )
  stem <- file.path(VIOLIN_DIR, paste0("CRC__", score))
  ggsave(paste0(stem, ".pdf"), p, width = 7.2, height = 4.8)
  ggsave(paste0(stem, ".png"), p, width = 7.2, height = 4.8, dpi = 300)
}

for (score in keep_scores) {
  plot_one(score)
}

message("=== ALL CancerSEA Near vs Boundary ===")
print(
  as.data.frame(
    stats %>%
      filter(contrast == "Near_vs_Boundary") %>%
      select(score_name, n_patients, n_patients_higher_near, wilcox_p_value, significant_raw_p0.05) %>%
      arrange(wilcox_p_value)
  )
)
message("DONE ", OUT_DIR)
