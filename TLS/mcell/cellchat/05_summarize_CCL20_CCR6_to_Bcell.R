#!/usr/bin/env Rscript

# Summarize CCL20-CCR6 communication from Mcell-tumor or otherTumor to B_cell
# using per-sample CellChat sig_interaction.tsv files.

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(tidyr)
})

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) {
  normalizePath(args[[1]], mustWork = TRUE)
} else {
  normalizePath(
    file.path(
      dirname(normalizePath(
        sub(
          "^--file=",
          "",
          grep(
            "^--file=",
            commandArgs(trailingOnly = FALSE),
            value = TRUE
          )[[1]]
        ),
        mustWork = TRUE
      )),
      "cellchat_output"
    ),
    mustWork = TRUE
  )
}

out_dir <- file.path(
  root,
  "CCL20_CCR6_Mcell_otherTumor_to_Bcell_summary"
)
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

senders <- c("Mcell-tumor", "otherTumor")
min_cells <- if (length(args) >= 2) as.integer(args[[2]]) else 5L
if (is.na(min_cells) || min_cells < 1) {
  stop("Minimum cell count must be a positive integer")
}
interaction_paths <- list.files(
  root,
  pattern = "^sig_interaction\\.tsv$",
  recursive = TRUE,
  full.names = TRUE
)

if (length(interaction_paths) == 0) {
  stop("No sig_interaction.tsv files found under: ", root)
}

read_one_sample <- function(interaction_path) {
  sample_id <- basename(dirname(interaction_path))
  counts_path <- file.path(dirname(interaction_path), "cell_type_counts.tsv")
  if (!file.exists(counts_path)) {
    warning("Missing cell_type_counts.tsv for ", sample_id)
    counts <- tibble(cell_type = character(), n_cells = integer())
  } else {
    counts <- read_tsv(counts_path, show_col_types = FALSE)
  }

  interactions <- read_tsv(
    interaction_path,
    show_col_types = FALSE,
    progress = FALSE
  )

  bind_rows(lapply(senders, function(sender_name) {
    n_sender <- counts$n_cells[match(sender_name, counts$cell_type)]
    n_b_cell <- counts$n_cells[match("B_cell", counts$cell_type)]
    n_sender <- ifelse(is.na(n_sender), 0L, as.integer(n_sender))
    n_b_cell <- ifelse(is.na(n_b_cell), 0L, as.integer(n_b_cell))
    eligible <- n_sender >= min_cells && n_b_cell >= min_cells

    hit <- interactions %>%
      filter(
        source == sender_name,
        target == "B_cell",
        ligand == "CCL20",
        receptor == "CCR6"
      ) %>%
      arrange(pval, desc(prob))

    detected <- eligible && nrow(hit) > 0
    probability <- if (!eligible) {
      NA_real_
    } else if (nrow(hit) == 0) {
      0
    } else {
      as.numeric(hit$prob[[1]])
    }
    p_value <- if (nrow(hit) == 0) NA_real_ else as.numeric(hit$pval[[1]])

    tibble(
      sample = sample_id,
      sender = sender_name,
      receiver = "B_cell",
      ligand = "CCL20",
      receptor = "CCR6",
      n_sender = n_sender,
      n_receiver = n_b_cell,
      eligible = eligible,
      detected_in_sig_interaction = detected,
      probability = probability,
      cellchat_p_value = p_value
    )
  }))
}

summary_long <- bind_rows(lapply(interaction_paths, read_one_sample)) %>%
  mutate(
    sender = factor(sender, levels = senders),
    log10_probability = ifelse(
      eligible,
      log10(probability + 1e-6),
      NA_real_
    )
  ) %>%
  arrange(sample, sender)

write_tsv(
  summary_long,
  file.path(out_dir, "CCL20_CCR6_sample_summary.tsv")
)

paired <- summary_long %>%
  select(sample, sender, probability, detected_in_sig_interaction) %>%
  pivot_wider(
    names_from = sender,
    values_from = c(probability, detected_in_sig_interaction)
  ) %>%
  filter(
    !is.na(`probability_Mcell-tumor`),
    !is.na(probability_otherTumor)
  )

wilcox_result <- if (nrow(paired) >= 3) {
  wilcox.test(
    paired$`probability_Mcell-tumor`,
    paired$probability_otherTumor,
    paired = TRUE,
    exact = FALSE
  )
} else {
  NULL
}

mcell_only <- sum(
  paired$`detected_in_sig_interaction_Mcell-tumor` &
    !paired$detected_in_sig_interaction_otherTumor
)
other_only <- sum(
  !paired$`detected_in_sig_interaction_Mcell-tumor` &
    paired$detected_in_sig_interaction_otherTumor
)
discordant <- mcell_only + other_only
mcnemar_exact_p <- if (discordant > 0) {
  binom.test(mcell_only, discordant, p = 0.5)$p.value
} else {
  1
}

statistics <- tibble(
  comparison = "Mcell-tumor vs otherTumor -> B_cell CCL20-CCR6",
  n_paired_eligible_samples = nrow(paired),
  n_detected_mcell = sum(
    paired$`detected_in_sig_interaction_Mcell-tumor`
  ),
  n_detected_other_tumor = sum(
    paired$detected_in_sig_interaction_otherTumor
  ),
  n_mcell_only_detected = mcell_only,
  n_other_tumor_only_detected = other_only,
  median_probability_mcell = median(
    paired$`probability_Mcell-tumor`
  ),
  median_probability_other_tumor = median(
    paired$probability_otherTumor
  ),
  median_paired_difference = median(
    paired$`probability_Mcell-tumor` -
      paired$probability_otherTumor
  ),
  paired_wilcoxon_p = if (is.null(wilcox_result)) {
    NA_real_
  } else {
    wilcox_result$p.value
  },
  exact_mcnemar_p = mcnemar_exact_p
)

write_tsv(
  statistics,
  file.path(out_dir, "CCL20_CCR6_comparison_statistics.tsv")
)
write_tsv(
  paired,
  file.path(out_dir, "CCL20_CCR6_paired_eligible_samples.tsv")
)

paired_plot_data <- summary_long %>%
  filter(sample %in% paired$sample, eligible)

p1 <- ggplot(
  paired_plot_data,
  aes(
    x = sender,
    y = probability,
    group = sample
  )
) +
  geom_boxplot(
    aes(group = sender, fill = sender),
    width = 0.55,
    alpha = 0.28,
    outlier.shape = NA,
    linewidth = 0.5
  ) +
  geom_line(color = "grey70", linewidth = 0.45, alpha = 0.8) +
  geom_point(
    aes(color = sender),
    size = 2.2,
    alpha = 0.85
  ) +
  scale_color_manual(
    values = c(
      "Mcell-tumor" = "#C61E89",
      "otherTumor" = "#2C7FBA"
    )
  ) +
  scale_fill_manual(
    values = c(
      "Mcell-tumor" = "#C61E89",
      "otherTumor" = "#2C7FBA"
    )
  ) +
  labs(
    title = "CCL20-CCR6 signaling to B cells",
    subtitle = paste0(
      "Paired eligible samples; Wilcoxon p=",
      signif(statistics$paired_wilcoxon_p, 3)
    ),
    x = "CCL20-expressing sender",
    y = "CellChat communication probability",
    color = NULL
  ) +
  theme_classic(base_size = 9) +
  theme(legend.position = "none")

detection <- summary_long %>%
  group_by(sender) %>%
  summarise(
    n_eligible = sum(eligible),
    n_detected = sum(detected_in_sig_interaction),
    detection_fraction = ifelse(
      n_eligible > 0,
      n_detected / n_eligible,
      NA_real_
    ),
    .groups = "drop"
  )

write_tsv(
  detection,
  file.path(out_dir, "CCL20_CCR6_detection_summary.tsv")
)

p2 <- ggplot(
  detection,
  aes(x = sender, y = detection_fraction, fill = sender)
) +
  geom_col(width = 0.65) +
  geom_text(
    aes(label = paste0(n_detected, "/", n_eligible)),
    vjust = -0.3,
    size = 3
  ) +
  scale_fill_manual(
    values = c(
      "Mcell-tumor" = "#C61E89",
      "otherTumor" = "#2C7FBA"
    )
  ) +
  scale_y_continuous(
    limits = c(0, 1.08),
    breaks = seq(0, 1, 0.25)
  ) +
  labs(
    title = "Significant-interaction detection",
    subtitle = paste0(
      "Exact paired detection p=",
      signif(statistics$exact_mcnemar_p, 3)
    ),
    x = "CCL20-expressing sender",
    y = "Fraction of eligible samples",
    fill = NULL
  ) +
  theme_classic(base_size = 9) +
  theme(legend.position = "none")

donut_data <- detection %>%
  filter(sender %in% senders) %>%
  transmute(
    sender,
    Detected = n_detected,
    `Not detected` = n_eligible - n_detected
  ) %>%
  pivot_longer(
    cols = c(Detected, `Not detected`),
    names_to = "status",
    values_to = "n_samples"
  ) %>%
  mutate(
    sender = factor(sender, levels = senders),
    status = factor(status, levels = c("Detected", "Not detected"))
  ) %>%
  group_by(sender) %>%
  mutate(
    fraction = n_samples / sum(n_samples),
    label = paste0(
      n_samples,
      "/",
      sum(n_samples),
      "\n(",
      scales::percent(fraction, accuracy = 0.1),
      ")"
    )
  ) %>%
  ungroup()

donut_labels <- donut_data %>%
  filter(n_samples > 0)
donut_plot_data <- donut_data %>%
  filter(n_samples > 0)
fisher_table <- detection %>%
  filter(sender %in% senders) %>%
  arrange(factor(sender, levels = senders)) %>%
  transmute(
    detected = n_detected,
    not_detected = n_eligible - n_detected
  ) %>%
  as.matrix()
fisher_p <- fisher.test(fisher_table)$p.value

p3 <- ggplot(
  donut_plot_data,
  aes(x = 2, y = n_samples, fill = status)
) +
  geom_col(width = 1, color = "white", linewidth = 0.7) +
  coord_polar(theta = "y") +
  xlim(0.5, 2.5) +
  geom_text(
    data = donut_labels,
    aes(label = label),
    position = position_stack(vjust = 0.5),
    size = 3.2
  ) +
  facet_wrap(~sender, nrow = 1, scales = "free") +
  scale_fill_manual(
    values = c(
      "Detected" = "#C61E89",
      "Not detected" = "#D9D9D9"
    )
  ) +
  labs(
    title = "Tumor-to-B_cell CCL20-CCR6 detection",
    subtitle = paste0(
      "Frequency across eligible samples; Fisher exact p=",
      signif(fisher_p, 3)
    ),
    fill = NULL
  ) +
  theme_void(base_size = 9) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    plot.background = element_rect(fill = "white", color = NA)
  )

probability_threshold <- if (length(args) >= 3) {
  as.numeric(args[[3]])
} else {
  0.1
}
if (is.na(probability_threshold) || probability_threshold < 0) {
  stop("Probability threshold must be a non-negative number")
}
threshold_file_label <- format(
  probability_threshold,
  scientific = FALSE,
  trim = TRUE
)
threshold_detection <- summary_long %>%
  filter(eligible, sender %in% senders) %>%
  group_by(sender) %>%
  summarise(
    n_eligible = n_distinct(sample),
    n_detected = sum(probability > probability_threshold),
    .groups = "drop"
  ) %>%
  mutate(sender = factor(sender, levels = senders))

threshold_donut_data <- threshold_detection %>%
  transmute(
    sender,
    Detected = n_detected,
    `Not detected` = n_eligible - n_detected
  ) %>%
  pivot_longer(
    cols = c(Detected, `Not detected`),
    names_to = "status",
    values_to = "n_samples"
  ) %>%
  mutate(
    status = factor(status, levels = c("Detected", "Not detected"))
  ) %>%
  filter(n_samples > 0)

threshold_labels <- threshold_detection %>%
  mutate(
    label = paste0(
      n_detected,
      "/",
      n_eligible,
      " detected\n(",
      scales::percent(n_detected / n_eligible, accuracy = 0.1),
      ")"
    )
  )

threshold_fisher <- fisher.test(
  threshold_detection %>%
    transmute(
      detected = n_detected,
      not_detected = n_eligible - n_detected
    ) %>%
    as.matrix()
)$p.value

p4 <- ggplot(
  threshold_donut_data,
  aes(x = 2, y = n_samples, fill = status)
) +
  geom_col(width = 1, color = "white", linewidth = 0.7) +
  coord_polar(theta = "y") +
  xlim(0.5, 2.5) +
  geom_text(
    data = threshold_labels,
    aes(x = 2, y = n_eligible / 2, label = label),
    inherit.aes = FALSE,
    size = 3.2
  ) +
  facet_wrap(~sender, nrow = 1, scales = "free") +
  scale_fill_manual(
    values = c(
      "Detected" = "#C61E89",
      "Not detected" = "#D9D9D9"
    ),
    drop = FALSE
  ) +
  labs(
    title = "Tumor-to-B_cell CCL20-CCR6 detection",
    subtitle = paste0(
      "CellChat probability > ",
      probability_threshold,
      "; Fisher exact p=",
      signif(threshold_fisher, 3)
    ),
    fill = NULL
  ) +
  theme_void(base_size = 9) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5),
    plot.background = element_rect(fill = "white", color = NA)
  )

pdf(
  file.path(out_dir, "CCL20_CCR6_Mcell_otherTumor_to_Bcell.pdf"),
  width = 5.5,
  height = 4.8,
  onefile = TRUE
)
print(p1)
print(p2)
print(p3)
print(p4)
dev.off()

ggsave(
  file.path(out_dir, "CCL20_CCR6_paired_probability_boxplot.pdf"),
  p1,
  width = 5.5,
  height = 4.8
)
ggsave(
  file.path(out_dir, "CCL20_CCR6_paired_probability.png"),
  p1,
  width = 5.5,
  height = 4.8,
  dpi = 300
)
ggsave(
  file.path(out_dir, "CCL20_CCR6_detection_fraction.png"),
  p2,
  width = 5.5,
  height = 4.8,
  dpi = 300
)
ggsave(
  file.path(out_dir, "CCL20_CCR6_two_sender_detection_donuts.pdf"),
  p3,
  width = 7,
  height = 4.2
)
ggsave(
  file.path(out_dir, "CCL20_CCR6_two_sender_detection_donuts.png"),
  p3,
  width = 7,
  height = 4.2,
  dpi = 300
)
ggsave(
  file.path(
    out_dir,
    paste0(
      "CCL20_CCR6_two_sender_probability_gt_",
      threshold_file_label,
      "_donuts.pdf"
    )
  ),
  p4,
  width = 7,
  height = 4.2
)
ggsave(
  file.path(
    out_dir,
    paste0(
      "CCL20_CCR6_two_sender_probability_gt_",
      threshold_file_label,
      "_donuts.png"
    )
  ),
  p4,
  width = 7,
  height = 4.2,
  dpi = 300
)

message("Wrote outputs to: ", out_dir)
