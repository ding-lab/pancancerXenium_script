#!/usr/bin/env Rscript

# min5 overall survival under TCGA/test/min5:
#   per-cancer survminer ggforest for proximal-50 tumoral area fraction
#   PRAD uses age + Gleason (6-7 vs 8-10) instead of gender/stage.
#   LUAD/LUSC also adjust for Ever-smoker (tobacco_smoking_history).

suppressPackageStartupMessages({
  library(tidyverse)
  library(survival)
  library(survminer)
  library(grid)
  library(ggpubr)
})

clinical_dir <- "/storage1/fs1/dinglab/Active/Projects/yuweiz/projects/xenium1k/01_cell_typing/01_analysis/04_schwann/clinical"
out_root <- file.path(clinical_dir, "TCGA", "test", "min5")
cmp_dir <- file.path(out_root, "cancer_type_metric_comparison")
dir.create(cmp_dir, recursive = TRUE, showWarnings = FALSE)
clinical_file <- file.path(clinical_dir, "TCGA_info", "TCGA-PRAD.clinical.tsv")
clin_matrix_root <- file.path(clinical_dir, "TCGA", "clinical")

min_nerves_for_fraction <- 5L
no_gender_cancers <- c("TCGA-BRCA", "TCGA-PRAD")
smoking_cancers <- c("TCGA-LUAD", "TCGA-LUSC")

metric_specs <- tibble(
  metric = "proximal_50_tumoral_area_fraction",
  short_name = "proximal_50",
  plot_name = "Proximal-50 fraction"
)

transform_metric <- function(x) asin(sqrt(pmin(pmax(x, 0), 1)))

load_gleason <- function() {
  read_tsv(
    clinical_file,
    col_select = c(
      submitter_id,
      sample_type_id.samples,
      primary_gleason_grade.diagnoses,
      secondary_gleason_grade.diagnoses
    ),
    show_col_types = FALSE
  ) %>%
    filter(as.character(sample_type_id.samples) == "01") %>%
    transmute(
      case_id = str_trim(submitter_id),
      primary_gleason = parse_number(as.character(primary_gleason_grade.diagnoses)),
      secondary_gleason = parse_number(as.character(secondary_gleason_grade.diagnoses))
    ) %>%
    filter(is.finite(primary_gleason), is.finite(secondary_gleason)) %>%
    distinct() %>%
    group_by(case_id) %>%
    summarise(
      gleason_score_numeric = dplyr::first(primary_gleason) + dplyr::first(secondary_gleason),
      .groups = "drop"
    ) %>%
    mutate(
      gleason_score_binary = factor(
        case_when(
          gleason_score_numeric %in% c(6, 7) ~ "Gleason 6-7",
          gleason_score_numeric %in% c(8, 9, 10) ~ "Gleason 8-10",
          TRUE ~ NA_character_
        ),
        levels = c("Gleason 6-7", "Gleason 8-10")
      )
    ) %>%
    filter(!is.na(gleason_score_binary)) %>%
    select(case_id, gleason_score_binary)
}

load_smoking <- function(cancer) {
  code <- str_remove(cancer, "^TCGA-")
  f <- file.path(
    clin_matrix_root,
    sprintf("TCGA.%s.sampleMap_%s_clinicalMatrix", code, code)
  )
  read_tsv(f, show_col_types = FALSE) %>%
    transmute(
      sample_id = str_trim(sampleID),
      case_id = str_sub(sample_id, 1, 12),
      sample_type_id = coalesce(
        str_pad(as.character(sample_type_id), 2, pad = "0"),
        str_sub(sample_id, 14, 15)
      ),
      smoke_code = str_trim(as.character(tobacco_smoking_history))
    ) %>%
    filter(smoke_code %in% c("1", "2", "3", "4", "5"), case_id != "") %>%
    mutate(
      smoking_status = factor(
        if_else(smoke_code == "1", "Nonsmoker", "Ever-smoker"),
        levels = c("Nonsmoker", "Ever-smoker")
      ),
      priority = if_else(sample_type_id == "01", 1L, 2L)
    ) %>%
    arrange(priority) %>%
    group_by(case_id) %>%
    slice(1) %>%
    ungroup() %>%
    select(case_id, smoking_status)
}

extract_coef <- function(fit, cancer, metric, short_name, formula_text) {
  s <- summary(fit)
  coefs <- as.data.frame(s$coefficients)
  conf <- as.data.frame(s$conf.int)
  tibble(
    cancer = cancer,
    metric = metric,
    short_name = short_name,
    formula = formula_text,
    term = rownames(coefs),
    n = as.integer(s$n),
    events = as.integer(s$nevent),
    n_parameters = nrow(coefs),
    hazard_ratio = conf[["exp(coef)"]],
    conf_low = conf[["lower .95"]],
    conf_high = conf[["upper .95"]],
    p_value = coefs[["Pr(>|z|)"]]
  )
}

choose_terms_standard <- function(data, cancer) {
  terms <- character()
  x <- data
  if (sum(is.finite(x$age_per_10y)) >= 2 &&
      n_distinct(x$age_per_10y[is.finite(x$age_per_10y)]) > 1) {
    terms <- c(terms, "age_per_10y")
    x <- x %>% filter(is.finite(age_per_10y))
  }
  use_gender <- !(cancer %in% no_gender_cancers) &&
    sum(!is.na(x$gender)) >= 2 &&
    n_distinct(droplevels(x$gender[!is.na(x$gender)])) > 1
  if (use_gender) {
    terms <- c(terms, "gender")
    x <- x %>% filter(!is.na(gender))
  }
  if (sum(!is.na(x$stage_group)) >= 2 &&
      n_distinct(droplevels(x$stage_group[!is.na(x$stage_group)])) > 1) {
    terms <- c(terms, "stage_group")
    x <- x %>% filter(!is.na(stage_group))
  }
  if (cancer %in% smoking_cancers &&
      sum(!is.na(x$smoking_status)) >= 2 &&
      n_distinct(droplevels(x$smoking_status[!is.na(x$smoking_status)])) > 1) {
    terms <- c(terms, "smoking_status")
    x <- x %>% filter(!is.na(smoking_status))
  }
  list(terms = terms, data = droplevels(x))
}

choose_terms_prad <- function(data) {
  terms <- character()
  x <- data
  if (sum(is.finite(x$age_per_10y)) >= 2 &&
      n_distinct(x$age_per_10y[is.finite(x$age_per_10y)]) > 1) {
    terms <- c(terms, "age_per_10y")
    x <- x %>% filter(is.finite(age_per_10y))
  }
  if (sum(!is.na(x$gleason_score_binary)) >= 2 &&
      n_distinct(droplevels(x$gleason_score_binary[!is.na(x$gleason_score_binary)])) > 1) {
    terms <- c(terms, "gleason_score_binary")
    x <- x %>% filter(!is.na(gleason_score_binary))
  }
  list(terms = terms, data = droplevels(x))
}

make_one_forest <- function(cancer, analysis_data, spec) {
  out_dir <- file.path(out_root, cancer)
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

  model_data <- analysis_data %>%
    transmute(
      time = os_time,
      event = os_event,
      value = .data[[spec$metric]],
      age_per_10y,
      gender,
      stage_group,
      gleason_score_binary = if ("gleason_score_binary" %in% names(analysis_data)) {
        gleason_score_binary
      } else {
        factor(NA_character_, levels = c("Gleason 6-7", "Gleason 8-10"))
      },
      smoking_status = if ("smoking_status" %in% names(analysis_data)) {
        smoking_status
      } else {
        factor(NA_character_, levels = c("Nonsmoker", "Ever-smoker"))
      }
    ) %>%
    filter(
      is.finite(time),
      time > 0,
      event %in% c(0L, 1L),
      is.finite(value)
    ) %>%
    mutate(metric_z = as.numeric(scale(transform_metric(value)))) %>%
    filter(is.finite(metric_z)) %>%
    droplevels()

  if (nrow(model_data) < 10 ||
      sum(model_data$event) < 2 ||
      n_distinct(model_data$metric_z) < 2) {
    return(tibble(
      cancer = cancer, metric = spec$metric, short_name = spec$short_name,
      formula = NA_character_, term = NA_character_,
      n = nrow(model_data), events = sum(model_data$event),
      n_parameters = NA_integer_, hazard_ratio = NA_real_,
      conf_low = NA_real_, conf_high = NA_real_, p_value = NA_real_,
      note = "Skipped: too few cases/events or no metric variation"
    ))
  }

  chosen <- if (cancer == "TCGA-PRAD") {
    choose_terms_prad(model_data)
  } else {
    choose_terms_standard(model_data, cancer)
  }

  plot_data <- as.data.frame(chosen$data)
  names(plot_data)[names(plot_data) == "metric_z"] <- "Proximal50"
  if ("age_per_10y" %in% chosen$terms) {
    names(plot_data)[names(plot_data) == "age_per_10y"] <- "Age"
    plot_data$Age <- as.numeric(plot_data$Age)
  }
  if ("gender" %in% chosen$terms) {
    names(plot_data)[names(plot_data) == "gender"] <- "Gender"
  }
  if ("stage_group" %in% chosen$terms) {
    names(plot_data)[names(plot_data) == "stage_group"] <- "Stage"
  }
  if ("gleason_score_binary" %in% chosen$terms) {
    names(plot_data)[names(plot_data) == "gleason_score_binary"] <- "Gleason"
  }
  if ("smoking_status" %in% chosen$terms) {
    names(plot_data)[names(plot_data) == "smoking_status"] <- "Smoking"
  }

  metric_term <- "Proximal50"
  rhs <- c(
    metric_term,
    if ("age_per_10y" %in% chosen$terms) "Age",
    if ("gender" %in% chosen$terms) "Gender",
    if ("stage_group" %in% chosen$terms) "Stage",
    if ("gleason_score_binary" %in% chosen$terms) "Gleason",
    if ("smoking_status" %in% chosen$terms) "Smoking"
  )
  formula_text <- paste("Surv(time, event) ~", paste(rhs, collapse = " + "))
  fit <- tryCatch(
    coxph(as.formula(formula_text), data = plot_data, ties = "efron", x = TRUE),
    error = function(e) NULL
  )
  if (is.null(fit)) {
    return(tibble(
      cancer = cancer, metric = spec$metric, short_name = spec$short_name,
      formula = formula_text, term = NA_character_,
      n = nrow(chosen$data), events = sum(chosen$data$event),
      n_parameters = NA_integer_, hazard_ratio = NA_real_,
      conf_low = NA_real_, conf_high = NA_real_, p_value = NA_real_,
      note = "Adjusted Cox model failed"
    ))
  }

  results <- extract_coef(
    fit, cancer, spec$metric, spec$short_name, formula_text
  ) %>%
    mutate(note = NA_character_)
  write_csv(
    results,
    file.path(
      out_dir,
      paste0(spec$short_name, "_age_gender_stage_adjusted_cox_results.csv")
    )
  )

  covariate_label <- paste(
    c(
      spec$plot_name,
      if ("age_per_10y" %in% chosen$terms) "age",
      if ("gender" %in% chosen$terms) "gender",
      if ("stage_group" %in% chosen$terms) "stage",
      if ("gleason_score_binary" %in% chosen$terms) "Gleason",
      if ("smoking_status" %in% chosen$terms) "Ever-smoker"
    ),
    collapse = " + "
  )
  p <- ggforest(
    fit,
    data = plot_data,
    main = paste0(cancer, " OS (min5): ", covariate_label),
    cpositions = c(0.02, 0.22, 0.4),
    fontsize = 0.85,
    refLabel = "reference",
    noDigits = 2
  )
  ggsave(
    file.path(
      out_dir,
      paste0(spec$short_name, "_age_gender_stage_adjusted_forest_plot.pdf")
    ),
    p,
    width = 8.6,
    height = max(3.6, 1.7 + 0.72 * length(coef(fit)))
  )
  message("  wrote ", spec$short_name, " ggforest for ", cancer)
  results
}

# ---------------------------------------------------------------------------
# 1) per-cancer ggforest
# ---------------------------------------------------------------------------
gleason <- load_gleason()
cohorts <- list.dirs(out_root, recursive = FALSE, full.names = FALSE)
cohorts <- cohorts[str_starts(cohorts, "TCGA-")]
cohorts <- cohorts[
  file.exists(file.path(out_root, cohorts, "survival_analysis_data.csv"))
]
cohorts <- sort(cohorts)
message("ggforest cohorts: ", paste(cohorts, collapse = ", "))

all_rows <- map_dfr(cohorts, function(cancer) {
  message("Fitting ", cancer)
  analysis_data <- read_csv(
    file.path(out_root, cancer, "survival_analysis_data.csv"),
    show_col_types = FALSE
  ) %>%
    mutate(
      gender = factor(gender, levels = c("Female", "Male")),
      stage_group = factor(stage_group, levels = c("Stage I/II", "Stage III/IV"))
    )
  if (cancer == "TCGA-PRAD") {
    analysis_data <- analysis_data %>%
      left_join(gleason, by = "case_id")
  }
  if (cancer %in% smoking_cancers) {
    analysis_data <- analysis_data %>%
      left_join(load_smoking(cancer), by = "case_id")
  }
  map_dfr(seq_len(nrow(metric_specs)), function(i) {
    tryCatch(
      make_one_forest(cancer, analysis_data, metric_specs[i, ]),
      error = function(e) {
        message("  failed ", metric_specs$short_name[i], ": ", conditionMessage(e))
        tibble(
          cancer = cancer,
          metric = metric_specs$metric[i],
          short_name = metric_specs$short_name[i],
          formula = NA_character_,
          term = NA_character_,
          n = NA_integer_,
          events = NA_integer_,
          n_parameters = NA_integer_,
          hazard_ratio = NA_real_,
          conf_low = NA_real_,
          conf_high = NA_real_,
          p_value = NA_real_,
          note = conditionMessage(e)
        )
      }
    )
  })
})

write_csv(
  all_rows,
  file.path(out_root, "proximal_adjusted_ggforest_results.csv")
)
message("Wrote: ", file.path(out_root, "proximal_adjusted_ggforest_results.csv"))

# ---------------------------------------------------------------------------
# all-cancers forest
# ---------------------------------------------------------------------------
message("==== summary forests ====")

# Same ggforest layout as TCGA/cancer_type_metric_comparison
# (tcga_v5_proximal50_all_cancers_ggforest.R).
make_all_cancers_forest <- function(plot_data, outfile, title, footnote) {
  no_digits <- 2
  fontsize <- 0.78
  cpositions <- c(0.02, 0.22, 0.4)
  plot_data <- plot_data %>%
    mutate(
      estimate = log(hazard_ratio),
      conf.low = log(conf_low),
      conf.high = log(conf_high),
      estimate.1 = format(hazard_ratio, digits = no_digits),
      ci = paste0(
        "(",
        format(conf_low, digits = no_digits),
        " - ",
        format(conf_high, digits = no_digits),
        ")"
      ),
      stars = case_when(
        p_value < 0.001 ~ "<0.001 ***",
        TRUE ~ paste0(
          signif(p_value, no_digits + 1),
          " ",
          strrep("*", (p_value < 0.05) + (p_value < 0.01) + (p_value < 0.001))
        )
      ),
      N = paste0("(N=", n, ", events=", events, ")")
    ) %>%
    arrange(hazard_ratio) %>%
    mutate(row_id = row_number())

  write_csv(
    plot_data %>%
      select(
        cancer, cancer_short, n, events, hazard_ratio, conf_low, conf_high,
        p_value
      ),
    sub("\\.pdf$", ".csv", outfile)
  )

  rangeb <- range(plot_data$conf.low, plot_data$conf.high, na.rm = TRUE)
  breaks <- axisTicks(rangeb / 2, log = TRUE, nint = 7)
  rangeplot <- rangeb
  rangeplot[1] <- rangeplot[1] - diff(rangeb)
  rangeplot[2] <- rangeplot[2] + 0.15 * diff(rangeb)
  width <- diff(rangeplot)
  y_variable <- rangeplot[1] + cpositions[1] * width
  y_nlevel <- rangeplot[1] + cpositions[2] * width
  y_cistring <- rangeplot[1] + cpositions[3] * width
  y_stars <- rangeb[2]
  annot_size_mm <- fontsize *
    as.numeric(convertX(unit(theme_get()$text$size, "pt"), "mm"))

  p <- ggplot(plot_data, aes(row_id, exp(estimate))) +
    geom_rect(
      aes(
        xmin = row_id - 0.5,
        xmax = row_id + 0.5,
        ymin = exp(rangeplot[1]),
        ymax = exp(rangeplot[2]),
        fill = ordered(row_id %% 2 + 1)
      )
    ) +
    scale_fill_manual(values = c("#FFFFFF33", "#00000033"), guide = "none") +
    geom_point(pch = 15, size = 4) +
    geom_errorbar(aes(ymin = exp(conf.low), ymax = exp(conf.high)), width = 0.15) +
    geom_hline(yintercept = 1, linetype = 3) +
    coord_flip(ylim = exp(rangeplot)) +
    ggtitle(title) +
    scale_y_log10(
      name = "",
      labels = sprintf("%g", breaks),
      expand = c(0.02, 0.02),
      breaks = breaks
    ) +
    theme_light() +
    theme(
      panel.grid.minor.y = element_blank(),
      panel.grid.minor.x = element_blank(),
      panel.grid.major.y = element_blank(),
      legend.position = "none",
      panel.border = element_blank(),
      axis.title.y = element_blank(),
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank(),
      plot.title = element_text(hjust = 0.5)
    ) +
    xlab("") +
    annotate(
      geom = "text",
      x = plot_data$row_id,
      y = exp(y_variable),
      label = plot_data$cancer_short,
      fontface = "bold",
      hjust = 0,
      size = annot_size_mm
    ) +
    annotate(
      geom = "text",
      x = plot_data$row_id,
      y = exp(y_nlevel),
      label = plot_data$N,
      fontface = "italic",
      hjust = 0,
      size = annot_size_mm
    ) +
    annotate(
      geom = "text",
      x = plot_data$row_id,
      y = exp(y_cistring),
      label = plot_data$estimate.1,
      size = annot_size_mm,
      vjust = -0.1
    ) +
    annotate(
      geom = "text",
      x = plot_data$row_id,
      y = exp(y_cistring),
      label = plot_data$ci,
      size = annot_size_mm,
      vjust = 1.1,
      fontface = "italic"
    ) +
    annotate(
      geom = "text",
      x = plot_data$row_id,
      y = exp(y_stars),
      label = plot_data$stars,
      size = annot_size_mm,
      hjust = -0.2,
      fontface = "italic"
    ) +
    annotate(
      geom = "text",
      x = 0.35,
      y = exp(y_variable),
      label = footnote,
      size = annot_size_mm,
      hjust = 0,
      vjust = 1.2,
      fontface = "italic"
    )

  gt <- ggplot_gtable(ggplot_build(p))
  gt$layout$clip[gt$layout$name == "panel"] <- "off"
  p_out <- ggpubr::as_ggplot(gt)
  ggsave(outfile, p_out, width = 9.2, height = 6.4)
  message("Wrote: ", outfile)
}

# proximal-50 all cancers from ggforest
p50_all <- all_rows %>%
  filter(
    short_name == "proximal_50",
    term == "Proximal50",
    is.finite(hazard_ratio),
    is.finite(p_value)
  ) %>%
  mutate(cancer_short = str_remove(cancer, "^TCGA-"))
make_all_cancers_forest(
  p50_all,
  file.path(cmp_dir, "proximal_50_all_cancers_adjusted_forest.pdf"),
  title = "OS: proximal-50 fraction by cancer",
  footnote = paste0(
    "Age + gender + stage; LUAD/LUSC +Ever-smoker; BRCA omit gender; PRAD: age + Gleason\n",
    "Ranked by adjusted HR; each row is Proximal-50 fraction (min5, >=5 nerves)"
  )
)

# significant-only proximal forests
sn <- "proximal_50"
term_name <- "Proximal50"
sig <- all_rows %>%
  filter(
    short_name == sn,
    term == term_name,
    is.finite(p_value),
    p_value < 0.05,
    is.finite(hazard_ratio)
  ) %>%
  mutate(
    cancer_short = str_remove(cancer, "^TCGA-"),
    plot_label = paste0(
      cancer_short,
      "\nHR ",
      sprintf("%.2f", hazard_ratio),
      " (",
      sprintf("%.2f", conf_low),
      "-",
      sprintf("%.2f", conf_high),
      "), p=",
      format.pval(p_value, digits = 3)
    )
  ) %>%
  arrange(hazard_ratio) %>%
  mutate(plot_label = factor(plot_label, levels = plot_label))
write_csv(
  sig,
  file.path(cmp_dir, paste0(sn, "_significant_cancers_adjusted_forest.csv"))
)
if (nrow(sig) > 0) {
  p <- ggplot(
    sig,
    aes(x = hazard_ratio, y = plot_label, xmin = conf_low, xmax = conf_high)
  ) +
    geom_vline(xintercept = 1, linetype = 2, color = "grey50") +
    geom_errorbarh(height = 0.2, color = "#2C7BB6") +
    geom_point(size = 2.8, color = "#2C7BB6") +
    scale_x_log10() +
    labs(
      x = "Hazard ratio (log scale)",
      y = NULL,
      title = paste0(
        "Significant ",
        sn,
        " adjusted HRs (min5, p<0.05)"
      )
    ) +
    theme_classic(base_size = 11) +
    theme(plot.title = element_text(face = "bold", size = 11))
  ggsave(
    file.path(cmp_dir, paste0(sn, "_significant_cancers_adjusted_forest.pdf")),
    p,
    width = 8.4,
    height = max(3.6, 1.5 + 0.72 * nrow(sig))
  )
  message(
    "Significant ",
    sn,
    ": ",
    paste(sig$cancer_short, collapse = ", ")
  )
}

message("Done. Outputs under: ", out_root)
message("  ggforest: ", out_root, "/TCGA-*/proximal_*_age_gender_stage_adjusted_forest_plot.pdf")
message("  comparison: ", cmp_dir)
