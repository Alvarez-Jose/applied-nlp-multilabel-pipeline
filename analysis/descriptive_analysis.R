# =============================================================================
# Palestine Violence Archive — Descriptive Analysis
# =============================================================================
#
# Purpose:
#   Produce frequency tables, cross-tabulations, and basic visualizations
#   from the cleaned incident database. Designed to be run after
#   export_for_analysis.py has produced the CSV/SAV files.
#
# Input:
#   analysis/Palestine_Violence_Archive_2023_2024_2025.csv
#   (or the .sav file via haven::read_sav)
#
# Usage:
#   Rscript analysis/descriptive_analysis.R
#   # or open in RStudio and run interactively
#
# Output:
#   analysis/tables/         — frequency tables as CSV
#   analysis/figures/        — PNG figures
#
# Required packages (install once):
#   install.packages(c("tidyverse", "haven", "janitor", "gt", "ggplot2"))
# =============================================================================

library(tidyverse)
library(haven)     # read .sav with labels
library(janitor)   # tabyl(), adorn_*
library(ggplot2)

# -----------------------------------------------------------------------------
# 0. Configuration
# -----------------------------------------------------------------------------

ANALYSIS_DIR <- dirname(sys.frame(1)$ofile) |> normalizePath()
if (!exists("ANALYSIS_DIR") || is.null(ANALYSIS_DIR)) {
  ANALYSIS_DIR <- getwd()  # fallback for RStudio interactive use
}

CSV_FILE <- file.path(ANALYSIS_DIR, "Palestine_Violence_Archive_2023_2024_2025.csv")
SAV_FILE <- file.path(ANALYSIS_DIR, "Palestine_Violence_Archive_2023_2024_2025.sav")

TABLE_DIR  <- file.path(ANALYSIS_DIR, "tables")
FIGURE_DIR <- file.path(ANALYSIS_DIR, "figures")
dir.create(TABLE_DIR,  showWarnings = FALSE, recursive = TRUE)
dir.create(FIGURE_DIR, showWarnings = FALSE, recursive = TRUE)

# Binary label columns
LABELS <- c(
  "raid", "arrest_detention", "physical_assault",
  "coercive_actions", "restriction_of_freedoms", "religious_encroachment",
  "harm_to_property", "dispossession", "multi_community_incident", "protest"
)

LABEL_NAMES <- c(
  raid                   = "Raid",
  arrest_detention       = "Arrest / Detention",
  physical_assault       = "Physical Assault",
  coercive_actions       = "Coercive Actions",
  restriction_of_freedoms = "Restriction of Freedoms",
  religious_encroachment = "Religious Encroachment",
  harm_to_property       = "Harm to Property",
  dispossession          = "Dispossession",
  multi_community_incident = "Multi-Community Incident",
  protest                = "Protest"
)

cat("=== Palestine Violence Archive — Descriptive Analysis ===\n\n")

# -----------------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------------

if (file.exists(CSV_FILE)) {
  df <- read_csv(CSV_FILE, show_col_types = FALSE)
  cat("Loaded CSV:", CSV_FILE, "\n")
} else if (file.exists(SAV_FILE)) {
  df <- read_sav(SAV_FILE) |> zap_labels()  # strip SPSS labels → plain R data frame
  cat("Loaded SAV:", SAV_FILE, "\n")
} else {
  stop(
    "No input file found. Run first:\n",
    "  python data_pipeline/export_for_analysis.py\n"
  )
}

df$date <- as.Date(df$date)

cat("Total incidents loaded:", nrow(df), "\n\n")

# -----------------------------------------------------------------------------
# 2. Helper: save table as CSV and print to console
# -----------------------------------------------------------------------------

save_table <- function(tbl, filename, caption = NULL) {
  path <- file.path(TABLE_DIR, filename)
  write_csv(tbl, path)
  if (!is.null(caption)) cat("\n---", caption, "---\n")
  print(tbl, n = Inf)
  cat("  -> saved:", path, "\n")
  invisible(tbl)
}

save_figure <- function(p, filename, width = 8, height = 5) {
  path <- file.path(FIGURE_DIR, filename)
  ggsave(path, plot = p, width = width, height = height, dpi = 150)
  cat("  -> saved:", path, "\n")
  invisible(p)
}

# -----------------------------------------------------------------------------
# 3. Incident counts by year
# -----------------------------------------------------------------------------

tbl_year <- df |>
  count(year, name = "n_incidents") |>
  arrange(year)

save_table(tbl_year, "01_incidents_by_year.csv", "Incidents by Year")

# -----------------------------------------------------------------------------
# 4. Incidents by governorate
# -----------------------------------------------------------------------------

tbl_gov <- df |>
  filter(!is.na(governorate)) |>
  count(governorate, name = "n") |>
  arrange(desc(n)) |>
  mutate(pct = round(n / sum(n) * 100, 1))

save_table(tbl_gov, "02_incidents_by_governorate.csv", "Incidents by Governorate")

# Figure: bar chart
p_gov <- tbl_gov |>
  mutate(governorate = fct_reorder(governorate, n)) |>
  ggplot(aes(x = n, y = governorate)) +
  geom_col(fill = "#2c7bb6") +
  geom_text(aes(label = n), hjust = -0.2, size = 3) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.12))) +
  labs(
    title = "Incident Count by Governorate",
    x = "Number of Incidents",
    y = NULL
  ) +
  theme_minimal(base_size = 12)

save_figure(p_gov, "02_incidents_by_governorate.png")

# -----------------------------------------------------------------------------
# 5. Incidents by perpetrator
# -----------------------------------------------------------------------------

tbl_perp <- df |>
  filter(!is.na(perpetrator)) |>
  count(perpetrator, name = "n") |>
  arrange(desc(n)) |>
  mutate(pct = round(n / sum(n) * 100, 1))

save_table(tbl_perp, "03_incidents_by_perpetrator.csv", "Incidents by Perpetrator")

# -----------------------------------------------------------------------------
# 6. Incidents by jurisdiction (Area A / B / C)
# -----------------------------------------------------------------------------

tbl_jur <- df |>
  filter(!is.na(jurisdiction)) |>
  count(jurisdiction, name = "n") |>
  arrange(desc(n)) |>
  mutate(pct = round(n / sum(n) * 100, 1))

save_table(tbl_jur, "04_incidents_by_jurisdiction.csv", "Incidents by Oslo Jurisdiction")

# -----------------------------------------------------------------------------
# 7. Label prevalence (% of incidents with each label)
# -----------------------------------------------------------------------------

tbl_labels <- tibble(
  label       = LABELS,
  label_name  = LABEL_NAMES[LABELS],
  n           = map_int(LABELS, \(l) sum(df[[l]] == 1, na.rm = TRUE)),
  pct         = map_dbl(LABELS, \(l) mean(df[[l]] == 1, na.rm = TRUE) * 100) |> round(1)
) |>
  arrange(desc(pct))

save_table(tbl_labels, "05_label_prevalence.csv", "Label Prevalence")

# Figure: lollipop chart
p_labels <- tbl_labels |>
  mutate(label_name = fct_reorder(label_name, pct)) |>
  ggplot(aes(x = pct, y = label_name)) +
  geom_segment(aes(x = 0, xend = pct, y = label_name, yend = label_name),
               color = "grey70") +
  geom_point(size = 4, color = "#d7191c") +
  geom_text(aes(label = paste0(pct, "%")), hjust = -0.3, size = 3) +
  scale_x_continuous(limits = c(0, 70), labels = \(x) paste0(x, "%")) +
  labs(
    title = "Prevalence of Incident Labels",
    x     = "% of Incidents",
    y     = NULL
  ) +
  theme_minimal(base_size = 12)

save_figure(p_labels, "05_label_prevalence.png")

# -----------------------------------------------------------------------------
# 8. Label prevalence by year
# -----------------------------------------------------------------------------

tbl_labels_year <- df |>
  group_by(year) |>
  summarise(
    across(all_of(LABELS), \(x) round(mean(x == 1, na.rm = TRUE) * 100, 1)),
    n_incidents = n(),
    .groups = "drop"
  ) |>
  pivot_longer(all_of(LABELS), names_to = "label", values_to = "pct") |>
  mutate(label_name = LABEL_NAMES[label])

save_table(
  tbl_labels_year |> pivot_wider(names_from = year, values_from = pct),
  "06_label_prevalence_by_year.csv",
  "Label Prevalence by Year (%)"
)

# Figure: faceted bar chart
p_labels_year <- tbl_labels_year |>
  mutate(label_name = fct_reorder(label_name, pct, .fun = mean)) |>
  ggplot(aes(x = factor(year), y = pct, fill = factor(year))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~label_name, ncol = 2, scales = "free_y") +
  scale_fill_manual(values = c("2023" = "#4575b4", "2024" = "#d73027", "2025" = "#1a9850")) +
  labs(
    title = "Incident Label Prevalence by Year",
    x     = "Year",
    y     = "% of Incidents"
  ) +
  theme_minimal(base_size = 10) +
  theme(strip.text = element_text(size = 8))

save_figure(p_labels_year, "06_label_prevalence_by_year.png", width = 10, height = 12)

# -----------------------------------------------------------------------------
# 9. Monthly trend (incidents over time)
# -----------------------------------------------------------------------------

tbl_monthly <- df |>
  filter(!is.na(date)) |>
  mutate(month = floor_date(date, "month")) |>
  count(month, name = "n_incidents")

save_table(tbl_monthly, "07_monthly_trend.csv", "Monthly Incident Count")

p_monthly <- tbl_monthly |>
  ggplot(aes(x = month, y = n_incidents)) +
  geom_line(color = "#2c7bb6", linewidth = 0.8) +
  geom_point(color = "#2c7bb6", size = 1.5) +
  scale_x_date(date_breaks = "3 months", date_labels = "%b %Y") +
  labs(
    title = "Monthly Incident Count",
    x     = NULL,
    y     = "Incidents"
  ) +
  theme_minimal(base_size = 12) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

save_figure(p_monthly, "07_monthly_trend.png", width = 10, height = 4)

# -----------------------------------------------------------------------------
# 10. Cross-tab: perpetrator x labels
# -----------------------------------------------------------------------------

tbl_perp_labels <- df |>
  filter(!is.na(perpetrator)) |>
  group_by(perpetrator) |>
  summarise(
    n = n(),
    across(all_of(LABELS), \(x) round(mean(x == 1, na.rm = TRUE) * 100, 1)),
    .groups = "drop"
  ) |>
  arrange(desc(n))

colnames(tbl_perp_labels) <- c(
  "Perpetrator", "N",
  LABEL_NAMES[LABELS]
)

save_table(tbl_perp_labels, "08_perpetrator_x_labels.csv",
           "Label Prevalence (%) by Perpetrator")

# -----------------------------------------------------------------------------
# 11. Label co-occurrence matrix
# -----------------------------------------------------------------------------

label_mat <- df |>
  select(all_of(LABELS)) |>
  mutate(across(everything(), \(x) as.integer(x == 1)))

cooc <- t(as.matrix(label_mat)) %*% as.matrix(label_mat)
diag(cooc) <- NA  # zero out diagonal for display

cooc_df <- as.data.frame(cooc) |>
  rownames_to_column("label_a") |>
  pivot_longer(-label_a, names_to = "label_b", values_to = "count") |>
  filter(!is.na(count)) |>
  mutate(
    label_a = LABEL_NAMES[label_a],
    label_b = LABEL_NAMES[label_b]
  )

save_table(
  as.data.frame(cooc) |> rownames_to_column("label"),
  "09_label_cooccurrence.csv",
  "Label Co-occurrence Matrix (count of incidents with both labels)"
)

# Figure: heatmap
p_cooc <- cooc_df |>
  ggplot(aes(x = label_b, y = label_a, fill = count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = count), size = 2.5) +
  scale_fill_gradient(low = "#f7fbff", high = "#084594", na.value = "grey90") +
  labs(
    title = "Label Co-occurrence Matrix",
    x     = NULL,
    y     = NULL,
    fill  = "Count"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    axis.text.x = element_text(angle = 40, hjust = 1),
    panel.grid  = element_blank()
  )

save_figure(p_cooc, "09_label_cooccurrence.png", width = 10, height = 8)

# -----------------------------------------------------------------------------
# 12. Incidents per governorate per year (heat map)
# -----------------------------------------------------------------------------

tbl_gov_year <- df |>
  filter(!is.na(governorate)) |>
  count(governorate, year, name = "n") |>
  pivot_wider(names_from = year, values_from = n, values_fill = 0)

save_table(tbl_gov_year, "10_governorate_by_year.csv",
           "Incident Counts by Governorate and Year")

# -----------------------------------------------------------------------------
# 13. Summary statistics for count variables
# -----------------------------------------------------------------------------

count_vars <- c("killed", "number_injured", "number_arrested")
present_count_vars <- intersect(count_vars, colnames(df))

if (length(present_count_vars) > 0) {
  tbl_counts <- map_dfr(present_count_vars, \(v) {
    x <- df[[v]]
    tibble(
      variable    = v,
      n_non_zero  = sum(x > 0, na.rm = TRUE),
      total       = sum(x, na.rm = TRUE),
      mean        = round(mean(x, na.rm = TRUE), 2),
      median      = median(x, na.rm = TRUE),
      max         = max(x, na.rm = TRUE),
      n_missing   = sum(is.na(x))
    )
  })
  save_table(tbl_counts, "11_count_variable_summary.csv",
             "Summary Statistics: Killed, Injured, Arrested")
}

# -----------------------------------------------------------------------------
# Done
# -----------------------------------------------------------------------------

cat("\n=== Analysis complete ===\n")
cat("Tables saved to:", TABLE_DIR, "\n")
cat("Figures saved to:", FIGURE_DIR, "\n\n")
cat("To open in SPSS:\n")
cat("  File > Open > Data > Palestine_Violence_Archive_2023_2024_2025.sav\n\n")
cat("To open in R with value labels intact:\n")
cat("  library(haven)\n")
cat("  df <- read_sav('analysis/Palestine_Violence_Archive_2023_2024_2025.sav')\n\n")
