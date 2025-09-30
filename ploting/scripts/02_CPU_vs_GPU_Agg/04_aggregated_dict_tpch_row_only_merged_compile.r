# --- 0. Setup: Load Libraries and Configure Parameters ---
# This script generates an aggregated plot for ROW-BASED dictionary
# encodings, merging COMPILED and NOT-COMPILED implementations.

# Install necessary packages if they are not already installed
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("dplyr")) install.packages("dplyr")
if (!require("readr")) install.packages("readr")
if (!require("stringr")) install.packages("stringr")
if (!require("scales")) install.packages("scales")

library(ggplot2)
library(dplyr)
library(readr)
library(stringr)
library(scales)

# --- Configuration ---
# --- MODIFIED: Arguments no longer include torch_compile ---
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  stop("Usage: Rscript 04_aggregated_dict_tpch_row_only_merged_compile.r <input_file> <use_mask> <gray_out>", call. = FALSE)
}

input_file <- args[1]
USE_MASK <- as.logical(args[2])
GRAY_OUT_NON_COMPILED <- as.logical(args[3])

PANEL_SPACING <- unit(2, "lines")
output_dir <- "plots"

# --- MODIFIED: Select only ROW-BASED dictionary encodings ---
PREDICATES_TO_PLOT <- c("Eq", "Lt", "Prefix")
DICT_ENCODINGS_TO_PLOT <- c(
  "UnsortedDictionaryEncoding", 
  "DictionaryEncoding"
)

# --- Plotting options ---
SHOW_THEORETICAL_BANDWIDTH <- TRUE
TITLE_HJUST <- 0.0
SUBTITLE_HJUST <- 0.0
TITLE_SIZE <- 22
SUBTITLE_SIZE <- 15
LEGEND_MARGIN_VAL <- margin(t = -10, unit = "pt")
AXIS_TEXT_SIZE <- 16
AXIS_TITLE_SIZE <- 18
LEGEND_TITLE_SIZE <- 18
LEGEND_TEXT_SIZE <- 16
STRIP_TEXT_SIZE <- 16
X_AXIS_BREAKS <- c(1, 50, 100, 150, 200)

# Theoretical Memory Bandwidth (in bytes per second)
THEORETICAL_GPU_BANDWIDTH <- 1.1e12 # 1.1 TB/s
TIME_METRIC <- "mean"
BACKGROUND_STYLE <- "transparent"

# --- 1. Data Loading and Preparation ---
data <- read_csv(input_file, show_col_types = FALSE)

predicate_map <- c("Eq" = "Equal", "Lt" = "Less Than", "Prefix" = "Prefix Match")

# --- MODIFIED: Filter data and create new grouping columns for compile status ---
plot_data <- data %>%
  filter(
    `param:device` == "cuda",
    pred %in% PREDICATES_TO_PLOT,
    `param:tensor_cls` %in% DICT_ENCODINGS_TO_PLOT,
    `param:return_mask` == USE_MASK
  ) %>%
  mutate(
    throughput_gb_per_sec = (total_size_bytes / 1e9) / .data[[TIME_METRIC]],
    SortType = if_else(str_detect(`param:tensor_cls`, "Unsorted"), "Unsorted", "Sorted"),
    CompileStatus = if_else(`param:torch_compile`, "Compiled", "Not Compiled"),
    # Create a combined group for plotting lines correctly
    DisplayType = factor(paste(SortType, CompileStatus, sep = " - ")),
    pred_name = recode(pred, !!!predicate_map)
  )

# --- 2. Plotting Function ---
create_aggregated_plot <- function(df) {
  df <- df %>% mutate(`param:scale` = as.factor(`param:scale`))

  # --- MODIFIED: Conditionally define colors based on GRAY_OUT_NON_COMPILED ---
  if (GRAY_OUT_NON_COMPILED) {
    plot_colors <- c(
      "Sorted - Compiled" = "#f8766d", "Unsorted - Compiled" = "#00bfc4",
      "Sorted - Not Compiled" = "grey70", "Unsorted - Not Compiled" = "grey40"
    )
  } else {
    # Use the same base colors for a direct line comparison
    plot_colors <- c(
      "Sorted - Compiled" = "#f8766d", "Unsorted - Compiled" = "#00bfc4",
      "Sorted - Not Compiled" = "#f8766d", "Unsorted - Not Compiled" = "#00bfc4"
    )
  }
  
  # Shapes always differ to distinguish compile status
  plot_shapes <- c(
    "Sorted - Compiled" = 17, "Unsorted - Compiled" = 17, # Square for compiled
    "Sorted - Not Compiled" = 16, "Unsorted - Not Compiled" = 16  # Circle for not-compiled
  )
  
  p <- ggplot(df, aes(x = `param:scale`, y = throughput_gb_per_sec, color = DisplayType, shape = DisplayType, group = DisplayType)) +
    geom_line(linewidth = 1) +
    geom_point(size = 3)

  if (SHOW_THEORETICAL_BANDWIDTH) {
    p <- p + geom_hline(
      yintercept = THEORETICAL_GPU_BANDWIDTH / 1e9, 
      linetype = "dashed", 
      color = "#afabab", 
      linewidth = 1
    )
  }

  p <- p +
    scale_x_discrete(breaks = as.character(X_AXIS_BREAKS)) +
    scale_color_manual(name = "Implementation", values = plot_colors) +
    scale_shape_manual(name = "Implementation", values = plot_shapes) +
    guides(
      color = guide_legend(ncol = 2),
      shape = guide_legend(ncol = 2)
    ) +
    labs(
      x = "TPC-H Scale Factor",
      y = "Throughput (GB/s)"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", hjust = TITLE_HJUST, size = TITLE_SIZE),
      plot.subtitle = element_text(hjust = SUBTITLE_HJUST, size = SUBTITLE_SIZE),
      legend.position = "bottom",
      legend.box = "vertical",
      legend.margin = LEGEND_MARGIN_VAL,
      legend.title = element_text(size = LEGEND_TITLE_SIZE),
      legend.text = element_text(size = LEGEND_TEXT_SIZE),
      axis.text = element_text(size = AXIS_TEXT_SIZE),
      axis.title = element_text(size = AXIS_TITLE_SIZE),
      strip.text = element_text(size = STRIP_TEXT_SIZE),
      panel.spacing = PANEL_SPACING
    )
  
  return(p)
}

# --- 3. Generate and Save Plot ---
dir.create(output_dir, showWarnings = FALSE)

if (nrow(plot_data) > 0) {
  nonzero_label <- if (USE_MASK) "Return Mask" else "With Nonzero"
  # --- MODIFIED: Move compile info to subtitle ---
  subtitle_text <- paste("Operation:", nonzero_label, "| Compiled vs. Not-Compiled")

  p <- create_aggregated_plot(plot_data) +
    facet_wrap(~pred_name, ncol = length(PREDICATES_TO_PLOT)) +
    labs(
      # --- MODIFIED: Shortened main title ---
      title = "TPC-H Throughput for Row-Based Dictionary Encodings",
      subtitle = subtitle_text
    )

  filename_suffix_mask <- if (USE_MASK) "_Mask" else "_Nonzero"
  # --- MODIFIED: Add suffix for color style ---
  filename_suffix_color <- if (GRAY_OUT_NON_COMPILED) "_GrayOut" else "_FullColor"
  # --- MODIFIED: New filename for the merged compile plot ---
  output_filename <- file.path(output_dir, paste0("04_aggregated_TPCH_Dict_RowOnly_MergedCompile", filename_suffix_mask, filename_suffix_color, ".png"))
  
  ggsave(output_filename, p, width = 14, height = 7, dpi = 300, bg = BACKGROUND_STYLE)
  print(paste("Aggregated plot saved to:", output_filename))
} else {
  print("No data available for the specified configuration. No plot generated.")
}