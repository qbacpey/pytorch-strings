# --- 0. Setup: Load Libraries and Configure Parameters ---
# This script generates an aggregated plot comparing different dictionary
# encodings for specified TPC-H predicates.

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
# --- Arguments passed from the command line ---
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  stop("Usage: Rscript 02_aggregated_dict_tpch.r <input_file> <use_mask> <use_torch_compile>", call. = FALSE)
}

input_file <- args[1]
USE_MASK <- as.logical(args[2])
USE_TORCH_COMPILE <- as.logical(args[3])

PANEL_SPACING <- unit(2, "lines")

output_dir <- "plots"

# --- NEW: Select which predicates and encodings to plot ---
PREDICATES_TO_PLOT <- c("Eq", "Lt", "Prefix")
DICT_ENCODINGS_TO_PLOT <- c(
  "UnsortedDictionaryEncoding", "UnsortedCDictionaryEncoding", 
  "DictionaryEncoding", "CDictionaryEncoding"
)

# --- Plotting options ---
SHOW_THEORETICAL_BANDWIDTH <- TRUE
TITLE_HJUST <- 0.0
SUBTITLE_HJUST <- 0.0
TITLE_SIZE <- 22
SUBTITLE_SIZE <- 15
LEGEND_MARGIN_VAL <- margin(t = -10, unit = "pt")
AXIS_TEXT_SIZE <- 14
AXIS_TITLE_SIZE <- 16
X_AXIS_BREAKS <- c(1, 10, 50, 100, 150, 200)

# Theoretical Memory Bandwidth (in bytes per second)
THEORETICAL_GPU_BANDWIDTH <- 1.1e12 # 1.1 TB/s
TIME_METRIC <- "mean"
BACKGROUND_STYLE <- "white"

# --- 1. Data Loading and Preparation ---
data <- read_csv(input_file, show_col_types = FALSE)

predicate_map <- c("Eq" = "Equal", "Lt" = "Less Than", "Prefix" = "Prefix Match")
encoding_name_map <- c(
  "UnsortedDictionaryEncoding" = "Unsorted Row Dict",
  "UnsortedCDictionaryEncoding" = "Unsorted Column Dict",
  "DictionaryEncoding" = "Sorted Row Dict",
  "CDictionaryEncoding" = "Sorted Column Dict"
)

# --- MODIFIED: Filter for specific predicates, encodings, and CUDA only ---
plot_data <- data %>%
  filter(
    `param:device` == "cuda", # Only CUDA
    pred %in% PREDICATES_TO_PLOT,
    `param:tensor_cls` %in% DICT_ENCODINGS_TO_PLOT, # Only specified dict encodings
    `param:return_mask` == USE_MASK,
    `param:torch_compile` == USE_TORCH_COMPILE
  ) %>%
  mutate(
    throughput_gb_per_sec = (total_size_bytes / 1e9) / .data[[TIME_METRIC]],
    encoding_type_raw = str_replace(`param:tensor_cls`, "StringColumnTensor", ""),
    # --- NEW: Create columns for color (Sort) and shape (Implementation) ---
    SortType = if_else(str_detect(encoding_type_raw, "Unsorted"), "Unsorted", "Sorted"),
    ImplType = factor(
      if_else(str_detect(encoding_type_raw, "C"), "Column", "Row"),
      levels = c("Row", "Column")
    ),
    # Use the full encoding name for grouping lines correctly
    encoding_type = recode(encoding_type_raw, !!!encoding_name_map),
    pred_name = recode(pred, !!!predicate_map)
  )

# --- 2. Plotting Function ---
create_aggregated_plot <- function(df) {
  df <- df %>% mutate(`param:scale` = as.factor(`param:scale`))

  # --- MODIFIED: Map color to SortType and shape to ImplType ---
  p <- ggplot(df, aes(x = `param:scale`, y = throughput_gb_per_sec, color = SortType, shape = ImplType, group = encoding_type)) +
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
    scale_color_manual(name = "Sort Type", values = c("Sorted" = "#f8766d", "Unsorted" = "#00bfc4")) +
    scale_shape_manual(name = "Implementation", values = c("Row" = 17, "Column" = 15)) +
    # --- MODIFIED: Removed scale_linetype_manual as it's no longer needed ---
    guides(
      # --- MODIFIED: Set order to control legend stacking ---
      color = guide_legend(ncol = 2, order = 1),
      shape = guide_legend(ncol = 2, order = 2)
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
      axis.text = element_text(size = AXIS_TEXT_SIZE),
      axis.title = element_text(size = AXIS_TITLE_SIZE),
      panel.spacing = PANEL_SPACING
    )
  
  return(p)
}

# --- 3. Generate and Save Plot ---
dir.create(output_dir, showWarnings = FALSE)

if (nrow(plot_data) > 0) {
  nonzero_label <- if (USE_MASK) "Return Mask" else "With Nonzero"
  subtitle_text <- paste("Operation:", nonzero_label)

  p <- create_aggregated_plot(plot_data) +
    # --- MODIFIED: Facet by predicate name ---
    facet_wrap(~pred_name, ncol = length(PREDICATES_TO_PLOT)) +
    labs(
      title = "TPC-H Throughput for Dictionary Encodings",
      subtitle = subtitle_text
    )

  filename_suffix_mask <- if (USE_MASK) "_Mask" else "_Nonzero"
  filename_suffix_compile <- if (USE_TORCH_COMPILE) "_Compiled" else "_NoCompile"
  output_filename <- file.path(output_dir, paste0("02_aggregated_TPCH_Dict_", filename_suffix_mask, filename_suffix_compile, ".png"))
  
  ggsave(output_filename, p, width = 14, height = 7, dpi = 300, bg = BACKGROUND_STYLE)
  print(paste("Aggregated plot saved to:", output_filename))
} else {
  print("No data available for the specified configuration. No plot generated.")
}