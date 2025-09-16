# --- 0. Setup: Load Libraries and Configure Parameters ---
#
# This script generates a boxplot of throughput for different encodings
# based on a specific predicate, mask, and compile configuration.
#
# Usage:
# Rscript 01_boxplot_mssb.r <input_file> <use_mask> <use_torch_compile> <predicate>
# Example:
# Rscript 01_boxplot_mssb.r 20250913_mssb_all_runs.csv TRUE TRUE Eq

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

if (length(args) != 4) {
  stop("Usage: Rscript 01_boxplot_mssb.r <input_file> <use_mask> <use_torch_compile> <predicate>", call. = FALSE)
}

input_file <- args[1]
USE_MASK <- as.logical(args[2])
USE_TORCH_COMPILE <- as.logical(args[3])
PREDICATE_TO_PLOT <- args[4]

output_dir <- "plots"

# 0 = left, 0.5 = center, 1 = right
TITLE_HJUST <- 0
SUBTITLE_HJUST <- 0
TITLE_SIZE <- 22
SUBTITLE_SIZE <- 15
LEGEND_SPACING_Y <- unit(0, "cm")
SHOW_LEGEND <- FALSE

# Theoretical Memory Bandwidth (in bytes per second)
THEORETICAL_GPU_BANDWIDTH <- 1.1e12 # 1.1 TB/s

# Time metric to use for throughput calculation
TIME_METRIC <- "mean"
BACKGROUND_STYLE <- "white"

# --- Define the desired order of encodings for the plot ---
encoding_order <- c(
  "PlainEncoding", "CPlainEncoding", "UnsortedDictionaryEncoding",
  "UnsortedCDictionaryEncoding", "DictionaryEncoding", "CDictionaryEncoding"
)

encoding_name_map <- c(
  "PlainEncoding" = "Row Plain",
  "CPlainEncoding" = "Column Plain",
  "UnsortedDictionaryEncoding" = "Unsorted Row Dict",
  "UnsortedCDictionaryEncoding" = "Unsorted Column Dict",
  "DictionaryEncoding" = "Sorted Row Dict",
  "CDictionaryEncoding" = "Sorted Column Dict"
)

predicate_map <- c(
  "Eq" = "Equal",
  "Lt" = "Less_Than",
  "Prefix" = "Prefix_Match"
)

# --- 1. Data Loading and Preparation ---
data <- read_csv(input_file, show_col_types = FALSE)

# Filter data based on configuration and calculate throughput
plot_data <- data %>%
  filter(
    `param:device` == "cuda", # Only show CUDA results
    pred == PREDICATE_TO_PLOT,
    `param:return_mask` == USE_MASK,
    `param:torch_compile` == USE_TORCH_COMPILE
  ) %>%
  mutate(
    throughput_gb_per_sec = (total_size_bytes / .data[[TIME_METRIC]]) / 1e9,
    encoding_type = str_replace(`param:tensor_cls`, "StringColumnTensor", ""),
    encoding_type = factor(encoding_type, levels = encoding_order),
    encoding_type = recode(encoding_type, !!!encoding_name_map),
    pred_name = recode(pred, !!!predicate_map)
  )

# --- 2. Generate and Save Plot ---
dir.create(output_dir, showWarnings = FALSE)

if (nrow(plot_data) > 0) {
  # Get descriptive names for titles and filenames
  descriptive_pred_name <- unique(plot_data$pred_name)
  nonzero_label <- if (USE_MASK) "Return Mask" else "With Nonzero"
  compile_label <- if (USE_TORCH_COMPILE) "Compiled" else "Not Compiled"

  # Create the plot
  p <- ggplot(plot_data, aes(x = encoding_type, y = throughput_gb_per_sec, fill = encoding_type)) +
    geom_boxplot() +
    # --- MODIFIED: Use a fixed string for the linetype aesthetic ---
    geom_hline(
      aes(yintercept = THEORETICAL_GPU_BANDWIDTH / 1e9, linetype = "Theoretical Bandwidth (DGX)"),
      color = "#D55E00", linewidth = 1
    ) +
    # --- MODIFIED: Define the linetype and its legend entry ---
    scale_linetype_manual(
      name = "Line Type",
      values = c("Theoretical Bandwidth (DGX)" = "dashed"),
      labels = c(paste("Theoretical Max:", round(THEORETICAL_GPU_BANDWIDTH / 1e9), "GB/s"))
    ) +
    # --- NEW: Explicitly define the fill legend for encodings ---
    scale_fill_discrete(name = "Encoding Type") +
    # --- MODIFIED: Ensure all legends are shown and arranged vertically ---
    guides(
      fill = guide_legend(title.position = "left", ncol = 7),
      linetype = guide_legend(title.position = "left", ncol = 2)
    ) +
    labs(
      title = paste("MSSB Throughput Distribution for Predicate:", descriptive_pred_name),
      subtitle = paste("Operation:", nonzero_label, "| Torch Compile:", compile_label),
      x = "Encoding Type",
      y = "Throughput (GB/s)"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", hjust = TITLE_HJUST, size = TITLE_SIZE),
      plot.subtitle = element_text(hjust = SUBTITLE_HJUST, size = SUBTITLE_SIZE),
      axis.text.x = element_text(angle = 0, hjust = 0.5), # Rotate labels to prevent overlap
      legend.position = if (SHOW_LEGEND) "bottom" else "none",
      legend.box = "vertical",
      legend.spacing.y = LEGEND_SPACING_Y
    )

  # Generate a descriptive filename
  filename_suffix_mask <- if (USE_MASK) "_Mask" else "_Nonzero"
  filename_suffix_compile <- if (USE_TORCH_COMPILE) "_Compiled" else "_NoCompile"
  output_filename <- file.path(output_dir, paste0("01_boxplot_MSSB_", descriptive_pred_name, filename_suffix_mask, filename_suffix_compile, ".png"))
  
  ggsave(output_filename, p, width = 12, height = 8, dpi = 300, bg = BACKGROUND_STYLE)
  print(paste("Boxplot saved to:", output_filename))

} else {
  print(paste("No data available for configuration: Predicate=", PREDICATE_TO_PLOT, ", Mask=", USE_MASK, ", Compile=", USE_TORCH_COMPILE))
}