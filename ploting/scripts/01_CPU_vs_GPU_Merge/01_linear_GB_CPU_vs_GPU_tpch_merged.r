# --- 0. Setup: Load Libraries and Configure Parameters ---
# This script generates a plot merging "Mask" and "Nonzero" implementations.

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
# --- MODIFIED: Arguments are now passed from the command line (3 args) ---
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 3) {
  stop("Usage: Rscript 00_linear_GB_CPU_vs_GPU_tpch_merged.r <input_file> <use_torch_compile> <predicate>", call. = FALSE)
}

input_file <- args[1]
USE_TORCH_COMPILE <- as.logical(args[2])
PREDICATE_TO_PLOT <- args[3]

output_dir <- "plots"

# --- Plotting options ---
SHOW_THEORETICAL_BANDWIDTH <- TRUE
TITLE_HJUST <- 0
SUBTITLE_HJUST <- 0
TITLE_SIZE <- 22
SUBTITLE_SIZE <- 15
LEGEND_MARGIN_VAL <- margin(t = -10, unit = "pt")
X_AXIS_BREAKS <- c(1, 10, 50, 100, 150, 200)

PANEL_SPACING <- unit(1, "lines")

# --- NEW: Color for the 'Nonzero' implementation ---
NONZERO_COLOR <- "grey60"
MASK_COLOR <- "#D55E00" # Standard CUDA color

# Theoretical Memory Bandwidth (in bytes per second)
THEORETICAL_GPU_BANDWIDTH <- 1.1e12 # 1.1 TB/s
THEORETICAL_CPU_BANDWIDTH <- 60e9 # 60 GB/s

TIME_METRIC <- "mean"
BACKGROUND_STYLE <- "transparent"

# --- 1. Data Loading and Preparation ---
data <- read_csv(input_file, show_col_types = FALSE)

predicate_map <- c("Eq" = "Equal", "Lt" = "Less_Than", "Prefix" = "Prefix_Match")
encoding_name_map <- c("PlainEncoding" = "Row Plain", "CPlainEncoding" = "Column Plain", "UnsortedDictionaryEncoding" = "Unsorted Row Dict", "UnsortedCDictionaryEncoding" = "Unsorted Column Dict", "DictionaryEncoding" = "Sorted Row Dict", "CDictionaryEncoding" = "Sorted Column Dict")
encoding_order <- c("PlainEncoding", "CPlainEncoding", "UnsortedDictionaryEncoding", "UnsortedCDictionaryEncoding", "DictionaryEncoding", "CDictionaryEncoding")

# --- MODIFIED: Filter data and create new grouping columns ---
plot_data <- data %>%
  filter(
    pred == PREDICATE_TO_PLOT,
    `param:torch_compile` == USE_TORCH_COMPILE
  ) %>%
  mutate(
    throughput_gb_per_sec = (total_size_bytes / 1e9) / .data[[TIME_METRIC]],
    Implementation = if_else(`param:return_mask`, "Mask", "Nonzero"),
    PlotGroup = case_when(
      `param:device` == "cuda" ~ paste("CUDA", Implementation, sep = " - "),
      TRUE ~ toupper(`param:device`)
    ),
    theoretical_throughput_gb_per_sec = case_when(
      `param:device` == "cpu" ~ THEORETICAL_CPU_BANDWIDTH / 1e9,
      `param:device` == "cuda" ~ THEORETICAL_GPU_BANDWIDTH / 1e9
    ),
    encoding_type = str_replace(`param:tensor_cls`, "StringColumnTensor", ""),
    encoding_type = factor(encoding_type, levels = encoding_order),
    encoding_type = recode(encoding_type, !!!encoding_name_map),
    pred_name = recode(pred, !!!predicate_map)
  )

# --- 2. Plotting Function (Linear Scale) ---
create_throughput_plot <- function(df) {
  df <- df %>% mutate(`param:scale` = as.factor(`param:scale`))

  # --- MODIFIED: Define aesthetics only for the items to be shown in the legend ---
  plot_colors <- c(
    "CUDA - Mask" = MASK_COLOR,
    "CUDA - Nonzero" = NONZERO_COLOR
  )
  plot_shapes <- c("CUDA - Mask" = 17, "CUDA - Nonzero" = 15)
  plot_labels <- c(
    "CUDA - Mask" = "CUDA - Mask",
    "CUDA - Nonzero" = "CUDA - Nonzero"
  )

  p <- ggplot(df, aes(x = `param:scale`)) +
    # --- MODIFIED: Plot measured CUDA data points ---
    geom_line(data = filter(df, `param:device` == "cuda"), aes(y = throughput_gb_per_sec, color = PlotGroup, group = PlotGroup), linewidth = 1) +
    geom_point(data = filter(df, `param:device` == "cuda"), aes(y = throughput_gb_per_sec, color = PlotGroup, shape = PlotGroup), size = 3)

  # --- MODIFIED: Add theoretical bandwidth lines without creating legend entries ---
  if (SHOW_THEORETICAL_BANDWIDTH) {
    p <- p +
      # Add CUDA theoretical line with fixed styling
      geom_hline(yintercept = THEORETICAL_GPU_BANDWIDTH / 1e9, color = "#D55E00", linetype = "dashed", linewidth = 1) +
      # Add CPU theoretical line with fixed styling
      geom_hline(yintercept = THEORETICAL_CPU_BANDWIDTH / 1e9, color = "#0072B2", linetype = "dashed", linewidth = 1)
  }

  # --- MODIFIED: Simplify scales to only manage the CUDA implementations ---
  p <- p +
    scale_color_manual(name = "Implementation", values = plot_colors, labels = plot_labels) +
    scale_shape_manual(name = "Implementation", values = plot_shapes, labels = plot_labels)

  if (!is.null(X_AXIS_BREAKS)) {
    p <- p + scale_x_discrete(breaks = as.character(X_AXIS_BREAKS))
  }

  # --- MODIFIED: Set legend to have 2 columns for a horizontal layout ---
  p <- p + guides(
    color = guide_legend(ncol = 2),
    shape = guide_legend(ncol = 2)
  ) +
    labs(x = "TPC-H Scale Factor", y = "Throughput (GB/s)") +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", hjust = TITLE_HJUST, size = TITLE_SIZE),
      plot.subtitle = element_text(hjust = SUBTITLE_HJUST, size = SUBTITLE_SIZE),
      legend.position = "bottom",
      legend.box = "vertical",
      legend.margin = LEGEND_MARGIN_VAL,
      axis.text = element_text(size = 12),
      # --- NEW: Apply custom panel spacing ---
      panel.spacing = PANEL_SPACING
    )

  return(p)
}

# --- 3. Generate and Save Plot ---
dir.create(output_dir, showWarnings = FALSE)

if (nrow(plot_data) > 0) {
  descriptive_pred_name <- unique(plot_data$pred_name)
  subtitle_text <- paste("Operation: Mask & Nonzero")

  p <- create_throughput_plot(plot_data) +
    facet_wrap(~encoding_type, scales = "fixed", ncol = 2) +
    labs(
      title = paste("TPC-H Throughput for Predicate:", descriptive_pred_name),
      subtitle = subtitle_text,
      caption = "Each panel represents a different string encoding algorithm."
    )

  filename_suffix_compile <- if (USE_TORCH_COMPILE) "_Compiled" else "_NoCompile"
  output_filename <- file.path(output_dir, paste0("00_linear_GB_TPCH_Merged_", descriptive_pred_name, filename_suffix_compile, ".png"))

  ggsave(output_filename, p, width = 12, height = 8, dpi = 300, bg = BACKGROUND_STYLE)
  print(paste("Merged plot saved to:", output_filename))
} else {
  print("No data available for the specified configuration. No plot generated.")
}