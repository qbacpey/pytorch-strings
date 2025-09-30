# --- 0. Setup: Load Libraries and Configure Parameters ---
# This script generates a plot showing ONLY CUDA performance.

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
# --- Arguments are now passed from the command line ---
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 4) {
  stop("Usage: Rscript 00_linear_GB_GPU_only_tpch.r <input_file> <use_mask> <use_torch_compile> <predicate>", call. = FALSE)
}

input_file <- args[1]
USE_MASK <- as.logical(args[2])
USE_TORCH_COMPILE <- as.logical(args[3])
PREDICATE_TO_PLOT <- args[4]

output_dir <- "plots_GPU_only"

# --- In-script plotting options ---
SHOW_THEORETICAL_BANDWIDTH <- TRUE
SHOW_COMPILE <- TRUE

# --- Title and Subtitle Position ---
TITLE_HJUST <- 0
SUBTITLE_HJUST <- 0

# --- Title, Subtitle, and Legend Styling ---
TITLE_SIZE <- 22
SUBTITLE_SIZE <- 15
LEGEND_MARGIN_VAL <- margin(t = -10, unit = "pt")
LEGEND_TITLE_SIZE <- 16
LEGEND_TEXT_SIZE <- 14
AXIS_TEXT_SIZE <- 15
AXIS_TITLE_SIZE <- 18
STRIP_TEXT_SIZE <- 16

# --- X-axis label configuration ---
X_AXIS_BREAKS <- c(1, 50, 100, 150, 200)

# Theoretical Memory Bandwidth (in bytes per second)
THEORETICAL_GPU_BANDWIDTH <- 1.1e12 # 1.1 TB/s

# Time metric to use for throughput calculation
TIME_METRIC <- "mean"
BACKGROUND_STYLE <- "transparent"

# --- 1. Data Loading and Preparation ---
data <- read_csv(input_file, show_col_types = FALSE)

predicate_map <- c("Eq" = "Equal", "Lt" = "Less_Than", "Prefix" = "Prefix_Match")
encoding_name_map <- c(
  "PlainEncoding" = "Row Plain", "CPlainEncoding" = "Column Plain",
  "UnsortedDictionaryEncoding" = "Unsorted Row Dictionary", "UnsortedCDictionaryEncoding" = "Unsorted Column Dictionary",
  "DictionaryEncoding" = "Sorted Row Dictionary", "CDictionaryEncoding" = "Sorted Column Dictionary"
)
encoding_order <- c(
  "PlainEncoding", "CPlainEncoding", "UnsortedDictionaryEncoding",
  "UnsortedCDictionaryEncoding", "DictionaryEncoding", "CDictionaryEncoding"
)

# --- MODIFIED: Filter data for CUDA ONLY ---
plot_data <- data %>%
  filter(
    pred == PREDICATE_TO_PLOT,
    `param:return_mask` == USE_MASK,
    `param:torch_compile` == USE_TORCH_COMPILE,
    `param:device` == "cuda" # Filter for GPU data
  ) %>%
  mutate(
    throughput_gb_per_sec = (total_size_bytes / 1e9) / .data[[TIME_METRIC]],
    theoretical_throughput_gb_per_sec = THEORETICAL_GPU_BANDWIDTH / 1e9,
    encoding_type = str_replace(`param:tensor_cls`, "StringColumnTensor", ""),
    encoding_type = factor(encoding_type, levels = encoding_order),
    encoding_type = recode(encoding_type, !!!encoding_name_map),
    pred_name = recode(pred, !!!predicate_map)
  )

# --- 2. Plotting Function (Linear Scale) ---
create_throughput_plot <- function(df) {
  df <- df %>% mutate(`param:scale` = as.factor(`param:scale`))

  # --- MODIFIED: Aesthetics are no longer mapped to Device ---
  p <- ggplot(df, aes(x = `param:scale`, y = throughput_gb_per_sec, group = 1)) +
    geom_line(linewidth = 1, color = "#D55E00") +
    geom_point(size = 3, shape = 17, color = "#D55E00")

  # --- MODIFIED: Only add GPU theoretical bandwidth line ---
  if (SHOW_THEORETICAL_BANDWIDTH) {
    p <- p + geom_line(
      aes(y = theoretical_throughput_gb_per_sec, linetype = "Theoretical Bandwidth (DGX)"),
      linewidth = 1
    ) +
    scale_linetype_manual(
      name = NULL, 
      values = c("Theoretical Bandwidth (DGX)" = "dashed"),
      labels = c(paste0("CUDA Theoretical (", THEORETICAL_GPU_BANDWIDTH / 1e9, " GB/s)"))
    )
  }

  if (!is.null(X_AXIS_BREAKS)) {
    p <- p + scale_x_discrete(breaks = as.character(X_AXIS_BREAKS))
  }

  p <- p + labs(
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
      strip.text = element_text(size = STRIP_TEXT_SIZE)
    )
  
  return(p)
}

# --- 3. Generate and Save Plot ---
dir.create(output_dir, showWarnings = FALSE)

if (nrow(plot_data) > 0) {
  descriptive_pred_name <- unique(plot_data$pred_name)
  nonzero_label <- if (USE_MASK) "Return Mask" else "With Nonzero"
  compile_label <- if (USE_TORCH_COMPILE) "Compiled" else "Not Compiled"
  
  subtitle_text <- paste("Operation:", nonzero_label)
  if (SHOW_COMPILE) {
    subtitle_text <- paste(subtitle_text, "| Torch Compile:", compile_label)
  }

  p <- create_throughput_plot(plot_data) +
    facet_wrap(~encoding_type, scales = "fixed", ncol = 2) +
    labs(
      # --- MODIFIED: Updated title for GPU-only plot ---
      title = paste("TPC-H GPU Throughput for all Encodings (Predicate: ", descriptive_pred_name, ")", sep = ""),
      subtitle = subtitle_text,
      caption = "Each panel represents a different string encoding algorithm."
    )

  # --- MODIFIED: Updated filename for GPU-only plot ---
  filename_suffix_mask <- if (USE_MASK) "_Mask" else "_Nonzero"
  filename_suffix_compile <- if (USE_TORCH_COMPILE) "_Compiled" else "_NoCompile"
  output_filename <- file.path(output_dir, paste0("00_linear_GB_TPCH_GPU_Only_", descriptive_pred_name, filename_suffix_mask, filename_suffix_compile, ".png"))
  
  ggsave(output_filename, p, width = 12, height = 8, dpi = 300, bg = BACKGROUND_STYLE)
  print(paste("GPU-only plot saved to:", output_filename))
} else {
  print("No data available for the specified configuration. No plot generated.")
}