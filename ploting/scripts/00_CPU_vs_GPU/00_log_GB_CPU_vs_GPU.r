# --- 0. Setup: Load Libraries and Configure Parameters ---
# install.packages(c("ggplot2", "dplyr", "readr", "stringr", "scales"))
# source("00_log_GB_CPU_vs_GPU.r")

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
# File Paths
input_file <- "0011_allPrefix.csv"
output_dir <- "plots"

# Theoretical Memory Bandwidth (in bytes per second)
THEORETICAL_GPU_BANDWIDTH <- 1.6e12 # 1.6 TB/s
THEORETICAL_CPU_BANDWIDTH <- 60e9  # 60 GB/s

# Time metric to use for throughput calculation
TIME_METRIC <- "mean"
PLOT_STYLE <- "facet"
BACKGROUND_STYLE <- "white"

# --- 1. Data Loading and Preparation ---
data <- read_csv(input_file, show_col_types = FALSE)

predicate_map <- c(
  "Eq" = "Equal",
  "Lt" = "Less_Than",
  "Prefix" = "Prefix_Match"
)

encoding_order <- c(
  "PlainEncoding", "CPlainEncoding", "UnsortedDictionaryEncoding",
  "UnsortedCDictionaryEncoding", "DictionaryEncoding", "CDictionaryEncoding"
)

plot_data <- data %>%
  mutate(
    throughput_gb_per_sec = (total_size_bytes / .data[[TIME_METRIC]]) / 1e9
  ) %>%
  mutate(
    theoretical_throughput_gb_per_sec = case_when(
      `param:device` == "cpu" ~ THEORETICAL_CPU_BANDWIDTH / 1e9,
      `param:device` == "cuda" ~ THEORETICAL_GPU_BANDWIDTH / 1e9
    )
  ) %>%
  mutate(
    encoding_type = str_replace(`param:tensor_cls`, "StringColumnTensor", ""),
    encoding_type = factor(encoding_type, levels = encoding_order),
    Device = factor(toupper(`param:device`)),
    pred_name = recode(pred, !!!predicate_map)
  )

# --- 2. Plotting Function ---
create_throughput_plot_log <- function(df) {
  df <- df %>% mutate(`param:scale` = as.factor(`param:scale`))

  # Create descriptive labels for the legend
  legend_labels <- c(
    "CPU" = paste0("CPU (", THEORETICAL_CPU_BANDWIDTH / 1e9, " GB/s)"),
    "CUDA" = paste0("CUDA (", THEORETICAL_GPU_BANDWIDTH / 1e9, " GB/s)")
  )

  ggplot(df, aes(x = `param:scale`, y = throughput_gb_per_sec, color = Device, group = Device)) +
    # --- Measured Performance (colored by Device) ---
    geom_line(linewidth = 1) +
    geom_point(size = 3, aes(shape = Device)) +

    # --- Theoretical Performance (colored by Device, with updated legend) ---
    geom_line(
      aes(y = theoretical_throughput_gb_per_sec, linetype = "Theoretical Bandwidth (DGX)"),
      # The 'color = Device' aesthetic is now inherited from the main ggplot() call
      linewidth = 1
    ) +

    # --- Y-Axis changed to Logarithmic Scale ---
    scale_y_log10(
      labels = trans_format("log10", math_format(10^.x)),
      breaks = trans_breaks("log10", function(x) 10^x, n = 8)
    ) +
    # --- Updated scales to include bandwidth in the legend labels ---
    scale_color_manual(
      name = "Device (Theoretical Bandwidth)",
      values = c("CPU" = "#0072B2", "CUDA" = "#D55E00"),
      labels = legend_labels
    ) +
    scale_shape_manual(
      name = "Device (Theoretical Bandwidth)",
      values = c("CPU" = 16, "CUDA" = 17),
      labels = legend_labels
    ) +

    # --- Updated linetype scale to match the new description ---
    scale_linetype_manual(name = "Line Type", values = c("Theoretical Bandwidth (DGX)" = "dashed")) +

    labs(
      title = "Measured vs. Theoretical Throughput (Log Scale)",
      subtitle = paste("Using", TIME_METRIC, "time for calculation"),
      x = "TPC-H Scale Factor",
      y = "Throughput (GB/s)",
      # Remove color and shape from labs() as they are now set in the scales
      linetype = "Line Type"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold"),
      legend.position = "bottom",
      legend.box = "vertical"
    )
}


# --- 3. Generate and Save Plots ---
dir.create(output_dir, showWarnings = FALSE)

if (PLOT_STYLE == "facet") {
  predicate_types <- unique(plot_data$pred)
  for (pred_type in predicate_types) {
    subset_data <- plot_data %>% filter(pred == pred_type)
    descriptive_pred_name <- unique(subset_data$pred_name)
    p <- create_throughput_plot_log(subset_data) +
      facet_wrap(~encoding_type, scales = "fixed", ncol = 2) +
      labs(
        title = paste("Throughput for Predicate:", descriptive_pred_name, "(Log Scale)"),
        caption = "Each panel represents a different string encoding algorithm."
      )
    output_filename <- file.path(output_dir, paste0("00_log_GB_TPCH_Faceted_", descriptive_pred_name, ".png"))
    ggsave(output_filename, p, width = 12, height = 8, dpi = 300, bg = BACKGROUND_STYLE)
    print(paste("Log-scale plot saved to:", output_filename))
  }
}