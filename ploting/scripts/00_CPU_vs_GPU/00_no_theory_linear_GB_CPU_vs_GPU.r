# --- 0. Setup: Load Libraries and Configure Parameters ---
# install.packages(c("ggplot2", "dplyr", "readr", "stringr"))
# source("00_no_theory_linear_GB_CPU_vs_GPU.r")

# Install necessary packages if they are not already installed
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("dplyr")) install.packages("dplyr")
if (!require("readr")) install.packages("readr")
if (!require("stringr")) install.packages("stringr")

library(ggplot2)
library(dplyr)
library(readr)
library(stringr)

# --- Configuration ---
# File Paths
input_file <- "0011_allPrefix.csv"
output_dir <- "plots"

# Time metric to use for throughput calculation ("mean", "min", or "max")
TIME_METRIC <- "mean"

# Plotting style: "facet" for a single plot with facets, or "separate" for one plot per file.
PLOT_STYLE <- "facet"

# Background style for saved plots: "white" or "transparent"
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

# Prepare the data for plotting (theoretical bandwidth calculation is removed)
plot_data <- data %>%
  mutate(
    throughput_gb_per_sec = (total_size_bytes / .data[[TIME_METRIC]]) / 1e9
  ) %>%
  mutate(
    encoding_type = str_replace(`param:tensor_cls`, "StringColumnTensor", ""),
    encoding_type = factor(encoding_type, levels = encoding_order),
    Device = factor(toupper(`param:device`)),
    pred_name = recode(pred, !!!predicate_map)
  )

# --- 2. Plotting Function (Simplified) ---
create_throughput_plot_no_theory <- function(df) {
  df <- df %>% mutate(`param:scale` = as.factor(`param:scale`))

  ggplot(df, aes(x = `param:scale`, y = throughput_gb_per_sec, color = Device, group = Device)) +
    geom_line(linewidth = 1) +
    geom_point(size = 3, aes(shape = Device)) +
    scale_y_continuous(breaks = scales::pretty_breaks(n = 8)) +
    scale_color_manual(name = "Device", values = c("CPU" = "#0072B2", "CUDA" = "#D55E00")) +
    scale_shape_manual(name = "Device", values = c("CPU" = 16, "CUDA" = 17)) +
    labs(
      title = "Measured Throughput for String Operations",
      subtitle = paste("Using", TIME_METRIC, "time for calculation"),
      x = "TPC-H Scale Factor",
      y = "Throughput (GB/s)"
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
    p <- create_throughput_plot_no_theory(subset_data) +
      facet_wrap(~encoding_type, scales = "fixed", ncol = 2) +
      labs(
        title = paste("Throughput for Predicate:", descriptive_pred_name),
        caption = "Each panel represents a different string encoding algorithm."
      )
    output_filename <- file.path(output_dir, paste0("00_no_theory_linear_GB_TPCH_Faceted_", descriptive_pred_name, ".png"))
    ggsave(output_filename, p, width = 12, height = 8, dpi = 300, bg = BACKGROUND_STYLE)
    print(paste("Plot saved to:", output_filename))
  }
}