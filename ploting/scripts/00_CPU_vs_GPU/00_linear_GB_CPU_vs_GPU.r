# --- 0. Setup: Load Libraries and Configure Parameters ---
# install.packages(c("ggplot2", "dplyr", "readr", "stringr"))
# source("00_linear_GB_CPU_vs_GPU.r")

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
# You can modify these parameters to change the plot's behavior.

# File Paths
input_file <- "0011_allPrefix.csv" # Path to the input CSV file
output_dir <- "plots"

# Theoretical Memory Bandwidth (in bytes per second)
# HBM on A100/H100 is ~1.6-2.0 TB/s.
# DDR5 on modern CPUs is ~60-80 GB/s.
THEORETICAL_GPU_BANDWIDTH <- 1.6e12 # 1.6 TB/s
THEORETICAL_CPU_BANDWIDTH <- 60e9 # 60 GB/s

# Time metric to use for throughput calculation ("mean", "min", or "max")
TIME_METRIC <- "mean"

# Plotting style: "facet" for a single plot with facets, or "separate" for one plot per file.
PLOT_STYLE <- "facet"

# Background style for saved plots: "white" or "transparent"
BACKGROUND_STYLE <- "white" 


# --- 1. Data Loading and Preparation ---

# Read the joined benchmark data
data <- read_csv(input_file, show_col_types = FALSE)

# Define a lookup table for descriptive predicate names
predicate_map <- c(
  "Eq" = "Equal",
  "Lt" = "Less_Than",
  "Prefix" = "Prefix_Match"
)

# Define the desired order for the encoding facets
encoding_order <- c(
  "PlainEncoding",
  "CPlainEncoding",
  "UnsortedDictionaryEncoding",
  "UnsortedCDictionaryEncoding",
  "DictionaryEncoding",
  "CDictionaryEncoding"
)


# Prepare the data for plotting
plot_data <- data %>%
  # Calculate measured throughput in GB/s
  mutate(
    throughput_gb_per_sec = (total_size_bytes / .data[[TIME_METRIC]]) / 1e9
  ) %>%
  # Define theoretical throughput in GB/s
  mutate(
    theoretical_throughput_gb_per_sec = case_when(
      `param:device` == "cpu" ~ THEORETICAL_CPU_BANDWIDTH / 1e9,
      `param:device` == "cuda" ~ THEORETICAL_GPU_BANDWIDTH / 1e9
    )
  ) %>%
  # Clean up encoding names for better plot titles
  mutate(
    encoding_type = str_replace(`param:tensor_cls`, "StringColumnTensor", "")
  ) %>%
  # Convert encoding_type to a factor with a specific order for plotting
  mutate(
    encoding_type = factor(encoding_type, levels = encoding_order)
  ) %>%
  # Convert device to a factor for consistent coloring
  mutate(
    Device = factor(toupper(`param:device`))
  ) %>%
  # Add a new column with descriptive predicate names using the lookup table
  mutate(
    pred_name = recode(pred, !!!predicate_map)
  )

# Validate that all predicates were successfully mapped
unmapped_preds <- plot_data %>%
  filter(!pred %in% names(predicate_map)) %>%
  distinct(pred)

print(paste("Unmapped predicates found:", unmapped_preds))

if (nrow(unmapped_preds) > 0) {
  stop(paste("Unexpected predicate type(s) found:", paste(unmapped_preds$pred, collapse = ", ")))
}


# --- 2. Plotting Function ---

# A reusable function to generate the plot structure
create_throughput_plot <- function(df) {
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

    # --- Theoretical Performance (colored by Device) ---
    geom_line(
      aes(y = theoretical_throughput_gb_per_sec, linetype = "Theoretical Bandwidth (DGX)"),
      linewidth = 1
    ) +

    # --- Scales and Labels ---
    scale_y_continuous(breaks = scales::pretty_breaks(n = 8)) +
    # Updated scales to include bandwidth in the legend labels
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
    # Updated linetype scale to match the new description
    scale_linetype_manual(name = "Line Type", values = c("Theoretical Bandwidth (DGX)" = "dashed")) +

    # --- Titles and Theme ---
    labs(
      title = "Measured vs. Theoretical Throughput for String Operations",
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
      legend.box = "vertical" # Helps organize multiple legends
    )
}
# --- 3. Generate and Save Plots ---

# Create the output directory if it doesn't exist
dir.create(output_dir, showWarnings = FALSE)


if (PLOT_STYLE == "facet") {
  # Get a list of all unique predicate short names
  predicate_types <- unique(plot_data$pred)

  # Loop through each predicate type and create a separate plot file for it.
  for (pred_type in predicate_types) {
    # Filter data for the current predicate
    subset_data <- plot_data %>% filter(pred == pred_type)

    # Get the descriptive name for the title
    descriptive_pred_name <- unique(subset_data$pred_name)

    # Generate the plot for the current predicate
    p <- create_throughput_plot(subset_data) +
      facet_wrap(~encoding_type, scales = "fixed", ncol = 2) +
      labs(
        title = paste("Throughput for Predicate:", descriptive_pred_name),
        caption = "Each panel represents a different string encoding algorithm."
      )

    # Save the faceted plot with a unique name for the predicate
    output_filename <- file.path(output_dir, paste0("00_linear_GB_TPCH_Faceted_", descriptive_pred_name, ".png"))
    ggsave(output_filename, p, width = 12, height = 8, dpi = 300, bg = BACKGROUND_STYLE)
    print(paste("Faceted plot saved to:", output_filename))
  }
} else if (PLOT_STYLE == "separate") {
  # Get unique combinations of encoding and predicate
  plot_combinations <- plot_data %>%
    distinct(encoding_type, pred, pred_name)

  # Generate a separate plot for each combination
  for (i in 1:nrow(plot_combinations)) {
    enc_type <- plot_combinations$encoding_type[i]
    pred_short_name <- plot_combinations$pred[i]
    pred_long_name <- plot_combinations$pred_name[i]

    # Filter data for the current combination
    subset_data <- plot_data %>%
      filter(encoding_type == enc_type, pred == pred_short_name)

    # Create the plot for the subset
    p <- create_throughput_plot(subset_data) +
      labs(
        title = paste("Throughput for Predicate:", pred_long_name),
        subtitle = paste("Encoding:", enc_type, "| Using", TIME_METRIC, "time")
      )

    # Save the individual plot with a more descriptive name
    output_filename <- file.path(output_dir, paste0("01_GB_TPCH_Separate_", pred_long_name, "_", enc_type, ".png"))
    ggsave(output_filename, p, width = 10, height = 6, dpi = 300, bg = BACKGROUND_STYLE)
    print(paste("Faceted plot saved to:", output_filename))
  }
}