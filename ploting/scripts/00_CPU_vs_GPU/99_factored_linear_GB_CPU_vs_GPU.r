# --- 0. Setup: Load Libraries and Configure Parameters ---
# install.packages(c("ggplot2", "dplyr", "readr", "stringr", "tidyr"))
# source("00_linear_GB_CPU_vs_GPU.r")

# Install necessary packages if they are not already installed
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("dplyr")) install.packages("dplyr")
if (!require("readr")) install.packages("readr")
if (!require("stringr")) install.packages("stringr")
if (!require("tidyr")) install.packages("tidyr") # For tribble

library(ggplot2)
library(dplyr)
library(readr)
library(stringr)
library(tidyr)

# --- Configuration ---

# File Paths
input_file <- "0009_allEq.csv"
output_dir <- "plots"

# --- NEW: Plotting Options ---
# Set to TRUE to generate separate plots for CPU and CUDA.
# Set to FALSE to plot both on the same graph.
SEPARATE_PLOTS_BY_DEVICE <- TRUE

# Theoretical Memory Bandwidth (in bytes per second)
THEORETICAL_GPU_BANDWIDTH <- 1.6e12 # 1.6 TB/s
THEORETICAL_CPU_BANDWIDTH <- 60e9  # 60 GB/s

STRING_LENGTH_BYTES <- 20  # Average string length in bytes (for theoretical calculations)

# Read TensorEq + Write TensorEq Mask + Read TensorEq Mask + Write Mask Result + Read Mask Result
PLAIN_EQ_BANDWIDTH_FACTOR <- 1 + 1 + 1 + 1 / 20 + 1 / 20 + (8 * 0.143 * (1 / 20))
DICT_EQ_BANDWIDTH_FACTOR <- 8 / 7 + (8 * 0.143 * (1 / 20))

# --- NEW: Fine-Grained Bandwidth Factors ---
# Define unique factors for each combination of encoding and predicate.
# The theoretical bandwidth will be divided by this factor.
# Add or modify rows as needed for your specific kernels.
factors_df <- tribble(
  ~encoding_type, ~pred, ~factor,
  # --- Plain Encodings (Example Factors) ---
  "PlainEncoding", "Eq", PLAIN_EQ_BANDWIDTH_FACTOR,
  "PlainEncoding", "Lt", 4.2,
  "PlainEncoding", "Prefix", 4.5,
  "CPlainEncoding", "Eq", PLAIN_EQ_BANDWIDTH_FACTOR,
  "CPlainEncoding", "Lt", 4.2,
  "CPlainEncoding", "Prefix", 4.5,
  # --- Unsorted Dictionary (Example Factors) ---
  "UnsortedDictionaryEncoding", "Eq", DICT_EQ_BANDWIDTH_FACTOR,
  "UnsortedDictionaryEncoding", "Lt", 2.1,
  "UnsortedDictionaryEncoding", "Prefix", 5.0,
  "UnsortedCDictionaryEncoding", "Eq", DICT_EQ_BANDWIDTH_FACTOR,
  "UnsortedCDictionaryEncoding", "Lt", 2.1,
  "UnsortedCDictionaryEncoding", "Prefix", 5.0,
  # --- Sorted Dictionary (Example Factors) ---
  "DictionaryEncoding", "Eq", DICT_EQ_BANDWIDTH_FACTOR,
  "DictionaryEncoding", "Lt", 2.1,
  "DictionaryEncoding", "Prefix", 5.0,
  "CDictionaryEncoding", "Eq", DICT_EQ_BANDWIDTH_FACTOR,
  "CDictionaryEncoding", "Lt", 2.1,
  "CDictionaryEncoding", "Prefix", 5.0
)

# Time metric to use for throughput calculation ("mean", "min", or "max")
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

# Prepare the data for plotting
plot_data <- data %>%
  # Clean up encoding names to match the factors table
  mutate(
    encoding_type = str_replace(`param:tensor_cls`, "StringColumnTensor", "")
  ) %>%
  # Join with the factors table to get the correct factor for each row
  left_join(factors_df, by = c("encoding_type", "pred")) %>%
  # Use a default factor of 1.0 if a combination is not in the table
  mutate(factor = coalesce(factor, 1.0)) %>%
  # Calculate measured throughput in GB/s
  mutate(
    throughput_gb_per_sec = (total_size_bytes / .data[[TIME_METRIC]]) / 1e9
  ) %>%
  # Define theoretical throughput using the new per-row factor
  mutate(
    theoretical_throughput_gb_per_sec = case_when(
      `param:device` == "cpu" ~ (THEORETICAL_CPU_BANDWIDTH / factor) / 1e9,
      `param:device` == "cuda" ~ (THEORETICAL_GPU_BANDWIDTH / factor) / 1e9
    )
  ) %>%
  mutate(
    encoding_type = factor(encoding_type, levels = encoding_order),
    Device = factor(toupper(`param:device`)),
    pred_name = recode(pred, !!!predicate_map)
  )

# --- 2. Plotting Function ---
create_throughput_plot <- function(df) {
  df <- df %>% mutate(`param:scale` = as.factor(`param:scale`))

  legend_labels <- c(
    "CPU" = paste0("CPU (", round(THEORETICAL_CPU_BANDWIDTH / 1e9), " GB/s)"),
    "CUDA" = paste0("CUDA (", round(THEORETICAL_GPU_BANDWIDTH / 1e9), " GB/s)")
  )

  ggplot(df, aes(x = `param:scale`, y = throughput_gb_per_sec, color = Device, group = Device)) +
    geom_line(linewidth = 1) +
    geom_point(size = 3, aes(shape = Device)) +
    geom_line(
      aes(y = theoretical_throughput_gb_per_sec, linetype = "Theoretical Bandwidth (Factored)"),
      linewidth = 1
    ) +
    scale_y_continuous(breaks = scales::pretty_breaks(n = 8)) +
    scale_color_manual(
      name = "Device (Peak Bandwidth)",
      values = c("CPU" = "#0072B2", "CUDA" = "#D55E00"),
      labels = legend_labels,
      drop = FALSE # Prevents dropping unused factor levels
    ) +
    scale_shape_manual(
      name = "Device (Peak Bandwidth)",
      values = c("CPU" = 16, "CUDA" = 17),
      labels = legend_labels,
      drop = FALSE
    ) +
    scale_linetype_manual(name = "Line Type", values = c("Theoretical Bandwidth (Factored)" = "dashed")) +
    labs(
      title = "Measured vs. Theoretical Throughput for String Operations",
      subtitle = paste("Using", TIME_METRIC, "time for calculation"),
      x = "TPC-H Scale Factor",
      y = "Throughput (GB/s)",
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

# Define devices to loop over. If not separating, this is just one "iteration" named "ALL".
devices_to_plot <- if (SEPARATE_PLOTS_BY_DEVICE) c("CPU", "CUDA") else "ALL"

for (device_name in devices_to_plot) {
  # Filter data for the current device if required
  device_plot_data <- if (device_name != "ALL") {
    plot_data %>% filter(Device == device_name)
  } else {
    plot_data
  }

  if (nrow(device_plot_data) == 0) {
    print(paste("Skipping", device_name, "- no data found."))
    next
  }

  if (PLOT_STYLE == "facet") {
    predicate_types <- unique(device_plot_data$pred)
    for (pred_type in predicate_types) {
      subset_data <- device_plot_data %>% filter(pred == pred_type)
      descriptive_pred_name <- unique(subset_data$pred_name)

      # --- NEW: Prepare data for y-axis labels ---
      # This creates a data frame with the theoretical max for each facet's y-axis label.
      y_axis_labels <- subset_data %>%
        group_by(encoding_type) %>%
        summarise(
          y_position = max(theoretical_throughput_gb_per_sec),
          .groups = 'drop'
        ) %>%
        mutate(label_text = paste("Theoretical Max:", round(y_position), "GB/s"))

      p <- create_throughput_plot(subset_data) +
        # --- MODIFIED: Use "free_y" to allow independent y-axes for each facet ---
        facet_wrap(~encoding_type, scales = "free_y", ncol = 2) +
        # --- NEW: Add text annotation for the theoretical maximum on the y-axis ---
        # This places a label at the top of the y-axis for each facet.
        # Note: We use the first scale factor for the x-position.
        geom_text(
          data = y_axis_labels,
          aes(x = 1, y = y_position * 1.15, label = label_text),
          hjust = 0, vjust = 1, size = 3, color = "black",
          inherit.aes = FALSE # Important to avoid inheriting aesthetics from the main plot
        ) +
        labs(
          title = paste("Throughput for Predicate:", descriptive_pred_name),
          subtitle = if (device_name != "ALL") paste("Device:", device_name) else "Devices: CPU & CUDA",
          caption = "Each panel represents a different string encoding algorithm. Y-axes are scaled independently."
        )

      # Create a filename that includes the device if plots are separated
      file_suffix <- if (device_name != "ALL") paste0("_", device_name) else ""
      output_filename <- file.path(output_dir, paste0("00_linear_GB_TPCH_Faceted_", descriptive_pred_name, file_suffix, ".png"))
      
      ggsave(output_filename, p, width = 12, height = 8, dpi = 300, bg = BACKGROUND_STYLE)
      print(paste("Faceted plot saved to:", output_filename))
    }
  }
  # The logic for PLOT_STYLE == "separate" would go here, similarly modified.
}