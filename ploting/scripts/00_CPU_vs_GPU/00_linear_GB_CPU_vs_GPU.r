# --- 0. Setup: Load Libraries and Configure Parameters ---
# install.packages(c("ggplot2", "dplyr", "readr", "stringr", "scales"))
# source("00_linear_GB_CPU_vs_GPU.r")

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
input_file <- "0908_tpch_10to200_Eq_CUDA.csv"
output_dir <- "plots"

# --- NEW: User-specific plotting choices ---
# Set to TRUE to plot data where nonzero() was used, FALSE for returning the mask
USE_MASK <- TRUE
# USE_MASK <- FALSE
# Specify the predicate to plot: "Eq", "Lt", or "Prefix"
PREDICATE_TO_PLOT <- "Eq"

USE_TORCH_COMPILE <- FALSE
# USE_TORCH_COMPILE <- TRUE

# --- NEW: X-axis label configuration ---
# This controls which labels are displayed on the x-axis.
# It does NOT filter the data; all data points will still be plotted.
# Set to NULL to let ggplot decide the labels automatically.
X_AXIS_BREAKS <- c(10, 50, 100, 150, 200)
# X_AXIS_BREAKS <- NULL

# Theoretical Memory Bandwidth (in bytes per second)
THEORETICAL_GPU_BANDWIDTH <- 1.1e12 # 1.1 TB/s
THEORETICAL_CPU_BANDWIDTH <- 60e9  # 60 GB/s

# Time metric to use for throughput calculation
TIME_METRIC <- "mean"
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

# --- MODIFIED: Filter data based on user configuration ---
plot_data <- data %>%
  filter(
    pred == PREDICATE_TO_PLOT,
    `param:return_mask` == USE_MASK,
    `param:torch_compile` == USE_TORCH_COMPILE
  ) %>%
  mutate(
    throughput_gb_per_sec = (total_size_bytes / 1e9) / (.data[[TIME_METRIC]])
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

# --- 2. Plotting Function (Linear Scale) ---
# --- RENAMED: from create_throughput_plot_log to create_throughput_plot ---
create_throughput_plot <- function(df) {
  df <- df %>% mutate(`param:scale` = as.factor(`param:scale`))

  # Create descriptive labels for the legend
  legend_labels <- c(
    "CPU" = paste0("CPU (", THEORETICAL_CPU_BANDWIDTH / 1e9, " GB/s)"),
    "CUDA" = paste0("CUDA (", THEORETICAL_GPU_BANDWIDTH / 1e9, " GB/s)")
  )

  p <- ggplot(df, aes(x = `param:scale`, y = throughput_gb_per_sec, color = Device, group = Device)) +
    # --- Measured Performance (colored by Device) ---
    geom_line(linewidth = 1) +
    geom_point(size = 3, aes(shape = Device)) +

    # --- Theoretical Performance (colored by Device, with updated legend) ---
    geom_line(
      aes(y = theoretical_throughput_gb_per_sec, linetype = "Theoretical Bandwidth (DGX)"),
      # The 'color = Device' aesthetic is now inherited from the main ggplot() call
      linewidth = 1
    ) +

    # --- REMOVED: scale_y_log10() for linear scale ---

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
    scale_linetype_manual(name = "Line Type", values = c("Theoretical Bandwidth (DGX)" = "dashed"))

  # --- NEW: Add custom x-axis breaks if specified ---
  # This only changes the labels shown on the axis, not the data being plotted.
  if (!is.null(X_AXIS_BREAKS)) {
    p <- p + scale_x_discrete(breaks = as.character(X_AXIS_BREAKS))
  }

  p <- p + labs(
      # Title and subtitle will be set later
      x = "TPC-H Scale Factor",
      y = "Throughput (GB/s)",
      linetype = "Line Type"
    ) +
    theme_minimal(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold"),
      legend.position = "bottom",
      legend.box = "vertical",
       axis.text = element_text(size = 12) # Controls both x and y axis labels
    )
  
  return(p)
}



# --- 3. Generate and Save Plot ---
dir.create(output_dir, showWarnings = FALSE)

# --- MODIFIED: Generate a single plot based on configuration ---
if (nrow(plot_data) > 0) {
  # Get descriptive names for titles and filenames
  descriptive_pred_name <- unique(plot_data$pred_name)
  nonzero_label <- if (USE_MASK) "Return Mask" else "With Nonzero"
  compile_label <- if (USE_TORCH_COMPILE) "Compiled" else "Not Compiled"

  # Create the plot
  p <- create_throughput_plot(plot_data) +
    facet_wrap(~encoding_type, scales = "fixed", ncol = 2) +
    labs(
      title = paste("Throughput for Predicate:", descriptive_pred_name),
      subtitle = paste("Operation:", nonzero_label, "| Torch Compile:", compile_label),
      caption = "Each panel represents a different string encoding algorithm."
    )

  # Generate a descriptive filename
  filename_suffix_mask <- if (USE_MASK) "_Mask" else "_Nonzero"
  filename_suffix_compile <- if (USE_TORCH_COMPILE) "_Compiled" else "_NoCompile"
  output_filename <- file.path(output_dir, paste0("00_linear_GB_TPCH_Faceted_", descriptive_pred_name, filename_suffix_mask, filename_suffix_compile, ".png"))
  
  ggsave(output_filename, p, width = 12, height = 8, dpi = 300, bg = BACKGROUND_STYLE)
  print(paste("Linear-scale plot saved to:", output_filename))
} else {
  print("No data available for the specified configuration (PREDICATE_TO_PLOT, USE_MASK, and USE_TORCH_COMPILE). No plot generated.")
}