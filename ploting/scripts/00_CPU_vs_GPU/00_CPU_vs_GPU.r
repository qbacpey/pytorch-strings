# --- 0. Setup: Load Libraries and Configure Parameters ---
# install.packages(c("ggplot2", "dplyr", "readr", "stringr"))

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
input_file <- "../10_Post_Benchmark_Process/01_TPCH_Joined.csv"
output_dir <- "plots"

# Theoretical Memory Bandwidth (in bytes per second)
# HBM on A100/H100 is ~1.6-2.0 TB/s.
# DDR5 on modern CPUs is ~60-80 GB/s.
THEORETICAL_GPU_BANDWIDTH <- 1.6e12 # 1.6 TB/s
THEORETICAL_CPU_BANDWIDTH <- 60e9   # 60 GB/s

# Time metric to use for throughput calculation ("mean", "min", or "max")
TIME_METRIC <- "mean"

# Plotting style: "facet" for a single plot with facets, or "separate" for one plot per file.
PLOT_STYLE <- "facet"

# Background style for saved plots: "white" or "transparent"
BACKGROUND_STYLE <- "white"


# --- 1. Data Loading and Preparation ---

# Read the joined benchmark data
data <- read_csv(input_file, show_col_types = FALSE)

# Prepare the data for plotting
plot_data <- data %>%
  # Calculate measured throughput (Tuples per Second)
  # The time metric is selected here based on the TIME_METRIC variable
  mutate(
    throughput_tuples_per_sec = tuple_count / .data[[TIME_METRIC]]
  ) %>%
  # Calculate theoretical throughput based on device and tuple size
  mutate(
    theoretical_throughput = case_when(
      `param:device` == 'cpu'  ~ THEORETICAL_CPU_BANDWIDTH / tuple_element_size_bytes,
      `param:device` == 'cuda' ~ THEORETICAL_GPU_BANDWIDTH / tuple_element_size_bytes
    )
  ) %>%
  # Clean up encoding names for better plot titles
  mutate(
    encoding_type = str_replace(`param:tensor_cls`, "StringColumnTensor", "")
  ) %>%
  # Convert device to a factor for consistent coloring
  mutate(
    Device = factor(toupper(`param:device`))
  )

# --- 2. Plotting Function ---

# A reusable function to generate the plot structure
create_throughput_plot <- function(df) {
  df <- df %>% mutate(`param:scale` = as.factor(`param:scale`))
  
  ggplot(df, aes(x = `param:scale`, y = throughput_tuples_per_sec, color = Device, group = Device)) +
    # --- Measured Performance (colored by Device) ---
    geom_line(linewidth = 1) +
    geom_point(size = 3, aes(shape = Device)) +

    # --- Theoretical Performance (black, dashed, with its own legend entry) ---
    geom_line(
      aes(y = theoretical_throughput, linetype = "Theoretical (Bandwidth / Tuple Size)"), 
      linewidth = 1
    ) +
    
    # --- Scales and Labels ---
    scale_y_log10(
      labels = scales::trans_format("log10", scales::math_format(10^.x)),
      breaks = scales::trans_breaks("log10", function(x) 10^x, n = 8)
    ) +
    # Manually define colors for devices
    scale_color_manual(values = c("CPU" = "#0072B2", "CUDA" = "#D55E00")) +
    # Manually define shapes for devices
    scale_shape_manual(values = c("CPU" = 16, "CUDA" = 17)) +
    # Manually define the linetype and its legend entry
    scale_linetype_manual(name = "Line Type", values = c("Theoretical (Bandwidth / Tuple Size)" = "dashed")) +
    
    # --- Titles and Theme ---
    labs(
      title = "Measured vs. Theoretical Throughput for String Operations",
      subtitle = paste("Using", TIME_METRIC, "time for calculation"),
      x = "TPC-H Scale Factor",
      y = "Throughput (Tuples / Second)",
      color = "Device",
      shape = "Device",
      linetype = "Line Type" # Title for the new legend
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
  # Generate a single plot with facets for each encoding type
  p <- create_throughput_plot(plot_data) +
    # Use scales = "fixed" to ensure a uniform Y-axis across all facets
    facet_wrap(~ encoding_type, scales = "fixed", ncol = 2) +
    labs(caption = "Each panel represents a different string encoding algorithm.")
  
  # Save the faceted plot
  output_filename <- file.path(output_dir, "00_TPCH_Throughput_Faceted.png")
  ggsave(output_filename, p, width = 12, height = 8, dpi = 300, bg = BACKGROUND_STYLE)
  print(paste("Faceted plot saved to:", output_filename))
  
} else if (PLOT_STYLE == "separate") {
  # Generate a separate plot for each encoding type
  encoding_types <- unique(plot_data$encoding_type)
  
  for (enc_type in encoding_types) {
    # Filter data for the current encoding type
    subset_data <- plot_data %>% filter(encoding_type == enc_type)
    
    # Create the plot for the subset
    p <- create_throughput_plot(subset_data) +
      labs(subtitle = paste("Encoding:", enc_type, "| Using", TIME_METRIC, "time"))
    
    # Save the individual plot
    output_filename <- file.path(output_dir, paste0("01_TPCH_Throughput_", enc_type, ".png"))
    ggsave(output_filename, p, width = 10, height = 6, dpi = 300, bg = BACKGROUND_STYLE)
    print(paste("Separate plot saved to:", output_filename))
  }
}
# ### How to Use This Script: 首先启动 R 环境，然后`source("00_CPU_vs_GPU.r")`

# 1.  **Prerequisites:**
#     *   Make sure you have R installed on your system.
#     *   Open an R console or RStudio.

# 2.  **Install Packages:** Run the following commands in your R console to install the necessary libraries. The script also includes a check to do this automatically.
#     ````r
#     install.packages(c("ggplot2", "dplyr", "readr", "stringr"))
#     ````

# 3.  **Save the Script:** Save the code above to the specified file path: 00_CPU_vs_GPU.r.

# 4.  **Run the Script:**
#     *   Open your R environment (like RStudio or a terminal).
#     *   Set your working directory to the script's location:
#         ````r
#         setwd("/home/qba/00_SS25/02_DBLab/10_DMLab/pytorch-strings/ploting/scripts/00_CPU_vs_GPU/")
#         ````
#     *   Execute the script:
#         ````r
#         source("00_CPU_vs_GPU.r")
#         ````

# ### What the Script Does:

# 1.  **Configuration:** At the top, you can easily change the input file, theoretical bandwidth values, the time metric (`mean`, `min`, `max`), and the plotting style (`facet` or `separate`).
# 2.  **Data Processing:** It reads your CSV, calculates the measured throughput in Tuples/Second, and also calculates the theoretical maximum throughput based on your configured bandwidth constants and the size of each tuple.
# 3.  **Plotting:**
#     *   It uses `ggplot2` to create a high-quality plot.
#     *   The Y-axis is on a `log10` scale to clearly show the large performance gap between CPU and GPU.
#     *   **Solid lines and points** represent your actual measured performance.
#     *   **Dashed lines** represent the theoretical memory bandwidth limit for CPU and GPU.
#     *   It uses distinct colors and shapes for CPU and GPU data.
# 4.  **Output:** Based on your `PLOT_STYLE` setting, it will either:
#     *   **`facet`**: Create a single PNG file (`00_TPCH_Throughput_Faceted.png`) with a grid of plots, one for each encoding type.
#     *   **`separate`**: Create multiple PNG files, one for each encoding type (e.g., `01_TPCH_Throughput_PlainEncoding.png`).
#     *   All plots will be saved in a new `plots` subdirectory.