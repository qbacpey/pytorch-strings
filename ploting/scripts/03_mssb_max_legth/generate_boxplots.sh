#!/bin/bash

# ==============================================================================
# Boxplot Generation Runner
#
# This script runs the R boxplot script for all relevant configurations.
#
# Usage:
# 1. Make the script executable: chmod +x generate_boxplots.sh
# 2. Run the script from your terminal: ./generate_boxplots.sh
# ==============================================================================

# --- Configuration ---

# Path to the aggregated CSV file from your experiments
INPUT_CSV="20250914_mssb_max_len_alls.csv"

# Path to the R script that generates the boxplots
R_SCRIPT="01_boxplot_mssb.r"

# Define the different configurations to loop through
PREDICATES=("Eq" "Lt" "Prefix")
MASKS=("TRUE" "FALSE")
COMPILES=("TRUE" "FALSE")

# --- Main Execution Logic ---

echo "================================================="
echo "Starting boxplot generation for all configurations..."
echo "Input CSV: ${INPUT_CSV}"
echo "================================================="

# Check if the input CSV exists
if [ ! -f "$INPUT_CSV" ]; then
    echo "Error: Input CSV file not found at ${INPUT_CSV}"
    exit 1
fi

# Check if the R script exists
if [ ! -f "$R_SCRIPT" ]; then
    echo "Error: R script not found at ${R_SCRIPT}"
    exit 1
fi

# Loop through all configurations and generate a plot for each
for pred in "${PREDICATES[@]}"; do
  for mask in "${MASKS[@]}"; do
    for compile in "${COMPILES[@]}"; do
      echo "--- Generating plot for: Predicate=$pred, Mask=$mask, Compile=$compile ---"
      Rscript "$R_SCRIPT" "$INPUT_CSV" "$mask" "$compile" "$pred"
    done
  done
done

echo ""
echo "================================================="
echo "All boxplots generated successfully!"
echo "Check the 'plots' directory for the output images."
echo "================================================="