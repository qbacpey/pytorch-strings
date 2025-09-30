#!/bin/bash

# This script runs the R plotting script to generate AGGREGATED plots for TPC-H,
# merging the COMPILED and NOT-COMPILED results into a single plot.

# --- Configuration ---
# Point this to your actual data file
TPCH_CSV="0919_CUDA-only_NoPlain_test_tpch_string_processing_output.csv"

# Define the different configurations to loop through
MASKS=("TRUE" "FALSE")
# --- MODIFIED: Create a loop for the gray-out option ---
GRAY_OUT_OPTIONS=("TRUE" "FALSE")

# R script path for the merged compile plot
TPCH_AGGREGATE_SCRIPT="04_aggregated_dict_tpch_row_only_merged_compile.r"

# --- Generate TPC-H Aggregated Plots ---
echo "--- Generating TPC-H Merged-Compile Aggregated Plots ---"
for mask in "${MASKS[@]}"; do
  # --- MODIFIED: Loop through gray-out options ---
  for gray_out in "${GRAY_OUT_OPTIONS[@]}"; do
    echo "Running TPC-H Merged-Compile Aggregated with: Mask=$mask, GrayOut=$gray_out"
    # --- MODIFIED: Pass the gray_out variable to the R script ---
    Rscript "$TPCH_AGGREGATE_SCRIPT" "$TPCH_CSV" "$mask" "$gray_out"
  done
done

echo "--- All merged-compile aggregated plots generated successfully! ---"