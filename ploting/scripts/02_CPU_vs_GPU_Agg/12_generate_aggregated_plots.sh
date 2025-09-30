#!/bin/bash

# This script runs the R plotting script to generate AGGREGATED plots for TPC-H.

# --- Configuration ---
# Point this to your actual data file
TPCH_CSV="0919_CUDA-only_NoPlain_test_tpch_string_processing_output.csv"

# Define the different configurations to loop through
MASKS=("TRUE" "FALSE")
COMPILES=("TRUE" "FALSE")

# R script path
TPCH_AGGREGATE_SCRIPT="03_aggregated_dict_tpch_row_only.r"

# --- Generate TPC-H Aggregated Plots ---
echo "--- Generating TPC-H Aggregated Plots ---"
for mask in "${MASKS[@]}"; do
  for compile in "${COMPILES[@]}"; do
    echo "Running TPC-H Aggregated with: Mask=$mask, Compile=$compile"
    Rscript "$TPCH_AGGREGATE_SCRIPT" "$TPCH_CSV" "$mask" "$compile"
  done
done

echo "--- All aggregated plots generated successfully! ---"