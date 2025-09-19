#!/bin/bash

# This script runs the R plotting script to generate MERGED plots for all configurations.

# --- Configuration ---
# Point this to your actual data file
TPCH_CSV="0914_test_tpch_string_processing_output.csv"

# Define the different configurations to loop through
PREDICATES=("Eq" "Lt" "Prefix")
COMPILES=("FALSE")

# R script path
TPCH_SCRIPT="01_linear_GB_CPU_vs_GPU_tpch_merged.r"

# --- Generate TPC-H Merged Plots ---
echo "--- Generating TPC-H Merged Plots ---"
for pred in "${PREDICATES[@]}"; do
  for compile in "${COMPILES[@]}"; do
    echo "Running TPC-H Merged with: Predicate=$pred, Compile=$compile"
    Rscript "$TPCH_SCRIPT" "$TPCH_CSV" "$compile" "$pred"
  done
done

echo "--- All merged plots generated successfully! ---"