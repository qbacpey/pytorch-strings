#!/bin/bash

# This script runs the R plotting script to generate PLAIN and UNSORTED
# ENCODING plots for TPC-H.

# --- Configuration ---
# Point this to your actual data file
TPCH_CSV="0914_test_tpch_string_processing_output.csv"

# Define the different configurations to loop through
PREDICATES=("Eq" "Lt" "Prefix")
MASKS=("TRUE" "FALSE")
COMPILES=("TRUE" "FALSE")

# --- MODIFIED: R script path for the plain & unsorted plot ---
TPCH_SCRIPT="00_linear_GB_CPU_vs_GPU_tpch_plain_unsorted.r"

# --- Generate TPC-H Plain & Unsorted Plots ---
echo "--- Generating TPC-H Plain & Unsorted Plots ---"
for pred in "${PREDICATES[@]}"; do
  for mask in "${MASKS[@]}"; do
    for compile in "${COMPILES[@]}"; do
      echo "Running TPC-H Plain & Unsorted with: Predicate=$pred, Mask=$mask, Compile=$compile"
      Rscript "$TPCH_SCRIPT" "$TPCH_CSV" "$mask" "$compile" "$pred"
    done
  done
done

echo "--- All plain & unsorted plots generated successfully! ---"