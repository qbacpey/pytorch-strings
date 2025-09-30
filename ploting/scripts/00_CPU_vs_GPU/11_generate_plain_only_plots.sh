#!/bin/bash

# This script runs the R plotting script to generate PLAIN-ENCODING-ONLY plots for TPC-H.

# --- Configuration ---
# Point this to your actual data file
TPCH_CSV="0914_test_tpch_string_processing_output.csv"

# Define the different configurations to loop through
PREDICATES=("Eq" "Lt" "Prefix")
MASKS=("TRUE" "FALSE")
COMPILES=("TRUE" "FALSE")

# --- MODIFIED: R script path for the plain-only plot ---
TPCH_SCRIPT="00_linear_GB_CPU_vs_GPU_tpch_plain_only.r"

# --- Generate TPC-H Plain-Only Plots ---
echo "--- Generating TPC-H Plain-Only Plots ---"
for pred in "${PREDICATES[@]}"; do
  for mask in "${MASKS[@]}"; do
    for compile in "${COMPILES[@]}"; do
      echo "Running TPC-H Plain-Only with: Predicate=$pred, Mask=$mask, Compile=$compile"
      Rscript "$TPCH_SCRIPT" "$TPCH_CSV" "$mask" "$compile" "$pred"
    done
  done
done

echo "--- All plain-only plots generated successfully! ---"