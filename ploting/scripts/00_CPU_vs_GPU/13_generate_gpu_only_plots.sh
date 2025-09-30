#!/bin/bash

# This script runs the R plotting script for GPU-ONLY configurations.

# --- Configuration ---
# Point these to your actual data files
TPCH_CSV="0923_CUDA-only_test_tpch_string_processing_output.csv"

# Define the different configurations to loop through
PREDICATES=("Eq" "Lt" "Prefix")
MASKS=("TRUE" "FALSE")
COMPILES=("TRUE" "FALSE")

# --- MODIFIED: R script path for the GPU-only plot ---
TPCH_SCRIPT="00_linear_GB_GPU_only_tpch.r"

# --- Generate TPC-H GPU-Only Plots ---
echo "--- Generating TPC-H GPU-Only Plots ---"
for pred in "${PREDICATES[@]}"; do
  for mask in "${MASKS[@]}"; do
    for compile in "${COMPILES[@]}"; do
      echo "Running TPC-H GPU-Only with: Predicate=$pred, Mask=$mask, Compile=$compile"
      Rscript "$TPCH_SCRIPT" "$TPCH_CSV" "$mask" "$compile" "$pred"
    done
  done
done

echo "--- All GPU-only plots generated successfully! ---"