#!/bin/bash

# This script runs the R plotting scripts for all configurations.

# --- Configuration ---
# Point these to your actual data files
TPCH_CSV="0914_test_tpch_string_processing_output.csv"
MSSB_CSV="0909_mssb_10to200_all.csv"

# Define the different configurations to loop through
PREDICATES=("Eq" "Lt" "Prefix")
MASKS=("TRUE" "FALSE")
COMPILES=("FALSE")

# R script paths
TPCH_SCRIPT="00_linear_GB_CPU_vs_GPU_tpch.r"
MSSB_SCRIPT="00_linear_GB_CPU_vs_GPU_mssb.r"

# --- Generate TPC-H Plots ---
echo "--- Generating TPC-H Plots ---"
for pred in "${PREDICATES[@]}"; do
  for mask in "${MASKS[@]}"; do
    for compile in "${COMPILES[@]}"; do
      echo "Running TPC-H with: Predicate=$pred, Mask=$mask, Compile=$compile"
      Rscript "$TPCH_SCRIPT" "$TPCH_CSV" "$mask" "$compile" "$pred"
    done
  done
done

# --- Generate MSSB Plots ---
echo "--- Generating MSSB Plots ---"
for pred in "${PREDICATES[@]}"; do
  for mask in "${MASKS[@]}"; do
    for compile in "${COMPILES[@]}"; do
      echo "Running MSSB with: Predicate=$pred, Mask=$mask, Compile=$compile"
      Rscript "$MSSB_SCRIPT" "$MSSB_CSV" "$mask" "$compile" "$pred"
    done
  done
done

echo "--- All plots generated successfully! ---"