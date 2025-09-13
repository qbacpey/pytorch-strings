#!/bin/bash

# ==============================================================================
# Pytest Benchmark Experiment Runner
#
# This script runs a pytest benchmark multiple times, accumulating the results
# into a single CSV and a single log file.
#
# Usage:
# 1. Configure the variables in the "Configuration" section below.
# 2. Make the script executable: chmod +x run_experiment.sh
# 3. Submit the entire script to SLURM:
#    srun --ntasks=1 --cpus-per-task=32 --gres=gpu:1 --pty ./run_experiment.sh
# ==============================================================================

# --- Configuration ---

# Number of times to repeat the experiment
# 2025-09-13 21:59:47
# 2025-09-13 22:13:18
# 300 = 8h

NUM_RUNS=300

# Base names for the final accumulated output files
BASE_CSV_NAME="mssb_max_len_alls"
BASE_LOG_NAME="mssb_max_len_alls_log"

# Directory where the raw CSV files will be temporarily stored
RAW_DATA_DIR="/mnt/labstore/qchen/pytorch-strings/ploting/raw_data/csv"
# Directory where the final log file will be stored
LOG_DIR="." # Using the current directory for the log file

# Path to the data directory to be cleared before each run.
# This forces the benchmark script to regenerate the dataset.
# Based on your bench_test.py, this seems to be the correct path.
DATA_DIR_TO_CLEAR="/mnt/labstore/qchen/pytorch-strings/dataset/mssb_data/catalog.txt"

# --- Script Initialization ---

# Get the current date in YYYYMMDD format
DATE=$(date +%Y%m%d)

# Define the final, accumulated output file paths
FINAL_CSV_FILE="${RAW_DATA_DIR}/${DATE}_${BASE_CSV_NAME}.csv"
FINAL_LOG_FILE="${LOG_DIR}/${DATE}_${BASE_LOG_NAME}.ansi"

# Define the temporary file paths that will be created on each run
TEMP_CSV_FILE="${RAW_DATA_DIR}/temp_run_result.csv"
TEMP_LOG_FILE="temp_run_log.ansi"

# Pytest command arguments
# The temporary CSV and log files are specified here.
PYTEST_CMD="pytest -k test_mssb_string_processing -s --csv=\"${TEMP_CSV_FILE}\" --benchmark-storage='file:///mnt/labstore/qchen/pytorch-strings/ploting/raw_data/json/' --benchmark-autosave"

# --- Main Execution Logic ---

CURRENT_DATE=$(date +"%Y-%m-%d %H:%M:%S")

echo "================================================="
echo "Starting benchmark experiment..."
echo "Number of runs: ${NUM_RUNS}"
echo "Current Date: ${CURRENT_DATE}"
echo "Accumulated CSV will be saved to: ${FINAL_CSV_FILE}"
echo "Accumulated Log will be saved to: ${FINAL_LOG_FILE}"
echo "================================================="

# Create or clear the final output files at the very beginning
# This ensures we start fresh each time the main script is run.
> "${FINAL_LOG_FILE}"
> "${FINAL_CSV_FILE}"

# Loop for the specified number of runs
for (( i=1; i<=NUM_RUNS; i++ ))
do
    echo ""
    echo "--- Starting Run ${i} of ${NUM_RUNS} ---"

    # 1. Delete the old data to force regeneration
    echo "Clearing old dataset at: ${DATA_DIR_TO_CLEAR}"
    rm -rf "${DATA_DIR_TO_CLEAR}"
    # The benchmark script will now generate new data on its own.

    # 2. Run the pytest benchmark
    # The 'eval' is used to correctly handle the quotes in the PYTEST_CMD string.
    # Output (stdout and stderr) is redirected to the temporary log file.
    echo "Running pytest benchmark..."
    eval ${PYTEST_CMD} > "${TEMP_LOG_FILE}" 2>&1

    # 3. Append results to the accumulated files
    echo "Appending results to master files..."

    # Append the temporary log to the final log file
    cat "${TEMP_LOG_FILE}" >> "${FINAL_LOG_FILE}"

    # Append the temporary CSV to the final CSV file
    # If the final CSV is empty (first run), include the header.
    if [ ! -s "${FINAL_CSV_FILE}" ]; then
        cat "${TEMP_CSV_FILE}" >> "${FINAL_CSV_FILE}"
    else
        # Otherwise, skip the header line of the temporary CSV.
        tail -n +2 "${TEMP_CSV_FILE}" >> "${FINAL_CSV_FILE}"
    fi

    # 4. Clean up the temporary files for this run
    echo "Cleaning up temporary files..."
    rm "${TEMP_CSV_FILE}"
    rm "${TEMP_LOG_FILE}"

    echo "--- Finished Run ${i} of ${NUM_RUNS} ---"
done

CURRENT_DATE=$(date +"%Y-%m-%d %H:%M:%S")

echo ""
echo "================================================="
echo "All ${NUM_RUNS} benchmark runs are complete."
echo "Current Date: ${CURRENT_DATE}"
echo "Final results are located at:"
echo "CSV: ${FINAL_CSV_FILE}"
echo "Log: ${FINAL_LOG_FILE}"
echo