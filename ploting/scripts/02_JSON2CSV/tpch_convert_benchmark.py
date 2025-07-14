import json
import csv
import os

# --- Configuration ---
# You can change these values to match your needs.

# 1. The name of the top-level array in the JSON file you want to process.
JSON_BENCHMARK_FIELD = "benchmarks"

# 2. The list of column headers for your output CSV file.
#    The script will only look for these keys in the JSON data.
CSV_HEADERS = [
    "group",
    "name",
    "fullname",
    "param",
    "min",
    "max",
    "mean",
    "stddev",
    "rounds",
    "median",
    "iqr",
    "q1",
    "q3",
    "iqr_outliers",
    "stddev_outliers",
    "outliers",
    "ld15iqr",
    "hd15iqr",
    "ops",
    "total",
    "iterations",
    "col",
    "op",
    "pred",
    "val",
    "tuple_count",
    "query_result_size",
    "tuple_element_size_bytes",
    "total_size_bytes",
    "param:operators",
    "param:device",
    "param:tensor_cls",
    "param:scale",
]

# 3. Define the input and output file names.
INPUT_JSON_FILE = "../../raw_data/json/0002_0.1-100_cpu_20250710_164031.json"
OUTPUT_CSV_FILE = "0002_cpu.csv"

# --- End of Configuration ---


def flatten_benchmark_data(benchmark_record):
    """
    Extracts data from a single benchmark record (a dictionary)
    and flattens the nested 'stats' and 'params' objects.
    """
    flat_data = {}

    # Extract top-level keys
    for key in CSV_HEADERS:
        flat_data[key] = benchmark_record.get(key)

    # Extract nested 'params' keys
    if "params" in benchmark_record and benchmark_record["params"] is not None:
        for key in ["operators", "device", "tensor_cls", "scale"]:
            flat_data["param:" + key] = benchmark_record["params"].get(key)

    return flat_data


def convert_json_to_csv(json_file, csv_file, benchmark_field, headers):
    """
    Loads data from a JSON file, processes the specified benchmark array,
    and writes the result to a CSV file.
    """
    # Check if the input file exists
    if not os.path.exists(json_file):
        print(f"Error: Input file '{json_file}' not found.")
        return

    print(f"Loading JSON data from '{json_file}'...")
    with open(json_file, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error: Could not decode JSON from '{json_file}'. Details: {e}")
            return

    # Check if the specified benchmark field exists in the JSON
    if benchmark_field not in data:
        print(f"Error: The key '{benchmark_field}' was not found in the JSON file.")
        return

    benchmark_list = data[benchmark_field]

    print(f"Found {len(benchmark_list)} records in the '{benchmark_field}' array.")
    print(f"Writing data to '{csv_file}'...")

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        # Use DictWriter to easily map dictionary keys to CSV columns
        writer = csv.DictWriter(f, fieldnames=headers)

        # Write the header row
        writer.writeheader()

        # Process and write each benchmark record
        for record in benchmark_list:
            # Flatten the nested JSON object into a single-level dictionary
            flat_record = flatten_benchmark_data(record)

            # Create a row dictionary containing only the keys specified in headers
            row_to_write = {header: flat_record.get(header) for header in headers}

            writer.writerow(row_to_write)

    print("\nConversion complete!")
    print(f"Successfully created '{csv_file}'.")


# --- Main execution block ---
if __name__ == "__main__":
    convert_json_to_csv(
        json_file=INPUT_JSON_FILE,
        csv_file=OUTPUT_CSV_FILE,
        benchmark_field=JSON_BENCHMARK_FIELD,
        headers=CSV_HEADERS,
    )
