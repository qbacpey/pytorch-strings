import pandas as pd
import sys
import os

def join_benchmark_data(size_info_file: str, perf_info_file: str, output_file: str):
    """
    Reads benchmark size and performance data, performs a deduplicated inner join
    on the test name, and saves the result to a new CSV file.

    Args:
        size_info_file: Path to the CSV with memory/size info.
        perf_info_file: Path to the CSV with performance (timing) info.
        output_file: Path where the merged CSV will be saved.
    """
    try:
        df_size = pd.read_csv(size_info_file)
        df_perf = pd.read_csv(perf_info_file)
    except FileNotFoundError as e:
        print(f"Error: Could not find an input file: {e.filename}", file=sys.stderr)
        return

    # --- 1. Deduplicate Data ---
    # Before joining, remove any duplicate test runs. We keep the 'last' entry,
    # assuming it's the most recent run.
    df_size.drop_duplicates(subset=['fullname'], keep='last', inplace=True)
    df_perf.drop_duplicates(subset=['fullname'], keep='last', inplace=True)

    print(f"Loaded {len(df_size)} unique rows from size info file.")
    print(f"Loaded {len(df_perf)} unique rows from performance info file.")

    # --- 2. Perform Inner Join ---
    # Merge the two dataframes. 'how="inner"' ensures that only rows with matching
    # keys in BOTH files are kept.
    df_merged = pd.merge(
        left=df_size,
        right=df_perf,
        left_on='fullname',
        right_on='fullname',
        how='inner'
    )

    # --- 3. Clean Up and Save ---
    # Remove the "bench_test.py::TestStringColumnTensor::test_tpch_string_processing" from fullname
    # This is optional, depending on how you want to structure the final output.
    df_merged['fullname'] = df_merged['fullname'].str.replace(r"bench_test.py::TestStringColumnTensor::test_tpch_string_processing", "", regex=True)
    # df_merged.drop(columns=['name'], inplace=True)

    # Ensure the output directory exists
    # os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    df_merged.to_csv(output_file, index=False)
    
    print(f"\nSuccessfully joined data, resulting in {len(df_merged)} rows.")
    print(f"Merged data saved to: '{output_file}'")


if __name__ == '__main__':
    # Check if pandas is installed
    try:
        import pandas
    except ImportError:
        print("Error: pandas is not installed. Please install it using: pip install pandas", file=sys.stderr)
        sys.exit(1)

    # --- Define File Paths ---
    # Paths are relative to the script's location.
    
    # Input file with size/memory information
    size_file = 'benchmark_info_from_string_processing.csv'
    
    # Input file with performance/timing information (e.g., NAME.csv)
    # NOTE: Update this path if your performance data is in a different file.
    perf_file = 'cleaned_benchmark_results.csv'
    
    # Path for the final, merged output file
    output_file = '01_TPCH_Joined.csv'

    join_benchmark_data(size_file, perf_file, output_file)