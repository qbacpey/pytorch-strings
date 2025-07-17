import pandas as pd
import re
import sys


def process_benchmark_data(input_file: str, output_file: str):
    """
    Reads a benchmark CSV, cleans and structures the parameter columns,
    and saves the result to a new CSV file.

    Args:
        input_file: The path to the raw benchmark CSV file.
        output_file: The path where the cleaned CSV will be saved.
    """
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_file}'")
        return

    # 1. Clean up the 'param:tensor_cls' column
    known_classes = [
        "CPlainEncodingStringColumnTensor",
        "CDictionaryEncodingStringColumnTensor",
        "PlainEncodingStringColumnTensor",
        "DictionaryEncodingStringColumnTensor",
        "UnsortedDictionaryEncodingStringColumnTensor",
        "UnsortedCDictionaryEncodingStringColumnTensor",
    ]

    def get_tensor_name(cls_str: str) -> str:
        """Finds a known class name within the raw parameter string."""
        for name in known_classes:
            if name in cls_str:
                return name
        return cls_str  # Return original if no match found

    df["param:tensor_cls"] = df["param:tensor_cls"].apply(get_tensor_name)

    # 3. Create the final DataFrame
    # Drop the original long 'name' and 'param:operators' columns
    df_cleaned = df.drop(columns=["param:operators"])

    # Rename the 'name' column to 'fullname' for consistency
    df_cleaned.rename(columns={"name": "fullname"}, inplace=True)

    # Reorder columns for better readability
    final_cols = [
        "fullname",
        "param:tensor_cls",
        "param:device",
        "param:scale",
        "col",
        "op",
        "pred",
        "val",
        "mean",
        "min",
        "max",
        "stddev",
        "ops",
        "rounds",
        "query_result_size",
        "tuple_element_size_bytes",
        "tuple_count",
        "total_size_bytes",
    ]
    df_final = df_cleaned[final_cols]

    # 4. Save the cleaned data to a new CSV
    df_final.to_csv(output_file, index=False)
    print(
        f"Successfully processed '{input_file}' and saved cleaned data to '{output_file}'"
    )


if __name__ == "__main__":
    # Check if pandas is installed
    try:
        import pandas
    except ImportError:
        print("Pandas is not installed. Please install it using: pip install pandas")
        sys.exit(1)

    # Define input and output file names
    input_csv = "0006_unsortD_cpu_cuda.csv"
    output_csv = "0006_cleaned_unsortD_cpu_cuda.csv"

    process_benchmark_data(input_csv, output_csv)
