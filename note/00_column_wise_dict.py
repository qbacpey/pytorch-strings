from typing import List, Dict, Any
import numpy
import torch
import duckdb, duckdb.typing
import pytest, pytest_benchmark.fixture












# Example usage
if __name__ == "__main__":
    encoders = [DictionaryEncoder(), PlainEncoder(), RowWiseDictionaryEncoder()]
    for encoder in encoders:
        col = encoder.encode(
            [
                "apwho",
                "Initially",
                "apple",
                "applppp",
                "bpple",
                "each",
                "encoding",
                "method",
                "will",
                "will",
                "be",
                "applppp",
            ]
        )
        # print("Encoded tensor:", encoder.encoded_tensor)
        # print("Inverse indices:", encoder.inverse_indices)

        # Query for equality
        row_ids = col.query_equals("will")
        print("Row IDs for 'will':", row_ids)

        # Query for less than
        row_ids_lt = col.query_less_than("bpple")
        print("Row IDs for strings less than 'bpple':", row_ids_lt)

        # Query for prefix
        row_ids_prefix = col.query_prefix("ap")
        print("Row IDs for strings starting with 'ap':", row_ids_prefix)

        print(encoder.decode(col))
