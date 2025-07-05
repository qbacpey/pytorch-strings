import pytest
from pytest_benchmark.stats import Metadata
import re


# hooks need to be placed in a special file named conftest.py for pytest to discover and execute them.


def pytest_benchmark_update_json(config: pytest.Config, benchmarks: list[Metadata], output_json: dict) -> None:
    """
    Cleans up the benchmark parameters in the final JSON output by removing
    the "UNSERIALIZABLE" wrapper and simplifying class paths.
    """
    # The 'benchmarks' key in output_json holds the list of test results.
    for benchmark_data in output_json['benchmarks']:
        params = benchmark_data.get('params')
        if not params:
            continue

        # Clean up the 'operators' parameter
        if "operators" in params:
            cleaned_operators = []
            # The parameter is a list of strings
            for op_str in params["operators"]:
                if isinstance(op_str, str) and op_str.startswith("UNSERIALIZABLE["):
                    # Extract the readable part from "UNSERIALIZABLE[...]"
                    cleaned_operators.append(op_str[len("UNSERIALIZABLE["):-1])
                else:
                    cleaned_operators.append(str(op_str))
            # Update the params dictionary for the current benchmark
            params["operators"] = cleaned_operators

        # Clean up the 'tensor_cls' parameter
        if "tensor_cls" in params:
            cls_str = params["tensor_cls"]
            if isinstance(cls_str, str) and cls_str.startswith("UNSERIALIZABLE["):
                print(f"Original tensor_cls: {cls_str}")
                known_classes = [
                    "PlainEncodingStringColumnTensor", 
                    "CPlainEncodingStringColumnTensor", 
                    "DictionaryEncodingStringColumnTensor", 
                    "CDictionaryEncodingStringColumnTensor",
                    "UnsortedDictionaryEncodingStringColumnTensor",
                    "UnsortedCDictionaryEncodingStringColumnTensor"
                ]
                for class_name in known_classes:
                    if class_name in cls_str:
                        params["tensor_cls"] = class_name
                        break  # Stop after finding the first match

def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """
    Modify the collected test items.
    Example:
    .. sourcecode:: python
        def pytest_collection_modifyitems(config, items):
            for item in items:
                item.name = item.name.upper()
    """
    # Print each test name to the console
    # print("Collected tests:")
    # for item in items:
    #     print(item.name)