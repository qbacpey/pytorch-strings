import pytest
import pytest_benchmark.stats
import pytest_benchmark.csv


# hooks need to be placed in a special file named conftest.py for pytest to discover and execute them.
def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--benchmark-csv", "--csv", action="store", default=None, help="Path to output CSV file")
    parser.addoption("--torch-profile", action="store_true", default=False, help="Enable PyTorch profiling")
    parser.addoption("--toy-trace", action="store_true", default=False, help="Enable toy tracing")

def pytest_benchmark_update_json(config: pytest.Config, benchmarks: list[pytest_benchmark.stats.Metadata], output_json: dict) -> None:
    benchmarks_dict: list[dict] = output_json["benchmarks"]
    for bench in benchmarks_dict:
        if "operators" in bench["params"]:
            bench["params"]["operators"] = str(bench["params"]["operators"])
        if "tensor_cls" in bench["params"] and isinstance(bench["params"]["tensor_cls"], type):
            bench["params"]["tensor_cls"] = bench["params"]["tensor_cls"].__name__.replace("StringColumnTensor", "")

    # Flatten the stats and extra_info in the benchmarks, only to be used for CSV output but not for JSON output
    benchmarks_dict = benchmarks_dict.copy()
    for bench in benchmarks_dict:
        # flatten the stats
        bench.update(bench.pop('stats'))
        # flatten the extra_info
        bench.update(bench.pop('extra_info'))

    extra_fields = ["col", "op", "pred", "val", "tuple_count", "query_result_size", "tuple_element_size_bytes", "total_size_bytes"]
    # see pytest_benchmark/plugin.py add_display_options --columns
    stats_fields = ['min', 'max', 'mean', 'stddev', 'median', 'iqr', 'outliers', 'ops', 'rounds', 'iterations']
    # see pytest_benchmark/plugin.py add_display_options --sort
    sort_by_field = 'min'
    # see pytest_benchmark/plugin.py add_display_options --group_by
    group_by = 'group'
    # set output file name here
    csv_file = config.getoption("benchmark_csv") or "results.csv"
    print(f"[Benchmark] Writing Benchmark CSV results to: {csv_file}")

    results_csv = pytest_benchmark.csv.CSVResults(extra_fields + stats_fields, sort_by_field, config._benchmarksession.logger) # type: ignore
    groups = config.hook.pytest_benchmark_group_stats(
            benchmarks=benchmarks_dict,
            group_by=group_by,
            config=None,
        )
    # write the results to CSV
    results_csv.render(csv_file, groups)

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
