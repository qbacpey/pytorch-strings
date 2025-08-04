import torch
import pytest
import contextlib
import time
from pytest_benchmark.plugin import BenchmarkFixture, BenchmarkSession
from typing import List, Dict, Any, Callable
import csv
import os

from string_tensor import *
from mock_operator import *
from dataset import load_tpch_col, load_mssb_col

@pytest.fixture(scope="class")
def tpch_data(operators: List[MockOperator], scale: float, device: str) -> MockTable:
    with torch.device(device):
        cols = {}
        for operator in operators:
            col_name = operator.col_name
            col = load_tpch_col(col_name, scale, "dataset/tpch_data")
            print(f"Encoding column {col_name} with scale {scale} on device {device}...")
            cols[col_name] = PlainEncodingStringColumnTensor.Encoder.encode(col)
        
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return MockTable(cols)

@pytest.fixture(scope="class")
def mssb_data(operators: List[MockOperator], device: str) -> MockTable:
    with torch.device(device):
        cols = {}
        for operator in operators:
            col_name = operator.col_name
            col = load_mssb_col(col_name, "dataset/mssb_data")
            print(f"Encoding column {col_name} on device {device}...")
            cols[col_name] = PlainEncodingStringColumnTensor.Encoder.encode(col)
        
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return MockTable(cols)

@pytest.fixture(scope="function")
def make_benchmark(request):
    bs: BenchmarkSession = request.config._benchmarksession

    if bs.skip:
        pytest.skip('Benchmarks are skipped (--benchmark-skip was used).')

    benchmarks: list[BenchmarkFixture] = []
    node = request.node
    marker = node.get_closest_marker('benchmark')
    options: dict[str, object] = dict(marker.kwargs) if marker else {}
    # if 'timer' in options:
    #     options['timer'] = NameWrapper(options['timer'])
    def make_benchmark(group: str | None = None, name: str | None = None, mode: str | None = None) -> BenchmarkFixture:
        fixture = BenchmarkFixture(
            node,
            add_stats=bs.benchmarks.append,
            logger=bs.logger,
            warner=request.node.warn,
            disabled=bs.disabled,
            **dict(bs.options, **options),
        )
        fixture.group = fixture.group + " | " + group if fixture.group and group else (fixture.group or group)
        fixture.name = fixture.name + " <" + name + ">" if fixture.name and name else (fixture.name or name)
        fixture._mode = mode or None
        benchmarks.append(fixture)
        return fixture

    yield make_benchmark

    for benchmark in benchmarks:
        benchmark._cleanup()

def string_processing(benchmark: BenchmarkFixture, data: MockTable, encoder: type[StringColumnTensor.Encoder], operators: List[MockOperator], device: str):
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    with torch.device(device):

        col_names = [op.col_name for op in operators]
        cols = {col_name: encoder.encode(data.cols[col_name]) for col_name in col_names}

        for operator in operators:
            # Calculate metrics
            col: StringColumnTensor = cols[operator.col_name]
            tuple_count = len(col)
            query_result_size = len(operator.apply(col))
            tuple_element_size = col.tuple_size()
            total_size = tuple_count * tuple_element_size

            # Print to console
            print(f"Applying operator on column with {tuple_count} elements...")
            print(f"Produced {query_result_size} results.")
            print(f"Tuple element size: {tuple_element_size} bytes")
            print(f"Total size: {total_size} bytes")

            # Only the last operator's info will be stored in the benchmark's extra_info
            benchmark.extra_info["col"] = operator.col_name
            benchmark.extra_info["op"] = operator.__class__.__name__
            benchmark.extra_info["pred"] = operator.predicate.__class__.__name__.replace("Predicate", "") if isinstance(operator, FilterScan) else None
            benchmark.extra_info["val"] = operator.predicate.value if isinstance(operator, FilterScan) else None
            benchmark.extra_info["tuple_count"] = tuple_count
            benchmark.extra_info["query_result_size"] = query_result_size
            benchmark.extra_info["tuple_element_size_bytes"] = tuple_element_size
            benchmark.extra_info["total_size_bytes"] = total_size

        benchmark(lambda: (
            [operator.apply(cols[operator.col_name]) for operator in operators],
            torch.cuda.synchronize() if device == "cuda" else None
        ))

@contextlib.contextmanager
def wrap_query_methods(target_cls, predicates: list[str], benchmark: BenchmarkFixture, stage1_benchmark: BenchmarkFixture, stage2_benchmark: BenchmarkFixture):
    """
    Wraps the query methods of the target class to add additional staged statistics for benchmarking.
    """
    stage1_stats = stage1_benchmark.stats or stage1_benchmark._make_stats(1)
    stage2_stats = stage2_benchmark.stats or stage2_benchmark._make_stats(1)
    query_methods = {}

    for predicate_name in predicates:
        query_dict = getattr(target_cls, f"query_{predicate_name}")
        query_codes = getattr(target_cls, f"query_{predicate_name}_codes")
        query_methods[predicate_name] = query_dict

        def wrapped_query(self, query: str) -> torch.Tensor:
            t = time.perf_counter()
            codes = query_dict(self, query)
            t_query_dict = time.perf_counter() - t
            t += t_query_dict
            match_index = query_codes(self, codes)
            t_query_codes = time.perf_counter() - t
            if benchmark.stats and benchmark.stats.stats.data: # skip warmup iterations, where benchmark.stats.stats.data is empty
                stage1_stats.update(t_query_dict)
                stage2_stats.update(t_query_codes)
            return match_index

        setattr(target_cls, f"query_{predicate_name}", wrapped_query)

    yield

    for predicate_name, query_dict in query_methods.items():
        setattr(target_cls, f"query_{predicate_name}", query_dict)

@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=3
)
class TestStringColumnTensor:

    @staticmethod
    def encoding_name(tensor_cls: type[StringColumnTensor]) -> str:
        name = tensor_cls.__name__[:-len("StringColumnTensor")]
        tensor_cls.__name__.replace("StringColumnTensor", "")
        return f"encoding={name}"
    
    @staticmethod
    def operator_id(operators: list[MockOperator]) -> str:
        """Generates a structured, parsable ID for a list of operators."""
        id_parts = []
        for op in operators:
            op_name = op.__class__.__name__
            # Start with common attributes
            parts = [f"op={op_name}", f"col={op.col_name}"]

            # Add type-specific attributes
            if isinstance(op, FilterScan):
                pred_name = op.predicate.__class__.__name__.replace("Predicate", "")
                parts.append(f"pred={pred_name}")
                parts.append(f"val={op.predicate.value}")
            elif isinstance(op, Sort):
                parts.append(f"asc={op.ascending}")
            
            id_parts.append(",".join(parts))
        return "|".join(id_parts)

    @pytest.mark.parametrize(
        "scale",
        [i for i in range(1, 10)] + [i * 10 for i in range(1, 11)],
        # [round(i * 0.1, 1) for i in range(1, 11)],
        scope="class",
        ids=lambda scale: f"scale={scale}",
    )
    @pytest.mark.parametrize(
        "tensor_cls",
        [
            UnsortedDictionaryEncodingStringColumnTensor, 
            UnsortedCDictionaryEncodingStringColumnTensor,
            PlainEncodingStringColumnTensor,
            CPlainEncodingStringColumnTensor,
            DictionaryEncodingStringColumnTensor,
            CDictionaryEncodingStringColumnTensor,
        ],
        ids=encoding_name,
    )
    @pytest.mark.parametrize("device", ["cpu","cuda"], scope="class", ids=lambda dev: f"device={dev}")
    @pytest.mark.parametrize(
        "operators",
        [
            # [FilterScan("l_shipmode", PredicateEq("AIR"))],
            # [FilterScan("l_shipmode", PredicateLt("REG AIR"))],
            [FilterScan("l_shipmode", PredicatePrefix("TR"))],
        ],
        scope="class",
        ids=operator_id,
    )
    @pytest.mark.benchmark(group="string_tensor_query_processing | TPCH")

    def test_tpch_string_processing(self, benchmark, tpch_data, operators: List[MockOperator], tensor_cls: type[StringColumnTensor], scale: float, device: str):
        print(f"Testing string query processing with {tensor_cls.__name__} on scale {scale} and device {device}...")
        # benchmark.group += f" | FilterScan(l_shipmode==AIR)"
        # benchmark.group += f" | scale-{scale}"
        string_processing(benchmark, tpch_data, tensor_cls.Encoder, operators, device) 

    @pytest.mark.parametrize("tensor_cls", [PlainEncodingStringColumnTensor, CPlainEncodingStringColumnTensor, DictionaryEncodingStringColumnTensor, CDictionaryEncodingStringColumnTensor], ids=encoding_name)
    @pytest.mark.parametrize("device", ["cpu", "cuda"], scope="class")
    @pytest.mark.parametrize("operators", [
        # [FilterScan('0001', PredicateEq('tsokmptbcza'))],
        # [FilterScan('0002', PredicateEq('tsokmptbcza'))],
        # [FilterScan('0003', PredicateEq('tsokmptbcza'))],
        # [FilterScan('0004', PredicateEq('tsokmptbcza'))],
        # [FilterScan('0005', PredicateEq('tsokmptbcza'))],
        # [FilterScan('0006', PredicateEq('tsokmptbcza'))],
    ], scope="class", ids=lambda ops: "")
    @pytest.mark.benchmark(group="string_tensor_query_processing | MSSB")

    def test_mssb_string_processing(self, benchmark, mssb_data, operators: List[MockOperator], tensor_cls: type[StringColumnTensor], device: str):
        print(f"Testing string query processing with {tensor_cls.__name__} on device {device}...")
        string_processing(benchmark, mssb_data, tensor_cls.Encoder, operators, device)

    @pytest.mark.parametrize("tensor_cls", [UnsortedDictionaryEncodingStringColumnTensor, UnsortedCDictionaryEncodingStringColumnTensor, DictionaryEncodingStringColumnTensor, CDictionaryEncodingStringColumnTensor], ids=encoding_name)
    @pytest.mark.parametrize("device", ["cpu", "cuda"], scope="class")
    @pytest.mark.parametrize("operators", [
        # [FilterScan('0001', PredicateEq('tsokmptbcza'))],
        # [FilterScan('0002', PredicateEq('tsokmptbcza'))],
        # [FilterScan('0003', PredicateEq('tsokmptbcza'))],
        # [FilterScan('0004', PredicateEq('tsokmptbcza'))],
        # [FilterScan('0005', PredicateEq('tsokmptbcza'))],
        # [FilterScan('0006', PredicateEq('tsokmptbcza'))],
    ], scope="class", ids=lambda ops: "")
    @pytest.mark.benchmark(group="string_tensor_query_processing | MSSB Staged")

    def test_mssb_staged_string_processing(self, make_benchmark, mssb_data, operators: List[MockOperator], tensor_cls: type[StringColumnTensor], device: str):
        print(f"Testing string query processing with {tensor_cls.__name__} on device {device}...")

        benchmark = make_benchmark(self.encoding_name(tensor_cls), "total", "")
        stage1_benchmark = make_benchmark(self.encoding_name(tensor_cls), "lookup_dict", "benchmark.pedantic(...)")
        stage2_benchmark = make_benchmark(self.encoding_name(tensor_cls), "query_codes", "benchmark.pedantic(...)")

        with wrap_query_methods(tensor_cls, ["equals", "less_than", "prefix"], benchmark, stage1_benchmark, stage2_benchmark):
            string_processing(benchmark, mssb_data, tensor_cls.Encoder, operators, device)
