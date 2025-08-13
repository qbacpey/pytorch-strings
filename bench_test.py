import torch
import pytest
import contextlib
import time
from pytest_benchmark.plugin import BenchmarkFixture, BenchmarkSession
from itertools import chain, product

from string_tensor import *
from mock_operator import *
from dataset import *


# 0.1 - 0.9, 1 - 10, 10 - 100
tpch_scale = [[*np.arange(0.1, 1, 0.1), *range(1, 10), *range(10, 110, 10)]]
tpch_col = [["l_shipmode"]]
tpch_predicate = [["eq"]]

# total_count, unique_count, max_length, predicate, selectivity_list
# Each outer list represents a series of parameter groups.
# The i-th inner list of each parameter (e.g., total_count[i], unique_count[i], etc.) forms a group.
# Within each group, the parameters are combined to form a grid of parameters.
# The groups are independent from each other.
# For example, you can pair total_count values ranging from 1e7 to 1e9 with unique_count values like 1e5 and 1e6 and test on only gpu,
# while pairing total_count values from 1e5 to 1e6 with a unique_count of 1e3 and test on both cpu and gpu.

# 10^4 to 10^9
mssb_total_count = [[10 ** i for i in range(4, 10)]]
mssb_unique_count = [[1000]]
mssb_max_length = [[20]]
mssb_predicate = [["eq"]]
mssb_selectivity = [[0.3]]


tensor_cls = [[
    PlainEncodingStringColumnTensor,
    CPlainEncodingStringColumnTensor,
    DictionaryEncodingStringColumnTensor,
    CDictionaryEncodingStringColumnTensor,
    UnsortedDictionaryEncodingStringColumnTensor,
    UnsortedCDictionaryEncodingStringColumnTensor
]]

device = [["cpu", "cuda"]]

tpch_params = [
    param_set
    for group in zip(
        tensor_cls,
        tpch_scale,
        tpch_col,
        tpch_predicate,
        device
    )
    for param_set in product(*group)
]

mssb_params = [
    param_set
    for group in zip(
        tensor_cls,
        mssb_total_count,
        mssb_unique_count,
        mssb_max_length,
        mssb_predicate,
        mssb_selectivity,
        device,
    )
    for param_set in product(*group)
]

tpch_col_gen = {
    s: {col
        for scales, cols in zip(tpch_scale, tpch_col)
        if s in scales
        for col in cols
    }
    for s in chain.from_iterable(tpch_scale)
}

class StringTensorTestContext(NamedTuple):
    meta: StringColumnMetadata
    tensors: StringTensorDict
    op: MockOperator
    expected: torch.Tensor | None

@pytest.fixture(scope="class")
def tpch_context(col: str, scale: float, predicate: str,device: str) -> StringTensorTestContext:
    gen_tpch_col(scale, tpch_col_gen[scale], "dataset/tpch_data")
    meta, tensors = load_tpch_col(col, scale, device)

    query = meta.query_candidates[0][1]
    pred = MockPredicate[predicate](query)
    op = FilterScan(meta.column, pred)

    if meta.total_count <= 1000_000 and (strs := tensors["list"]):
        with torch.device(device):
            expected = op.apply(MockStringColumnTensor(strs))
    else:
        expected = None

    return StringTensorTestContext(meta, tensors, op, expected)

@pytest.fixture(scope="class")
def mssb_context(total_count, unique_count, max_length, predicate, selectivity, device) -> StringTensorTestContext:

    gen_mssb_col(total_count, unique_count, max_length, predicate, [selectivity], "dataset/mssb_data")
    meta, tensors = load_mssb_col(total_count, unique_count, max_length, predicate, [selectivity], device)

    query = next(q for s, q in meta.query_candidates if s == selectivity)
    pred = MockPredicate[predicate](query)
    op = FilterScan(meta.column, pred)

    if meta.total_count <= 1000_000 and (strs := tensors["list"]):
        with torch.device(device):
            expected = op.apply(MockStringColumnTensor(strs))
    else:
        expected = None

    return StringTensorTestContext(meta, tensors, op, expected)

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
        fixture.fullname = fixture.fullname  + " <" + name + ">" if fixture.fullname and name else (fixture.fullname or name)
        fixture._mode = mode or None
        benchmarks.append(fixture)
        return fixture

    yield make_benchmark

    for benchmark in benchmarks:
        benchmark._cleanup()

def string_processing(benchmark: BenchmarkFixture, ctx: StringTensorTestContext, encoder: type[StringColumnTensor], device: str):
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    with torch.device(device):
        _, tensors, op, expected = ctx
        tensor = tensors[encoder.__name__]
        tuple_count = len(tensor)
        result = op.apply(tensor)
        query_result_size = len(result)
        tuple_element_size = tensor.tuple_size()
        total_size = tuple_count * tuple_element_size

        print(f"Applying operator on column with {tuple_count} elements...")
        print(f"Produced {query_result_size} results.")
        print(f"Tuple element size: {tuple_element_size} bytes")
        print(f"Total size: {total_size} bytes")

        benchmark.extra_info["col"] = op.col_name
        benchmark.extra_info["op"] = op.__class__.__name__
        benchmark.extra_info["pred"] = op.predicate.__class__.__name__.replace("Predicate", "") if isinstance(op, FilterScan) else None
        benchmark.extra_info["val"] = op.predicate.value if isinstance(op, FilterScan) else None
        benchmark.extra_info["tuple_count"] = tuple_count
        benchmark.extra_info["query_result_size"] = query_result_size
        benchmark.extra_info["tuple_element_size_bytes"] = tuple_element_size
        benchmark.extra_info["total_size_bytes"] = total_size

        benchmark(lambda: (
            op.apply(tensor),
            torch.cuda.synchronize() if device == "cuda" else None
        ))

        if expected is not None:
            assert torch.equal(expected, result), f"{encoder.__name__} did not produce expected results."

@contextlib.contextmanager
def wrap_query_methods(target_cls, predicates: list[str], benchmark: BenchmarkFixture, stage1_benchmark: BenchmarkFixture, stage2_benchmark: BenchmarkFixture):
    """
    Wraps the query methods of the target class to add additional staged statistics for benchmarking.
    """
    stage1_stats = stage1_benchmark.stats or stage1_benchmark._make_stats(1)
    stage2_stats = stage2_benchmark.stats or stage2_benchmark._make_stats(1)
    query_methods = {}

    for predicate_name in predicates:
        query_method = getattr(target_cls, f"query_{predicate_name}")
        query_dict = getattr(target_cls.dictionary_cls, f"query_{predicate_name}")
        query_codes = getattr(target_cls, f"query_{predicate_name}_codes")
        query_methods[predicate_name] = query_method

        def wrapped_query(self, query: str, query_dict=query_dict, query_codes=query_codes) -> torch.Tensor:
            t = time.perf_counter()
            codes = query_dict(self.dictionary, query)
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

    for predicate_name, query_method in query_methods.items():
        setattr(target_cls, f"query_{predicate_name}", query_method)

def tpch_param_id(params: tuple[type[StringColumnTensor], float, str, str, str]) -> str:
    tensor_cls, scale, col, predicate, device = params
    return "-".join([
        tensor_cls.__name__.replace("StringColumnTensor", ""),
        f"{scale:.8g}",
        col,
        predicate,
        device
    ])

def mssb_param_id(params: tuple[type[StringColumnTensor], int, int, int, str, float, str]) -> str:
    tensor_cls, total_count, unique_count, max_length, predicate, selectivity, device = params
    return "-".join([
        tensor_cls.Encoding,
        f"{total_count:.0g}".replace("e+", "e"),
        f"{unique_count:0g}".replace("e+0", "e"),
        f"{max_length}",
        predicate,
        f"{selectivity:.3g}",
        device
    ])

@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=3
)
class TestStringColumnTensor:

    @pytest.mark.parametrize("tensor_cls,scale,col,predicate,device", tpch_params, scope="class", ids=map(tpch_param_id, tpch_params))
    @pytest.mark.benchmark(group="string_tensor_query_processing | TPCH")
    def test_tpch_string_processing(self, benchmark, tpch_context, tensor_cls: type[StringColumnTensor], scale, col, predicate, device):
        print(f"Testing string query processing with {tensor_cls.__name__} on scale {scale} and device {device}...")
        # benchmark.group += f" | FilterScan(l_shipmode==AIR)"
        # benchmark.group += f" | scale-{scale}"
        string_processing(benchmark, tpch_context, tensor_cls, device)

    @pytest.mark.parametrize("tensor_cls,total_count,unique_count,max_length,predicate,selectivity,device", mssb_params, scope="class", ids=map(mssb_param_id, mssb_params))
    @pytest.mark.benchmark(group="string_tensor_query_processing | MSSB")
    def test_mssb_string_processing(self, benchmark, mssb_context, tensor_cls: type[StringColumnTensor], total_count, unique_count, max_length, predicate, selectivity, device):
        print(f"Testing string query processing with {tensor_cls.__name__} on device {device}...")
        string_processing(benchmark, mssb_context, tensor_cls, device)

    @pytest.mark.parametrize("tensor_cls,total_count,unique_count,max_length,predicate,selectivity,device", mssb_params, scope="class", ids=map(mssb_param_id, mssb_params))
    @pytest.mark.benchmark(group="string_tensor_query_processing | MSSB Staged")
    def test_mssb_staged_string_processing(self, make_benchmark, mssb_context, tensor_cls: type[StringColumnTensor], total_count, unique_count, max_length, predicate, selectivity, device):
        if not hasattr(tensor_cls, "query_equals_codes") or not hasattr(tensor_cls, "dictionary_cls"):
            pytest.skip(f"{tensor_cls.__name__} does not support staged query processing.")

        print(f"Testing string query processing with {tensor_cls.__name__} on device {device}...")

        benchmark = make_benchmark(tensor_cls.Encoding, "total", "")
        stage1_benchmark = make_benchmark(tensor_cls.Encoding, "lookup_dict", "benchmark.pedantic(...)")
        stage2_benchmark = make_benchmark(tensor_cls.Encoding, "query_codes", "benchmark.pedantic(...)")

        stage1_benchmark.extra_info = benchmark.extra_info
        stage2_benchmark.extra_info = benchmark.extra_info

        with wrap_query_methods(tensor_cls, ["equals", "less_than", "prefix"], benchmark, stage1_benchmark, stage2_benchmark):
            string_processing(benchmark, mssb_context, tensor_cls, device)
