import torch
import pytest
import contextlib
import time
from pytest_benchmark.plugin import BenchmarkFixture, BenchmarkSession
from itertools import product

from string_tensor import *
from mock_operator import *
from dataset import *
from toy import *


# 0.1 - 0.9, 1 - 10, 10 - 100
tpch_scale = [[*np.arange(0.1, 1, 0.1), *range(1, 10), *range(10, 110, 10)]]
tpch_col = [["l_shipmode"]]
tpch_predicate = [["eq"]]
tpch_group_name = [[""]]

# total_count, unique_count, max_length, predicate, selectivity
# Each outer list represents a series of parameter groups.
# The i-th inner list of each parameter (e.g., total_count[i], unique_count[i], etc.) forms a group.
# Within each group, the parameters are combined to form a grid of parameters.
# The groups are independent from each other.
# For example, you can pair total_count values ranging from 1e7 to 1e9 with unique_count values like 1e5 and 1e6 and test on only gpu,
# while pairing total_count values from 1e5 to 1e6 with a unique_count of 1e3 and test on both cpu and gpu.

# Parameters that remain constant across all groups can be specified as:
#   - A single value (e.g., "cpu") → will be automatically lifted to [[value]] and broadcasted to all groups.
#   - A one-dimensional list (e.g., ["cpu", "cuda"]) → will be lifted to [[...]] and broadcasted.
# This allows you to keep the configuration concise.

# Each parameter group can optionally be assigned a name (e.g., "fix-total-vary-unique", "fix-unique-vary-total").
# These names will be included in benchmark output CSVs to help categorize and analyze results across groups.

# 10^4 to 10^9
mssb_total_count = [[10 ** i for i in range(4, 10)]]
mssb_unique_count = [[1000]]
mssb_max_length = [[20]]
mssb_predicate = [["eq"]]
mssb_selectivity = [[0.3]]
mssb_group_name = [[""]]

tensor_cls = [[
    PlainEncodingStringColumnTensor,
    CPlainEncodingStringColumnTensor,
    DictionaryEncodingStringColumnTensor,
    CDictionaryEncodingStringColumnTensor,
    UnsortedDictionaryEncodingStringColumnTensor,
    UnsortedCDictionaryEncodingStringColumnTensor
]]

device = [["cpu", "cuda"]]

def params_grid_groups(*args):
    args = [
        [[arg]] if not isinstance(arg, list) else 
        [arg] if not isinstance(arg[0], list) else
        arg
        for arg in args
    ]
    n_groups = max(len(arg) for arg in args)
    args = [arg * n_groups if len(arg) == 1 and n_groups > 1 else arg for arg in args]

    return [
        param_set
        for group in zip(*args)
        for param_set in product(*group)
    ]

tpch_params = params_grid_groups(
    tpch_group_name,
    tpch_scale,
    tpch_col,
    tpch_predicate,
    device,
    tensor_cls,
)

mssb_params = params_grid_groups(
    mssb_group_name,
    mssb_total_count,
    mssb_unique_count,
    mssb_max_length,
    mssb_predicate,
    mssb_selectivity,
    device,
    tensor_cls,
)

tpch_col_gen = {
    scale: {col for _, s, col, _, _, _ in tpch_params if s == scale}
    for _, scale, _, _, _, _ in tpch_params
}

class StringTensorTestContext(NamedTuple):
    meta: StringColumnMetadata
    tensors: StringTensorDict
    op: MockOperator
    expected: torch.Tensor | None


toy_tracing_enabled = lambda *_, **__: os.getenv("TOY_TRACING", "").lower() in {"1", "true", "yes", "on"}

@pytest.fixture(scope="class")
@conditional(toy_tracing_enabled)
@cuda_trace
def tpch_context(col: str, scale: float, predicate: str) -> StringTensorTestContext:
    gen_tpch_col(scale, tpch_col_gen[scale], "dataset/tpch_data")
    meta, tensors = load_tpch_col(col, scale)

    query = meta.query_candidates[0][1]
    pred = MockPredicate[predicate](query)
    op = FilterScan(meta.column, pred)

    if meta.total_count <= 1000_000 and (strs := tensors["list"]):
        expected = op.apply(MockStringColumnTensor(strs))
    else:
        expected = None

    return StringTensorTestContext(meta, tensors, op, expected)

@pytest.fixture(scope="class")
@conditional(toy_tracing_enabled)
@cuda_trace
def mssb_context(total_count, unique_count, max_length, predicate, selectivity) -> StringTensorTestContext:

    gen_mssb_col(total_count, unique_count, max_length, predicate, [selectivity], "dataset/mssb_data")
    meta, tensors = load_mssb_col(total_count, unique_count, max_length, predicate, [selectivity])

    query = next(q for s, q in meta.query_candidates if s == selectivity)
    pred = MockPredicate[predicate](query)
    op = FilterScan(meta.column, pred)

    if meta.total_count <= 1000_000 and (strs := tensors["list"]):
        expected = op.apply(MockStringColumnTensor(strs))
    else:
        expected = None

    return StringTensorTestContext(meta, tensors, op, expected)

@pytest.fixture(scope="function")
def make_benchmark(request: pytest.FixtureRequest):
    bs: BenchmarkSession = request.config._benchmarksession  # type: ignore
    request._pyfuncitem.session

    if bs.skip:
        pytest.skip('Benchmarks are skipped (--benchmark-skip was used).')

    benchmarks: list[BenchmarkFixture] = []
    node = request.node
    marker = node.get_closest_marker('benchmark')
    options: dict[str, object] = dict(marker.kwargs) if marker else {}
    if 'timer' in options:
        from pytest_benchmark.utils import NameWrapper
        options['timer'] = NameWrapper(options['timer'])

    def make_benchmark(group: list[str], name: str | None = None, mode: str | None = None) -> BenchmarkFixture:
        fixture = BenchmarkFixture(
            node,
            add_stats=bs.benchmarks.append,
            logger=bs.logger,
            warner=request.node.warn,
            disabled=bs.disabled,
            **dict(bs.options, **options),
        )
        group = [g for g in (group if isinstance(group, list) else [group]) if g]
        fixture.group = " | ".join([fixture.group, *group]) if fixture.group and group else (fixture.group or " | ".join(group))
        fixture.name = fixture.name + " <" + name + ">" if fixture.name and name else (fixture.name or name)
        fixture.fullname = fixture.fullname  + " <" + name + ">" if fixture.fullname and name else (fixture.fullname or name)
        fixture._mode = mode or None
        benchmarks.append(fixture)
        return fixture

    yield make_benchmark

    for benchmark in benchmarks:
        benchmark._cleanup()

@pytest.fixture(scope="function")
def torch_profile(request: pytest.FixtureRequest):
    if not request.config.getoption("torch_profile"):
        yield
        return

    with (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # schedule=torch.profiler.schedule(wait=0, warmup=1, active=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./trace", request.node.name),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof,
    ):
        yield prof

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=100))

def torch_timer() -> float:
    if torch.empty(()).device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()

@conditional(toy_tracing_enabled)
@cuda_trace
def string_processing(benchmark: BenchmarkFixture, ctx: StringTensorTestContext, tensor_cls: type[StringColumnTensor], device: str):
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    with torch.device(device):
        _, tensors, op, expected = ctx
        tensor = transfer_col(tensors[tensor_cls.__name__], device)
        result = benchmark(op.apply, tensor)

        tuple_count = len(tensor)
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

        if expected is not None:
            assert torch.equal(expected.to(result.device), result), f"{tensor_cls.__name__} did not produce expected results."

def string_transfer(benchmark: BenchmarkFixture, ctx: StringTensorTestContext, tensor_cls: type[StringColumnTensor], device: str):
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    with torch.device(device):
        _, tensors, op, _ = ctx
        tensor = benchmark(transfer_col, tensors[tensor_cls.__name__], device)
        result = op.apply(tensor)

        tuple_count = len(tensor)
        query_result_size = len(result)
        tuple_element_size = tensor.tuple_size()
        total_size = tuple_count * tuple_element_size

        benchmark.extra_info["col"] = op.col_name
        benchmark.extra_info["op"] = op.__class__.__name__
        benchmark.extra_info["pred"] = op.predicate.__class__.__name__.replace("Predicate", "") if isinstance(op, FilterScan) else None
        benchmark.extra_info["val"] = op.predicate.value if isinstance(op, FilterScan) else None
        benchmark.extra_info["tuple_count"] = tuple_count
        benchmark.extra_info["query_result_size"] = query_result_size
        benchmark.extra_info["tuple_element_size_bytes"] = tuple_element_size
        benchmark.extra_info["total_size_bytes"] = total_size

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

@pytest.fixture(scope="session", autouse=True)
def toy_tracing(request: pytest.FixtureRequest):
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("TOY_TRACING", str(request.config.getoption("toy_trace")))
        yield

def tpch_param_id(params: tuple[str, float, str, str, str, type[StringColumnTensor]]) -> str:
    _, scale, col, predicate, device, tensor_cls = params
    return "-".join([
        f"{scale:.8g}",
        col,
        predicate,
        device,
        tensor_cls.Encoding
    ])

def mssb_param_id(params: tuple[str, int, int, int, str, float, str, type[StringColumnTensor]]) -> str:
    _, total_count, unique_count, max_length, predicate, selectivity, device, tensor_cls = params
    return "-".join([
        f"{total_count:.0g}".replace("e+", "e"),
        f"{unique_count:.0g}".replace("e+0", "e"),
        f"{max_length}",
        predicate,
        f"{selectivity:.3g}",
        device,
        tensor_cls.Encoding,
    ])

@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=3,
    timer=torch_timer,
)
class TestStringColumnTensor:

    @pytest.mark.parametrize("group,scale,col,predicate,device,tensor_cls", tpch_params, scope="class", ids=map(tpch_param_id, tpch_params))
    @pytest.mark.usefixtures("torch_profile")
    def test_tpch_string_processing(self, benchmark, tpch_context, tensor_cls: type[StringColumnTensor], scale, col, predicate, device, group):
        print(f"Testing string query processing with {tensor_cls.__name__} on scale {scale} and device {device}...")
        # benchmark.group += f" | FilterScan(l_shipmode==AIR)"
        # benchmark.group += f" | scale-{scale}"
        benchmark.group = "string_tensor_query_processing | TPCH"
        benchmark.group += f" | {group}" if group else ""
        string_processing(benchmark, tpch_context, tensor_cls, device)

    @pytest.mark.parametrize("group,total_count,unique_count,max_length,predicate,selectivity,device,tensor_cls", mssb_params, scope="class", ids=map(mssb_param_id, mssb_params))
    @pytest.mark.usefixtures("torch_profile")
    def test_mssb_string_processing(self, benchmark, mssb_context, tensor_cls: type[StringColumnTensor], total_count, unique_count, max_length, predicate, selectivity, device, group):
        print(f"Testing string query processing with {tensor_cls.__name__} on device {device}...")
        benchmark.group = "string_tensor_query_processing | MSSB"
        benchmark.group += f" | {group}" if group else ""
        string_processing(benchmark, mssb_context, tensor_cls, device)

    @pytest.mark.parametrize("group,total_count,unique_count,max_length,predicate,selectivity,device,tensor_cls", mssb_params, scope="class", ids=map(mssb_param_id, mssb_params))
    def test_mssb_staged_string_processing(self, make_benchmark, mssb_context, tensor_cls: type[StringColumnTensor], total_count, unique_count, max_length, predicate, selectivity, device, group):
        if not hasattr(tensor_cls, "query_equals_codes") or not hasattr(tensor_cls, "dictionary_cls"):
            pytest.skip(f"{tensor_cls.__name__} does not support staged query processing.")

        print(f"Testing string query processing with {tensor_cls.__name__} on device {device}...")

        group = "string_tensor_query_processing | MSSB Staged" + (f" | {group}" if group else "")
        benchmark = make_benchmark(group, "total", "")
        stage1_benchmark = make_benchmark(group, "lookup_dict", "benchmark.pedantic(...)")
        stage2_benchmark = make_benchmark(group, "query_codes", "benchmark.pedantic(...)")

        with wrap_query_methods(tensor_cls, ["equals", "less_than", "prefix"], benchmark, stage1_benchmark, stage2_benchmark):
            string_processing(benchmark, mssb_context, tensor_cls, device)

        stage1_benchmark.stats.extra_info = benchmark.stats.extra_info.copy()
        stage2_benchmark.stats.extra_info = benchmark.stats.extra_info.copy()

    @pytest.mark.parametrize("group,scale,col,predicate,device,tensor_cls", tpch_params, scope="class", ids=map(tpch_param_id, tpch_params))
    def test_tpch_string_transfer(self, benchmark, tpch_context, tensor_cls: type[StringColumnTensor], scale, col, predicate, device, group):
        if device == "cpu":
            pytest.skip("Skipping string transfer test on CPU")
        print(f"Testing string tensor transfer with {tensor_cls.__name__} on scale {scale} and device {device}...")
        benchmark.group = "string_tensor_query_processing | TPCH"
        benchmark.group += f" | {group}" if group else ""
        string_transfer(benchmark, tpch_context, tensor_cls, device)

    @pytest.mark.parametrize("group,total_count,unique_count,max_length,predicate,selectivity,device,tensor_cls", mssb_params, scope="class", ids=map(mssb_param_id, mssb_params))
    def test_mssb_string_transfer(self, benchmark, mssb_context, tensor_cls: type[StringColumnTensor], total_count, unique_count, max_length, predicate, selectivity, device, group):
        if device == "cpu":
            pytest.skip("Skipping string transfer test on CPU")
        print(f"Testing string tensor transfer with {tensor_cls.__name__} on device {device}...")
        benchmark.group = "string_tensor_query_processing | MSSB"
        benchmark.group += f" | {group}" if group else ""
        string_transfer(benchmark, mssb_context, tensor_cls, device)
