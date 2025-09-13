import torch
import pytest
import contextlib
import time
import gc
from pytest_benchmark.plugin import BenchmarkFixture, BenchmarkSession
from itertools import product

from string_tensor import *
from mock_operator import *
from dataset import *
from toy import *


# 0.1 - 0.9, 1 - 10, 10 - 100
# tpch_scale = [[*np.arange(0.1, 1, 0.1), *range(1, 10), *range(10, 110, 10)][0:2]]
tpch_scale = [[1, *range(10, 210, 10)]]
# tpch_scale = [[1]]
tpch_col = [["l_shipmode"]]
# tpch_predicate = [["lt","prefix"]]
tpch_predicate = [["eq","lt","prefix"]]
# tpch_predicate = [["eq"]]
tpch_return_mask = [[False, True]]
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
# mssb_total_count = [[10 ** i for i in range(6, 10)]]
mssb_total_count = [10**8]
mssb_unique_count = [10000]
mssb_max_length = [[20]]
mssb_predicate = [["eq","lt","prefix"]]
mssb_selectivity = [[0.15]]
mssb_return_mask = [[False, True]]
mssb_group_name = [[""]]

full_encoding = [
    PlainEncodingStringColumnTensor,
    CPlainEncodingStringColumnTensor,
    DictionaryEncodingStringColumnTensor,
    CDictionaryEncodingStringColumnTensor,
    UnsortedDictionaryEncodingStringColumnTensor,
    UnsortedCDictionaryEncodingStringColumnTensor
]
# tensor_cls = [full_encoding[0]]
tensor_cls = [[
    PlainEncodingStringColumnTensor,
    CPlainEncodingStringColumnTensor,
    DictionaryEncodingStringColumnTensor,
    CDictionaryEncodingStringColumnTensor,
    UnsortedDictionaryEncodingStringColumnTensor,
    UnsortedCDictionaryEncodingStringColumnTensor
]]
# tensor_cls = [[
#     CPlainEncodingStringColumnTensor
# ]]

# device = [["cpu", "cuda"][-1]]
device = [["cpu", "cuda"]]
# device = [["cuda"]]

# torch_compile = [[True]]
torch_compile = [[False, True]]

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
    tpch_return_mask,
    torch_compile,
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
    mssb_return_mask,
    torch_compile,
)

tpch_col_gen = {
    scale: {col for _, s, col, _, _, _, _, _ in tpch_params if s == scale}
    for _, scale, _, _, _, _, _, _ in tpch_params
}

class StringTensorTestContext(NamedTuple):
    meta: StringColumnMetadata
    tensors: StringTensorDict
    op: MockOperator
    expected: torch.Tensor | None

@pytest.fixture(scope="class")
@conditional(toy.on_global("toy_tracing"))
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
@conditional(toy.on_global("toy_tracing"))
@cuda_trace
def mssb_context(total_count, unique_count, max_length, predicate, selectivity) -> StringTensorTestContext:

    gen_mssb_col(total_count, unique_count, max_length * 1 // 2, max_length, predicate, [selectivity], "dataset/mssb_data")
    meta, tensors = load_mssb_col(total_count, unique_count, max_length, predicate, [selectivity])

    query = next(q for s, q in meta.query_candidates if s == float(format(selectivity, ".3g")))
    pred = MockPredicate[predicate](query)
    op = FilterScan(meta.column, pred)

    if meta.total_count <= 1000_000 and (strs := tensors["list"]):
        expected = op.apply(MockStringColumnTensor(strs))
    else:
        expected = None

    return StringTensorTestContext(meta, tensors, op, expected)

def clear_inductor_caches():
    import torch._inductor.utils
    import torch._inductor.codecache
    import torch._inductor.codegen.simd
    import torch._inductor.metrics

    for mod in torch._inductor.codecache.PyCodeCache.cache.values():
        sys.modules.pop(mod.__name__, None)

    torch._inductor.utils.clear_inductor_caches()
    torch._inductor.codegen.simd.SIMDScheduling.candidate_tilings.cache_clear()
    torch._inductor.metrics.reset()

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

@contextlib.contextmanager
def handle_error(benchmark: BenchmarkFixture):

    def is_referencing_tensors(obj, cache={}):
        if id(obj) in cache:
            return cache[id(obj)], 0
        cache[id(obj)] = False

        if isinstance(obj, torch.Tensor):
            if obj.is_cuda and obj.numel() > 0 and not isinstance(obj, torch._subclasses.FakeTensor):
                cache[id(obj)] = True
                return True, 1
            return False, 1

        refs = gc.get_referents(obj)
        if not refs:
            return False, 1

        steps = 0
        for ref in refs:
            found, count = is_referencing_tensors(ref, cache)
            steps += count
            if found:
                cache[id(obj)] = True
                return True, steps
    
        return False, steps

    def objects_referencing_tensors():
        objs = gc.get_objects()
        steps = [len(objs)]

        # referrers = {id(obj): gc.get_referrers(obj) for obj in objs}
        referrers: dict[int, list] = {}
        for o in objs:
            for r in gc.get_referents(o):
                referrers.setdefault(id(r), []).append(o)
        steps.append(sum(len(v) for v in referrers.values()))

        reachable_ids = set()
        new_ids = {id(obj) for obj in objs if isinstance(obj, torch.Tensor) and 
                   obj.is_cuda and obj.numel() > 0 and not isinstance(obj, torch._subclasses.FakeTensor)}

        while new_ids:
            reachable_ids |= new_ids
            steps.append(len(new_ids))
            new_ids = {id(ref) for oid in new_ids for ref in referrers.get(oid, [])}
            new_ids -= reachable_ids

        return reachable_ids, steps

    def stringify_and_release_traceback_locals_referencing_tensors(exc: BaseException):
        # This function will stringify and overwrite traceback locals to drop large tensor refs
        # freeing GPU memory while keeping the traceback printed normally
        import ctypes
        from _pytest._io.saferepr import saferepr, safeformat

        if exc.__cause__:
            stringify_and_release_traceback_locals_referencing_tensors(exc.__cause__)
        if exc.__context__ and not exc.__suppress_context__:
            stringify_and_release_traceback_locals_referencing_tensors(exc.__context__)

        _PyFrame_LocalsToFast = ctypes.pythonapi.PyFrame_LocalsToFast
        _PyFrame_LocalsToFast.argtypes = [ctypes.py_object, ctypes.c_int]
        _PyFrame_LocalsToFast.restype = None

        truncate_locals = live_instance_of[pytest.Config].get_verbosity() <= 1
        # truncate_args = live_instance_of[pytest.Config].get_verbosity() <= 2

        assert(exc.__traceback__)
        # Skip *this* frame
        tb = exc.__traceback__.tb_next

        # referencing_ids, _ = objects_referencing_tensors()
        while tb is not None:
            f = tb.tb_frame
            f_locals = f.f_locals
            for k, v in list(f_locals.items()):
                # if id(v) in referencing_ids:
                if is_referencing_tensors(v)[0]:
                    f_locals[k] = saferepr(v) if truncate_locals else safeformat(v)
            _PyFrame_LocalsToFast(f, 1)
            tb = tb.tb_next

    try:

        yield

    except Exception as e:
        stats = benchmark._make_stats(0) if benchmark.stats is None else benchmark.stats
        stats.extra_info["error"] = repr(e)
        stats = stats.stats
        stats.as_dict = lambda: {field: getattr(stats, field) if stats.data else 0 for field in stats.fields}
        stringify_and_release_traceback_locals_referencing_tensors(e)
        raise e

@contextlib.contextmanager
def torch_profile(benchmark: BenchmarkFixture):
    node: pytest.Function = benchmark._warner.__self__

    if not node.config.getoption("torch_profile"):
        yield
        return

    with (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            # schedule=torch.profiler.schedule(wait=0, warmup=1, active=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./trace", node.name),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof,
    ):
        yield prof

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=100))

@contextlib.contextmanager
def toy_trace():
    if not toy.on_global("toy_tracing")():
        yield
        return
    
    with toy.trace(toy.cuda_tracer()):
        yield

@contextlib.contextmanager
def torch_device(device: str):
    with torch.device(device):
        yield

@contextlib.contextmanager
def torch_clean(device: str):
    if device == "cuda":
        torch.cuda.synchronize()

    torch.compiler.reset()
    clear_inductor_caches()
    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()
    yield

def torch_timer() -> float:
    if torch.empty(()).is_cuda:
        torch.cuda.synchronize()
    return time.perf_counter()

# @conditional(toy.on_global("toy_tracing"), cuda_trace)
def string_processing(benchmark: BenchmarkFixture, ctx: StringTensorTestContext, tensor_cls: type[StringColumnTensor], device: str, return_mask: bool, torch_compile: bool):

    with torch_device(device), torch_clean(device), handle_error(benchmark), torch_profile(benchmark), toy_trace():
        meta, tensors, op, expected = ctx
        tensor = transfer_col(tensors[tensor_cls.__name__], device)

        if torch_compile:
            apply_op = torch.compile(op.apply)
        else:
            apply_op = op.apply

        result = benchmark(apply_op, tensor, return_mask)
        if return_mask:
            result = result.nonzero().view(-1)

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

        total, uniq, sel = meta.total_count, meta.unique_count, meta.selectivity_list[0]
        assert query_result_size == pytest.approx(
            uniq + (total - uniq) * sel, rel=0.05
        ), f"Result size {query_result_size} not matching expected {(total - uniq) * sel} for selectivity {sel}, total {total}, unique {uniq}"

        if expected is not None:
            assert torch.equal(expected.to(result.device), result), f"{tensor_cls.__name__} did not produce expected results."

def string_transfer(benchmark: BenchmarkFixture, ctx: StringTensorTestContext, tensor_cls: type[StringColumnTensor], device: str):

    with torch_device(device), torch_clean(device), handle_error(benchmark):
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
        query_dict = getattr(target_cls, f"query_{predicate_name}_lookup_dict")
        query_encoded = getattr(target_cls, f"query_{predicate_name}_match_encoded")
        query_methods[predicate_name] = query_method

        def wrapped_query(self, query: str, return_mask=False, query_dict=query_dict, query_encoded=query_encoded) -> torch.Tensor:
            t = torch_timer()
            selector = query_dict(self, query, return_mask)
            t_query_dict = torch_timer() - t
            t += t_query_dict
            match = query_encoded(self, selector, return_mask)
            t_query_codes = torch_timer() - t
            if benchmark.stats: # skip warmup iterations, where benchmark.stats is empty
                stage1_stats.update(t_query_dict)
                stage2_stats.update(t_query_codes)
            return match

        setattr(target_cls, f"query_{predicate_name}", wrapped_query)

    yield

    for predicate_name, query_method in query_methods.items():
        setattr(target_cls, f"query_{predicate_name}", query_method)

@pytest.fixture(scope="session", autouse=True)
def toy_tracing(request: pytest.FixtureRequest):
    with toy.GlobalContext().set_global("toy_tracing", request.config.getoption("toy_trace")):
        yield

def tpch_param_id(params: tuple[str, float, str, str, str, type[StringColumnTensor], bool, bool]) -> str:
    _, scale, col, predicate, device, tensor_cls, return_mask, torch_compile = params
    return "-".join([
        f"{scale:.8g}",
        col,
        predicate,
        device,
        tensor_cls.Encoding,
        "mask" if return_mask else "index",
        "compile" if torch_compile else "eager",
    ])

def mssb_param_id(params: tuple[str, int, int, int, str, float, str, type[StringColumnTensor], bool, bool]) -> str:
    _, total_count, unique_count, max_length, predicate, selectivity, device, tensor_cls, return_mask, torch_compile = params
    return "-".join([
        f"{total_count:.0g}".replace("e+", "e"),
        f"{unique_count:.0g}".replace("e+0", "e"),
        f"{max_length}",
        predicate,
        f"{selectivity:.3g}",
        device,
        tensor_cls.Encoding,
        "mask" if return_mask else "index",
        "compile" if torch_compile else "eager",
    ])

@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=3,
    timer=torch_timer,
)
class TestStringColumnTensor:

    @pytest.mark.parametrize("group,scale,col,predicate,device,tensor_cls,return_mask,torch_compile", tpch_params, scope="class", ids=map(tpch_param_id, tpch_params))
    def test_tpch_string_processing(self, benchmark, tpch_context, tensor_cls: type[StringColumnTensor], scale, col, predicate, device, return_mask, torch_compile, group):
        print(f"Testing TPCH string tensor query: scale={scale}, col={col}, predicate={predicate}, device={device}, tensor={tensor_cls.__name__}, return_mask={return_mask}, torch_compile={torch_compile}")
        # benchmark.group += f" | FilterScan(l_shipmode==AIR)"
        # benchmark.group += f" | scale-{scale}"
        benchmark.group = "string_tensor_query_processing | TPCH"
        benchmark.group += f" | {group}" if group else ""
        string_processing(benchmark, tpch_context, tensor_cls, device, return_mask, torch_compile)

    @pytest.mark.parametrize("group,total_count,unique_count,max_length,predicate,selectivity,device,tensor_cls,return_mask,torch_compile", mssb_params, scope="class", ids=map(mssb_param_id, mssb_params))
    def test_mssb_string_processing(self, benchmark, mssb_context, tensor_cls: type[StringColumnTensor], total_count, unique_count, max_length, predicate, selectivity, device, return_mask, torch_compile, group):
        print(f"Testing MSSB string tensor query: total_count={total_count}, unique_count={unique_count}, max_length={max_length}, predicate={predicate}, selectivity={selectivity}, device={device}, tensor={tensor_cls.__name__}, return_mask={return_mask}, torch_compile={torch_compile}")
        benchmark.group = "string_tensor_query_processing | MSSB"
        benchmark.group += f" | {group}" if group else ""
        string_processing(benchmark, mssb_context, tensor_cls, device, return_mask, torch_compile)

    @pytest.mark.parametrize("group,total_count,unique_count,max_length,predicate,selectivity,device,tensor_cls,return_mask,torch_compile", mssb_params, scope="class", ids=map(mssb_param_id, mssb_params))
    def test_mssb_staged_string_processing(self, make_benchmark, mssb_context, tensor_cls: type[StringColumnTensor], total_count, unique_count, max_length, predicate, selectivity, device, return_mask, torch_compile, group):
        if not hasattr(tensor_cls, "query_equals_match_encoded") or not hasattr(tensor_cls, "dictionary_cls"):
            pytest.skip(f"{tensor_cls.__name__} does not support staged query processing.")

        print(f"Testing MSSB staged string tensor query: total_count={total_count}, unique_count={unique_count}, max_length={max_length}, predicate={predicate}, selectivity={selectivity}, device={device}, tensor={tensor_cls.__name__}, return_mask={return_mask}, torch_compile={torch_compile}")

        group = "string_tensor_query_processing | MSSB Staged" + (f" | {group}" if group else "")
        benchmark = make_benchmark(group, "total", "")
        stage1_benchmark = make_benchmark(group, "lookup_dict", "benchmark.pedantic(...)")
        stage2_benchmark = make_benchmark(group, "match_codes", "benchmark.pedantic(...)")

        with wrap_query_methods(tensor_cls, ["equals", "less_than", "prefix"], benchmark, stage1_benchmark, stage2_benchmark):
            string_processing(benchmark, mssb_context, tensor_cls, device, return_mask, torch_compile)

        stage1_benchmark.stats.extra_info = benchmark.stats.extra_info.copy()
        stage2_benchmark.stats.extra_info = benchmark.stats.extra_info.copy()
        benchmark.extra_info["stage"] = "total"
        stage1_benchmark.extra_info["stage"] = "lookup_dict"
        stage2_benchmark.extra_info["stage"] = "match_codes"

    @pytest.mark.parametrize("group,scale,col,predicate,device,tensor_cls,return_mask,torch_compile", tpch_params, scope="class", ids=map(tpch_param_id, tpch_params))
    def test_tpch_string_transfer(self, benchmark, tpch_context, tensor_cls: type[StringColumnTensor], scale, col, predicate, device, return_mask, torch_compile, group):
        if device == "cpu" or return_mask or torch_compile:
            pytest.skip("Skipping string transfer test on CPU or with return mask or with torch compile")
        print(f"Testing TPCH string tensor transfer: scale={scale}, col={col}, predicate={predicate}, device={device}, tensor={tensor_cls.__name__}, return_mask={return_mask}, torch_compile={torch_compile}")
        benchmark.group = "string_tensor_query_processing | TPCH"
        benchmark.group += f" | {group}" if group else ""
        string_transfer(benchmark, tpch_context, tensor_cls, device)

    @pytest.mark.parametrize("group,total_count,unique_count,max_length,predicate,selectivity,device,tensor_cls,return_mask,torch_compile", mssb_params, scope="class", ids=map(mssb_param_id, mssb_params))
    def test_mssb_string_transfer(self, benchmark, mssb_context, tensor_cls: type[StringColumnTensor], total_count, unique_count, max_length, predicate, selectivity, device, return_mask, torch_compile, group):
        if device == "cpu" or return_mask or torch_compile:
            pytest.skip("Skipping string transfer test on CPU or with return mask or with torch compile")
        print(f"Testing MSSB string tensor transfer: total_count={total_count}, unique_count={unique_count}, max_length={max_length}, predicate={predicate}, selectivity={selectivity}, device={device}, tensor={tensor_cls.__name__}, return_mask={return_mask}, torch_compile={torch_compile}")
        benchmark.group = "string_tensor_query_processing | MSSB"
        benchmark.group += f" | {group}" if group else ""
        string_transfer(benchmark, mssb_context, tensor_cls, device)
