import torch
from typing import List, Dict, Any
from encodings import StringColumnTensor, StringEncoder, PlainEncoder, DictionaryEncoder, RowWiseDictionaryEncoder

class TestOperator:
    __test__ = False
    col_name: str
    def apply(self, col: StringColumnTensor) -> Any:
        raise NotImplementedError

class TestPredicate:
    __test__ = False
    def __init__(self, value: str):
        self.value: str = value
    def apply(self, col: StringColumnTensor) -> Any:
        raise NotImplementedError

class FilterScan(TestOperator):
    def __init__(self, col_name: str, predicate: TestPredicate):
        self.col_name: str = col_name
        self.predicate: TestPredicate = predicate
    def apply(self, col: StringColumnTensor) -> Any:
        return self.predicate.apply(col)

class Aggregate(TestOperator):
    def __init__(self, col_name: str):
        self.col_name: str = col_name
    def apply(self, col: StringColumnTensor) -> Any:
        return col.query_aggregate()

class Sort(TestOperator):
    def __init__(self, col_name: str, ascending: bool = True):
        self.col_name: str = col_name
        self.ascending: bool = ascending
    
    def apply(self, col: StringColumnTensor) -> Any:
        return col.query_sort(self.ascending)
    
class PredicateEq(TestPredicate):
    def apply(self, col: StringColumnTensor) -> Any:
        return col.query_equals(self.value)

class PredicateLt(TestPredicate):
    def apply(self, col: StringColumnTensor) -> Any:
        return col.query_less_than(self.value)

class PredicatePrefix(TestPredicate):
    def apply(self, col: StringColumnTensor) -> Any:
        return col.query_prefix(self.value)

@pytest.fixture(scope="class")
def tpch_data(scale: float) -> Dict[str, List[str]]:
    str_cols: Dict[str, List[str]] = {}
    print(f"Loading TPCH data with scale factor {scale}...")
    with duckdb.connect(":memory:") as con:
        con.execute(f"INSTALL tpch; LOAD tpch;CALL dbgen(sf = {scale});")
        for table_name in ['customer','orders','lineitem','supplier','part','partsupp','nation','region']:
            table = con.table(table_name)
            cols, names, types = table.fetchnumpy(), table.columns, table.types
            for col_name, col_type in zip(names, types):
                if col_type == duckdb.typing.VARCHAR:
                    str_cols[col_name] = cols[col_name].tolist()
    return str_cols

def string_processing(benchmark: pytest_benchmark.fixture.BenchmarkFixture, tpch_data: Dict[str, List[str]], encoder: StringEncoder, operators: List[TestOperator], device: str):
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    with torch.device(device):
        col_names = [op.col_name for op in operators]
        cols = {col_name: encoder.encode(tpch_data[col_name]) for col_name in col_names}
        for operator in operators:
            print(len(cols[operator.col_name]))
            print(len(operator.apply(cols[operator.col_name])))

        benchmark(lambda: (
            [operator.apply(cols[operator.col_name]) for operator in operators],
            torch.cuda.synchronize() if device == "cuda" else None
        ))

@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=3,
    group="string-query-processing",
)
class TestStringColumnTensor:

    # todo: scale and operators grouping/ids
    # todo: deal with large deviation
    # todo: compare with duckdb
    # todo: add dictionary encoding

    @pytest.mark.parametrize("scale", [0.01, 0.1, 1], scope="class", ids=lambda scale: f"scale-{scale}")
    @pytest.mark.parametrize("encoder", [PlainEncoder(), DictionaryEncoder(), RowWiseDictionaryEncoder()], ids=lambda encoder: encoder.__class__.__name__)
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    @pytest.mark.parametrize("operators", [
        [FilterScan('l_shipmode', PredicateEq('AIR'))],
    ])
    def test_string_processing(self, benchmark, tpch_data, operators: List[TestOperator], encoder: StringEncoder, scale: float, device: str):
        print(f"Testing string query processing with encoder {encoder.__class__.__name__} on scale {scale} and device {device}...")
        # benchmark.group += f" | FilterScan(l_shipmode==AIR)"
        # benchmark.group += f" | scale-{scale}"
        string_processing(benchmark, tpch_data, encoder, operators, device)