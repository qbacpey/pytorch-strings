import torch
import pytest, pytest_benchmark.fixture
from typing import List, Dict, Any
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
            cols[col_name] = PlainEncodingStringColumnTensor.Encoder.encode(col)
        
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return MockTable(cols)

def string_processing(benchmark: pytest_benchmark.fixture.BenchmarkFixture, data: MockTable, encoder: type[StringColumnTensor.Encoder], operators: List[MockOperator], device: str):
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    with torch.device(device):
        col_names = [op.col_name for op in operators]
        cols = {col_name: encoder.encode(data.cols[col_name]) for col_name in col_names}
        for operator in operators:
            print(len(cols[operator.col_name]))
            print(len(operator.apply(cols[operator.col_name])))

        benchmark(lambda: (
            [operator.apply(cols[operator.col_name]) for operator in operators],
            torch.cuda.synchronize() if device == "cuda" else None
        ))

@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=3
)
class TestStringColumnTensor:

    @pytest.mark.parametrize("scale", [10], scope="class", ids=lambda scale: f"scale-{scale}")
    @pytest.mark.parametrize("tensor_cls", [PlainEncodingStringColumnTensor, CPlainEncodingStringColumnTensor, DictionaryEncodingStringColumnTensor, CDictionaryEncodingStringColumnTensor], ids=lambda cls: cls.__name__)
    @pytest.mark.parametrize("device", ["cuda"], scope="class")
    @pytest.mark.parametrize("operators", [
        [FilterScan('l_shipmode', PredicateEq('AIR'))],
    ], scope="class", ids=lambda ops: "")
    @pytest.mark.benchmark(group="string_tensor_query_processing | TPCH")
    def test_tpch_string_processing(self, benchmark, tpch_data, operators: List[MockOperator], tensor_cls: type[StringColumnTensor], scale: float, device: str):
        print(f"Testing string query processing with encoder {tensor_cls.__name__} on scale {scale} and device {device}...")
        # benchmark.group += f" | FilterScan(l_shipmode==AIR)"
        # benchmark.group += f" | scale-{scale}"
        string_processing(benchmark, tpch_data, tensor_cls.Encoder, operators, device) 

    @pytest.mark.parametrize("tensor_cls", [PlainEncodingStringColumnTensor, CPlainEncodingStringColumnTensor, DictionaryEncodingStringColumnTensor, CDictionaryEncodingStringColumnTensor], ids=lambda cls: cls.__name__)
    @pytest.mark.parametrize("device", ["cuda"], scope="class")
    @pytest.mark.parametrize("operators", [
        [FilterScan('0001', PredicateEq('tsokmptbcza'))],
        [FilterScan('0002', PredicateEq('tsokmptbcza'))],
        [FilterScan('0003', PredicateEq('tsokmptbcza'))],
        [FilterScan('0004', PredicateEq('tsokmptbcza'))],
        [FilterScan('0005', PredicateEq('tsokmptbcza'))],
    ], scope="class", ids=["nrows-1e4","nrows-1e5","nrows-1e6","nrows-1e7","nrows-1e8"])
    @pytest.mark.benchmark(group="string_tensor_query_processing | MSSB")
    def test_mssb_string_processing(self, benchmark, mssb_data, operators: List[MockOperator], tensor_cls: type[StringColumnTensor], device: str):
        print(f"Testing string query processing with encoder {tensor_cls.__name__} on device {device}...")
        # benchmark.group += f" | FilterScan(l_shipmode==AIR)"
        # benchmark.group += f" | scale-{scale}"
        string_processing(benchmark, mssb_data, tensor_cls.Encoder, operators, device)
