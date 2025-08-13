from typing import NamedTuple, TypedDict
from string_tensor import *

class StringColumnMetadata(NamedTuple):
    file: str
    table: str
    column: str
    total_count: int
    unique_count: int
    max_length: int
    predicate: str
    selectivity_list: list[float]
    query_candidates: list[tuple[float, str]]

class StringTensorDict(TypedDict):
    list: list[str]
    PlainEncodingStringColumnTensor: PlainEncodingStringColumnTensor
    CPlainEncodingStringColumnTensor: CPlainEncodingStringColumnTensor
    DictionaryEncodingStringColumnTensor: DictionaryEncodingStringColumnTensor
    CDictionaryEncodingStringColumnTensor: CDictionaryEncodingStringColumnTensor
    UnsortedDictionaryEncodingStringColumnTensor: UnsortedDictionaryEncodingStringColumnTensor
    UnsortedCDictionaryEncodingStringColumnTensor: UnsortedCDictionaryEncodingStringColumnTensor

class StringTensorData(NamedTuple):
    meta: StringColumnMetadata
    tensors: StringTensorDict

def gen_tpch_col(scale: float, cols: list[str] | set[str], path: str = "dataset/tpch_data"):
    from .gen_tpch_data import generate_and_save_tpch_data
    generate_and_save_tpch_data(scale, cols, path)

def gen_mssb_col(total_count: int, unique_count: int, max_length: int, predicate: str, selectivity_list: list[float], path: str = "dataset/mssb_data"):
    from .gen_mssb_data import generate_and_save_mssb_data
    generate_and_save_mssb_data(total_count, unique_count, max_length, predicate, selectivity_list, path)

def load_tpch_col(col_name: str, scale: float, device: str, path: str = "dataset/tpch_data") -> StringTensorData:
    """
    Get the specified column of TPCH data
    :param col_name: column name
    :param scale: data scale factor
    :return: data of the specified column
    """
    from .gen_tpch_data import get_tpch_catalog
    catalog = get_tpch_catalog(scale, path)
    if rec := catalog.get_col_record(col_name):
        return catalog.get_col_data(rec.column, device)
    raise ValueError(f"Column '{col_name}' not found in catalog.")

def load_mssb_col(total_count: int, unique_count: int, max_length: int, predicate: str, selectivity_list: list[float], device: str, path: str = "dataset/mssb_data") -> StringTensorData:
    """
    Get the specified column of MSSB data
    :param col_name: column name
    :return: data of the specified column
    """
    from .gen_mssb_data import get_mssb_catalog
    catalog = get_mssb_catalog(path)
    if rec := catalog.get_col_record(total_count, unique_count, max_length, predicate, selectivity_list):
        return catalog.get_col_data(rec.column, device)
    raise ValueError(f"Column with total_count={total_count}, unique_count={unique_count}, max_length={max_length}, predicate='{predicate}', selectivity_list={selectivity_list} not found in catalog.")
