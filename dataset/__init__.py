from typing import NamedTuple, TypedDict, cast
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

def gen_mssb_col(total_count: int, unique_count: int, min_length: int, max_length: int, predicate: str, selectivity_list: list[float], path: str = "dataset/mssb_data"):
    from .gen_mssb_data import generate_and_save_mssb_data
    generate_and_save_mssb_data(total_count, unique_count, min_length, max_length, predicate, selectivity_list, path)

def construct_tensors(tensors: StringTensorDict) -> StringTensorDict:
    strs = tensors["list"]
    dict = tensors["DictionaryEncodingStringColumnTensor"]

    src = dict.dictionary.tensor()
    idxs = dict.encoded_tensor

    plain = PlainEncodingStringColumnTensor(src[idxs])
    cplain = CPlainEncodingStringColumnTensor(None, src.t()[:, idxs])

    src = PlainEncodingStringColumnTensor(src)
    dict = DictionaryEncodingStringColumnTensor(src, idxs)
    udict = UnsortedDictionaryEncodingStringColumnTensor(src, idxs).shuffle()

    src = CPlainEncodingStringColumnTensor(src.tensor())
    cdict = CDictionaryEncodingStringColumnTensor(src, idxs)
    ucdict = UnsortedCDictionaryEncodingStringColumnTensor(src, idxs).shuffle()

    tensors = cast(StringTensorDict,
        {tensor.__class__.__name__: tensor
            for tensor in [strs, plain, cplain, dict, udict, cdict, ucdict]})
    return tensors

def load_tpch_col(col_name: str, scale: float, path: str = "dataset/tpch_data") -> StringTensorData:
    """
    Get the specified column of TPCH data
    :param col_name: column name
    :param scale: data scale factor
    :return: data of the specified column
    """
    from .gen_tpch_data import get_tpch_catalog
    catalog = get_tpch_catalog(scale, path)
    if meta := catalog.get_col_record(col_name):
        tensors = catalog.get_col_data(meta.column)
        tensors = construct_tensors(tensors)
        return StringTensorData(meta, tensors)
    raise ValueError(f"Column '{col_name}' not found in catalog.")

def load_mssb_col(total_count: int, unique_count: int, max_length: int, predicate: str, selectivity_list: list[float], path: str = "dataset/mssb_data") -> StringTensorData:
    """
    Get the specified column of MSSB data
    :param col_name: column name
    :return: data of the specified column
    """
    from .gen_mssb_data import get_mssb_catalog
    catalog = get_mssb_catalog(path)
    if rec := catalog.get_col_record(total_count, unique_count, max_length, predicate, selectivity_list):
        tensors = catalog.get_col_data(rec.column)
        tensors = construct_tensors(tensors)
        return StringTensorData(rec, tensors)
    raise ValueError(f"Column with total_count={total_count}, unique_count={unique_count}, max_length={max_length}, predicate='{predicate}', selectivity_list={selectivity_list} not found in catalog.")

def transfer_col(src: T, device: str) -> T:
    match src:
        case PlainEncodingStringColumnTensor():
            return src.from_tensor(src.tensor().to(device))
        case DictionaryEncodingStringColumnTensor():
            return src.__class__(transfer_col(src.dictionary, device), src.encoded_tensor.to(device))
        case _:
            raise TypeError(f"Unsupported StringColumnTensor type: {type(src)}")
