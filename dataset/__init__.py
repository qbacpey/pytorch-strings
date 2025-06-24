import numpy as np
from .gen_mssb_data import generate_and_save_mssb_data, MSSB_DataGenArgs, get_mssb_catalog
from .gen_tpch_data import generate_and_save_tpch_data, TPCH_DataGenArgs, get_tpch_catalog

tpch_data_gen_list: list[TPCH_DataGenArgs] = [
    # scale
    TPCH_DataGenArgs(scale=1),
    TPCH_DataGenArgs(scale=10),
]

mssb_data_gen_list: list[MSSB_DataGenArgs] = [
    # total_count, unique_count, max_length, predicate, selectivity_list, unit_test
    MSSB_DataGenArgs(10000, 1000, 20, "equal", [0.01, 0.1, 0.3, 0.5], False)
]

def load_tpch_col(col_name: str, scale: float = 1.0, path: str = "") -> list[str]:
    """
    Get the specified column of TPCH data
    :param col_name: column name
    :param scale: data scale factor
    :return: data of the specified column
    """
    for gen_args in tpch_data_gen_list:
        generate_and_save_tpch_data(gen_args.scale, path)
    catalog = get_tpch_catalog(scale, path)
    col_file = catalog.get_col_file(col_name)
    return np.load(col_file, allow_pickle=True)

def load_mssb_col(col_name: str, path: str = "") -> list[str]:
    """
    Get the specified column of MSSB data
    :param col_name: column name
    :return: data of the specified column
    """
    for gen_args in mssb_data_gen_list:
        generate_and_save_mssb_data(
            total_count=gen_args.total_count,
            unique_count=gen_args.unique_count,
            max_length=gen_args.max_length,
            predicate=gen_args.predicate,
            selectivity_list=gen_args.selectivity_list,
            path=path
        )
    catalog = get_mssb_catalog(path)
    col_file = catalog.get_col_file(col_name)
    return np.load(col_file, allow_pickle=True)

__all__ = [
    "load_tpch_col",
    "load_mssb_col",
]
