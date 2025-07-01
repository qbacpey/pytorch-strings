import numpy as np
from .gen_mssb_data import generate_and_save_mssb_data, MSSB_DataGenArgs, get_mssb_catalog
from .gen_tpch_data import generate_and_save_tpch_data, TPCH_DataGenArgs, get_tpch_catalog

tpch_data_gen_list: list[TPCH_DataGenArgs] = [
    # scale
    # TPCH_DataGenArgs(scale=1),
    # TPCH_DataGenArgs(scale=10),
    # TPCH_DataGenArgs(scale=50),
    # TPCH_DataGenArgs(scale=100),
]

mssb_data_gen_list: list[MSSB_DataGenArgs] = [
    # total_count, unique_count, max_length, predicate, selectivity_list, unit_test
    MSSB_DataGenArgs(10_000, 1000, 20, "equal", [0.01, 0.1, 0.3, 0.5], False),
    MSSB_DataGenArgs(100_000, 1000, 20, "equal", [0.01, 0.1, 0.3, 0.5], False),
    MSSB_DataGenArgs(1_000_000, 1000, 20, "equal", [0.01, 0.1, 0.3, 0.5], False),
    MSSB_DataGenArgs(10_000_000, 1000, 20, "equal", [0.01, 0.1, 0.3, 0.5], False),
    MSSB_DataGenArgs(100_000_000, 1000, 20, "equal", [0.01, 0.1, 0.3, 0.5], False),
    MSSB_DataGenArgs(500_000_000, 1000, 20, "equal", [0.01, 0.1, 0.3, 0.5], False),
]

gen_path = "dataset"

def generate_all_tpch_data(path: str = "dataset/tpch_data") -> None:
    for gen_args in tpch_data_gen_list:
        generate_and_save_tpch_data(gen_args.scale, path)

def generate_all_mssb_data(path: str = "dataset/mssb_data") -> None:
    for gen_args in mssb_data_gen_list:
        generate_and_save_mssb_data(
            total_count=gen_args.total_count,
            unique_count=gen_args.unique_count,
            max_length=gen_args.max_length,
            predicate=gen_args.predicate,
            selectivity_list=gen_args.selectivity_list,
            path=path
        )

generate_all_mssb_data(gen_path + "/mssb_data")
generate_all_tpch_data(gen_path + "/tpch_data")

def load_tpch_col(col_name: str, scale: float = 1.0, path: str = "dataset/tpch_data") -> list[str]:
    """
    Get the specified column of TPCH data
    :param col_name: column name
    :param scale: data scale factor
    :return: data of the specified column
    """
    catalog = get_tpch_catalog(scale, path)
    col_file = catalog.get_col_file(col_name)
    return np.load(col_file, allow_pickle=True)

def load_mssb_col(col_name: str, path: str = "dataset/mssb_data") -> list[str]:
    """
    Get the specified column of MSSB data
    :param col_name: column name
    :return: data of the specified column
    """
    catalog = get_mssb_catalog(path)
    col_file = catalog.get_col_file(col_name)
    return np.load(col_file, allow_pickle=True)

__all__ = [
    "load_tpch_col",
    "load_mssb_col",
]

if __name__ == "__main__":
    gen_all_tpch_data()
    gen_all_mssb_data()
