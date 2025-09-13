import argparse
import json
import os
import random
import string
import unittest
import duckdb
import numpy as np
from contextlib import redirect_stdout
from typing import Optional, cast
from collections import namedtuple
from typing import TYPE_CHECKING

from string_tensor import *
from dataset import StringColumnMetadata, StringTensorDict

def generate_mssb_data(total_count, unique_count, min_length, max_length, predicate: Optional[str] = None, selectivity_list: Optional[list[float]] = None) -> tuple[StringTensorDict, list[tuple[float, str]]]:
    """
    Generate string columns for OLAP testing.
    
    Parameters:
      - total_count: Total number of rows
      - unique_count: Target number of unique strings
      - max_length: Maximum length of each string (random length 1 ~ max_length)
      - predicate: Specified predicate type for reaching target selectivity, currently supports "equal", "less_than", "prefix"
      - selectivity_list: A list of target selectivities (decimal values between 0 and 1), e.g., [0.01, 0.1, 0.5]
    
    When predicate and selectivity_list are provided, the function ensures certain candidate values are reserved,
    so that queries with the corresponding predicate (equal, less than, prefix) will return approximately 
    round(s * total_count) rows, and returns a mapping of candidate query values (dictionary).
    
    Returns:
       data: List of strings (length is total_count)
       query_candidates: Dictionary with selectivity as key and candidate query string as value
    """
    def random_string(min_length: int = min_length, max_length: int = max_length) -> str:
        """Generate a random string with length between min_length and max_length."""
        l = random.randint(min_length, max_length)
        return ''.join(random.choices(string.ascii_letters + string.digits, k=l))

    def unique_random_strings(count: int, min_length: int = min_length, max_length: int = max_length, prefix="", exclude=[]) -> list[str]:
        """Generate count unique random strings with lengths between min_length and max_length."""
        uniq_strs = set()
        exclude = set(exclude)
        while len(uniq_strs) < count:
            s = prefix + random_string(max(min_length - len(prefix), 0), max_length - len(prefix))
            if s not in uniq_strs and s not in exclude:
                uniq_strs.add(s)
        return list(uniq_strs)

    def zipf_distribution(n: int, a: float = 1.2) -> np.ndarray:
        """Generate Zipf distribution frequency weights"""
        x = np.arange(1, n + 1)
        weights = 1 / (x ** a)
        weights = weights / weights.sum()
        return weights

    def sample(uniq_strs: list[str], weights: np.ndarray, total_count: int) -> StringTensorDict:
        """Sample from unique_strs according to weights to generate total_count data."""
        if total_count >= 2 ** 31:
            raise ValueError("Cannot generate more than 2^31 elements.")

        counts = ((total_count - len(uniq_strs)) * weights // weights.sum()).astype(int) + 1
        if counts.sum() < total_count:
            extra = total_count - counts.sum()
            counts += (extra // len(counts)).astype(int)
            counts[:(extra % len(counts)).astype(int)] += 1

        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            perm = torch.randperm(total_count, dtype=torch.int32).to(torch.int64) # requires 10x temporary space, so put first to ensure no other allocation happens before

            src = PlainEncodingStringColumnTensor.from_strings(uniq_strs).tensor()
            src, sorted_idxs = src.unique(dim=0, return_inverse=True)
            sorted_idxs = torch.empty_like(sorted_idxs).scatter_(0, sorted_idxs, torch.arange(len(sorted_idxs)))
            src = PlainEncodingStringColumnTensor.from_tensor(src)

            counts = torch.tensor(counts)[sorted_idxs]
            idxs = torch.arange(len(src))
            idxs = torch.repeat_interleave(idxs, counts) # repeat each index according to its count
            idxs = idxs[perm]  # shuffle the indices and we get the final distributed indices, also the encoded_tensor in dictionary encoding
            # del perm

            # # construct each StringColumnTensor based on src and idxs
            # plain = (PlainEncodingStringColumnTensor(src.tensor()[idxs].to("cpu")) if total_count > 1000_000 else
            #         PlainEncodingStringColumnTensor(src.tensor()[idxs]))
            # cplain = (CPlainEncodingStringColumnTensor(None, src.tensor().t()[:, idxs].to("cpu")) if total_count > 1000_000 else
            #         CPlainEncodingStringColumnTensor(None, src.tensor().t()[:, idxs]))

            dict = DictionaryEncodingStringColumnTensor(src, idxs)
            # udict = UnsortedDictionaryEncodingStringColumnTensor(src, idxs).shuffle()

            # src = CPlainEncodingStringColumnTensor(src.tensor())
            # cdict = CDictionaryEncodingStringColumnTensor(src, idxs)
            # ucdict = UnsortedCDictionaryEncodingStringColumnTensor(src, idxs).shuffle()

        final_indices = sorted_idxs[idxs].tolist() if total_count <= 1000_000 else []
        strs = [uniq_strs[i] for i in final_indices]
        tensors = [strs, dict]

        return cast(StringTensorDict, {tensor.__class__.__name__: tensor for tensor in tensors})

    print(f"Generating MSSB data: total_count={total_count}, unique_count={unique_count}, max_length={max_length}, predicate={predicate}, selectivity_list={selectivity_list}")

    # ===================================================
    # If no predicate is specified, use the simplest generation logic:
    if predicate is None or selectivity_list is None:
        # Generate unique_count different strings (ensure random length between 1~max_length)
        uniq_strs = unique_random_strings(unique_count)
        # Construct final data (each string repeated by its frequency), then shuffle
        weights = zipf_distribution(unique_count)
        data = sample(uniq_strs, weights, total_count)
        return data, []

    # ===================================================
    # Predicate is "equal"
    # Let the selectivity_list in front for the weight arr of uni string, the rest value would be broken down by Zipf distribution
    elif predicate.lower() in ["equal", "eq"]:
        uniq_strs = unique_random_strings(unique_count)
        # selectivity_list = sorted(selectivity_list)
        total_selectivity = sum(selectivity_list)
        num_selectivities = len(selectivity_list)
        if total_selectivity > 1:
            raise ValueError("Sum of selectivities cannot exceed 1")
        if num_selectivities > unique_count:
            raise ValueError("Number of selectivities cannot exceed unique_count")
        weights = np.append(selectivity_list, (zipf_distribution(unique_count - num_selectivities) * (1 - total_selectivity)))
        data = sample(uniq_strs, weights, total_count)
        query_candidates = [(sel, str) for sel, str in zip(selectivity_list, uniq_strs)]
        return data, query_candidates

    # ===================================================
    # Predicate is "less_than"
    elif predicate.lower() in ["less_than", "lt"]:
        uniq_strs = sorted(unique_random_strings(unique_count))
        selectivity_list = sorted(selectivity_list)
        total_selectivity = selectivity_list[-1]
        num_selectivities = len(selectivity_list)
        if total_selectivity > 1:
            raise ValueError("Sum of selectivities cannot exceed 1")
        if num_selectivities > unique_count:
            raise ValueError("Number of selectivities cannot exceed unique_count")

        diff_selectivities = np.diff(selectivity_list, prepend=0)
        cand_uniq_counts = np.floor(np.array(diff_selectivities) * unique_count).astype(int)
        cand_uniq_counts = np.ceil((cand_uniq_counts + 1) / 2).astype(int)
        cand_total_count = sum(cand_uniq_counts)
        weights = []
        for i, count in enumerate(cand_uniq_counts):
            weights.extend(zipf_distribution(count) * diff_selectivities[i])

        weights = np.append(weights, (zipf_distribution(unique_count - cand_total_count) * (1 - total_selectivity)))
        data = sample(uniq_strs, weights, total_count)

        ext_uniq_strs = uniq_strs + [chr(ord(uniq_strs[-1][0]) + 1)]  # Ensure the last string is larger than all others
        bound_strs = [ext_uniq_strs[i] for i in np.cumsum(cand_uniq_counts)]
        query_candidates = [(sel, str) for sel, str in zip(selectivity_list, bound_strs)]
        return data, query_candidates

    # ===================================================
    # Predicate is "prefix"
    elif predicate.lower().startswith("pre"):
        total_selectivity = sum(selectivity_list)
        num_selectivities = len(selectivity_list)
        if total_selectivity > 1:
            raise ValueError("Sum of selectivities cannot exceed 1")
        if num_selectivities > unique_count:
            raise ValueError("Number of selectivities cannot exceed unique_count")

        cand_uniq_counts = np.floor(np.array(selectivity_list) * unique_count).astype(int)
        cand_uniq_counts = np.ceil((cand_uniq_counts + 1) / 2).astype(int)
        cand_total_count = sum(cand_uniq_counts)
        uniq_strs, weights, prefixes = [], [], []
        for i, count in enumerate(cand_uniq_counts):
            prefix = random_string(min_length, min_length + (max_length - min_length) * 1 // 2)
            uniq_strs.extend(unique_random_strings(count, min_length, max_length, prefix, exclude=uniq_strs))
            weights.extend(zipf_distribution(count) * selectivity_list[i])
            prefixes.append(prefix)
        uniq_strs.extend(unique_random_strings(unique_count - cand_total_count, exclude=uniq_strs))
        weights = np.append(weights, zipf_distribution(unique_count - cand_total_count) * (1 - total_selectivity))
        data = sample(uniq_strs, weights, total_count)
        query_candidates = [(sel, str) for sel, str in zip(selectivity_list, prefixes)]
        return data, query_candidates
    else:
        raise ValueError("Unsupported predicate type")

# ================= Unit Tests ====================
class TestStringGenerator(unittest.TestCase):
    total_count: Optional[int] = None
    unique_count: Optional[int] = None
    max_length: Optional[int] = None
    selectivity_list: Optional[list] = None

    def test_general_generation(self):
        # total_count, unique_count, max_length = 1000, 100, 20
        total_count = self.total_count or 100
        unique_count = self.unique_count or 10
        max_length = self.max_length or 20

        data, candidates = generate_mssb_data(total_count, unique_count, 1, max_length)
        data = data["list"]
        self.assertEqual(len(data), total_count)
        for s in data:
            self.assertTrue(1 <= len(s) <= max_length)
        self.assertEqual(candidates, [])  # No candidate query values

    def test_equal_predicate(self):
        # total_count, unique_count, max_length = 1000, 120, 20
        total_count = self.total_count or 100
        unique_count = self.unique_count or 20
        max_length = self.max_length or 20
        selectivities = self.selectivity_list or [0.05, 0.2, 0.5]  # For example 5%, 20%, 50%
        data, candidates = generate_mssb_data(total_count, unique_count, 1, max_length,
                                                 predicate="equal",
                                                 selectivity_list=selectivities)
        data = data["list"]  
        candidates = {s: v for s, v in candidates}
        print("data:", data[:10])
        print("Equal query candidates:", candidates)

        self.assertEqual(len(data), total_count)
        self.assertEqual(len(set(data)), unique_count) 
        self.assertTrue(all(1 <= len(s) <= max_length for s in data))
        self.assertEqual(len(candidates), len(selectivities))
        # Verify that candidate queries return expected row counts for equality queries
        for s in selectivities:
            candidate_val = candidates[s]
            expected = round(s * (total_count - unique_count)) + 1
            actual = data.count(candidate_val)
            print(f"Selectivity {s}: expected {expected}, actual {actual}")
            self.assertAlmostEqual(actual, expected, delta=round(0.05 * total_count))  # Allow 5% error

    def test_less_than_predicate(self):
        # total_count, unique_count, max_length = 10000, 300, 15
        total_count = self.total_count or 100
        unique_count = self.unique_count or 20
        max_length = self.max_length or 20
        selectivities = self.selectivity_list or [0.1, 0.3, 0.7]
        data, candidates = generate_mssb_data(total_count, unique_count, 1, max_length,
                                                 predicate="less_than",
                                                 selectivity_list=selectivities)
        data = data["list"]
        candidates = {s: v for s, v in candidates}
        print("data:", data[:10])
        print("Less than query candidates:", candidates)

        self.assertEqual(len(data), total_count)
        self.assertEqual(len(set(data)), unique_count) 
        self.assertTrue(all(1 <= len(s) <= max_length for s in data))
        self.assertEqual(len(candidates), len(selectivities))
        for s in selectivities:
            candidate_val = candidates[s]
            # For less_than queries, count rows strictly less than candidate_val
            actual = sum(1 for x in data if x < candidate_val)
            expected = round(s * (total_count - unique_count)) + 1
            print(f"Selectivity {s}: expected {expected}, actual {actual}")
            self.assertAlmostEqual(actual, expected, delta=round(0.05 * total_count))

    def test_prefix_predicate(self):
        # total_count, unique_count, max_length = 1000, 120, 20
        total_count = self.total_count or 100
        unique_count = self.unique_count or 20
        max_length = self.max_length or 20
        selectivities = self.selectivity_list or [0.1, 0.4]
        data, candidates = generate_mssb_data(total_count, unique_count, 1, max_length,
                                                 predicate="prefix",
                                                 selectivity_list=selectivities)
        data = data["list"]
        candidates = {s: v for s, v in candidates}
        print("data:", data[:10])
        print("Prefix query candidates:", candidates)

        self.assertEqual(len(data), total_count)
        self.assertEqual(len(set(data)), unique_count) 
        self.assertTrue(all(1 <= len(s) <= max_length for s in data))
        self.assertEqual(len(candidates), len(selectivities))
        for s in selectivities:
            candidate_val = candidates[s]
            actual = sum(1 for x in data if x.startswith(candidate_val))
            expected = round(s * (total_count - unique_count)) + 1
            print(f"Selectivity {s}: expected {expected}, actual {actual}")
            self.assertAlmostEqual(actual, expected, delta=round(0.05 * total_count))
# ==============================================

def generate_and_save_mssb_data(total_count, unique_count, min_length, max_length, predicate, selectivity_list, path="mssb_data"):
    """
    Generate test data columns based on input parameters:
      total_count, unique_count, max_length, predicate, selectivity
    Call generate_mssb_data() to generate data columns (selectivity passed as list [selectivity]),
    then save as pt format to a unified directory, and save parameters and query candidate mapping to the mapping file catalog.txt,
    file names use sequentially incrementing numbers (e.g., 0001.pt).
    
    If a file with the same parameters already exists, don't generate again, just return the path to the corresponding file.
    
    Returns: Path to the generated or existing pt file.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    catalog = Catalog.load(path)
    
    # Check if a record with the same parameters already exists
    if rec := catalog.get_col_record(total_count, unique_count, max_length, predicate, selectivity_list):
        print(f"Dataset already exists: {rec.file}, skipping generation.")
        return

    # No matching record found, generate a new dataset
    # Call external generate_mssb_data function (returns data, query_candidates)
    data, query_candidates = generate_mssb_data(total_count, unique_count, min_length, max_length, predicate, selectivity_list)

    if data["list"]:
        with torch.device("cuda" if torch.cuda.is_available() else "cpu"):
            strs = data["list"]
            # plain = data["PlainEncodingStringColumnTensor"]
            dict = data["DictionaryEncodingStringColumnTensor"]
            # udict = data["UnsortedDictionaryEncodingStringColumnTensor"]

            # plain_expected = PlainEncodingStringColumnTensor.from_strings(strs)
            dict_expected = DictionaryEncodingStringColumnTensor.from_strings(strs)
            # udict_expected = UnsortedDictionaryEncodingStringColumnTensor.from_string_tensor(plain_expected)

            # assert (
            #     plain.tensor().equal(plain_expected.encoded_tensor)
            # ), "PlainEncodingStringColumnTensor does not match expected."
            assert (
                dict.encoded_tensor.equal(dict_expected.encoded_tensor) and
                dict.dictionary.tensor().equal(dict_expected.dictionary.tensor())
            ), "DictionaryEncodingStringColumnTensor does not match expected."
            # assert (
            #     udict.encoded_tensor.equal(udict_expected.encoded_tensor) and
            #     udict.dictionary.encoded_tensor.equal(udict_expected.dictionary.encoded_tensor)
            # ), "UnsortedDictionaryEncodingStringColumnTensor does not match expected."
    

    # Determine new File ID (sequentially incrementing, like 0001,0002,...)
    if catalog.records:
        new_id = max(int(rec.column) for rec in catalog.records) + 1
    else:
        new_id = 1
    column = f"{new_id:04d}"
    file_name = f"{column}.pt"
    file_path = os.path.join(path, file_name)

    # Save data as pytorch file
    torch.save(data, file_path)

    new_record = CatalogRecord(file_name, "MSSB", column, total_count, unique_count, max_length, predicate, selectivity_list, query_candidates)
    catalog.records.append(new_record)
    catalog.save()

class CatalogRecord(StringColumnMetadata):
    pass

class Catalog:
    def __init__(self, records: list[CatalogRecord], path: str):
        self.path = path
        self.records = records

    def get_col_record(self, total_count: int, unique_count: int, max_length: int, predicate: str, selectivity_list: list[float]) -> CatalogRecord | None:
        for rec in self.records:
            if (rec.total_count == total_count and
                rec.unique_count == unique_count and
                rec.max_length == max_length and
                rec.predicate == predicate and
                rec.selectivity_list == [float(format(sel, ".3g")) for sel in selectivity_list]):
                return rec
        return None

    def get_col_data(self, col_name: str) -> StringTensorDict:
        for record in self.records:
            if record.column == col_name:
                file_name = os.path.join(self.path, record.file)
                data: StringTensorDict = torch.load(file_name, map_location="cpu")
                return data
        raise ValueError(f"Column {col_name} not found in catalog.")

    def save(self):
        # Write to self.path (output all records in tabular format)
        catalog_file = os.path.join(self.path, "catalog.txt")
        with duckdb.connect(":memory:") as con:
            con.execute("CREATE TABLE catalog (file_name VARCHAR, table_name VARCHAR, column_name VARCHAR, total_count INTEGER, unique_count INTEGER, max_length INTEGER, predicate VARCHAR, selectivity_list JSON, query_candidates JSON)")
            for rec in self.records:
                con.execute(
                    "INSERT INTO catalog VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (rec.file, rec.table, rec.column, rec.total_count, rec.unique_count,
                    rec.max_length, rec.predicate,
                    json.dumps([format(x, ".3g") for x in rec.selectivity_list]),
                    json.dumps([(format(x, ".3g"), y) for x, y in rec.query_candidates]))
                )
            with open(catalog_file, "w", encoding="utf-8") as f, redirect_stdout(f):
                con.table("catalog").show(max_rows=10000, max_width=10000) # type: ignore

    @classmethod
    def load(cls, path: str) -> "Catalog":
        """
        Load catalog from file
        :return: self
        """
        catalog_file = os.path.join(path, "catalog.txt")
        if not os.path.exists(catalog_file):
            return cls([], path=path)

        with open(catalog_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        records = []        
        for line in lines[4:]:  # Skip header and separator lines
            values = [part.strip() for part in line.split("│")][1:-1]  # Remove leading/trailing whitespace and empty parts
            if len(values) < 7:
                continue
            record = CatalogRecord(
                file=values[0],
                table=values[1],
                column=values[2],
                total_count=int(values[3]),
                unique_count=int(values[4]),
                max_length=int(values[5]),
                predicate=values[6],
                selectivity_list=[float(x) for x in json.loads(values[7])],
                query_candidates=[(float(x), y) for x, y in json.loads(values[8])]
            )
            records.append(record)
        return cls(records, path=path)

def get_mssb_catalog(path: str = "mssb_data") -> Catalog:
    """
    Load dataset mapping records from catalog.txt file and return a Catalog object.
    If catalog.txt doesn't exist, return an empty Catalog object.
    """
    catalog_file = os.path.join(path, "catalog.txt")
    
    if not os.path.exists(catalog_file):
        print(f"Catalog file {catalog_file} does not exist. Please generate MSSB data first.")
    return Catalog.load(path)

# --------------------------------------------------
# If running this module directly, parameters can be passed via command line

# Create a named tuple type for default arguments
MSSB_DataGenArgs = namedtuple('MSSB_DataGenArgs', ['total_count', 'unique_count', 'max_length', 'predicate', 'selectivity_list', 'unit_test'])

# Initialize with default values
# specified_args = MSSB_DataGenArgs(100, 10, 20, "equal", [0.1, 0.3, 0.5], False)
specified_args = MSSB_DataGenArgs(1_000_000, int(2e5), 20, "prefix", [0.3], True)
# specified_args = None

def main():
    if specified_args is None:
        parser = argparse.ArgumentParser(description="Generate and save string dataset for OLAP testing.")
        parser.add_argument("--total_count", type=int, required=True, help="Total number of strings")
        parser.add_argument("--unique_count", type=int, required=True, help="Number of unique strings")
        parser.add_argument("--max_length", type=int, required=True, help="Maximum string length")
        parser.add_argument("--predicate", type=str, required=True, choices=["equal", "less_than", "prefix"],
                            help="Predicate type to reserve candidate values")
        parser.add_argument("--selectivity_list", type=float, nargs='+', required=True, help="Target selectivity (0 < s < 1)")
        parser.add_argument("--unit_test", action='store_true',
                            help="Run unit tests instead of generating dataset")
        args = parser.parse_args()
    else:
        args = specified_args

    if args.unit_test:
        TestStringGenerator.total_count = args.total_count
        TestStringGenerator.unique_count = args.unique_count
        TestStringGenerator.max_length = args.max_length
        TestStringGenerator.selectivity_list = args.selectivity_list
        unittest.main()
    else:
        generate_and_save_mssb_data(args.total_count, args.unique_count, 1, args.max_length, args.predicate, args.selectivity_list)

if __name__ == '__main__':
    main()
