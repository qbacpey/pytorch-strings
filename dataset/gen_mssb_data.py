import argparse
import json
import os
import random
import string
import unittest
import duckdb
import numpy as np
from contextlib import redirect_stdout
from typing import Optional

from collections import namedtuple

def generate_mssb_data(total_count, unique_count, max_length, predicate: Optional[str] = None, selectivity_list: Optional[list[float]] = None):
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
    def random_string(max_length: int = max_length) -> str:
        """Generate a random string with length between 1 and max_length."""
        l = random.randint(1, max_length)
        return ''.join(random.choices(string.ascii_letters + string.digits, k=l))
    
    def unique_random_strings(count: int, max_length: int = max_length) -> list[str]:
        """Generate count unique random strings with lengths between 1 and max_length."""
        uniq_strs = set()
        while len(uniq_strs) < count:
            s = random_string(max_length)
            if s not in uniq_strs:
                uniq_strs.add(s)
        return list(uniq_strs)
    
    def zipf_distribution(n: int, a: float = 1.2) -> np.ndarray:
        """Generate Zipf distribution frequency weights"""
        x = np.arange(1, n + 1)
        weights = 1 / (x ** a)
        weights = weights / weights.sum()
        return weights

    def sample(unique_strs: list[str], weights: np.ndarray, total_count: int) -> list[str]:
        """Sample from unique_strs according to weights to generate total_count data."""
        data = []
        counts = ((total_count - len(unique_strs)) * weights // weights.sum()).astype(int) + 1
        if counts.sum() < total_count:
            extra = total_count - counts.sum()
            counts += (extra // len(counts)).astype(int)
            counts[:(extra % len(counts)).astype(int)] += 1

        for str, count in zip(unique_strs, counts):
            data.extend([str] * count)
        random.shuffle(data)
        return data

    # ===================================================
    # If no predicate is specified, use the simplest generation logic:
    if predicate is None or selectivity_list is None:
        # Generate unique_count different strings (ensure random length between 1~max_length)
        uniq_strs = unique_random_strings(unique_count)
        # Construct final data (each string repeated by its frequency), then shuffle
        weights = zipf_distribution(unique_count)
        data = sample(uniq_strs, weights, total_count)
        return data, {}

    # ===================================================
    # Predicate is "equal"
    # Let the selectivity_list in front for the weight arr of uni string, the rest value would be broken down by Zipf distribution
    elif predicate == "equal":
        uniq_strs = sorted(unique_random_strings(unique_count))
        selectivity_list = sorted(selectivity_list)
        total_selectivity = sum(selectivity_list)
        num_selectivities = len(selectivity_list)
        if total_selectivity > 1:
            raise ValueError("Sum of selectivities cannot exceed 1")
        if num_selectivities > unique_count:
            raise ValueError("Number of selectivities cannot exceed unique_count")
        weights = np.append(selectivity_list, (zipf_distribution(unique_count - num_selectivities) * (1 - total_selectivity)))
        data = sample(uniq_strs, weights, total_count)
        query_candidates = {sel: str for sel, str in zip(selectivity_list, uniq_strs)}
        return data, query_candidates

    # ===================================================
    # Predicate is "less_than"
    elif predicate == "less_than":
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
        query_candidates = {sel: str for sel, str in zip(selectivity_list, bound_strs)}
        return data, query_candidates

    # ===================================================
    # Predicate is "prefix"
    elif predicate == "prefix":
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
            prefix = random_string()[:random.randint(1, max_length * 4 // 5)]
            uniq_strs.extend([prefix + random_string(max_length - len(prefix)) for _ in range(count)])
            weights.extend(zipf_distribution(count) * selectivity_list[i])
            prefixes.append(prefix)
        
        uniq_strs.extend(unique_random_strings(unique_count - cand_total_count))
        weights = np.append(weights, zipf_distribution(unique_count - cand_total_count) * (1 - total_selectivity))
        data = sample(uniq_strs, weights, total_count)
        query_candidates = {sel: str for sel, str in zip(selectivity_list, prefixes)}
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

        data, candidates = generate_mssb_data(total_count, unique_count, max_length)
        self.assertEqual(len(data), total_count)
        for s in data:
            self.assertTrue(1 <= len(s) <= max_length)
        self.assertEqual(candidates, {})  # No candidate query values
    
    def test_equal_predicate(self):
        # total_count, unique_count, max_length = 1000, 120, 20
        total_count = self.total_count or 100
        unique_count = self.unique_count or 20
        max_length = self.max_length or 20
        selectivities = self.selectivity_list or [0.05, 0.2, 0.5]  # For example 5%, 20%, 50%
        data, candidates = generate_mssb_data(total_count, unique_count, max_length,
                                                 predicate="equal",
                                                 selectivity_list=selectivities)   
        print("data:", data)
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
            self.assertAlmostEqual(actual, expected, delta=round(0.05 * total_count))  # Allow 5% error

    def test_less_than_predicate(self):
        # total_count, unique_count, max_length = 10000, 300, 15
        total_count = self.total_count or 100
        unique_count = self.unique_count or 20
        max_length = self.max_length or 20
        selectivities = self.selectivity_list or [0.1, 0.3, 0.7]
        data, candidates = generate_mssb_data(total_count, unique_count, max_length,
                                                 predicate="less_than",
                                                 selectivity_list=selectivities)
        print("data:", data)
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
            self.assertAlmostEqual(actual, expected, delta=round(0.05 * total_count))

    def test_prefix_predicate(self):
        # total_count, unique_count, max_length = 1000, 120, 20
        total_count = self.total_count or 100
        unique_count = self.unique_count or 20
        max_length = self.max_length or 20
        selectivities = self.selectivity_list or [0.1, 0.4]
        data, candidates = generate_mssb_data(total_count, unique_count, max_length,
                                                 predicate="prefix",
                                                 selectivity_list=selectivities)
        print("data:", data)
        print("Prefix query candidates:", candidates)

        self.assertEqual(len(data), total_count)
        self.assertEqual(len(set(data)), unique_count) 
        self.assertTrue(all(1 <= len(s) <= max_length for s in data))
        self.assertEqual(len(candidates), len(selectivities))
        for s in selectivities:
            candidate_val = candidates[s]
            actual = sum(1 for x in data if x.startswith(candidate_val))
            expected = round(s * (total_count - unique_count)) + 1
            self.assertAlmostEqual(actual, expected, delta=round(0.05 * total_count))
# ==============================================

def generate_and_save_mssb_data(total_count, unique_count, max_length, predicate, selectivity_list, path="mssb_data"):
    """
    Generate test data columns based on input parameters:
      total_count, unique_count, max_length, predicate, selectivity
    Call generate_mssb_data() to generate data columns (selectivity passed as list [selectivity]),
    then save as npy format to a unified directory, and save parameters and query candidate mapping to the mapping file catalog.txt,
    file names use sequentially incrementing numbers (e.g., 0001.npy).
    
    If a file with the same parameters already exists, don't generate again, just return the path to the corresponding file.
    
    Returns: Path to the generated or existing npy file.
    """
    if not os.path.exists(path):
        os.makedirs(path)

    catalog = Catalog.load(path)
    
    # Check if a record with the same parameters already exists
    for rec in catalog.records:
        if (rec.total_count == total_count and
            rec.unique_count == unique_count and
            rec.max_length == max_length and
            rec.predicate == predicate and
            rec.selectivity_list == selectivity_list):
            file_name = f"{rec.file_id}.npy"
            print(f"Dataset already exists: {file_name}, skipping generation.")
            return
    
    # No matching record found, generate a new dataset
    # Call external generate_mssb_data function (returns data, query_candidates)
    data, query_candidates = generate_mssb_data(total_count, unique_count, max_length, predicate, selectivity_list)
    
    # Determine new File ID (sequentially incrementing, like 0001,0002,...)
    if catalog.records:
        new_id = max(int(rec.file_id) for rec in catalog.records) + 1
    else:
        new_id = 1
    file_id = f"{new_id:04d}"
    file_name = f"{file_id}.npy"
    file_path = os.path.join(path, file_name)
    
    # Save data as npy file (convert to numpy array)
    np.save(file_path, np.array(data))
    
    new_record = CatalogRecord(file_id, total_count, unique_count, max_length, predicate, selectivity_list, query_candidates)
    catalog.records.append(new_record)
    catalog.save()

class CatalogRecord:
    """
    Class for storing dataset mapping records.
    Includes file ID, total rows, unique string count, max length, predicate type, selectivity list, and query candidate mapping.
    """
    def __init__(self, file_id, total_count, unique_count, max_length, predicate, selectivity_list, query_candidates):
        self.file_id = file_id
        self.total_count = total_count
        self.unique_count = unique_count
        self.max_length = max_length
        self.predicate = predicate
        self.selectivity_list = selectivity_list
        self.query_candidates = query_candidates

class Catalog:
    def __init__(self, records: list[CatalogRecord], path: str):
        self.path = path
        self.records = records
    
    def get_col_file(self, col_name: str) -> str:
        for record in self.records:
            if record.file_id == col_name:
                return os.path.join(self.path, f"{record.file_id}.npy")
        raise ValueError(f"Column {col_name} not found in catalog.")
    
    def save(self):
        # Write to self.path (output all records in tabular format)
        catalog_file = os.path.join(self.path, "catalog.txt")
        with duckdb.connect(":memory:") as con:
            con.execute("CREATE TABLE catalog (file_id VARCHAR, total_count INTEGER, unique_count INTEGER, max_length INTEGER, predicate VARCHAR, selectivity_list JSON, query_candidates JSON)")
            for rec in self.records:
                con.execute(
                    "INSERT INTO catalog VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (rec.file_id, rec.total_count, rec.unique_count,
                     rec.max_length, rec.predicate,
                     json.dumps(rec.selectivity_list), json.dumps(rec.query_candidates))
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
                file_id=values[0],
                total_count=int(values[1]),
                unique_count=int(values[2]),
                max_length=int(values[3]),
                predicate=values[4],
                selectivity_list=json.loads(values[5]),
                query_candidates=json.loads(values[6])
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
specified_args = MSSB_DataGenArgs(100, 10, 20, "equal", [0.1, 0.3, 0.5], False)
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
        generate_and_save_mssb_data(args.total_count, args.unique_count, args.max_length, args.predicate, args.selectivity_list)

if __name__ == '__main__':
    main()
