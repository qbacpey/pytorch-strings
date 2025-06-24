from collections import namedtuple
from contextlib import redirect_stdout
import gc
import json
import os
import duckdb, duckdb.typing
import numpy as np

def generate_and_save_tpch_data(scale: float, path: str = "tpch_data"):
    print(f"Loading TPCH data with scale factor {scale}...")
    db_path = os.path.join(path, "tmp.db")
    out_path = os.path.join(path, f"sf-{scale}")
    os.makedirs(out_path, exist_ok=True)
    
    catalog_file = os.path.join(out_path, "catalog.txt")
    if os.path.exists(catalog_file):
        print(f"Catalog file {catalog_file} already exists, skipping generation.")
        return

    with duckdb.connect(db_path) as con:
        # Predefined TPCH data table names
        tables = ['customer', 'orders', 'lineitem', 'supplier', 
                            'part', 'partsupp', 'nation', 'region']
        # Delete existing tables to ensure data regeneration
        for table_name in tables:
            con.execute(f"DROP TABLE IF EXISTS {table_name}")

        # Install and load TPCH extension, generate data
        con.execute(f"INSTALL tpch; LOAD tpch; CALL dbgen(sf = {scale});")

        catalog_records: list[CatalogRecord] = []

        # Process string columns for each table
        for table_name in tables:
            table = con.table(table_name)
            names = table.columns
            types = table.types

            for col_name, col_type in zip(names, types):
                if col_type != duckdb.typing.VARCHAR:
                    continue
                file_id = f"{table_name}.{col_name}.npy"
                file_path = os.path.join(out_path, file_id)
                # Save string columns to file first (avoid loading everything into memory later)
                if not os.path.exists(file_path):
                    print(f"Saving column {table_name}.{col_name} ...")
                    col_data = table.select(col_name).fetchnumpy()[col_name]
                    np.save(file_path, col_data)
                    print(f"Saved to {file_path}")
                    del col_data
                    gc.collect()
                else:
                    print(f"File {file_path} exists, skipping saving of {table_name}.{col_name}")
                
                # Calculate statistics for this column via SQL (without loading the entire table into Python)
                total_count = table.count(col_name).fetchall()[0][0]
                unique_count = table.count(f"distinct {col_name}").fetchall()[0][0]
                max_length = table.max(f"length({col_name})").fetchall()[0][0]
                
                # Get the first 5 distinct values (sorted) as candidate query values
                uniq_rows = table.select(f"distinct {col_name}").limit(5).fetchall()
                uniq_values = [row[0] for row in uniq_rows if row[0] is not None]
                # Generate catalog records for each predicate
                for predicate in ["equal", "less_than", "prefix"]:
                    sel_list = []    # Store selectivity (float values) for each candidate query
                    sel_queries = []  # mapping: formatted selectivity -> query statement
                    for val in uniq_values:
                        if predicate == "equal":
                            cnt = table.filter(duckdb.ColumnExpression(col_name) == duckdb.ConstantExpression(val)).count("").fetchall()[0][0]
                            query_str = val
                        elif predicate == "less_than":
                            cnt = table.filter(duckdb.ColumnExpression(col_name) < duckdb.ConstantExpression(val)).count("").fetchall()[0][0]
                            query_str = val
                        elif predicate == "prefix":
                            prefix = val[:max(1, len(val)//2)]
                            cnt = table.filter(f"{col_name} like '{prefix}%'").count("").fetchall()[0][0]
                            query_str = prefix
                        else:
                            continue
                        # Execute query and calculate selectivity
                        measured_sel = cnt / total_count if total_count > 0 else 0
                        measured_sel = f"{measured_sel:.3g}"
                        sel_list.append(measured_sel)
                        sel_queries.append((measured_sel, query_str))
                    # Construct current record
                    record = CatalogRecord(file_id, table_name, col_name, total_count, unique_count, max_length, predicate, sel_list, sel_queries)
                    catalog_records.append(record)

    catalog = Catalog(catalog_records, out_path)
    catalog.save()

class CatalogRecord:
    def __init__(self, file_id: str, table: str, column: str, total_count: int, 
                 unique_count: int, max_length: int, predicate: str, 
                 selectivity_list: list[str], query_candidates: list[tuple[str, str]]):
        self.file_id = file_id
        self.table = table
        self.column = column
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
            if record.column == col_name:
                return os.path.join(self.path, record.file_id)
        raise ValueError(f"Column {col_name} not found in catalog.")
    
    def save(self):
        # Write to catalog.txt (output all records in tabular format)
        catalog_file = os.path.join(self.path, "catalog.txt")
        with duckdb.connect(":memory:") as con:
            con.execute("CREATE TABLE catalog (file_id VARCHAR, table_name VARCHAR, column_name VARCHAR, total_count INTEGER, unique_count INTEGER, max_length INTEGER, predicate VARCHAR, selectivity_list JSON, query_candidates JSON)")
            for rec in self.records:
                con.execute(
                    "INSERT INTO catalog VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (rec.file_id, rec.table, rec.column, rec.total_count, rec.unique_count,
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
            if len(values) < 9:
                continue
            record = CatalogRecord(
                file_id=values[0],
                table=values[1],
                column=values[2],
                total_count=int(values[3]),
                unique_count=int(values[4]),
                max_length=int(values[5]),
                predicate=values[6],
                selectivity_list=json.loads(values[7]),
                query_candidates=json.loads(values[8])
            )
            records.append(record)
        return cls(records, path=path)

def get_tpch_catalog(scale: float = 1.0, path: str = "tpch_data") -> Catalog:
    """
    Get catalog information for TPCH data
    :param scale: data scale factor
    :return: catalog information list
    """
    path = os.path.join(path, f"sf-{scale}")
    catalog_file = os.path.join(path, "catalog.txt")

    if not os.path.exists(catalog_file):
        print(f"Catalog file {catalog_file} does not exist. Please generate TPCH data first.")
    return Catalog.load(path)

TPCH_DataGenArgs = namedtuple('TPCH_DataGenArgs', ['scale'])
