from typing import List, Dict, Any
import numpy
import torch
import duckdb, duckdb.typing
import pytest, pytest_benchmark.fixture


class StringColumnTensor:
    """
    Base class for string encoding strategies.
    """

    def query_equals(self, query: str) -> torch.Tensor:
        """Return RowIDs where the string equals the query."""
        raise NotImplementedError

    def query_less_than(self, query: str) -> torch.Tensor:
        """Return RowIDs where the string is less than the query."""
        raise NotImplementedError

    def query_prefix(self, prefix: str) -> torch.Tensor:
        """Return RowIDs where the string starts with the given prefix."""
        raise NotImplementedError
    
    def query_aggregate(self) -> torch.Tensor:
        """
        Generate inverse indices that map each element's original position
        to its aggregated group index.
        
        Returns:
        List[int]: A list of length N (the number of original elements), 
                   where the i-th entry is the index of the aggregate 
                   group to which element i belongs.
        """
        raise NotImplementedError
    
    def query_sort(self, ascending: bool = True) -> torch.Tensor:
        """
        Generate a permutation index that maps sorted positions 
        back to their original positions.

        Args:
            ascending (bool, optional): If True (default), sort in ascending order; 
                                        if False, sort in descending order.

        Returns:
            List[int]: A list of length N (the number of elements), 
                    where the i-th entry is the original index 
                    of the element now at sorted position i.
        """
        raise NotImplementedError
    
    def index_select(self, *indices: Any) -> 'StringColumnTensor':
        """
        Select a subset of the encoded strings based on the provided indices.
        
        Args:
            indices: List of indices to select from the encoded column.
        
        Returns:
            StringColumnTensor: A new instance containing only the selected strings.
        """
        raise NotImplementedError
    
    def __len__(self) -> int:
        """
        Return number of rows.
        """
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        """Return encoder configuration for benchmarking/reporting."""
        raise NotImplementedError

class StringEncoder:
    """
    Base class for string encoding strategies.
    """

    def encode(self, strings: List[str]) -> StringColumnTensor:
        """Encode and store the given list of strings."""
        raise NotImplementedError

    def decode(self, encoded_tensor: StringColumnTensor) -> List[str]:
        """Decode the given indices back to strings."""
        raise NotImplementedError

class PlainEncodingStringColumnTensor(StringColumnTensor):
    encoded_tensor: torch.Tensor
    max_length: int

    def __init__(self, encoded_tensor=torch.empty(0, dtype=torch.uint8), max_length=0):
        self.encoded_tensor = encoded_tensor
        self.max_length = max_length
    
    def query_equals(self, query: str) -> torch.Tensor:
        query_tensor = torch.tensor(list(bytes(query, "ascii")), dtype=torch.uint8)
        query_tensor = torch.nn.functional.pad(query_tensor, (0, self.max_length - len(query_tensor)), value=0)
        matches = (self.encoded_tensor == query_tensor).all(dim=1)
        return matches.nonzero().view(-1)

    def query_less_than(self, query: str) -> torch.Tensor:
        # Create properly padded query tensor
        query_tensor = torch.tensor(list(bytes(query, "ascii")), dtype=torch.uint8)
        query_tensor = torch.nn.functional.pad(query_tensor, (0, self.max_length - len(query_tensor)), value=0)

        ne_mask = self.encoded_tensor != query_tensor
        # Find the first position where they differ, or the very first position if they are equal
        first_ne = ne_mask.int().argmax(dim=1)

        first_ne_tensor = self.encoded_tensor[torch.arange(len(first_ne)), first_ne]
        # Get whether the first differing position is less than the query, or must be false if they are equal
        first_lt = first_ne_tensor < query_tensor[first_ne]

        return first_lt.nonzero().view(-1)
    
    def query_prefix(self, prefix: str) -> torch.Tensor:
        prefix_len = len(prefix)
        prefix_tensor = torch.tensor(list(bytes(prefix, "ascii")), dtype=torch.uint8)

        matches = (self.encoded_tensor[:, :prefix_len] == prefix_tensor).all(dim=1)
        return matches.nonzero().view(-1)
    
    def query_aggregate(self) -> torch.Tensor:
        _, inverse_indices = torch.unique(self.encoded_tensor, dim=0, return_inverse=True)
        return inverse_indices
    
    def query_sort(self, ascending: bool = True) -> torch.Tensor:
        _, inverse_indices = torch.unique(self.encoded_tensor, dim=0, return_inverse=True)
        sorted_indices = torch.argsort(inverse_indices, descending=not ascending)
        return sorted_indices
    
    def index_select(self, *indices: Any) -> StringColumnTensor:
        return PlainEncodingStringColumnTensor(
            encoded_tensor=self.encoded_tensor[indices],
            max_length=self.max_length
        )
    
    def __getitem__(self, *indices: Any) -> torch.Tensor:
        return self.encoded_tensor[indices]
    
    def __len__(self) -> int:
        return len(self.encoded_tensor)

class PlainEncoder(StringEncoder):
    def encode(self, strings: List[str]) -> PlainEncodingStringColumnTensor:
        ts = [torch.tensor(list(bytes(s, "ascii")), dtype=torch.uint8) for s in strings]
        max_length = max(len(s) for s in strings)
        encoded_tensor = torch.nn.utils.rnn.pad_sequence(ts, batch_first=True, padding_value=0)
        return PlainEncodingStringColumnTensor(encoded_tensor, max_length)
    
    def decode(self, encoded_tensor: StringColumnTensor) -> List[str]:
        """
        Decode the given indices back to strings.
        """
        if not isinstance(encoded_tensor, PlainEncodingStringColumnTensor):
            raise TypeError("Expected PlainEncodingStringColumnTensor for decoding.")
        return [bytes(encoded_tensor[i][encoded_tensor[i] > 0].tolist()).decode("ascii") for i in range(len(encoded_tensor))]

class DictionaryEncodingStringColumnTensor(StringColumnTensor):
    """
    Dictionary-based string encoding implementation.
    """
    unique_words: torch.Tensor
    encoded_tensor: torch.Tensor
    inverse_indices: torch.Tensor
    max_length: int

    def __init__(self, unique_words: torch.Tensor, encoded_tensor: torch.Tensor, inverse_indices: torch.Tensor, max_length: int):
        self.unique_words = unique_words
        self.encoded_tensor = encoded_tensor
        self.inverse_indices = inverse_indices
        self.max_length = max_length

    def query_equals(self, query: str) -> torch.Tensor:
        """
        Return RowIDs where the string equals the query.
        """
        # Convert the query string to a padded tensor
        # query_tensor = torch.ByteTensor(list(bytes(query, "ascii")))
        query_tensor = torch.tensor(list(bytes(query, "ascii")), dtype=torch.uint8)
        query_tensor = torch.cat(
            (
                query_tensor,
                torch.zeros(
                    self.max_length - query_tensor.size()[0], dtype=torch.uint8
                ),
            )
        )

        # Perform equality checking column by column
        num_unique_words = self.encoded_tensor.shape[1]
        potential_match_indices = torch.arange(
            num_unique_words, device=self.encoded_tensor.device
        )

        for char_idx in range(self.max_length):
            if len(potential_match_indices) == 0:
                break

            current_query_char = query_tensor[char_idx]
            chars_from_dict_candidates = self.encoded_tensor[
                char_idx, potential_match_indices
            ]
            current_char_match_mask = chars_from_dict_candidates == current_query_char
            potential_match_indices = potential_match_indices[current_char_match_mask]

        # If a single match is found, return the corresponding RowIDs
        if len(potential_match_indices) == 1:
            matched_unique_word_idx = potential_match_indices[0].item()
            row_ids = torch.where(self.inverse_indices == matched_unique_word_idx)[
                0
            ]
            return row_ids
        elif len(potential_match_indices) == 0:
            return torch.empty(0, dtype=torch.long)  # No match found
        else:
            raise ValueError("Ambiguous or multiple matches found for the query.")

    def query_less_than(self, query: str) -> torch.Tensor:
        """
        Return RowIDs where the string is less than the query.
        This is a simplified version that assumes lexicographical order.
        """
        # Convert the query strings to a padded tensor
        query_tensor = torch.ByteTensor(list(bytes(query, "ascii")))
        query_tensor = torch.cat(
            (
                query_tensor,
                torch.zeros(
                    self.max_length - query_tensor.size()[0], dtype=torch.uint8
                ),
            )
        )

        # Perform equality checking column by column
        num_unique_words = self.encoded_tensor.shape[1]
        potential_exactly_match_indices = torch.arange(
            num_unique_words, device=self.encoded_tensor.device
        )
        # All empty tensor for already less than matches
        less_than_indices = torch.tensor(
            [], dtype=torch.long, device=self.encoded_tensor.device
        )

        for char_idx in range(self.max_length):
            if len(potential_exactly_match_indices) == 0:
                break

            current_query_char = query_tensor[char_idx]
            chars_from_dict_candidates = self.encoded_tensor[
                char_idx, potential_exactly_match_indices
            ]

            current_char_lt_mask = chars_from_dict_candidates < current_query_char
            current_char_match_mask = chars_from_dict_candidates == current_query_char

            # Collect indices where the current character is less than the query character
            less_than_indices = torch.cat(
                (
                    less_than_indices,
                    potential_exactly_match_indices[current_char_lt_mask],
                )
            )

            # Update potential_exactly_match_indices to only those that match the current character
            potential_exactly_match_indices = potential_exactly_match_indices[
                current_char_match_mask
            ]

        # Return the corresponding RowIDs
        if len(less_than_indices) > 0:
            # Check unsqueeze
            # print(f"Less than indices unsqueeze: {less_than_indices.unsqueeze(1)}")
            # print(f"Inverse indices unsqueeze: {self.inverse_indices.unsqueeze(0)}")

            # Get the index of the element of self.inverse_indices when it matches any of less than indices
            row_ids = torch.where(
                self.inverse_indices.unsqueeze(0) == less_than_indices.unsqueeze(1)
            )[1]
            # for matched_idx in less_than_indices:
            #     original_row_ids = torch.where(self.inverse_indices == matched_idx.item())[0]
            #     row_ids.extend(original_row_ids.tolist())
            return row_ids
        else:
            return torch.empty(0, dtype=torch.long)

    def query_prefix(self, prefix: str) -> torch.Tensor:
        """
        Return RowIDs where the string starts with the given prefix.
        """
        prefix_len = len(prefix)
        
        # Convert the query string to a padded tensor
        prefix_tensor = torch.ByteTensor(list(bytes(prefix, "ascii")))
        prefix_tensor = torch.cat(
            (
                prefix_tensor,
                torch.zeros(
                    self.max_length - prefix_tensor.size()[0], dtype=torch.uint8
                ),
            )
        )

        # Perform equality checking column by column
        num_unique_words = self.encoded_tensor.shape[1]
        potential_match_indices = torch.arange(
            num_unique_words, device=self.encoded_tensor.device
        )

        for char_idx in range(prefix_len):
            if len(potential_match_indices) == 0:
                break

            current_query_char = prefix_tensor[char_idx]
            chars_from_dict_candidates = self.encoded_tensor[
                char_idx, potential_match_indices
            ]
            current_char_match_mask = chars_from_dict_candidates == current_query_char
            potential_match_indices = potential_match_indices[current_char_match_mask]

        # If a single match is found, return the corresponding RowIDs
        if len(potential_match_indices) >= 0:
            row_ids = torch.where(
                self.inverse_indices.unsqueeze(0)
                == potential_match_indices.unsqueeze(1)
            )[1]
            return row_ids
        else:
            return torch.empty(0, dtype=torch.long)
    
    def query_aggregate(self) -> torch.Tensor:
        return self.inverse_indices
    
    def query_sort(self, ascending: bool = True) -> torch.Tensor:
        return torch.argsort(self.inverse_indices, descending=not ascending)

    def get_config(self) -> Dict[str, Any]:
        """
        Return encoder configuration for benchmarking/reporting.
        """
        return {"type": "DictionaryEncoder"}
    
    def index_select(self, *indices: Any) -> StringColumnTensor:
        return DictionaryEncodingStringColumnTensor(
            unique_words=self.unique_words[indices],
            encoded_tensor=self.unique_words[indices].t(),
            inverse_indices=self.inverse_indices[indices],
            max_length=self.max_length
        )
    
    def __getitem__(self, *indices: Any) -> torch.Tensor:
        return self.unique_words[self.inverse_indices[indices]]
    
    def __len__(self) -> int:
        return len(self.inverse_indices)
    
    
class DictionaryEncoder(StringEncoder):
    def encode(self, strings: List[str]) -> DictionaryEncodingStringColumnTensor:
        """
        Encode the given list of strings into a tensor representation.
        """
        # Convert strings to byte tensors
        ts_list = []
        max_length = 0
        for w in strings:
            ts_list.append(torch.ByteTensor(list(bytes(w, "ascii"))))
            max_length = max(ts_list[-1].size()[0], max_length)

        # Create a padded tensor for all strings
        w_t = torch.zeros((len(ts_list), max_length), dtype=torch.uint8)
        for i, ts in enumerate(ts_list):
            w_t[i, 0 : ts.size()[0]] = ts

        # Use torch.unique to find unique words and inverse indices
        unique_words, inverse_indices = torch.unique(
            w_t, dim=0, return_inverse=True
        )
        encoded_tensor = unique_words.t()  # Transpose for column-wise access

        return DictionaryEncodingStringColumnTensor(unique_words, encoded_tensor, inverse_indices, max_length)

    def decode(self, encoded_tensor: StringColumnTensor) -> List[str]:
        """
        Decode the given indices back to their original strings.
        """
        if not isinstance(encoded_tensor, DictionaryEncodingStringColumnTensor):
            raise TypeError("Expected DictionaryEncodingStringColumnTensor for decoding.")

        decoded_strings = []
        for idx in range(len(encoded_tensor)):
            byte_tensor = encoded_tensor[idx]
            decoded_strings.append(
                bytes(byte_tensor[byte_tensor > 0].tolist()).decode("ascii")
            )
        return decoded_strings

class RowWiseDictionaryEncodingStringColumnTensor(StringColumnTensor):
    dictionary: torch.Tensor
    encoded_tensor: torch.Tensor
    max_length: int

    def __init__(self, dictionary: torch.Tensor, encoded_tensor: torch.Tensor, max_length: int):
        self.dictionary = dictionary
        self.encoded_tensor = encoded_tensor
        self.max_length = max_length

    def query_equals(self, query: str) -> torch.Tensor:
        query_tensor = torch.tensor(list(bytes(query, "ascii")), dtype=torch.uint8)
        query_tensor = torch.nn.functional.pad(query_tensor, (0, self.max_length - len(query_tensor)), value=0)
        
        # Find the matching code in the dictionary by comparing with each row
        matches_in_dict = (self.dictionary == query_tensor).all(dim=1)
        codes = matches_in_dict.nonzero().view(-1)
        
        if len(codes) > 0:
            # If found in dictionary, get index and check against encoded tensor
            code = codes[0]
            matches = (self.encoded_tensor == code)
            return matches.nonzero().view(-1)
        else:
            # If not found in dictionary, return empty tensor
            return torch.empty(0, dtype=torch.long)

    def query_less_than(self, query: str) -> torch.Tensor:
        # Create properly padded query tensor
        query_tensor = torch.tensor(list(bytes(query, "ascii")), dtype=torch.uint8)
        query_tensor = torch.nn.functional.pad(query_tensor, (0, self.max_length - len(query_tensor)), value=0)

        ne_mask = self.dictionary != query_tensor
        # Find the first position where they differ, or the very first position if they are equal
        first_ne = ne_mask.int().argmax(dim=1)

        first_ne_tensor = self.dictionary[torch.arange(len(first_ne)), first_ne]
        # Get whether the first differing position is less than the query, or must be false if they are equal
        first_lt = first_ne_tensor < query_tensor[first_ne]
        lt_codes = first_lt.nonzero().view(-1)
        if len(lt_codes) > 0:
            matches = (self.encoded_tensor.view(-1, 1) == lt_codes).any(dim=1)
            return matches.nonzero().view(-1)
        else:
            return torch.empty(0, dtype=torch.long)

    def query_prefix(self, prefix: str) -> torch.Tensor:
        prefix_len = len(prefix)
        prefix_tensor = torch.tensor(list(bytes(prefix, "ascii")), dtype=torch.uint8)

        matches_in_dict = (self.dictionary[:, :prefix_len] == prefix_tensor).all(dim=1)
        codes = matches_in_dict.nonzero().view(-1)
        if len(codes) > 0:
            matches = (self.encoded_tensor.view(-1, 1) == codes).any(dim=1)
            return matches.nonzero().view(-1)
        else:
            return torch.empty(0, dtype=torch.long)
    
    def query_aggregate(self) -> torch.Tensor:
        return self.encoded_tensor
    
    def query_sort(self, ascending: bool = True) -> torch.Tensor:
        return torch.argsort(self.encoded_tensor, descending=not ascending)

    def get_config(self) -> Dict[str, Any]:
        """
        Return encoder configuration for benchmarking/reporting.
        """
        return {"type": "DictionaryEncoder"}
    
    def index_select(self, *indices: Any) -> StringColumnTensor:
        return RowWiseDictionaryEncodingStringColumnTensor(
            dictionary=self.dictionary,
            encoded_tensor=self.encoded_tensor[indices],
            max_length=self.max_length
        )
    
    def __getitem__(self, *indices: Any) -> torch.Tensor:
        return self.dictionary[self.encoded_tensor[indices]]
    
    def __len__(self) -> int:
        return len(self.encoded_tensor)
    
    
class RowWiseDictionaryEncoder(StringEncoder):
    def encode(self, strings: List[str]) -> RowWiseDictionaryEncodingStringColumnTensor:
        ts = [torch.tensor(list(bytes(s, "ascii")), dtype=torch.uint8) for s in strings]
        max_length = max(len(s) for s in strings)
        plain_tensor = torch.nn.utils.rnn.pad_sequence(ts, batch_first=True, padding_value=0)
        # Use torch.unique to find unique words and inverse indices
        dictionary, encoded_tensor = torch.unique(
            plain_tensor, dim=0, return_inverse=True
        )
        return RowWiseDictionaryEncodingStringColumnTensor(dictionary, encoded_tensor, max_length)

    def decode(self, encoded_tensor: StringColumnTensor) -> List[str]:
        if not isinstance(encoded_tensor, RowWiseDictionaryEncodingStringColumnTensor):
            raise TypeError("Expected RowWiseDictionaryEncodingStringColumnTensor for decoding.")
        return [bytes(encoded_tensor[i][encoded_tensor[i] > 0].tolist()).decode("ascii") for i in range(len(encoded_tensor))]

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

# Example usage
if __name__ == "__main__":
    encoders = [DictionaryEncoder(), PlainEncoder(), RowWiseDictionaryEncoder()]
    for encoder in encoders:
        col = encoder.encode(
            [
                "apwho",
                "Initially",
                "apple",
                "applppp",
                "bpple",
                "each",
                "encoding",
                "method",
                "will",
                "will",
                "be",
                "applppp",
            ]
        )
        # print("Encoded tensor:", encoder.encoded_tensor)
        # print("Inverse indices:", encoder.inverse_indices)

        # Query for equality
        row_ids = col.query_equals("will")
        print("Row IDs for 'will':", row_ids)

        # Query for less than
        row_ids_lt = col.query_less_than("bpple")
        print("Row IDs for strings less than 'bpple':", row_ids_lt)

        # Query for prefix
        row_ids_prefix = col.query_prefix("ap")
        print("Row IDs for strings starting with 'ap':", row_ids_prefix)

        print(encoder.decode(col))
