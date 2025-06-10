from typing import List, Dict, Any
import torch


class StringEncoder:
    """
    Base class for string encoding strategies.
    """

    def encode(self, strings: List[str]) -> None:
        """Encode and store the given list of strings."""
        raise NotImplementedError

    def decode(self, indices: List[int]) -> List[str]:
        """Decode the given indices back to strings."""
        raise NotImplementedError

    def query_equals(self, query: str) -> List[int]:
        """Return RowIDs where the string equals the query."""
        raise NotImplementedError

    def query_less_than(self, query: str) -> List[int]:
        """Return RowIDs where the string is less than the query."""
        raise NotImplementedError

    def query_prefix(self, prefix: str) -> List[int]:
        """Return RowIDs where the string starts with the given prefix."""
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        """Return encoder configuration for benchmarking/reporting."""
        raise NotImplementedError


class DictionaryEncoder(StringEncoder):
    """
    Dictionary-based string encoding implementation.
    """

    def __init__(self):
        self.encoded_tensor = None
        self.inverse_indices = None
        self.max_length = 0
        self.unique_words = None

    def encode(self, strings: List[str]) -> None:
        """
        Encode the given list of strings into a tensor representation.
        """
        # Convert strings to byte tensors
        ts_list = []
        self.max_length = 0
        for w in strings:
            ts_list.append(torch.ByteTensor(list(bytes(w, "ascii"))))
            self.max_length = max(ts_list[-1].size()[0], self.max_length)

        # Create a padded tensor for all strings
        w_t = torch.zeros((len(ts_list), self.max_length), dtype=torch.uint8)
        for i, ts in enumerate(ts_list):
            w_t[i, 0 : ts.size()[0]] = ts

        # Use torch.unique to find unique words and inverse indices
        self.unique_words, self.inverse_indices = torch.unique(
            w_t, dim=0, return_inverse=True
        )
        self.encoded_tensor = self.unique_words.t()  # Transpose for column-wise access

    def decode(self, indices: List[int]) -> List[str]:
        """
        Decode the given indices back to their original strings.
        """
        decoded_strings = []
        for idx in indices:
            byte_tensor = self.unique_words[idx]
            decoded_strings.append(
                bytes(byte_tensor[byte_tensor > 0].tolist()).decode("ascii")
            )
        return decoded_strings

    def query_equals(self, query: str) -> List[int]:
        """
        Return RowIDs where the string equals the query.
        """
        # Convert the query string to a padded tensor
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
            ].tolist()
            return row_ids
        elif len(potential_match_indices) == 0:
            return []  # No match found
        else:
            raise ValueError("Ambiguous or multiple matches found for the query.")

    def query_less_than(self, query: str) -> List[int]:
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
            )[1].tolist()
            # for matched_idx in less_than_indices:
            #     original_row_ids = torch.where(self.inverse_indices == matched_idx.item())[0]
            #     row_ids.extend(original_row_ids.tolist())
            return row_ids
        else:
            return []

    def query_prefix(self, prefix: str) -> List[int]:
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
            )[1].tolist()
            return row_ids
        else:
            return []

    def get_config(self) -> Dict[str, Any]:
        """
        Return encoder configuration for benchmarking/reporting.
        """
        return {"type": "DictionaryEncoder"}


class EncodingBenchmark:
    """
    Benchmarking framework for comparing encoding/query strategies.
    """

    def __init__(self):
        self.encoders: Dict[str, StringEncoder] = {}

    def register_encoder(self, name: str, encoder: StringEncoder) -> None:
        self.encoders[name] = encoder

    def run_correctness_tests(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run encode-decode fidelity and query correctness tests.
        """
        pass

    def run_benchmarks(
        self, datasets: List[List[str]], queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Use torch benchmarking utilities to measure performance.
        """
        pass


# Example usage
if __name__ == "__main__":
    encoder = DictionaryEncoder()
    encoder.encode(
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
    print("Encoded tensor:", encoder.encoded_tensor)
    print("Inverse indices:", encoder.inverse_indices)

    # Query for equality
    row_ids = encoder.query_equals("will")
    print("Row IDs for 'will':", row_ids)

    # Query for less than
    row_ids_lt = encoder.query_less_than("bpple")
    print("Row IDs for strings less than 'bpple':", row_ids_lt)

    # Query for prefix
    row_ids_prefix = encoder.query_prefix("ap")
    print("Row IDs for strings starting with 'ap':", row_ids_prefix)
