import torch
from typing import List, Dict, Any
from . import StringColumnTensor, StringEncoder


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