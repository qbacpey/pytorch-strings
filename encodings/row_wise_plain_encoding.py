import torch
from typing import List, Dict, Any
from . import StringColumnTensor, StringEncoder

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