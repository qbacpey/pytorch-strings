import numpy as np
import torch
from typing import List, Any, Self
from . import StringColumnTensor

class PlainEncodingStringColumnTensor(StringColumnTensor):
    encoded_tensor: torch.Tensor
    max_length: int

    def __init__(self, encoded_tensor: torch.Tensor):
        self.encoded_tensor = encoded_tensor
        if encoded_tensor.dim() != 2 or encoded_tensor.dtype != torch.uint8:
            raise ValueError("encoded_tensor must be a 2D tensor of type torch.uint8")
        self.max_length = encoded_tensor.size(1)

    def tuple_size(self) -> int:
        """
        Return the size of the tuple representing each string in bytes.

        Returns:
            int: The size of the tuple for each string.
        """
        return self.encoded_tensor.element_size() * self.max_length

    def tuple_counts(self) -> int:
        """
        Return the number of tuples in the encoded tensor.

        Returns:
            int: The number of tuples in the encoded tensor.
        """
        return self.encoded_tensor.numel() // self.max_length

    def __repr__(self) -> str:
        return f"PlainEncodingStringColumnTensor(max_length={self.max_length}, encoded_tensor_shape={self.encoded_tensor.shape})"

    def query_equals(self, query: str, return_mask=False) -> torch.Tensor:
        query_tensor = torch.tensor(list(bytes(query, "ascii")), dtype=torch.uint8)
        query_tensor = torch.nn.functional.pad(query_tensor, (0, self.max_length - len(query_tensor)), value=0)

        eq_mask = (self.encoded_tensor == query_tensor)
        match_mask = eq_mask.all(dim=1)
        del eq_mask  # Free memory

        if return_mask:
            return match_mask
        return match_mask.nonzero().view(-1)

    def query_less_than(self, query: str, return_mask=False) -> torch.Tensor:
        # Create properly padded query tensor
        query_tensor = torch.tensor(list(bytes(query, "ascii")), dtype=torch.uint8)
        query_tensor = torch.nn.functional.pad(query_tensor, (0, self.max_length - len(query_tensor)), value=0)

        ne_mask = self.encoded_tensor != query_tensor
        # Find the first position where they differ, or the very first position if they are equal
        if torch.compiler.is_compiling():
            first_diff_index = ne_mask.to(torch.uint8).argmax(dim=1)
        else:
            first_diff_index = ne_mask.view(torch.uint8).argmax(dim=1)
        del ne_mask  # Free memory

        first_diff_tensor = self.encoded_tensor[torch.arange(len(self)), first_diff_index]
        # Get whether the first differing position is less than the query, or must be false if they are equal
        lt_mask = first_diff_tensor < query_tensor[first_diff_index]

        if return_mask:
            return lt_mask
        return lt_mask.nonzero().view(-1)

    def query_prefix(self, prefix: str, return_mask=False) -> torch.Tensor:
        prefix_len = len(prefix)
        prefix_tensor = torch.tensor(list(bytes(prefix, "ascii")), dtype=torch.uint8)

        prefix_eq_mask = (self.encoded_tensor[:, :prefix_len] == prefix_tensor)
        match_mask = prefix_eq_mask.all(dim=1)
        del prefix_eq_mask  # Free memory

        if return_mask:
            return match_mask
        return match_mask.nonzero().view(-1)

    def query_aggregate(self) -> torch.Tensor:
        _, inverse_indices = torch.unique(self.encoded_tensor, dim=0, return_inverse=True)
        return inverse_indices

    def query_sort(self, ascending: bool = True) -> torch.Tensor:
        _, inverse_indices = torch.unique(self.encoded_tensor, dim=0, return_inverse=True)
        sorted_indices = torch.argsort(inverse_indices, descending=not ascending)
        return sorted_indices

    def index_select(self, *indices: Any) -> Self:
        return self.__class__(
            encoded_tensor=self.encoded_tensor[indices]
        )

    def __len__(self) -> int:
        return len(self.encoded_tensor)

    def tensor(self) -> torch.Tensor:
        return self.encoded_tensor

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> Self:
        return cls(tensor)

    @classmethod
    def from_strings(cls, strings: List[str] | np.ndarray) -> Self:
        max_length = max(len(s) for s in strings)
        batch_size = len(strings)
        arr = np.zeros((batch_size, max_length), dtype=np.uint8)
        for i, s in enumerate(strings):
            arr[i, :len(s)] = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
        encoded_tensor = torch.tensor(arr, dtype=torch.uint8)

        # ts = [torch.tensor(list(bytes(s, "ascii")), dtype=torch.uint8) for s in strings]
        # encoded_tensor = torch.nn.utils.rnn.pad_sequence(ts, batch_first=True, padding_value=0)
        return cls(encoded_tensor)

    def to_strings(self) -> List[str]:
        return [bytes(self.encoded_tensor[i][self.encoded_tensor[i] > 0].tolist()).decode("ascii") for i in range(len(self.encoded_tensor))]

    @classmethod
    def from_string_tensor(cls, string_tensor: StringColumnTensor) -> Self:
        if not isinstance(string_tensor, PlainEncodingStringColumnTensor):
            raise TypeError(
                    f"Unsupported type for PlainEncodingStringColumnTensor.from_string_tensor: {type(string_tensor)}. "
                    "Expected PlainEncodingStringColumnTensor."
                )
        return cls.from_tensor(string_tensor.tensor())
