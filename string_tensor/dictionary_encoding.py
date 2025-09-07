import numpy as np
import torch
from typing import List, Dict, Any, Self
from . import StringColumnTensor
from .plain_encoding import PlainEncodingStringColumnTensor
from .cplain_encoding import CPlainEncodingStringColumnTensor

class DictionaryEncodingStringColumnTensor(StringColumnTensor):
    dictionary: PlainEncodingStringColumnTensor
    dictionary_cls: type[PlainEncodingStringColumnTensor] = PlainEncodingStringColumnTensor
    encoded_tensor: torch.Tensor

    def __init__(self, dictionary: PlainEncodingStringColumnTensor, encoded_tensor: torch.Tensor):
        self.dictionary = dictionary
        self.encoded_tensor = encoded_tensor

    def __repr__(self) -> str:        return (
            f"DictionaryEncodingStringColumnTensor("
            f"dictionary_size={len(self.dictionary)}, "
            f"encoded_tensor_shape={self.encoded_tensor.shape})"
        )

    def tuple_size(self) -> int:
        
        # Use the actual element size of the tensor to calculate the tuple size
        # tuple_size = self.encoded_tensor.element_size()
        
        # Use the original max_length from the dictionary to calculate the tuple size
        tuple_size = self.dictionary.max_length
        
        # Print the shape of the encoded tensor for debugging
        # total_bytes = self.encoded_tensor.nbytes
        # print(f"Encoded tensor shape: {self.encoded_tensor.shape}")
        # print(f"Number of elements: {self.encoded_tensor.numel()}")
        # print(f"Size of one element: {self.encoded_tensor.element_size()} bytes")
        # print(f"Max length from dictionary: {self.dictionary.max_length} bytes")
        # print(f"Total memory size (in bytes): {total_bytes}")
        return tuple_size

    def tuple_counts(self) -> int:
        """
        Return the number of tuples in the encoded tensor.
        
        Returns:
            int: The number of tuples in the encoded tensor.
        """
        return self.encoded_tensor.numel()

    def query_equals(self, query: str, return_mask=False) -> torch.Tensor:
        codes = self.query_equals_lookup_dict(query, return_mask=False)
        return self.query_equals_match_encoded(codes, return_mask)

    def query_equals_lookup_dict(self, query: str, return_mask=False) -> torch.Tensor:
        return self.dictionary.query_equals(query, return_mask=False)

    def query_equals_match_encoded(self, selector: torch.Tensor, return_mask=False) -> torch.Tensor:
        codes = selector
        if len(codes) > 0:
            # If found in dictionary, get index and check against encoded tensor
            code = codes[0]
            matches = (self.encoded_tensor == code)

            if return_mask:
                return matches
            return matches.nonzero().view(-1)
        # If not found in dictionary, return empty tensor
        if return_mask:
            return torch.zeros(len(self),dtype=torch.bool)
        return torch.empty(0, dtype=torch.long)

    def query_less_than(self, query: str, return_mask=False) -> torch.Tensor:
        lt_codes = self.query_less_than_lookup_dict(query, return_mask=False)
        return self.query_less_than_match_encoded(lt_codes, return_mask)

    def query_less_than_lookup_dict(self, query: str, return_mask=False) -> torch.Tensor:
        return self.dictionary.query_less_than(query, return_mask=False)

    def query_less_than_match_encoded(self, selector: torch.Tensor, return_mask=False) -> torch.Tensor:
        lt_codes = selector
        if len(lt_codes) > 0:

            max_code = lt_codes[-1]
            matches = (self.encoded_tensor <= max_code)

            if return_mask:
                return matches
            return matches.nonzero().view(-1)

        if return_mask:
            return torch.zeros(len(self),dtype=torch.bool)
        return torch.empty(0, dtype=torch.long)

    def query_prefix(self, prefix: str, return_mask=False) -> torch.Tensor:
        prefix_codes = self.query_prefix_lookup_dict(prefix, return_mask=False)
        return self.query_prefix_match_encoded(prefix_codes, return_mask)

    def query_prefix_lookup_dict(self, prefix: str, return_mask=False) -> torch.Tensor:
        return self.dictionary.query_prefix(prefix, return_mask=False)

    def query_prefix_match_encoded(self, selector: torch.Tensor, return_mask=False) -> torch.Tensor:
        prefix_codes = selector
        if len(prefix_codes) > 0:

            min_code = prefix_codes[0]
            max_code = prefix_codes[-1]
            matches = (self.encoded_tensor >= min_code) & (self.encoded_tensor <= max_code)

            if return_mask:
                return matches
            return matches.nonzero().view(-1)

        if return_mask:
            return torch.zeros(len(self),dtype=torch.bool)
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

    def index_select(self, *indices: Any) -> Self:
        return self.__class__(
            dictionary=self.dictionary,
            encoded_tensor=self.encoded_tensor[indices]
        )

    def __len__(self) -> int:
        return len(self.encoded_tensor)

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> Self:
        dictionary, encoded_tensor = torch.unique(
            tensor, dim=0, return_inverse=True
        )
        return cls(
            dictionary=cls.dictionary_cls(dictionary),
            encoded_tensor=encoded_tensor
        )

    @classmethod
    def from_strings(cls, strings: List[str] | np.ndarray) -> Self:
        max_length = max(len(s) for s in strings)
        batch_size = len(strings)
        arr = np.zeros((batch_size, max_length), dtype=np.uint8)
        for i, s in enumerate(strings):
            arr[i, :len(s)] = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
        plain_tensor = torch.tensor(arr, dtype=torch.uint8)
        return cls.from_tensor(plain_tensor)

    def to_strings(self) -> List[str]:
        plain_tensor = self.dictionary.tensor()[self.encoded_tensor]
        return [bytes(plain_tensor[i][plain_tensor[i] > 0].tolist()).decode("ascii") for i in range(len(plain_tensor))]

    @classmethod
    def from_string_tensor(cls, string_tensor: StringColumnTensor) -> Self:
        match string_tensor:
            case PlainEncodingStringColumnTensor():
                return cls.from_tensor(string_tensor.tensor())
            case DictionaryEncodingStringColumnTensor():
                return cls(cls.dictionary_cls.from_string_tensor(string_tensor.dictionary), string_tensor.encoded_tensor)
            case _:
                raise TypeError(
                    f"Unsupported type for DictionaryEncodingStringColumnTensor.from_string_tensor: {type(string_tensor)}. "
                    "Expected PlainEncodingStringColumnTensor or DictionaryEncodingStringColumnTensor."
                )


class CDictionaryEncodingStringColumnTensor(DictionaryEncodingStringColumnTensor):
    dictionary_cls = CPlainEncodingStringColumnTensor
