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

    def query_equals(self, query: str) -> torch.Tensor:
        codes = self.dictionary.query_equals(query)
        return self.query_equals_codes(codes)
    
    def query_equals_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if len(codes) > 0:
            # If found in dictionary, get index and check against encoded tensor
            code = codes[0]
            matches = (self.encoded_tensor == code)
            return matches.nonzero().view(-1)
        # If not found in dictionary, return empty tensor
        return torch.empty(0, dtype=torch.long)

    def query_less_than(self, query: str) -> torch.Tensor:
        lt_codes = self.dictionary.query_less_than(query)
        return self.query_less_than_codes(lt_codes)
    
    def query_less_than_codes(self, lt_codes: torch.Tensor) -> torch.Tensor:
        if len(lt_codes) > 0:
            max_code = lt_codes[-1]
            matches = (self.encoded_tensor <= max_code)
            return matches.nonzero().view(-1)
        return torch.empty(0, dtype=torch.long)

    def query_prefix(self, prefix: str) -> torch.Tensor:
        prefix_codes = self.dictionary.query_prefix(prefix)
        return self.query_prefix_codes(prefix_codes)

    def query_prefix_codes(self, prefix_codes: torch.Tensor) -> torch.Tensor:
        if len(prefix_codes) > 0:
            min_code = prefix_codes[0]
            max_code = prefix_codes[-1]
            matches = (self.encoded_tensor >= min_code) & (self.encoded_tensor <= max_code)
            return matches.nonzero().view(-1)
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
    def from_tensor(cls, plain_tensor: torch.Tensor) -> Self:
        dictionary, encoded_tensor = torch.unique(
            plain_tensor, dim=0, return_inverse=True
        )
        return cls(
            dictionary=cls.dictionary_cls(dictionary),
            encoded_tensor=encoded_tensor
        )

    @classmethod
    def from_strings(cls, strings: List[str]) -> Self:
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
        if not isinstance(string_tensor, PlainEncodingStringColumnTensor):
            raise TypeError(
                    f"Unsupported type for DictionaryEncodingStringColumnTensor.from_string_tensor: {type(string_tensor)}. "
                    "Expected PlainEncodingStringColumnTensor."
                )
        return cls.from_tensor(string_tensor.tensor())

class CDictionaryEncodingStringColumnTensor(DictionaryEncodingStringColumnTensor):
    dictionary_cls = CPlainEncodingStringColumnTensor
