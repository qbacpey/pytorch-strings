import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Any, Self, Tuple
from . import StringColumnTensor
from .plain_encoding import PlainEncodingStringColumnTensor
from .cplain_encoding import CPlainEncodingStringColumnTensor
from .dictionary_encoding import DictionaryEncodingStringColumnTensor

class UnsortedDictionaryEncodingStringColumnTensor(DictionaryEncodingStringColumnTensor):

    def query_equals_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if len(codes) > 0:
            # If found in dictionary, get index and check against encoded tensor
            code = codes[0]
            matches = (self.encoded_tensor == code)
            return matches.nonzero().view(-1)
        # If not found in dictionary, return empty tensor
        return torch.empty(0, dtype=torch.long)

    def query_less_than_codes(self, lt_codes: torch.Tensor) -> torch.Tensor:
        if len(lt_codes) > 0:
            matches = (self.encoded_tensor.view(-1, 1) == lt_codes).any(dim=1)
            return matches.nonzero().view(-1)
        return torch.empty(0, dtype=torch.long)

    def query_prefix_codes(self, prefix_codes: torch.Tensor) -> torch.Tensor:
        if len(prefix_codes) > 0:
            matches = (self.encoded_tensor.view(-1, 1) == prefix_codes).any(dim=1)
            return matches.nonzero().view(-1)
        return torch.empty(0, dtype=torch.long)

    @classmethod
    def from_tensor(cls, plain_tensor: torch.Tensor) -> Self:
        unique_out: Tuple[torch.Tensor, torch.Tensor] = torch.unique(
            plain_tensor, dim=0, return_inverse=True
        )
        dictionary, encoded_tensor = unique_out

        perm = torch.randperm(len(dictionary))
        dictionary = dictionary.scatter(dim=0, index=perm.view(-1, 1), src=dictionary)
        encoded_tensor = perm[encoded_tensor]

        return cls(
            dictionary=cls.dictionary_cls(dictionary),
            encoded_tensor=encoded_tensor
        )

    @classmethod
    def from_strings(cls, strings: List[str]) -> Self:
        inverse_indices, pd_strings = pd.Series(strings).factorize(sort=False)
        max_length = max(len(s) for s in pd_strings)
        batch_size = len(pd_strings)
        arr = np.zeros((batch_size, max_length), dtype=np.uint8)
        for i, s in enumerate(pd_strings):
            arr[i, :len(s)] = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
        plain_tensor = torch.tensor(arr, dtype=torch.uint8)

        return cls(
            dictionary=cls.dictionary_cls(plain_tensor),
            encoded_tensor=torch.tensor(inverse_indices, dtype=torch.long)
        )

    @classmethod
    def from_string_tensor(cls, string_tensor: StringColumnTensor) -> Self:
        if not isinstance(string_tensor, PlainEncodingStringColumnTensor):
            raise TypeError(
                    f"Unsupported type for UnsortedDictionaryEncodingStringColumnTensor.from_string_tensor: {type(string_tensor)}. "
                    "Expected PlainEncodingStringColumnTensor."
                )
        return cls.from_tensor(string_tensor.tensor())

class UnsortedCDictionaryEncodingStringColumnTensor(UnsortedDictionaryEncodingStringColumnTensor):
    dictionary_cls = CPlainEncodingStringColumnTensor
