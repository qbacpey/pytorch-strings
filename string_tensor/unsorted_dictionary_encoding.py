import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Any, Self, Tuple
from . import StringColumnTensor
from .plain_encoding import PlainEncodingStringColumnTensor
from .cplain_encoding import CPlainEncodingStringColumnTensor
from .dictionary_encoding import DictionaryEncodingStringColumnTensor


class UnsortedDictionaryEncodingStringColumnTensor(
    DictionaryEncodingStringColumnTensor
):

    def query_equals_codes(self, codes: torch.Tensor, return_mask=False) -> torch.Tensor:
        if len(codes) > 0:
            # If found in dictionary, get index and check against encoded tensor
            code = codes[0]
            matches = self.encoded_tensor == code

            if return_mask:
                return matches
            return matches.nonzero().view(-1)

        # If not found in dictionary, return empty tensor
        if return_mask:
            return torch.zeros(len(self),dtype=torch.bool)
        return torch.empty(0, dtype=torch.long)

    def __repr__(self) -> str:
        return (
            f"UnsortedDictionaryEncodingStringColumnTensor("
            f"dictionary_size={len(self.dictionary)}, "
            f"encoded_tensor_shape={self.encoded_tensor.shape})"
        )

    def query_less_than_codes(self, lt_codes: torch.Tensor, return_mask=False) -> torch.Tensor:
        if len(lt_codes) > 0:

            if len(self) * lt_codes.numel() > 1e9:
                matches = torch.zeros(len(self),dtype=torch.bool)
                for code in lt_codes:
                    matches |= self.encoded_tensor == code
            else:
                matches = (self.encoded_tensor.view(-1, 1) == lt_codes).any(dim=1)

            if return_mask:
                return matches
            return matches.nonzero().view(-1)

        if return_mask:
            return torch.zeros(len(self),dtype=torch.bool)
        return torch.empty(0, dtype=torch.long)

    def query_prefix_codes(self, prefix_codes: torch.Tensor, return_mask=False) -> torch.Tensor:
        if len(prefix_codes) > 0:

            if len(self) * prefix_codes.numel() > 1e9:
                matches = torch.zeros(len(self),dtype=torch.bool)
                for code in prefix_codes:
                    matches |= self.encoded_tensor == code
            else:
                matches = (self.encoded_tensor.view(-1, 1) == prefix_codes).any(dim=1)

            if return_mask:
                return matches
            return matches.nonzero().view(-1)

        if return_mask:
            return torch.zeros(len(self),dtype=torch.bool)
        return torch.empty(0, dtype=torch.long)

    @staticmethod
    def lc_affine_ranbperm(n: int, seed: int | None = None, a: int | None = 1664525, c: int | None = 1013904223, dtype=None, device=None) -> torch.Tensor:
        gen = torch.Generator(device).manual_seed(seed) if seed is not None else torch.default_generator
        ta = torch.tensor(a, device=device) if a is not None else torch.randint(1, n, (1,), generator=gen, device=device)
        tc = torch.tensor(c, device=device) if c is not None else torch.randint(0, n, (1,), generator=gen, device=device)
        tn = torch.tensor(n, device=device)

        while torch.gcd(ta, tn) != 1:
            ta = torch.randint(1,n, (1,), generator=gen, device=device)

        idx = torch.arange(n, dtype=dtype, device=device)
        perm = (idx * ta + tc) % tn
        return perm

    def shuffle(self) -> Self:
        dictionary = self.dictionary.tensor()
        encoded_tensor = self.encoded_tensor

        # perm = torch.randperm(len(dictionary), generator=torch.Generator().manual_seed(42))
        # dictionary = dictionary.scatter(dim=0, index=perm.view(-1, 1).expand(-1, dictionary.size(1)), src=dictionary)
        # encoded_tensor = perm[encoded_tensor]
        perm = self.lc_affine_ranbperm(len(dictionary), device=encoded_tensor.device)
        # perm_inv = perm.argsort()
        perm_inv = torch.empty_like(perm).scatter_(0, perm, torch.arange(len(perm)))
        dictionary = dictionary[perm_inv]
        encoded_tensor = perm[encoded_tensor]

        return self.__class__(
            dictionary=self.dictionary_cls(dictionary),
            encoded_tensor=encoded_tensor
        )

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> Self:
        dict_tensor = super().from_tensor(tensor)
        return dict_tensor.shuffle()

    @classmethod
    def from_strings(cls, strings: List[str] | np.ndarray) -> Self:
        inverse_indices, pd_strings = pd.Series(strings).factorize(sort=False)
        max_length = max(len(s) for s in pd_strings)
        batch_size = len(pd_strings)
        arr = np.zeros((batch_size, max_length), dtype=np.uint8)
        for i, s in enumerate(pd_strings):
            arr[i, : len(s)] = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
        plain_tensor = torch.tensor(arr, dtype=torch.uint8)

        return cls(
            dictionary=cls.dictionary_cls(plain_tensor),
            encoded_tensor=torch.tensor(inverse_indices, dtype=torch.long),
        )

    @classmethod
    def from_string_tensor(cls, string_tensor: StringColumnTensor) -> Self:
        match string_tensor:
            case PlainEncodingStringColumnTensor():
                return cls.from_tensor(string_tensor.tensor())
            case DictionaryEncodingStringColumnTensor():
                return cls(cls.dictionary_cls.from_string_tensor(string_tensor.dictionary), string_tensor.encoded_tensor).shuffle()
            case _:
                raise TypeError(
                    f"Unsupported type for DictionaryEncodingStringColumnTensor.from_string_tensor: {type(string_tensor)}. "
                    "Expected PlainEncodingStringColumnTensor."
                )


class UnsortedCDictionaryEncodingStringColumnTensor(
    UnsortedDictionaryEncodingStringColumnTensor
):
    dictionary_cls = CPlainEncodingStringColumnTensor
