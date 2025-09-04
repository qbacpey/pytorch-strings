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
    def match_encoded_by_mask(self, mask: torch.Tensor, return_mask=False):
        matches = mask[self.encoded_tensor]

        if return_mask:
            return matches
        return matches.nonzero().view(-1)

    def match_encoded_by_codes(self, codes: torch.Tensor, return_mask=False) -> torch.Tensor:
        if len(codes) > 1:

            if len(self) * codes.numel() > 1e8:
                matches = torch.zeros(len(self),dtype=torch.bool)
                for code in codes:
                    matches |= self.encoded_tensor == code
            else:
                matches = (self.encoded_tensor.view(-1, 1) == codes).any(dim=1)

        elif len(codes) == 1:

            code = codes[0]
            matches = self.encoded_tensor == code

        else:

            matches = torch.zeros(len(self),dtype=torch.bool)

        if return_mask:
            return matches
        return matches.nonzero().view(-1)

    def __repr__(self) -> str:
        return (
            f"UnsortedDictionaryEncodingStringColumnTensor("
            f"dictionary_size={len(self.dictionary)}, "
            f"encoded_tensor_shape={self.encoded_tensor.shape})"
        )

    def query_equals_lookup_dict(self, query: str, return_mask=False) -> torch.Tensor:
        return self.dictionary.query_equals(query, return_mask=True)

    def query_equals_match_encoded(self, selector: torch.Tensor, return_mask=False) -> torch.Tensor:
        return self.match_encoded_by_mask(selector, return_mask)

    def query_less_than_lookup_dict(self, query: str, return_mask=False) -> torch.Tensor:
        return self.dictionary.query_less_than(query, return_mask=True)

    def query_less_than_match_encoded(self, selector: torch.Tensor, return_mask=False) -> torch.Tensor:
        return self.match_encoded_by_mask(selector, return_mask)

    def query_prefix_lookup_dict(self, prefix: str, return_mask=False) -> torch.Tensor:
        return self.dictionary.query_prefix(prefix, return_mask=True)

    def query_prefix_match_encoded(self, selector: torch.Tensor, return_mask=False) -> torch.Tensor:
        return self.match_encoded_by_mask(selector, return_mask)

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
