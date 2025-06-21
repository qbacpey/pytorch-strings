import torch
from typing import List, Dict, Any
from . import StringColumnTensor, StringEncoder

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