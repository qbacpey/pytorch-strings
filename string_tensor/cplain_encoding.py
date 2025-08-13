import torch
from .plain_encoding import PlainEncodingStringColumnTensor

class CPlainEncodingStringColumnTensor(PlainEncodingStringColumnTensor):
    encoded_tensor_transpose: torch.Tensor

    def __init__(self, encoded_tensor=torch.empty(0, dtype=torch.uint8), encoded_tensor_transpose=None):
        super().__init__(encoded_tensor=encoded_tensor)
        self.encoded_tensor_transpose = encoded_tensor.t() if encoded_tensor_transpose is None else encoded_tensor_transpose

    def __repr__(self) -> str:
        return f"CPlainEncodingStringColumnTensor(max_length={self.max_length}, encoded_tensor_shape={self.encoded_tensor_transpose.__len__})"
    
    def query_equals(self, query: str) -> torch.Tensor:
        query_tensor = torch.tensor(list(bytes(query, "ascii")), dtype=torch.uint8)
        query_tensor = torch.nn.functional.pad(query_tensor, (0, self.max_length - len(query_tensor)), value=0)
        
        match_index = torch.arange(len(self), dtype=torch.long)
        for i in range(self.max_length):
            if len(match_index) == 0:
                break
            # Filter the encoded tensor for the current character
            filtered_tensor = self.encoded_tensor_transpose[i][match_index]
            next_mask = (filtered_tensor == query_tensor[i])
            match_index = match_index[next_mask]
        return match_index


    def query_less_than(self, query: str) -> torch.Tensor:
        # Create properly padded query tensor
        query_tensor = torch.tensor(list(bytes(query, "ascii")), dtype=torch.uint8)
        query_tensor = torch.nn.functional.pad(query_tensor, (0, self.max_length - len(query_tensor)), value=0)

        lt_index = torch.empty(0, dtype=torch.long)
        match_index = torch.arange(len(self), dtype=torch.long)
        for i in range(self.max_length):
            if len(match_index) == 0:
                break
            # Filter the encoded tensor for the current character
            filtered_tensor = self.encoded_tensor_transpose[i][match_index]
            new_lt_mask = (filtered_tensor < query_tensor[i])
            lt_index = torch.cat((lt_index, match_index[new_lt_mask]))
            next_mask = (filtered_tensor == query_tensor[i])
            match_index = match_index[next_mask]

        lt_index, _ = lt_index.sort()
        return lt_index
    
    def query_prefix(self, prefix: str) -> torch.Tensor:
        prefix_len = len(prefix)
        prefix_tensor = torch.tensor(list(bytes(prefix, "ascii")), dtype=torch.uint8)

        match_index = torch.arange(len(self), dtype=torch.long)
        for i in range(prefix_len):
            if len(match_index) == 0:
                break
            # Filter the encoded tensor for the current character
            filtered_tensor = self.encoded_tensor_transpose[i][match_index]
            next_mask = (filtered_tensor == prefix_tensor[i])
            match_index = match_index[next_mask]
        return match_index
