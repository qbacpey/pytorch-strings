import torch
from typing import List, Dict, Any

class StringColumnTensor:
    """
    Base class for string encoding strategies.
    """

    def query_equals(self, query: str) -> torch.Tensor:
        """Return RowIDs where the string equals the query."""
        raise NotImplementedError

    def query_less_than(self, query: str) -> torch.Tensor:
        """Return RowIDs where the string is less than the query."""
        raise NotImplementedError

    def query_prefix(self, prefix: str) -> torch.Tensor:
        """Return RowIDs where the string starts with the given prefix."""
        raise NotImplementedError
    
    def query_aggregate(self) -> torch.Tensor:
        """
        Generate inverse indices that map each element's original position
        to its aggregated group index.
        
        Returns:
        List[int]: A list of length N (the number of original elements), 
                   where the i-th entry is the index of the aggregate 
                   group to which element i belongs.
        """
        raise NotImplementedError
    
    def query_sort(self, ascending: bool = True) -> torch.Tensor:
        """
        Generate a permutation index that maps sorted positions 
        back to their original positions.

        Args:
            ascending (bool, optional): If True (default), sort in ascending order; 
                                        if False, sort in descending order.

        Returns:
            List[int]: A list of length N (the number of elements), 
                    where the i-th entry is the original index 
                    of the element now at sorted position i.
        """
        raise NotImplementedError
    
    def index_select(self, *indices: Any) -> 'StringColumnTensor':
        """
        Select a subset of the encoded strings based on the provided indices.
        
        Args:
            indices: List of indices to select from the encoded column.
        
        Returns:
            StringColumnTensor: A new instance containing only the selected strings.
        """
        raise NotImplementedError
    
    def __len__(self) -> int:
        """
        Return number of rows.
        """
        raise NotImplementedError

    def get_config(self) -> Dict[str, Any]:
        """Return encoder configuration for benchmarking/reporting."""
        raise NotImplementedError

class StringEncoder:
    """
    Base class for string encoding strategies.
    """

    def encode(self, strings: List[str]) -> StringColumnTensor:
        """Encode and store the given list of strings."""
        raise NotImplementedError

    def decode(self, encoded_tensor: StringColumnTensor) -> List[str]:
        """Decode the given indices back to strings."""
        raise NotImplementedError