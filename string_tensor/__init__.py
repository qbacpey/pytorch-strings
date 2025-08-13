import numpy as np
import torch
from functools import cache
from typing import Generic, List, Dict, Any, Self, Type, TypeVar

T = TypeVar('T', bound='StringColumnTensor')

class StringColumnTensor:
    """
    Base class for string encoding strategies.
    """

    def __repr__(self) -> str:
        """
        Return a string representation of the StringColumnTensor.
        This should be implemented in subclasses to provide meaningful information.
        """
        raise NotImplementedError("Subclasses must implement __repr__ method.")

    def tuple_size(self) -> int:
        """
        Return the size of the tuple representing each string in bytes.
        
        Returns:
            int: The size of the tuple for each string.
        """
        raise NotImplementedError

    def tuple_counts(self) -> int:
        """
        Return the number of tuples in the encoded tensor.
        
        Returns:
            int: The number of tuples in the encoded tensor.
        """
        raise NotImplementedError

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

    def index_select(self, *indices: Any) -> Self:
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

    @classmethod
    def from_strings(cls, strings: List[str] | np.ndarray) -> Self:
        """
        Create a StringColumnTensor from a list of strings.
        
        Args:
            strings (List[str]): List of strings to encode.
        
        Returns:
            Self: An instance of StringColumnTensor containing the encoded strings.
        """
        raise NotImplementedError

    def to_strings(self) -> List[str]:
        """
        Convert the encoded tensor back to a list of strings.
        
        Returns:
            List[str]: The original strings represented by the encoded tensor.
        """
        raise NotImplementedError

    @classmethod
    def from_string_tensor(cls, string_tensor: 'StringColumnTensor') -> Self:
        """
        Create a StringColumnTensor from an existing tensor.
        
        Args:
            tensor (StringColumnTensor): The tensor to encode.
        
        Returns:
            Self: An instance of StringColumnTensor containing the encoded strings.
        """
        raise NotImplementedError

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> Self:
        """
        Create a StringColumnTensor from a PyTorch tensor.
        
        Args:
            tensor (torch.Tensor): The PyTorch tensor to encode.
        
        Returns:
            Self: An instance of StringColumnTensor containing the encoded strings.
        """
        raise NotImplementedError
    
    def to_string_tensor(self, target_cls: Type[T]) -> T:
        """
        Convert the current StringColumnTensor to the target class.

        Args:
            target_cls (Type[T]): The target class to convert to.

        Returns:
            T: An instance of the target class containing the encoded strings.
        """
        return target_cls.from_string_tensor(self)

    @classmethod
    @property
    @cache
    def Encoding(cls) -> str:
        return cls.__name__.replace("StringColumnTensor", "")

    class Encoder(Generic[T]):
        __outer__: Type[T]

        @classmethod
        def encode(cls, src: List[str] | np.ndarray | torch.Tensor | 'StringColumnTensor') -> T:
            """Encode and store the given list of strings."""
            if not hasattr(cls, '__outer__') or cls.__outer__ is None:
                raise NotImplementedError

            match src:
                case list() | np.ndarray():
                    return cls.__outer__.from_strings(src)
                case StringColumnTensor():
                    return cls.__outer__.from_string_tensor(src)
                case torch.Tensor():
                    return cls.__outer__.from_tensor(src)
                case _:
                    raise TypeError(
                        f"Unsupported type for encoding: {type(src)}. "
                        "Expected List[str], np.ndarray, torch.Tensor, or StringColumnTensor."
                    )
        @classmethod
        def decode(cls, encoded_tensor: Any) -> List[str]:
            """Decode the given indices back to strings."""
            return encoded_tensor.to_strings()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Automatically add an Encoder class to each subclass
        class Encoder(StringColumnTensor.Encoder):
            __outer__ = cls

        cls.Encoder = Encoder

# Import all encodings
from .plain_encoding import PlainEncodingStringColumnTensor
from .cplain_encoding import CPlainEncodingStringColumnTensor
from .dictionary_encoding import DictionaryEncodingStringColumnTensor, CDictionaryEncodingStringColumnTensor
from .unsorted_dictionary_encoding import UnsortedDictionaryEncodingStringColumnTensor, UnsortedCDictionaryEncodingStringColumnTensor
