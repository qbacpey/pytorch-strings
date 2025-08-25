import torch
from typing import List
from string_tensor import StringColumnTensor

class MockOperator:
    col_name: str
    def apply(self, col: StringColumnTensor) -> torch.Tensor:
        raise NotImplementedError

class MockPredicate:
    def __init__(self, value: str):
        self.value: str = value
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(value='{self.value}')"
    def apply(self, col: StringColumnTensor) -> torch.Tensor:
        raise NotImplementedError
    @classmethod
    def __class_getitem__(cls, key: str):
        _aliases = {
            "equal": "Eq",
            "eq": "Eq",
            "less_than": "Lt",
            "lt": "Lt",
            "prefix": "Prefix",
        }
        for subclass in cls.__subclasses__():
            if subclass.__name__ == f"Predicate{_aliases[key.lower()]}":
                return subclass
        raise KeyError(f"No predicate subclass found for key '{key}'")

class FilterScan(MockOperator):
    def __init__(self, col_name: str, predicate: MockPredicate):
        self.col_name: str = col_name
        self.predicate: MockPredicate = predicate
    def __repr__(self) -> str:
        return f"FilterScan(col='{self.col_name}', pred={self.predicate!r})"
    def apply(self, col: StringColumnTensor) -> torch.Tensor:
        return self.predicate.apply(col)

class Aggregate(MockOperator):
    def __init__(self, col_name: str):
        self.col_name: str = col_name
    def __repr__(self) -> str:
        return f"Aggregate(col='{self.col_name}')"
    def apply(self, col: StringColumnTensor) -> torch.Tensor:
        return col.query_aggregate()

class Sort(MockOperator):
    def __init__(self, col_name: str, ascending: bool = True):
        self.col_name: str = col_name
        self.ascending: bool = ascending
    def __repr__(self) -> str:
        return f"Sort(col='{self.col_name}', asc={self.ascending})"
    def apply(self, col: StringColumnTensor) -> torch.Tensor:
        return col.query_sort(self.ascending)
    
class PredicateEq(MockPredicate):
    def apply(self, col: StringColumnTensor) -> torch.Tensor:
        return col.query_equals(self.value)

class PredicateLt(MockPredicate):
    def apply(self, col: StringColumnTensor) -> torch.Tensor:
        return col.query_less_than(self.value)

class PredicatePrefix(MockPredicate):
    def apply(self, col: StringColumnTensor) -> torch.Tensor:
        return col.query_prefix(self.value)

class MockStringColumnTensor(StringColumnTensor):
    def __init__(self, strs: List[str]):
        self.strs = strs

    def query_equals(self, query: str) -> torch.Tensor:
        return torch.tensor([i for i, s in enumerate(self.strs) if s == query])

    def query_less_than(self, query: str) -> torch.Tensor:
        """Return RowIDs where the string is less than the query."""
        return torch.tensor([i for i, s in enumerate(self.strs) if s < query])

    def query_prefix(self, prefix: str) -> torch.Tensor:
        """Return RowIDs where the string starts with the given prefix."""
        return torch.tensor([i for i, s in enumerate(self.strs) if s.startswith(prefix)])
