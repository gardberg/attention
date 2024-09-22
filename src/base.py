from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from jax.numpy import ndarray


class Array(ndarray, Generic[TypeVar("Shape")]): ...


class BaseModule(ABC):
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Array:
        ...

    def __call__(self, *args, **kwargs) -> Array:
        return self.forward(*args, **kwargs)
