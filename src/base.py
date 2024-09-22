from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from jax.numpy import ndarray
import jax


class Array(ndarray, Generic[TypeVar("Shape")]): ...


class BaseModule(ABC):
    def __init__(self):
        self.use_jit = False

    def set_use_jit(self, use_jit: bool):
        self.use_jit = use_jit

    @abstractmethod
    def forward(self, *args, **kwargs) -> Array:
        ...

    def __call__(self, *args, **kwargs) -> Array:
        if hasattr(self, "_jit_forward") and self.use_jit:
            return self._jit_forward(*args, **kwargs)
        else:
            return self.forward(*args, **kwargs)
