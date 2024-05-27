from __future__ import annotations
from typing import TypeVar, Generic
import jax

class Array(jax.Array, Generic[TypeVar("Shape")]): ...
