import os
import jax
from jax import Array
import jax.numpy as jnp
import numpy as np
from typing import Union
from attention import NamedTupleSubclass
from log_utils import logger

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["TIKTOKEN_CACHE_DIR"] = os.path.join(ROOT_DIR, ".cache")

from tiktoken import get_encoding, Encoding


def get_tokenizer(name: str) -> Encoding:
    return get_encoding(name)


# TODO: Ugly at the moment
def state_to_str(state: Union[NamedTupleSubclass, Array, bool], indent=0):
    # state is a NamedTuple which contains several other NamedTuples, Arrays, or bools

    if isinstance(state, list):
        return "[\n" + ", ".join([state_to_str(s, indent) for s in state]) + "\n]"

    if state is None:
        return "None"

    if isinstance(state, Array):
        return f"{state.shape}"

    if isinstance(state, bool):
        return state

    result = [f"{state.__class__.__name__}:"] if state else []
    for name, value in state._asdict().items():
        field_str = f"  {name}: " if indent > 0 else f"{name}: "
        result.append(f"  {field_str}{state_to_str(value, indent + 1)}")

    return "\n".join(result)
