import os
import jax
from jax import Array
import jax.numpy as jnp
import numpy as np
from typing import Union
from attention import NamedTupleSubclass
from log_utils import logger

os.environ["TIKTOKEN_CACHE_DIR"] = "../.cache"
import tiktoken


class Tokenizer:
    # Wrapper class for tiktoken tokenizer

    def __init__(self, name="cl100k_base"):
        try:
            self.encoding = tiktoken.get_encoding(name)
        except:
            raise Exception(f"Could not download or find tiktoken tokenizer: '{name}'")

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(text)
    
    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return self.encoding.encode_batch(texts)
        
    def decode(self, tokens: list[int]) -> str:
        return self.encoding.decode(tokens)

    def decode_batch(self, tokens: list[list[int]]) -> list[str]:
        return self.encoding.decode_batch(tokens)


def state_to_str(state: Union[NamedTupleSubclass, Array, bool], indent=0):
    # state is a NamedTuple which contains several other NamedTuples, Arrays, or bools

    if state is None:
        return "None"

    if isinstance(state, Array):
        return f"{state.shape}"

    if isinstance(state, bool):
        return state

    result = [f"{state.__class__.__name__}:"] if state else []
    for name, value in state._asdict().items():
        field_str = f"\t{name}: " if indent > 0 else f"{name}: "
        result.append(f"\t{field_str}{state_to_str(value, indent + 1)}")

    return "\n".join(result)
