import os
import jax
from jax import Array
from typing import Union
from attention import NamedTupleSubclass

os.environ["TIKTOKEN_CACHE_DIR"] = "../.cache"
import tiktoken


class Tokenizer:
    # Wrapper class for tiktoken tokenizer

    def __init__(self, name="cl100k_base"):
        try:
            self.encoding = tiktoken.get_encoding(name)
        except:
            raise Exception(f"Could not download or find tiktoken tokenizer: '{name}'")

    def encode(self, text: str) -> Array:
        enc_list = self.encoding.encode(text)
        return jax.numpy.array(enc_list)

    # TODO: How do we handle batched array?
    def decode(self, enc: Array) -> str:
        return self.encoding.decode(enc.tolist())


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

