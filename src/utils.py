import os
from jax import Array
from typing import Union, NamedTuple
from attention import NamedTupleSubclass
from jax import tree_util
import torch.nn as nn

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


def count_params(state) -> int:

    if isinstance(state, Array):
        return state.size

    if isinstance(state, bool):
        return 0

    if isinstance(state, list):
        return sum([count_params(s) for s in state])

    if state is None:
        return 0

    return sum([count_params(v) for v in state._asdict().values()])

def torch_count_params(model: nn.Module, print_names: bool=False) -> int:
    s = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    if print_names:
        for name, p in model.named_parameters():
            print(name, p.numel())
    return s