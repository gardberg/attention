import os
from base import Array
from typing import Union, NamedTuple
from jax import Array as JaxArray
import torch.nn as nn
from log_utils import logger

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ["TIKTOKEN_CACHE_DIR"] = os.path.join(ROOT_DIR, ".cache")

from tiktoken import get_encoding, Encoding


def get_tokenizer(name: str) -> Encoding:
    return get_encoding(name)


# TODO: Ugly at the moment
def state_to_str(state: Union[NamedTuple, Array, bool], indent=0):
    # state is a NamedTuple which contains several other NamedTuples, Arrays, or bools

    if isinstance(state, list):
        return "[\n" + ", ".join([state_to_str(s, indent) for s in state]) + "\n]"

    if state is None:
        return "None"

    if isinstance(state, Array) or isinstance(state, JaxArray):
        return f"{state.shape}"

    if isinstance(state, bool):
        return state

    result = [f"{state.__class__.__name__}:"] if state is not None else []
    for name, value in state._asdict().items():
        field_str = f"  {name}: " if indent > 0 else f"{name}: "
        result.append(f"  {field_str}{state_to_str(value, indent + 1)}")

    return "\n".join(result)


# TODO: Atm counts all params. How do we count only learnable?
def count_params(state, seen_arrays=None) -> int:
    """Count unique parameters in a state object."""
    if seen_arrays is None:
        seen_arrays = set()

    if isinstance(state, int):
        return 1

    if isinstance(state, JaxArray) or isinstance(state, Array):
        # Get unique identifier using the array's data buffer
        array_id = id(state)

        if array_id not in seen_arrays:
            seen_arrays.add(array_id)
            return state.size
        logger.debug(
            f"Found duplicate array in {state.__class__.__name__} with id {array_id}"
        )
        return 0

    if isinstance(state, bool):
        return 0

    if isinstance(state, list):
        return sum([count_params(s, seen_arrays) for s in state])

    if state is None:
        return 0

    return sum([count_params(v, seen_arrays) for v in state._asdict().values()])


# For torch we count all params that require grad. Some, in e.g. batchnorm, are not learnabe,
# but are instead updated via a rule. These are counted in 'count_params' above, so
# there will be a difference
def torch_count_params(model: nn.Module, print_names: bool = False) -> int:
    s = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if print_names:
        for name, p in model.named_parameters():
            print(name, p.shape, p.numel())
    return s
