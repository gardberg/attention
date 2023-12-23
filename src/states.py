from typing import NamedTuple
import jax

class BatchNormState(NamedTuple):
    # TODO: make into nested dict?
    mean: jax.Array = 0
    var: jax.Array = 1
    gamma: jax.Array = 1
    beta: jax.Array = 0
    momentum: jax.Array = 0.1

class LinearState(NamedTuple):
    weights: jax.Array
    bias: jax.Array

class MultiHeadAttentionState(NamedTuple):
    query_state: LinearState
    key_state: LinearState
    value_state: LinearState
    output_state: LinearState
