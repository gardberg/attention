from typing import NamedTuple
import jax


class BatchNormState(NamedTuple):
    # TODO: make into nested dict?
    mean: jax.Array = 0
    var: jax.Array = 1
    gamma: jax.Array = 1
    beta: jax.Array = 0
    momentum: jax.Array = 0.1


# Layer norm does not keep a running mean
class LayerNormState(NamedTuple):
    gamma: jax.Array
    beta: jax.Array


class LinearState(NamedTuple):
    weights: jax.Array
    bias: jax.Array

    
class FeedForwardState(NamedTuple):
    linear1_state: LinearState
    linear2_state: LinearState


class MultiHeadAttentionState(NamedTuple):
    query_state: LinearState
    key_state: LinearState
    value_state: LinearState
    output_state: LinearState


class EncoderLayerState(NamedTuple):
    layer_norm1_state: LayerNormState
    self_attn_state: MultiHeadAttentionState
    layer_norm2_state: LayerNormState
    feed_forward_state: FeedForwardState
    training: bool=True
