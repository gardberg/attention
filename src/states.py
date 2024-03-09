from typing import NamedTuple, TypeVar, Type
import jax
from jax import Array
import jax.numpy as jnp

import torch
from torch import nn

class BatchNormState(NamedTuple):
    # TODO: make into nested dict?
    mean: Array = 0
    var: Array = 1
    gamma: Array = 1
    beta: Array = 0
    momentum: Array = 0.1


# Layer norm does not keep a running mean
class LayerNormState(NamedTuple):
    gamma: Array
    beta: Array


class RMSNormState(NamedTuple):
    gamma: Array


class SnakeState(NamedTuple):
    a: Array


class LinearState(NamedTuple):
    weights: Array
    bias: Array


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
    training: bool


# TODO: Move into separate file
# Requires torch import, which is a bit heavy
NamedTupleSubclass = TypeVar("NamedTupleSubclass", bound=NamedTuple)


def to_jax_state(torch_module: nn.Module) -> Type[NamedTupleSubclass]:
    if isinstance(torch_module, nn.MultiheadAttention):
        emb_size = torch_module.embed_dim
        w_in, b_in = torch_module.in_proj_weight, torch_module.in_proj_bias
        w_out, b_out = torch_module.out_proj.weight, torch_module.out_proj.bias

        # Unstack stacked q, k, and v weights
        linear_states = [
            LinearState(
                w_in[i * emb_size : (i + 1) * emb_size],
                b_in[i * emb_size : (i + 1) * emb_size] if b_in is not None else None,
            )
            for i in range(3)
        ]
        linear_states.append(LinearState(w_out, b_out if b_out is not None else None))

        return MultiHeadAttentionState(
            *(jax.tree_map(lambda x: jnp.array(x.detach()), linear_states))
        )

    elif isinstance(torch_module, nn.Linear):
        weight, bias = (
            jnp.array(torch_module.weight.detach()),
            jnp.array(torch_module.bias.detach())
            if torch_module.bias is not None
            else None,
        )
        return LinearState(weight, bias)

    elif isinstance(torch_module, nn.LayerNorm):
        return LayerNormState(
            jnp.array(torch_module.weight.detach()),
            jnp.array(torch_module.bias.detach()),
        )

    elif isinstance(torch_module, nn.TransformerEncoderLayer):
        return EncoderLayerState(
            layer_norm1_state=to_jax_state(torch_module.norm1),
            self_attn_state=to_jax_state(torch_module.self_attn),
            layer_norm2_state=to_jax_state(torch_module.norm2),
            feed_forward_state=FeedForwardState(
                to_jax_state(torch_module.linear1),
                to_jax_state(torch_module.linear2),
            ),
            training=torch_module.training,
        )

    else:
        raise NotImplementedError(
            f"to_jax_state not implemented for {type(torch_module)}"
        )
