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
    linear1: LinearState
    linear2: LinearState


class MultiHeadAttentionState(NamedTuple):
    query: LinearState
    key: LinearState
    value: LinearState
    output: LinearState


class EncoderLayerState(NamedTuple):
    layer_norm1: LayerNormState
    self_attn: MultiHeadAttentionState
    layer_norm2: LayerNormState
    feed_forward: FeedForwardState


class DecoderLayerState(NamedTuple):
    norm_attn: LayerNormState
    self_attn: MultiHeadAttentionState
    norm_src_attn: LayerNormState
    src_attn: MultiHeadAttentionState
    norm_ff: LayerNormState
    feed_forward: FeedForwardState


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
            layer_norm1=to_jax_state(torch_module.norm1),
            self_attn=to_jax_state(torch_module.self_attn),
            layer_norm2=to_jax_state(torch_module.norm2),
            feed_forward=FeedForwardState(
                to_jax_state(torch_module.linear1),
                to_jax_state(torch_module.linear2),
            ),
        )

    elif isinstance(torch_module, nn.TransformerDecoderLayer):
        return DecoderLayerState(
            norm_attn=to_jax_state(torch_module.norm1),
            self_attn=to_jax_state(torch_module.self_attn),
            norm_src_attn=to_jax_state(torch_module.norm2),
            src_attn=to_jax_state(torch_module.multihead_attn),
            norm_ff=to_jax_state(torch_module.norm3),
            feed_forward=FeedForwardState(
                to_jax_state(torch_module.linear1),
                to_jax_state(torch_module.linear2),
            ),
        )

    else:
        raise NotImplementedError(
            f"to_jax_state not implemented for {type(torch_module)}"
        )
