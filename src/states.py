from typing import NamedTuple, TypeVar, Type
import jax
from jax import Array
import jax.numpy as jnp
from transformers.models.t5.modeling_t5 import T5DenseActDense, T5LayerFF, T5Attention

from torch import nn


# Contains 3 more than torch
class BatchNormState(NamedTuple):
    mean: Array
    var: Array
    gamma: Array
    beta: Array
    momentum: Array


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


class EmbeddingState(NamedTuple):
    embeddings: Array


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


class EncoderState(NamedTuple):
    layers: list[EncoderLayerState]
    norm: LayerNormState


class DecoderLayerState(NamedTuple):
    norm_attn: LayerNormState
    self_attn: MultiHeadAttentionState
    norm_src_attn: LayerNormState
    src_attn: MultiHeadAttentionState
    norm_ff: LayerNormState
    feed_forward: FeedForwardState


class DecoderState(NamedTuple):
    layers: list[DecoderLayerState]
    norm: LayerNormState


class TransformerState(NamedTuple):
    encoder: EncoderState
    decoder: DecoderState


class Seq2SeqTransformerState(NamedTuple):
    transformer: TransformerState
    src_embedding: EmbeddingState
    tgt_embedding: EmbeddingState
    project_out: LinearState


# T5 States
class T5DenseState(NamedTuple):
    wi: LinearState
    wo: LinearState


class T5FeedForwardState(NamedTuple):
    dense: T5DenseState
    norm: RMSNormState


# TODO: Redundant with MultiHeadAttentionState?
class T5MultiHeadAttentionState(NamedTuple):
    query: LinearState
    key: LinearState
    value: LinearState
    output: LinearState
    pos_emb: EmbeddingState


# TODO: Move into separate file
# Requires torch import, which is a bit heavy
NamedTupleSubclass = TypeVar("NamedTupleSubclass", bound=NamedTuple)


def to_jax_state(module: nn.Module) -> Type[NamedTupleSubclass]:
    """
    Extracts parameters from an nn.Module to a NamedTuple state
    
    module: nn.Module
    """
    if isinstance(module, nn.MultiheadAttention):
        emb_size = module.embed_dim
        w_in, b_in = module.in_proj_weight, module.in_proj_bias
        w_out, b_out = module.out_proj.weight, module.out_proj.bias

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

    elif isinstance(module, nn.Linear):
        weight, bias = (
            jnp.array(module.weight.detach()),
            jnp.array(module.bias.detach())
            if module.bias is not None
            else None,
        )
        return LinearState(weight, bias)

    elif isinstance(module, nn.LayerNorm):
        return LayerNormState(
            jnp.array(module.weight.detach()),
            jnp.array(module.bias.detach()),
        )

    elif isinstance(module, nn.TransformerEncoderLayer):
        return EncoderLayerState(
            layer_norm1=to_jax_state(module.norm1),
            self_attn=to_jax_state(module.self_attn),
            layer_norm2=to_jax_state(module.norm2),
            feed_forward=FeedForwardState(
                to_jax_state(module.linear1),
                to_jax_state(module.linear2),
            ),
        )

    elif isinstance(module, nn.TransformerDecoderLayer):
        return DecoderLayerState(
            norm_attn=to_jax_state(module.norm1),
            self_attn=to_jax_state(module.self_attn),
            norm_src_attn=to_jax_state(module.norm2),
            src_attn=to_jax_state(module.multihead_attn),
            norm_ff=to_jax_state(module.norm3),
            feed_forward=FeedForwardState(
                to_jax_state(module.linear1),
                to_jax_state(module.linear2),
            ),
        )

    elif isinstance(module, nn.TransformerEncoder):
        return EncoderState(
            layers=[to_jax_state(layer) for layer in module.layers],
            norm=to_jax_state(module.norm)
            if module.norm is not None
            else None,
        )

    elif isinstance(module, nn.TransformerDecoder):
        return DecoderState(
            layers=[to_jax_state(layer) for layer in module.layers],
            norm=to_jax_state(module.norm)
            if module.norm is not None
            else None,
        )

    elif isinstance(module, nn.Transformer):
        return TransformerState(
            encoder=to_jax_state(module.encoder),
            decoder=to_jax_state(module.decoder),
        )

    elif isinstance(module, nn.Embedding):
        return EmbeddingState(jnp.array(module.weight.detach()))

    elif isinstance(module, T5DenseActDense):
        return T5DenseState(
            wi=LinearState(
                weights=jnp.array(module.wi.weight.detach()),
                bias=jnp.array(module.wi.bias.detach()) if module.wi.bias is not None else None,
            ),
            wo=LinearState(
                weights=jnp.array(module.wo.weight.detach()),
                bias=jnp.array(module.wo.bias.detach()) if module.wo.bias is not None else None,
            ),
        )
    elif isinstance(module, T5LayerFF):
        return T5FeedForwardState(
            dense=to_jax_state(module.DenseReluDense),
            norm=RMSNormState(jnp.array(module.layer_norm.weight.detach())),
        )

    elif isinstance(module, T5Attention):
        return T5MultiHeadAttentionState(
            query=to_jax_state(module.q),
            key=to_jax_state(module.k),
            value=to_jax_state(module.v),
            output=to_jax_state(module.o),
            pos_emb=to_jax_state(module.relative_attention_bias)
            if hasattr(module, "relative_attention_bias")
            else None,
        )

    # elif isinstance(module, T5Attention):
    #     return T5MultiHeadAttentionState(
    #         query=LinearState(jnp.array(module.q.weight.T.detach().numpy()), None),
    #         key=LinearState(jnp.array(module.k.weight.T.detach().numpy()), None),
    #         value=LinearState(jnp.array(module.v.weight.T.detach().numpy()), None),
    #         output=LinearState(jnp.array(module.o.weight.T.detach().numpy()), None),
    #         pos_emb=to_jax_state(module.relative_attention_bias)
    #         if hasattr(module, "relative_attention_bias")
    #         else None,
    #     )

    else:
        raise NotImplementedError(
            f"to_jax_state not implemented for {type(module)}"
        )
