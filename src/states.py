from typing import NamedTuple, Type
import jax
import jax.numpy as jnp
from transformers.models.t5.modeling_t5 import (
    T5DenseActDense,
    T5LayerFF,
    T5Attention,
    T5LayerSelfAttention,
    T5LayerCrossAttention,
    T5Block,
    T5Stack,
    T5Model,
    T5ForConditionalGeneration,
)
from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Block,
    GPT2MLP,
    GPT2Attention,
    GPT2Model,
    GPT2LMHeadModel,
)
from transformers.pytorch_utils import Conv1D
from base import Array

from torch import nn


def from_pretrained(hf_model: str) -> NamedTuple:
    SUPPORTED_MODELS = ["google-t5/t5-small"]
    if hf_model not in SUPPORTED_MODELS:
        raise NotImplementedError(
            f"Model {hf_model} not supported. Supported models are: {SUPPORTED_MODELS}"
        )

    if hf_model == "google-t5/t5-small":
        model = T5ForConditionalGeneration.from_pretrained(hf_model)
        state = to_jax_state(model)
        del model
        return state

    raise ValueError(f"Unexpected model {hf_model}")


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


class T5AttentionLayerState(NamedTuple):
    attention: T5MultiHeadAttentionState
    norm: RMSNormState


class T5EncoderBlockState(NamedTuple):
    self_attn_layer: T5AttentionLayerState
    feed_forward: T5FeedForwardState


class T5DecoderBlockState(NamedTuple):
    self_attn_layer: T5AttentionLayerState
    cross_attn_layer: T5AttentionLayerState
    feed_forward: T5FeedForwardState


class T5EncoderState(NamedTuple):
    embedding: EmbeddingState  # Shared with T5Decoder
    blocks: list[T5EncoderBlockState]
    norm: RMSNormState


class T5DecoderState(NamedTuple):
    embedding: EmbeddingState  # Shared with T5Encoder
    blocks: list[T5DecoderBlockState]
    norm: RMSNormState


class T5BaseModelState(NamedTuple):
    encoder: T5EncoderState
    decoder: T5DecoderState


class T5ModelState(NamedTuple):
    base_model: T5BaseModelState
    lm_head: LinearState


# GPT2 States
class GPT2DenseState(NamedTuple):
    c_fc: LinearState
    c_proj: LinearState


class GPT2AttentionState(NamedTuple):
    c_attn: LinearState
    c_proj: LinearState


class GPT2BlockState(NamedTuple):
    ln_1: LayerNormState
    attn: GPT2AttentionState
    ln_2: LayerNormState
    mlp: GPT2DenseState


class GPT2BaseModelState(NamedTuple):
    wte: EmbeddingState
    wpe: EmbeddingState
    blocks: list[GPT2BlockState]
    ln_f: LayerNormState


class GPT2State(NamedTuple):
    transformer: GPT2BaseModelState
    lm_head: LinearState


def to_jax_state(module: nn.Module) -> NamedTuple:
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
            jnp.array(module.bias.detach()) if module.bias is not None else None,
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
            norm=to_jax_state(module.norm) if module.norm is not None else None,
        )

    elif isinstance(module, nn.TransformerDecoder):
        return DecoderState(
            layers=[to_jax_state(layer) for layer in module.layers],
            norm=to_jax_state(module.norm) if module.norm is not None else None,
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
                bias=(
                    jnp.array(module.wi.bias.detach())
                    if module.wi.bias is not None
                    else None
                ),
            ),
            wo=LinearState(
                weights=jnp.array(module.wo.weight.detach()),
                bias=(
                    jnp.array(module.wo.bias.detach())
                    if module.wo.bias is not None
                    else None
                ),
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
            pos_emb=(
                to_jax_state(module.relative_attention_bias)
                if hasattr(module, "relative_attention_bias")
                else None
            ),
        )

    elif isinstance(module, T5LayerSelfAttention):
        return T5AttentionLayerState(
            attention=to_jax_state(module.SelfAttention),
            norm=RMSNormState(jnp.array(module.layer_norm.weight.detach())),
        )

    elif isinstance(module, T5LayerCrossAttention):
        return T5AttentionLayerState(
            attention=to_jax_state(module.EncDecAttention),
            norm=RMSNormState(jnp.array(module.layer_norm.weight.detach())),
        )

    # if is instace T5Block or has class name T5Block
    elif isinstance(module, T5Block) or module.__class__.__name__ == "T5Block":
        if module.is_decoder:
            return T5DecoderBlockState(
                self_attn_layer=to_jax_state(module.layer[0]),
                cross_attn_layer=to_jax_state(module.layer[1]),
                feed_forward=to_jax_state(module.layer[2]),
            )
        else:
            return T5EncoderBlockState(
                self_attn_layer=to_jax_state(module.layer[0]),
                feed_forward=to_jax_state(module.layer[-1]),
            )

    elif isinstance(module, T5Stack) or module.__class__.__name__ == "T5Stack":
        if module.is_decoder:
            return T5DecoderState(
                embedding=to_jax_state(module.embed_tokens),
                blocks=[to_jax_state(block) for block in module.block],
                norm=RMSNormState(jnp.array(module.final_layer_norm.weight.detach())),
            )
        else:
            return T5EncoderState(
                embedding=to_jax_state(module.embed_tokens),
                blocks=[to_jax_state(block) for block in module.block],
                norm=RMSNormState(jnp.array(module.final_layer_norm.weight.detach())),
            )

    elif isinstance(module, T5Model):
        # Assumes shared embedding is already set in T5Model via self.set_input_embeddings
        return T5BaseModelState(
            encoder=to_jax_state(module.encoder),
            decoder=to_jax_state(module.decoder),
        )

    elif isinstance(module, T5ForConditionalGeneration):
        return T5ModelState(
            base_model=T5BaseModelState(
                encoder=to_jax_state(module.encoder),
                decoder=to_jax_state(module.decoder),
            ),
            lm_head=to_jax_state(module.lm_head),
        )

    # GPT2

    elif isinstance(module, Conv1D):
        return LinearState(
            weights=jnp.array(module.weight.detach().transpose(0, 1)),
            bias=jnp.array(module.bias.detach()) if module.bias is not None else None,
        )

    elif isinstance(module, GPT2MLP):
        return GPT2DenseState(
            c_fc=to_jax_state(module.c_fc),
            c_proj=to_jax_state(module.c_proj),
        )

    elif isinstance(module, GPT2Attention):
        return GPT2AttentionState(
            c_attn=to_jax_state(module.c_attn),
            c_proj=to_jax_state(module.c_proj),
        )

    elif isinstance(module, GPT2Block):
        return GPT2BlockState(
            ln_1=to_jax_state(module.ln_1),
            attn=to_jax_state(module.attn),
            ln_2=to_jax_state(module.ln_2),
            mlp=to_jax_state(module.mlp),
        )

    elif isinstance(module, GPT2Model):
        return GPT2BaseModelState(
            wte=to_jax_state(module.wte),
            wpe=to_jax_state(module.wpe),
            blocks=[to_jax_state(block) for block in module.h],
            ln_f=to_jax_state(module.ln_f),
        )

    elif isinstance(module, GPT2LMHeadModel):
        transformer_state = to_jax_state(module.transformer)
        return GPT2State(
            transformer=transformer_state,
            lm_head=LinearState(
                weights=transformer_state.wte.embeddings,  # shared weights between lm_head and wte
                bias=None,
            ),
        )

    else:
        raise NotImplementedError(f"to_jax_state not implemented for {type(module)}")
