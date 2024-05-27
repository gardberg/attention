
# T5 translation model based on google t5-small for translation
# takes in input ids in the form of tokens, and returns predicted tokens (for translation)
from typing import Callable
from jax import random
import jax.numpy as jnp

from attention import Embedding, RMSNorm
from attention import Linear
from act import relu, dropout
from states import T5DenseState, T5FeedForwardState, T5MultiHeadAttentionState, EmbeddingState
from annotate import Array

# https://github.com/huggingface/transformers/blob/e4ea19b958c89d61e42461fac6ac8441787121f8/src/transformers/models/t5/modeling_t5.py#L646
class T5Encoder:
    def __init__(self):
        pass

class T5Model:
    def __init__(self, vocab_size: int, emb_size: int):
        self.shared_embedding = Embedding(vocab_size, emb_size) 
        self.encoder = None
        self.decoder = None


class T5MultiHeadAttention:
    def __init__(
        self, 
        emb_size: int,
        n_heads: int,
        use_rel_attn_bias: bool=False,
        rel_attn_n_buckets: int=32,
        rel_attn_max_distance: int=128
    ):
        self.emb_size = emb_size # d_model
        self.n_heads = n_heads
        self.use_rel_attn_bias = use_rel_attn_bias
        self.rel_attn_n_buckets = rel_attn_n_buckets
        self.rel_attn_max_distance = rel_attn_max_distance

        if self.use_rel_attn_bias:
            self.pos_emb = Embedding(self.rel_attn_n_buckets, self.n_heads)

    def compute_pos_bias(
        self, 
        pos_emb_state: EmbeddingState,
        query_len: int,
        key_len: int,
    ) -> Array["1, n_heads, query_len, key_len"]:

        # int64 in original
        context_pos = jnp.arange(query_len, dtype=jnp.int32)[:, None]
        memory_pos = jnp.arange(key_len, dtype=jnp.int32)[None, :]

        relative_pos: Array["q_len, k_len"] = memory_pos - context_pos

        relative_pos_bucket = self._calc_bucket(
            relative_pos,
            n_buckets=self.rel_attn_n_buckets,
            max_distance=self.rel_attn_max_distance,
        )

        values = self.pos_emb(pos_emb_state, relative_pos_bucket)
        values = values.transpose((2, 0, 1))[None, :, :, :]

        return values

    def _calc_bucket(
        self, relative_pos: Array["q_len, k_len"], 
        n_buckets: int=32, 
        max_distance: int=128
    ) -> Array["q_len, k_len"]:

        # for each index in relative_pos, return which bucket it corresponds to.
        # in range [0, n_buckets)

        rel_buckets = 0
        # assume bidirectional
        n_buckets //= 2
        rel_buckets += (relative_pos > 0).astype(jnp.int32) * n_buckets
        relative_pos = jnp.abs(relative_pos)

        max_exact = n_buckets // 2
        is_small = relative_pos < max_exact

        rel_pos_if_large = max_exact + (
            jnp.log(relative_pos.astype(jnp.float32) / max_exact)
            / jnp.log(max_distance / max_exact)
            * (n_buckets - max_exact)
        ).astype(jnp.int32)

        rel_pos_if_large = jnp.minimum(
            rel_pos_if_large, jnp.full_like(rel_pos_if_large, n_buckets - 1)
        )

        rel_buckets += jnp.where(is_small, relative_pos, rel_pos_if_large)
        return rel_buckets




class T5Dense:
    def __init__(self, n_in: int, d_ff: int, dropout: float=0.1, act: Callable=relu):
        self.wi = Linear(n_in, d_ff, bias=False)
        self.wo = Linear(d_ff, n_in, bias=False)
        self.dropout = dropout
        self.act = act

    def __call__(self, state: T5DenseState, x: Array, rng: Array, training=True) -> Array:
        x = self.wi(state.wi, x)
        x = self.act(x)
        x = dropout(x, self.dropout, rng, training)
        x = self.wo(state.wo, x)
        return x

    def init_state(self, rng: Array) -> T5DenseState:
        rng1, rng2 = random.split(rng, 2)
        return T5DenseState(
            wi=self.wi.init_state(rng1),
            wo=self.wo.init_state(rng2),
        )


class T5FeedForward:
    def __init__(self, n_in: int, d_ff: int, dropout: float=0.1):
        self.dense = T5Dense(n_in, d_ff, dropout)
        # T5LayerFF uses custom LayerNorm which is equivalent to RMSNorm
        self.norm = RMSNorm(n_in, eps=1e-6)
        self.dropout = dropout

    def __call__(self, state: T5FeedForwardState, x: Array, rng: Array, training=True) -> Array:
        rng1, rng2 = random.split(rng, 2)

        z = self.norm(state.norm, x)
        z = self.dense(state.dense, z, rng1, training)
        x += dropout(z, self.dropout, rng2, training)
        return x
