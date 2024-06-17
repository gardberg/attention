
# T5 translation model based on google t5-small for translation
# takes in input ids in the form of tokens, and returns predicted tokens (for translation)
from typing import Callable
from jax import random
import jax.numpy as jnp

from attention import Embedding, RMSNorm, Linear
from act import relu, dropout, softmax
from states import T5DenseState, T5FeedForwardState, T5MultiHeadAttentionState, EmbeddingState, LinearState, T5AttentionLayerState, T5EncoderBlockState, T5DecoderBlockState
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


# T5Stack with encoder config
class T5Encoder:
    def __init__(self, emb_size: int, n_layers: int, vocab_size: int, dropout_rate: float=0.1):
        self.emb_size = emb_size

        # Shared between enconder and decoder 
        # -> make sure to use same weights from state!
        self.embedding = Embedding(vocab_size, emb_size)

        # Only first block uses relative attention bias
        self.block = [
            T5EncoderBlock(use_rel_attn_bias=bool(i == 0)) for i in range(n_layers)
        ]

        self.norm = RMSNorm(self.emb_size, eps=1e-6)
        self.dropout_rate = dropout_rate

    def forward(self):
        # TODO
        pass


class T5EncoderBlock:
    def __init__(self, emb_size: int, n_heads: int, dropout_rate: float=0.1, use_rel_attn_bias: bool=False):
        self.self_attn_layer = T5SelfAttention(
            emb_size,
            n_heads,
            dropout=dropout_rate,
            use_rel_attn_bias=use_rel_attn_bias)

        self.feed_forward = T5FeedForward(emb_size, 4 * emb_size) # 2048

    def forward(
        self,
        state: T5EncoderBlockState,
        x: Array["batch_size, context_len, emb_size"],
        rng: Array,
        training = False,
    ) -> Array["batch_size, context_len, emb_size"]:

        rngs = random.split(rng, 2)
        
        self_attn_out = self.self_attn_layer(state.self_attn_layer, x, rngs[0], training=training)
        return self.feed_forward(state.feed_forward, self_attn_out, rngs[1], training=training)

    def __call__(self, *args, **kwargs) -> Array:
        return self.forward(*args, **kwargs)


class T5DecoderBlock:
    def __init__(self, emb_size: int, n_heads: int, dropout_rate: float=0.1, use_rel_attn_bias: bool=False):
        self.self_attn_layer = T5SelfAttention(
            emb_size,
            n_heads,
            dropout=dropout_rate,
            use_rel_attn_bias=use_rel_attn_bias,
            bidirectional=False)

        self.cross_attn_layer = T5CrossAttention(emb_size, n_heads, dropout=dropout_rate)
        self.feed_forward = T5FeedForward(emb_size, 4 * emb_size)

        self.debug_states = dict()

    def forward(
        self,
        state: T5DecoderBlockState,
        xq: Array["batch_size, tgt_len, emb_size"],
        xkv: Array["batch_size, src_len, emb_size"], # encoder hidden states
        rng: Array,
        training = False,
    ) -> Array["batch_size, tgt_len, emb_size"]:

        rngs = random.split(rng, 3)

        self.debug_states["inputs_q"] = xq.copy()
        self.debug_states["inputs_kv"] = xkv.copy()

        self_attn_out = self.self_attn_layer(state.self_attn_layer, xq, rngs[0], training=training)

        self.debug_states["self_attn_out"] = self_attn_out.copy()

        cross_attn_out = self.cross_attn_layer(state.cross_attn_layer, self_attn_out, xkv, rngs[1], training=training)

        self.debug_states["cross_attn_out"] = cross_attn_out.copy()

        ff_out = self.feed_forward(state.feed_forward, cross_attn_out, rngs[2], training=training)

        self.debug_states["ff_out"] = ff_out.copy()

        return ff_out

    def __call__(self, *args, **kwargs) -> Array:
        return self.forward(*args, **kwargs)


class T5CrossAttention:
    def __init__(self, emb_size: int, n_heads: int, dropout: float=0.1, use_rel_attn_bias: bool=False):
        self.attention = T5MultiHeadAttention(emb_size, n_heads, use_rel_attn_bias=use_rel_attn_bias, dropout=dropout)
        self.norm = RMSNorm(emb_size, eps=1e-6)
        self.dropout_rate = dropout

    def forward(
        self,
        state: T5AttentionLayerState,
        xq: Array["batch_size, tgt_len, emb_size"],
        xkv: Array["batch_size, src_len, emb_size"],
        rng: Array,
        training: bool=False,
    ) -> Array["batch_size, tgt_len, emb_size"]:
        xq_normed = self.norm(state.norm, xq)
        
        rngs = random.split(rng, 2)

        attn_out = self.attention(
            state.attention,
            xq_normed,
            xkv,
            xkv,
            rng=rngs[0],
            training=training
        )

        return xq + dropout(attn_out, self.dropout_rate, rngs[1], training)

    def __call__(self, *args, **kwargs) -> Array:
        return self.forward(*args, **kwargs)
        

class T5SelfAttention:
    def __init__(self, emb_size: int, n_heads: int, dropout: float=0.1, use_rel_attn_bias: bool=False, bidirectional: bool=True):
        self.attention = T5MultiHeadAttention(emb_size, n_heads, use_rel_attn_bias, dropout=dropout, bidirectional=bidirectional)
        self.norm = RMSNorm(emb_size, eps=1e-06)
        self.dropout_rate = dropout

    def forward(
        self,
        state: T5AttentionLayerState,
        x: Array["batch_size, context_len, emb_size"],
        rng: Array,
        training: bool=False,
    ) -> Array["batch_size, context_len, emb_size"]:
        x_normed = self.norm(state.norm, x)

        rngs = random.split(rng, 2)

        attn_out = self.attention(
            state.attention,
            x_normed,
            x_normed,
            x_normed,
            rng=rngs[0],
            training=training
        )
        
        return x + dropout(attn_out, self.dropout_rate, rngs[1], training)

    def __call__(self, *args, **kwargs) -> Array:
        return self.forward(*args, **kwargs)

class T5MultiHeadAttention:
    def __init__(
        self, 
        emb_size: int,
        n_heads: int,
        use_rel_attn_bias: bool=False,
        rel_attn_n_buckets: int=32,
        rel_attn_max_distance: int=128,
        dropout: float=0.1,
        bidirectional: bool=True
    ):
        
        # TODO: Add masking

        self.n_heads = n_heads
        self.emb_size = emb_size

        assert emb_size % n_heads == 0, "emb_size must be divisible by n_heads"

        self.d_k = emb_size // n_heads # Hidden inner dim per head
        self.inner_dim = self.n_heads * self.d_k

        self.query_dense = Linear(emb_size, self.inner_dim, bias=False)
        self.key_dense = Linear(emb_size, self.inner_dim, bias=False)
        self.value_dense = Linear(emb_size, self.inner_dim, bias=False)
        self.out_dense = Linear(self.inner_dim, emb_size, bias=False)

        self.use_rel_attn_bias = use_rel_attn_bias
        self.rel_attn_n_buckets = rel_attn_n_buckets
        self.rel_attn_max_distance = rel_attn_max_distance
        self.dropout_rate = dropout

        if self.use_rel_attn_bias:
            self.pos_emb = Embedding(self.rel_attn_n_buckets, self.n_heads)

        self.bidirectional = bidirectional

        self.debug_states = dict()

    def get_kv(
        self,
        state: T5MultiHeadAttentionState,
        xk: Array["batch_size, src_len, emb_size"],
        xv: Array,
        use_cache: bool,
        kv_cache: tuple[Array, Array]
    ) -> tuple[Array, Array]:
        if use_cache and kv_cache is not None:
            assert len(kv_cache) == 2, "kv_cache must be a tuple of length 2"
            cached_keys, cached_values = kv_cache
            
            next_k = xk[-1][None, ...]
            next_v = xv[-1][None, ...]

            key = jnp.concatenate(
                [cached_keys, self.apply_dense(state.key, next_k, self.key_dense)], axis=0
            )

            value = jnp.concatenate(
                [cached_values, self.apply_dense(state.value, next_v, self.value_dense)], axis=0
            )
        else:
            key = self.apply_dense(state.key, xk, self.key_dense)
            value = self.apply_dense(state.value, xv, self.value_dense)
            
        # (batch_size, n_heads, src_len, emb_size)
        return key, value

    def apply_dense(self, state: LinearState, x: Array, dense_fn: Linear) -> Array:
        # xq: (batch_size, context_len, emb_size)
        # dense_fn: Linear callable

        batch_size = x.shape[0]
        query: Array["batch_size, context_len, emb_size"] = dense_fn(state, x)
        query: Array["batch_size, n_heads, context_len, d_k"] = query.reshape(batch_size, -1, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        return query

    def forward(
        self,
        state: T5MultiHeadAttentionState,
        xq: Array["batch_size, tgt_len, emb_size"],
        xk: Array["batch_size, src_len, emb_size"],
        xv: Array["batch_size, src_len, emb_size"],
        rng: Array,
        mask: Array["tgt_len, src_len"] = None,
        use_cache: bool = False,
        kv_cache: tuple[Array, Array] = None,
        training = False,
    ) -> tuple[Array["batch_size, tgt_len, emb_size"], tuple[Array, Array]]:

        # ! Uses batch dimension first !

        # Returns (attn_output, kv_cache) if use_cache, else attn_output
        # kv_cache: tuple[()]
        
        batch_size, tgt_len = xq.shape[:2]
        src_len = xk.shape[1]

        # TODO: Masking

        query: Array["batch_size, n_heads, tgt_len, d_k"] = self.apply_dense(state.query, xq, self.query_dense)

        assert query.shape == (
            batch_size,
            self.n_heads,
            tgt_len,
            self.d_k,
        ), f"Expected shape {(batch_size, self.n_heads, tgt_len, self.d_k)}, got {query.shape}"

        # (batch_size, n_heads, src_len, d_k)
        key, value = self.get_kv(state, xk, xv, use_cache, kv_cache)

        if use_cache: kv_cache = (key, value)

        self.debug_states["query"] = query.copy()
        self.debug_states["key"] = key.copy()
        self.debug_states["value"] = value.copy()

        scores: Array["batch_size, n_heads, tgt_len, src_len"] = jnp.matmul(query, key.transpose((0, 1, 3, 2)))

        self.debug_states["scores"] = scores.copy()

        assert scores.shape == (
            batch_size,
            self.n_heads,
            tgt_len,
            src_len
        ), f"Expected shape {(batch_size, self.n_heads, tgt_len, src_len)}, got {scores.shape}"

        # TODO: Is scaling done in reference impl?
        # scores = scores * (1 / jnp.sqrt(self.d_k))

        # TODO: Only keep values relevant with kv cache
        position_bias: Array["1, n_heads, tgt_len, src_len"] = self.compute_pos_bias(state.pos_emb, tgt_len, src_len)

        self.debug_states["position_bias"] = position_bias.copy()

        assert position_bias.shape == (
            1,
            self.n_heads,
            tgt_len,
            src_len,
        ), f"Expected shape {(1, self.n_heads, tgt_len, src_len)}, got {position_bias.shape}"
        
        # broadcast to (batch_size, n_heads, tgt_len, src_len)
        position_bias = jnp.repeat(position_bias, batch_size, axis=0)

        scores += position_bias

        attn_weights = softmax(scores, dim=-1)

        self.debug_states["attn_weights"] = attn_weights.copy()

        attn_weights = dropout(attn_weights, self.dropout_rate, rng, training)

        self.debug_states["attn_weights_dropout"] = attn_weights.copy()

        attn: Array["batch_size, n_heads, tgt_len, d_k"] = jnp.matmul(attn_weights, value)
        attn: Array["batch_size, tgt_len, emb_size"] = attn.transpose((0, 2, 1, 3)).reshape(batch_size, tgt_len, self.inner_dim)

        self.debug_states["attn"] = attn.copy()

        assert attn.shape == (
            batch_size,
            tgt_len,
            self.inner_dim,
        ), f"Expected shape {(batch_size, tgt_len, self.inner_dim)}, got {attn.shape}"

        out: Array["tgt_len, batch_size, emb_size"] = self.out_dense(state.output, attn)

        self.debug_states["out"] = out.copy()

        return (out, kv_cache) if use_cache else out

    def __call__(self, *args, **kwargs) -> Array:
        return self.forward(*args, **kwargs)

    def init_state(self, rng: Array) -> T5MultiHeadAttentionState:
        rngs = random.split(rng, 5)
        return T5MultiHeadAttentionState(
            query=self.query_dense.init_state(rngs[0]),
            key=self.key_dense.init_state(rngs[1]),
            value=self.value_dense.init_state(rngs[2]),
            output=self.out_dense.init_state(rngs[3]),
            pos_emb=self.pos_emb.init_state(rngs[4]) if self.use_rel_attn_bias else None,
        )

    def compute_pos_bias(
        self, 
        pos_emb_state: EmbeddingState,
        query_len: int,
        key_len: int,
    ) -> Array["1, n_heads, query_len, key_len"]:

        if not self.use_rel_attn_bias:
            # return zero tensor
            return jnp.zeros((1, self.n_heads, query_len, key_len))

        # int64 in original
        context_pos = jnp.arange(query_len, dtype=jnp.int32)[:, None]
        memory_pos = jnp.arange(key_len, dtype=jnp.int32)[None, :]

        relative_pos: Array["q_len, k_len"] = memory_pos - context_pos

        relative_pos_bucket = self._calc_bucket(
            relative_pos,
            n_buckets=self.rel_attn_n_buckets,
            max_distance=self.rel_attn_max_distance,
            bidirectional=self.bidirectional
        )

        values = self.pos_emb(pos_emb_state, relative_pos_bucket)
        values = values.transpose((2, 0, 1))[None, :, :, :]

        return values

    def _calc_bucket(
        self, 
        relative_pos: Array["q_len, k_len"], 
        n_buckets: int=32, 
        max_distance: int=128,
        bidirectional: bool=True
    ) -> Array["q_len, k_len"]:

        # for each index in relative_pos, return which bucket it corresponds to.
        # in range [0, n_buckets)

        rel_buckets = 0
        # assume bidirectional
        if bidirectional:
            n_buckets //= 2
            rel_buckets += (relative_pos > 0).astype(jnp.int32) * n_buckets
            relative_pos = jnp.abs(relative_pos)
        else:
            relative_pos = -jnp.minimum(relative_pos, jnp.zeros_like(relative_pos))

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
