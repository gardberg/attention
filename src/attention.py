import jax.numpy as jnp
import jax
from jax import random, vmap
from typing import Callable, Union
from log_utils import logger
from act import *
from states import *
from functools import lru_cache

from base import BaseModule, Array

class BatchNorm1d(BaseModule):
    def __init__(self, dims: int):
        # dims: N, input dims aka number of input features
        self.dims = dims

    def forward(
        self, state: BatchNormState, x: Array, training: bool = True, eps: float = 1e-5
    ) -> tuple[Array, BatchNormState]:
        """
        :param Array x: (B, N) or (B, N, L), B batch size, N input dim, L input length
        :param BatchNormState state: NamedTuple with mean, var, gamma, beta
        """
        # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d

        if training:
            # Only update running mean and var during training
            mean = jnp.mean(x, axis=0)
            var = jnp.var(x, axis=0)  # ddof = 0, biased

            # update state.mean and state.var via jax.tree_map instead
            new_mean = (1 - state.momentum) * state.mean + state.momentum * mean
            new_var = (1 - state.momentum) * state.var + state.momentum * jnp.var(
                x, axis=0, ddof=1
            )

            x_norm = (x - mean) / jnp.sqrt(var + eps)
        else:
            x_norm = (x - state.mean) / jnp.sqrt(state.var + eps)

        # TODO: Update state with jax.tree_map instead?
        new_state = BatchNormState(
            mean=new_mean if training else state.mean,
            var=new_var if training else state.var,
            gamma=state.gamma,
            beta=state.beta,
            momentum=state.momentum,
        )

        return state.gamma * x_norm + state.beta, new_state

    def init_state(self, rng: Array = None) -> BatchNormState:
        return BatchNormState(
            mean=jnp.array(0),
            var=jnp.array(1),
            gamma=jnp.ones(self.dims),  # weight
            beta=jnp.zeros(self.dims),  # bias
            momentum=jnp.array(0.1),
        )


# (context_len, batch_size, emb_dim)
class LayerNorm(BaseModule):
    def __init__(self, norm_dims: Union[tuple[int, ...], int], eps=1e-5):
        assert isinstance(
            norm_dims, (tuple, int)
        ), f"norm_dims must be tuple or int, got {type(norm_dims)}"
        self.norm_dims = (norm_dims,) if isinstance(norm_dims, int) else norm_dims
        # tuple of dims to normalize over
        # Example: x.shape = (2, 3, 1), norm_dims = (3, 1)
        # => Normalize over last 2 dims

        # x.shape = (2, 3), norm_dims = (3,)
        # => Normalize over last dim

        # For example, for x: (context_len, batch_size, embed_dim)
        # you'd normally normalize over embed_dim:
        # LayerNorm((embed_dim))

        self.eps = eps

    def init_state(self, rng: Array = None):
        return LayerNormState(
            gamma=jnp.ones(self.norm_dims),
            beta=jnp.zeros(self.norm_dims),
        )

    # Do we need to pass the state as output?
    def forward(self, state: LayerNormState, x: Array) -> Array:
        assert (
            x.shape[-len(self.norm_dims) :] == self.norm_dims
        ), f"Input shape {x.shape} must have last dimension matching norm_dims: {self.norm_dims}"
        # x.shape: (*, *norm_dims), e.g. (context_len, batch_size, emb_size)

        logger.debug(f"x.dtype: {x.dtype}, gamma.dtype: {state.gamma.dtype}")

        # compute mean over last n = len(self.norm_dims) dimensions
        axes_to_reduce = tuple(range(-len(self.norm_dims), 0))  # (-n, ..., -1)
        means = jnp.mean(x, axis=axes_to_reduce, keepdims=True)

        vars = jnp.var(x, axis=axes_to_reduce, keepdims=True)
        x_norm = (x - means) / jnp.sqrt(vars + self.eps)
        return state.gamma * x_norm + state.beta


class RMSNorm(BaseModule):
    def __init__(self, norm_dims: int, eps=1e-5):
        self.norm_dim = norm_dims  # size of last dim to normalize over
        self.eps = eps

    def init_state(self, rng: Array = None):
        return RMSNormState(
            gamma=jnp.ones(self.norm_dim),
        )

    def forward(self, state: RMSNormState, x: Array) -> Array:
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True))
        return x / (rms + self.eps) * state.gamma


class Linear(BaseModule):
    def __init__(self, n_in: int, n_out: int, bias: bool = True, batch_dim: int = 0):
        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias
        self.batch_dim = batch_dim

    def init_state(self, rng: Array) -> LinearState:
        # Do we need to return a key here to ensure that the original one is not used again?
        w_key, b_key = random.split(rng)
        init_range = 1 / jnp.sqrt(self.n_in)
        w = random.uniform(
            w_key, (self.n_out, self.n_in), minval=-init_range, maxval=init_range
        )
        b = (
            random.uniform(b_key, (self.n_out,), minval=-init_range, maxval=init_range)
            if self.bias
            else None
        )
        return LinearState(w, b)

    # TODO: Rewrite
    def __call__(self, state: LinearState, x: Array) -> Array:
        """
        Batched forward pass along batch_dim
        """
        if x.ndim > 1:
            return jax.vmap(self.forward, in_axes=(None, self.batch_dim))(state, x)
        else:
            return self.forward(state, x)

    def forward(self, state: LinearState, x: Array) -> Array:
        """
        Non-batched forward pass

        x: input with shape (*, n_in) where * is any number of dimension, including None
        returns: output with shape (*, n_out)
        """
        dot = jnp.matmul(x, state.weights.T)
        return dot + state.bias if self.bias else dot


class FeedForward(BaseModule):
    def __init__(
        self, n_in: int, d_ff: int, act: Callable = relu, dropout: float = 0.0
    ):
        self.layer1 = Linear(n_in, d_ff)
        self.layer2 = Linear(d_ff, n_in)
        self.act = act
        self.dropout = dropout

    def init_state(self, rng: Array) -> FeedForwardState:
        rng1, rng2 = random.split(rng)
        return FeedForwardState(
            self.layer1.init_state(rng1),
            self.layer2.init_state(rng2),
        )

    def forward(
        self, state: FeedForwardState, x: Array, rng: Array, training=True
    ) -> Array:
        x = self.act(self.layer1(state.linear1, x))
        x = dropout(x, self.dropout, rng, training)
        return self.layer2(state.linear2, x)


class Embedding(BaseModule):
    """
    Learned lookup embeddings of size (n_embeddings, emb_size).
    Assumes that the input is a sequence of indices corresponding to items in vocabulary.
    """

    def __init__(self, n_embeddings: int, emb_size: int):
        self.n_embeddings = n_embeddings
        self.emb_size = emb_size

    def forward(self, state: EmbeddingState, indices: Array) -> Array:
        # indices.shape:            (*,), e.g. (context_len, batch_size)
        # output.shape:             (*, emb_size)
        # state.embeddings.shape:   (vocab_size, emb_size)
        assert jnp.max(indices) < self.n_embeddings, f"Index out of bounds"
        return jnp.take(state.embeddings, indices, axis=0)

    def init_state(self, rng: Array) -> EmbeddingState:
        return EmbeddingState(
            embeddings=random.normal(rng, (self.n_embeddings, self.emb_size))
        )


class PreAttention(BaseModule):
    """
    Linear layer transforming input to query, key or value
    """

    def __init__(
        self, emb_size: int, n_heads: int, d_k: int = None, bias: bool = False
    ):
        self.n_heads = n_heads
        self.d_k = emb_size if d_k is None else d_k

        self.dense = Linear(emb_size, n_heads * self.d_k, bias=bias)

    def forward(self, state: LinearState, x: Array) -> Array:
        # x.shape: (context_len, batch_size, emb_size)
        # returns: q, k, or v of shape (context_len, batch_size, n_heads, d_k)

        x = self.dense(state, x)

        head_shape = x.shape[:-1]
        # split the embedding size by the number of heads
        x = x.reshape((*head_shape, self.n_heads, self.d_k))
        return x

    def init_state(self, rng: Array) -> LinearState:
        return self.dense.init_state(rng)

    # TODO: Rewrite
    def __call__(self, weight_matrix: Array, x: Array) -> Array:
        # Batch batch_dim of x.shape: (context_len, batch_size, emd_size)
        if x.ndim > 3:
            raise ValueError(f"Input dim must be 2 or 3, got {x.ndim}")
        elif x.ndim == 3:
            # map forward over batch_dim
            x = jax.vmap(self.forward, in_axes=(None, 1), out_axes=1)(weight_matrix, x)
            return x
        else:
            return self.forward(weight_matrix, x)


def create_causal_mask(tgt_len: int, src_len: int = None) -> Array:
    """
    Creates a causal mask tensor of shape (tgt_len, src_len) to be used in MultiHeadAttention forward pass.
    """

    if src_len is None:
        src_len = tgt_len

    # return jnp.tril(jnp.ones((tgt_len, src_len), dtype=bool), k=0)
    # True -> Not allowed to attend
    return jnp.triu(jnp.ones((tgt_len, src_len), dtype=bool), k=1)


class MultiHeadAttention(BaseModule):
    """
    Attention with n_heads heads
    """

    def __init__(
        self,
        emb_size: int,
        n_heads: int,
        out_bias: bool = False,
        v_bias: bool = True,
        causal: bool = False,
    ):
        """
        emb_size:   Total size of query, key and value. Will be split over the number of heads
        n_heads:    Number of individual attention heads
        bias:       Whether to use bias in the output linear layer
        v_bias:     Whether to use bias in the value linear layer
        causal:     Whether to use a causal mask in the attention calculation.
                    If False, no mask is used. Manually specifying a mask overrides this setting.
        """

        self.emb_size = emb_size
        self.n_heads = n_heads
        # Features per head (head dim)
        assert emb_size % n_heads == 0, f"emb_size must be divisible by n_heads"
        self.d_k = emb_size // n_heads

        self.query_fn = PreAttention(emb_size, n_heads, d_k=self.d_k)
        self.key_fn = PreAttention(emb_size, n_heads, d_k=self.d_k)
        self.value_fn = PreAttention(emb_size, n_heads, d_k=self.d_k, bias=v_bias)

        self.out = Linear(emb_size, emb_size, bias=out_bias)
        self.causal = causal

        self.debug_states = dict()

    def init_state(self, rng: Array) -> MultiHeadAttentionState:
        rngs = random.split(rng, 4)
        return MultiHeadAttentionState(
            self.query_fn.init_state(rngs[0]),
            self.key_fn.init_state(rngs[1]),
            self.value_fn.init_state(rngs[2]),
            self.out.init_state(rngs[3]),
        )

    def _get_mask_batched(self, mask: Array, batch_size: int, n_heads: int) -> Array:
        assert mask.ndim == 2, f"Expected mask.ndim == 2, got {mask.ndim}"
        return jnp.tile(mask[:, :, None, None], (1, 1, batch_size, n_heads))

    def get_causal_mask(
        self, tgt_len: int, batch_size: int, src_len: int = None
    ) -> Array:
        # creates a causal mask of shape (tgt_len, src_len, batch_size, n_heads)
        # mask[i, j, ...] = true -> i can not attend to j
        base_mask = create_causal_mask(tgt_len, src_len=src_len)

        # tile into shape (tgt_len, src_len, batch_size, n_heads)
        return self._get_mask_batched(base_mask, batch_size, self.n_heads)

    def get_kv(
        self,
        state: MultiHeadAttentionState,
        k: Array,
        v: Array,
        use_cache: bool, 
        kv_cache: tuple[Array, Array]
    ) -> tuple[Array, Array]:

        if use_cache and kv_cache is not None:
            assert len(kv_cache) == 2, f"Expected kv_cache to be a tuple of length 2"
            cached_keys, cached_values = kv_cache

            next_k = k[-1][None, ...]
            next_v = v[-1][None, ...]

            key = jnp.concatenate(
                [cached_keys, self.key_fn(state.key, next_k)], axis=0
            )
            value = jnp.concatenate(
                [cached_values, self.value_fn(state.value, next_v)], axis=0
            )
        else:
            key = self.key_fn(state.key, k)
            value = self.value_fn(state.value, v)

        return key, value

    def forward(
        self,
        state: MultiHeadAttentionState,
        q: Array,
        k: Array,
        v: Array,
        mask: Array = None,
        use_cache: bool = False,
        kv_cache: tuple[Array, Array] = None,
    ) -> Array:
        """
        q.shape:        (tgt_len, batch_size, emb_size)
        k, v shape:     (src_len, batch_size, emb_size)
        mask.shape:     (tgt_len, src_len) or None
        kv_cache.shape: (cached_keys, cached_values) with shape (src_len - 1, batch_size, n_heads, d_k)
        """

        assert (
            k.shape == v.shape
        ), f"Expected k.shape == v.shape, got {k.shape} != {v.shape}"

        self.debug_states["input_query"] = q

        tgt_len, batch_size, emb_size = q.shape
        src_len = k.shape[0]

        query = self.query_fn(state.query, q)

        key, value = self.get_kv(state, k, v, use_cache, kv_cache)

        if use_cache: kv_cache = (key, value)

        self.debug_states["query"] = query
        self.debug_states["key"] = key
        self.debug_states["value"] = value

        # q.shape: (tgt_len, batch_size, n_heads, d_k)
        # k/v.shape: (src_len, batch_size, n_heads, d_k)
        # calc q * k^T = s with shape (tgt_len, src_len, batch_size, n_heads)

        scores = jnp.einsum("cbhd,Cbhd->cCbh", query, key)

        self.debug_states["scores"] = scores

        assert scores.shape == (
            tgt_len,
            src_len,
            batch_size,
            self.n_heads,
        ), f"Expected shape {(tgt_len, src_len, batch_size, self.n_heads)}, got {scores.shape}"

        scaled_scores = scores * (1 / jnp.sqrt(self.d_k))

        self.debug_states["scaled_scores"] = scaled_scores
        self.debug_states["mask"] = mask

        if mask is None and self.causal:
            mask = self.get_causal_mask(tgt_len, batch_size, src_len)

        if mask is not None:
            assert (
                mask.shape[:2] == scores.shape[:2]
            ), f"Mask shape {mask.shape[:2]} must match scores shape {scores.shape[:2]}. To create a causal mask, use create_causal_mask()"

            if mask.ndim == 2:
                mask = self._get_mask_batched(mask, batch_size, self.n_heads)

            # replace values in scores with float('-inf') where mask is True
            scaled_scores = jnp.where(mask, float("-inf"), scaled_scores)

        self.debug_states["masked_scaled_scores"] = scaled_scores

        s2 = softmax(scaled_scores, dim=1)

        self.debug_states["softmax"] = s2

        attn = jnp.einsum("cCbh,Cbhd->cbhd", s2, value)
        # attn.shape: (tgt_len, batch_size, n_heads, d_k)

        self.debug_states["scaled_values"] = attn

        # concat heads
        attn = attn.reshape((tgt_len, batch_size, emb_size))

        self.debug_states["concat_heads"] = attn

        # out = jnp.einsum("cbd,Dd->cbD", attn, state.output_state.weights)
        out = self.out(state.output, attn)

        self.debug_states["out"] = out

        # out.shape: (tgt_len, batch_size, emb_dim)
        # kv_cache.shape: ((src_len, batch_size, n_heads, d_k), *)
        return (out, kv_cache) if use_cache else out


class PositionalEncoding(BaseModule):
    def __init__(self, emb_size: int, dropout: float = 0.0, max_len: int = 5000):
        self.emb_size = emb_size
        self.dropout = dropout
        self.max_len = max_len

        self.embeds = self._create_embeds()

    def _create_embeds(self):
        # create embeds which is broadcastable with input x
        position = jnp.arange(self.max_len)
        position = jnp.expand_dims(position, axis=1)

        # Assumes emb_size to be big enough
        div_term = jnp.exp(
            jnp.arange(0, self.emb_size, 2) * (-jnp.log(10000.0) / self.emb_size)
        )
        pe = jnp.zeros((self.max_len, 1, self.emb_size))
        pe = pe.at[:, 0, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 0, 1::2].set(jnp.cos(position * div_term))
        return pe

    def forward(self, x: Array, rng: Array, training: bool = True) -> Array:
        # x.shape: (context_len, batch_size, embed_dim)
        assert (
            len(x.shape) == 3
        ), f"Expected input shape (context_len, batch_size, embed_dim), got {x.shape}"
        x = x + self.embeds[: x.shape[0]]
        return dropout(x, self.dropout, rng, training)


# https://arxiv.org/abs/2104.09864
# Adapted from https://github.com/google-deepmind/gemma
def apply_rope(x: Array) -> Array:
    """
    Applies
    x.shape: (seq_len, batch_size, n_heads, d)
    output.shape: (seq_len, batch_size, n_heads, d)
    """

    assert x.ndim == 4, f"Input dim must be 4, got {x.ndim}"

    seq_len = x.shape[0]
    d = x.shape[-1]
    batch_size = x.shape[1]

    # assume vectors come in sequential order
    positions = jnp.stack([jnp.arange(seq_len)] * batch_size, axis=-1)

    exponent = 2 * jnp.arange(0, d // 2) / d
    timescale = 10_000**exponent

    sinusoid_input = (
        positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
    )
    sinusoid_input = sinusoid_input[..., jnp.newaxis, :]

    sin = jnp.sin(sinusoid_input)
    cos = jnp.cos(sinusoid_input)

    first_half, second_half = jnp.split(x, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)

    return out.astype(x.dtype)
