import jax.numpy as jnp
import jax
import numpy as np
from jax import random, vmap
from typing import List, Callable, Tuple, NamedTuple
from utils import LOG_LEVEL, get_logger

logger = get_logger()

rng = random.PRNGKey(0)


# Naive
def softmax(x: jax.Array, dim: int) -> jax.Array:
    xe = jnp.exp(x)
    return xe / jnp.sum(xe, axis=dim, keepdims=True)


# Off by one
def softmax_one(x: jax.Array, dim: int) -> jax.Array:
    xe = jnp.exp(x)
    return xe / (jnp.sum(xe, axis=dim, keepdims=True) + 1)


# Stable
def softmax_stable(x: jax.Array, dim: int) -> jax.Array:
    maxes = jnp.max(x, axis=dim, keepdims=True)
    xm = jnp.exp(x - maxes)
    return xm / jnp.sum(xm, axis=dim, keepdims=True)


def sigmoid(x: jax.Array) -> jax.Array:
    return 1 / (1 + jnp.exp(-x))


class DenseState(NamedTuple):
    weights: jax.Array
    bias: jax.Array


class Dense:
    def __init__(self, n_in: int = None, n_out: int = None, bias: bool = True):
        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias

    def init_state(self, rng: jax.Array, scale=1e-2) -> DenseState:
        w_key, b_key = random.split(rng)
        w = scale * random.normal(w_key, (self.n_out, self.n_in))
        b = scale * random.normal(b_key, (self.n_out,)) if self.bias else None
        return DenseState(w, b)

    def __call__(self, state: DenseState, x: jax.Array) -> jax.Array:
        # Assume x.shape: (batch_size, n_in) or (n_in)

        # Batch if x has a batch dim
        if x.ndim > 2:
            raise ValueError(f"Input dim must be 1 or 2, got {x.ndim}")
        elif x.ndim == 2:
            return jax.vmap(self.forward, in_axes=(None, 0))(state, x)
        else:
            return self.forward(state, x)

    def forward(self, state: DenseState, x: jax.Array) -> jax.Array:
        # Linear forward pass of non-batched input
        # dot = jnp.dot(x, state.weights)
        dot = jnp.matmul(x, state.weights.T)
        return dot + state.bias if self.bias else dot


class PreAttention:
    def __init__(
        self, emb_size: int, n_heads: int, d_k: int = None, bias: bool = False
    ):
        self.n_heads = n_heads
        self.d_k = emb_size if d_k is None else d_k

        self.dense = Dense(emb_size, n_heads * self.d_k, bias=bias)

    # Assume weight matrix comes form SelfAttentionState
    def forward(self, state: DenseState, x: jax.Array) -> jax.Array:
        # x.shape: (context_len, batch_size, emd_size)
        head_shape = x.shape[:-1]

        # jnp dot along emb_size dim
        x = self.dense(state, x)

        x = x.reshape((*head_shape, self.n_heads, self.d_k))
        return x

    def __call__(self, weight_matrix: jax.Array, x: jax.Array) -> jax.Array:
        # Batch batch_dim of x.shape: (seq_len, batch_size, emd_size)
        if x.ndim > 3:
            raise ValueError(f"Input dim must be 2 or 3, got {x.ndim}")
        elif x.ndim == 3:
            # map forward over batch_dim
            x = jax.vmap(self.forward, in_axes=(None, 1), out_axes=1)(weight_matrix, x)
            return x
        else:
            return self.forward(weight_matrix, x)


# TODO: Use PreAttention
class SelfAttentionState(NamedTuple):
    W_q: jax.Array
    W_k: jax.Array
    W_v: jax.Array


class SelfAttention:
    """
    Non-matrix version of self-attention, i.e. assumes the input to be a sequence of vectors of size (emb_size).

    input: (context_len, emb_size) or (context_len, batch_size, emb_size)
    """

    def __init__(self, emb_size, d_k=None):
        # Size of each word embedding
        self.emb_size = emb_size

        # Size of each key/query/value vector
        self.d_k = emb_size if d_k is None else d_k

    def init_state(self, rng: jax.Array) -> SelfAttentionState:
        keys = random.split(rng, 3)

        W_q = random.normal(keys[0], (self.emb_size, self.d_k))
        W_k = random.normal(keys[1], (self.emb_size, self.d_k))
        W_v = random.normal(keys[2], (self.emb_size, self.d_k))

        return SelfAttentionState(W_q, W_k, W_v)

    def __call__(self, state: SelfAttentionState, x: jax.Array) -> jax.Array:
        # TODO: Make batch dim work

        q = jnp.dot(x, state.W_q)
        k = jnp.dot(x, state.W_k)
        v = jnp.dot(x, state.W_v)
        return self._attention(q, k, v)

    def _attention(self, q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        # Sum along sequence axis (context_len)
        return softmax(jnp.dot(q, k.T) / jnp.sqrt(self.d_k), dim=1) @ v


def relu(x: jax.Array) -> jax.Array:
    return jnp.maximum(x, 0)


class BatchNormState(NamedTuple):
    # TODO: make into nested dict?
    mean: jax.Array = 0
    var: jax.Array = 1
    gamma: jax.Array = 1
    beta: jax.Array = 0
    momentum: jax.Array = 0.1


def batchnorm_1d(
    x: jax.Array, state: BatchNormState, training: bool = True, eps=1e-5
) -> Tuple[jax.Array, BatchNormState]:
    """
    :param jax.Array x:           (B, N) or (B, N, L), B batch size, N input dim, L input length
    :param BatchNormState state:    NamedTuple with mean, var, gamma, beta
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
