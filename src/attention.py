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
    """
    Dense layer transforming input to query, key or value
    """
    def __init__(
        self, n_heads: int, emb_size: int, d_k: int = None, bias: bool = False
    ):
        self.n_heads = n_heads
        self.d_k = emb_size if d_k is None else d_k

        self.dense = Dense(emb_size, n_heads * self.d_k, bias=bias)

    def forward(self, state: DenseState, x: jax.Array) -> jax.Array:
        # x.shape: (context_len, batch_size, emb_size)
        # returns: q, k, or v of shape (context_len, batch_size, n_heads, d_k)

        # hej jag heter

        x = self.dense(state, x)

        head_shape = x.shape[:-1]
        # split the embedding size by the number of heads
        x = x.reshape((*head_shape, self.n_heads, self.d_k))
        return x

    def init_state(self, rng: jax.Array) -> DenseState:
        return self.dense.init_state(rng)

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


class MultiHeadAttentionState(NamedTuple):
    query_state: DenseState
    key_state: DenseState
    value_state: DenseState
    output_state: DenseState
    
    
class MultiHeadAttention:
    """
    Attention with n_heads heads
    """
    # Softmax along time / context length dimension
    
    def __init__(self, n_heads: int, emb_size: int, bias: bool = False):
        self.n_heads = n_heads
        # Features per head (head dim)
        self.d_k = emb_size // n_heads

        self.query_fn = PreAttention(n_heads, emb_size, d_k=self.d_k, bias=bias)
        self.key_fn = PreAttention(n_heads, emb_size, d_k=self.d_k, bias=bias)
        self.value_fn = PreAttention(n_heads, emb_size, d_k=self.d_k, bias=True) # Why bias here?
        
        self.out = Dense(emb_size, emb_size, bias=False)

        self.saved_steps = {}
        
    def init_state(self, rng: jax.Array) -> MultiHeadAttentionState:
        rngs = random.split(rng, 4)
        return MultiHeadAttentionState(
            self.query_fn.init_state(rngs[0]),
            self.key_fn.init_state(rngs[1]),
            self.value_fn.init_state(rngs[2]),
            self.out.init_state(rngs[3])
        )

    def forward(self, state: MultiHeadAttentionState, q: jax.Array, k: jax.Array, v: jax.Array) -> jax.Array:
        # TODO: Add masking
        # q, k, v shape: (context_len, batch_size, emb_size)
        print(f"q.shape = {q.shape}, k.shape = {k.shape}, v.shape = {v.shape}")
        self.saved_steps["input_query"] = q

        context_len, batch_size, emb_size = q.shape
        
        query = self.query_fn(state.query_state, q)
        key = self.key_fn(state.key_state, k)
        value = self.value_fn(state.value_state, v)

        self.saved_steps["transformed_query"] = query

        # shape: (context_len, batch_size, n_heads, d_k)

        print("# Shapes after linear transform and split into heads")
        print(f"query.shape = {query.shape}, key.shape = {key.shape}, value.shape = {value.shape}")
        
        # calc q * k^T = s with shape (contex_len, context_len, batch_size, n_heads)

        # scores = jnp.matmul(query, key.transpose((0, 1, 3, 2)))
        # scores = jnp.einsum("...id,...jd->...ij", query, key)
        scores = jnp.einsum("cbhd,Cbhd->cCbh", query, key)
        self.saved_steps['scores'] = scores
        assert scores.shape == (context_len, context_len, batch_size, self.n_heads), f"Expected shape {(context_len, context_len, batch_size, self.n_heads)}, got {scores.shape}"

        scores *= (1 / jnp.sqrt(self.d_k))
        self.saved_steps['scaled_scores'] = scores

        print(f"q * k^T = s.shape = {scores.shape}")

        s2 = softmax(scores, dim=1)
        self.saved_steps['softmax'] = s2

        print(f"Softmax attn.shape = {s2.shape}")
        attn = jnp.einsum("cCbh,Cbhd->cbhd", s2, value)
        print(f"*v shape = {attn.shape}")
        self.saved_steps['scaled_values'] = attn

        # attn.shape: (context_len, batch_size, n_heads, d_k)
        
        # concat heads
        attn = attn.reshape((context_len, batch_size, emb_size))        
        self.saved_steps['concat_heads'] = attn
        print(f"After reshape: x.shape = {attn.shape}")
        out = jnp.einsum("cbd,Dd->cbD", attn, state.output_state.weights)
        # add bias
        out += state.output_state.bias
        self.saved_steps['out'] = out
        print(f"out.shape = {out.shape}")
        return out

