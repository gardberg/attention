import jax.numpy as jnp
import jax
import numpy as np
from jax import random, vmap
from typing import Tuple, NamedTuple
from utils import LOG_LEVEL, get_logger
from act import softmax
from states import BatchNormState, LinearState, MultiHeadAttentionState
from typing import NamedTuple, TypeVar, Type
import torch

logger = get_logger()

rng = random.PRNGKey(0)


def batchnorm_1d(
    x: jax.Array, state: BatchNormState, training: bool = True, eps=1e-5
) -> Tuple[jax.Array, BatchNormState]:
    """
    :param jax.Array x: (B, N) or (B, N, L), B batch size, N input dim, L input length
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


class Linear:
    def __init__(self, n_in: int, n_out: int, bias: bool = True, batch_dim: int=0):
        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias
        self.batch_dim = batch_dim

    def init_state(self, rng: jax.Array) -> LinearState:
        w_key, b_key = random.split(rng)
        init_range = 1 / jnp.sqrt(self.n_in)
        w = random.uniform(w_key, (self.n_out, self.n_in), minval=-init_range, maxval=init_range)
        b = random.uniform(b_key, (self.n_out,), minval=-init_range, maxval=init_range) if self.bias else None
        return LinearState(w, b)

    def __call__(self, state: LinearState, x: jax.Array) -> jax.Array:
        """
        Batched forward pass along batch_dim
        """
        if x.ndim > 1:
            return jax.vmap(self._forward, in_axes=(None, self.batch_dim))(state, x)
        else:
            return self._forward(state, x)

    def _forward(self, state: LinearState, x: jax.Array) -> jax.Array: 
        """
        Non-batched forward pass

        x: input with shape (*, n_in) where * is any number of dimension, including None
        returns: output with shape (*, n_out)
        """
        dot = jnp.matmul(x, state.weights.T)
        return dot + state.bias if self.bias else dot


class PreAttention:
    """
    Linear layer transforming input to query, key or value
    """

    def __init__(
        self, emb_size: int, n_heads: int, d_k: int = None, bias: bool = False
    ):
        self.n_heads = n_heads
        self.d_k = emb_size if d_k is None else d_k

        self.dense = Linear(emb_size, n_heads * self.d_k, bias=bias)

    def forward(self, state: LinearState, x: jax.Array) -> jax.Array:
        # x.shape: (context_len, batch_size, emb_size)
        # returns: q, k, or v of shape (context_len, batch_size, n_heads, d_k)

        x = self.dense(state, x)

        head_shape = x.shape[:-1]
        # split the embedding size by the number of heads
        x = x.reshape((*head_shape, self.n_heads, self.d_k))
        return x

    def init_state(self, rng: jax.Array) -> LinearState:
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



class MultiHeadAttention:
    """
    Attention with n_heads heads
    """

    def __init__(
        self, emb_size: int, n_heads: int, bias: bool = False, v_bias: bool = True
    ):
        """
        emb_size:   Total size of query, key and value. Will be split over the number of heads
        n_heads:    Number of individual attention heads
        bias:       Whether to use bias in the output linear layer
        v_bias:     Whether to use bias in the value linear layer 
        """

        self.n_heads = n_heads
        # Features per head (head dim)
        assert emb_size % n_heads == 0, f"emb_size must be divisible by n_heads"
        self.d_k = emb_size // n_heads

        self.query_fn = PreAttention(emb_size, n_heads, d_k=self.d_k)
        self.key_fn = PreAttention(emb_size, n_heads, d_k=self.d_k)
        self.value_fn = PreAttention(emb_size, n_heads, d_k=self.d_k, bias=v_bias)

        self.out = Linear(emb_size, emb_size, bias=bias)

    def init_state(self, rng: jax.Array) -> MultiHeadAttentionState:
        rngs = random.split(rng, 4)
        return MultiHeadAttentionState(
            self.query_fn.init_state(rngs[0]),
            self.key_fn.init_state(rngs[1]),
            self.value_fn.init_state(rngs[2]),
            self.out.init_state(rngs[3]),
        )

    def get_causal_mask(self, context_len: int, batch_size: int) -> jax.Array:
        # creates a causal mask of shape (context_len, context_len, batch_size, n_heads)
        single_mask = jnp.tril(jnp.ones((context_len, context_len)), k=0)
        single_mask = single_mask.reshape((context_len, context_len, 1, 1))
        mask = jnp.tile(single_mask, (1, 1, batch_size, self.n_heads))
        return mask

    
    def forward(
        self, state: MultiHeadAttentionState, q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array = None 
    ) -> jax.Array:
        # q, k, v shape: (context_len, batch_size, emb_size)

        context_len, batch_size, emb_size = q.shape

        query = self.query_fn(state.query_state, q)
        key = self.key_fn(state.key_state, k)
        value = self.value_fn(state.value_state, v)

        # shape: (context_len, batch_size, n_heads, d_k)
        # calc q * k^T = s with shape (contex_len, context_len, batch_size, n_heads)
        scores = jnp.einsum("cbhd,Cbhd->cCbh", query, key)

        assert scores.shape == (
            context_len,
            context_len,
            batch_size,
            self.n_heads,
        ), f"Expected shape {(context_len, context_len, batch_size, self.n_heads)}, got {scores.shape}"

        scaled_scores = scores * (1 / jnp.sqrt(self.d_k))

        # TODO: Add tests
        if mask is not None:
            assert mask.shape == scores.shape, f"Mask shape {mask.shape} must match scores shape {scores.shape}. To create a mask, use MultiHeadAttention.get_causal_mask()"
            scaled_scores = jnp.where(mask, scores, float("-inf"))
        
        s2 = softmax(scaled_scores, dim=1)

        attn = jnp.einsum("cCbh,Cbhd->cbhd", s2, value)
        # attn.shape: (context_len, batch_size, n_heads, d_k)

        # concat heads
        attn = attn.reshape((context_len, batch_size, emb_size))

        # out = jnp.einsum("cbd,Dd->cbD", attn, state.output_state.weights)
        out = self.out(state.output_state, attn)

        # out.shape: (context_len, batch_size, emb_dim)
        return out

    def __call__(self, state: MultiHeadAttentionState, q: jax.Array, k: jax.Array, v: jax.Array, mask: jax.Array = None) -> jax.Array:
        return self.forward(state, q, k, v, mask) 
                 

# TODO: Move into separate file
# Requires torch import, which is a bit heavy
NamedTupleSubclass = TypeVar("NamedTupleSubclass", bound=NamedTuple)
def to_jax_state(torch_module: torch.nn.Module) -> Type[NamedTupleSubclass]:
    
    if isinstance(torch_module, torch.nn.MultiheadAttention):
        emb_size = torch_module.embed_dim

        torch_mha_weights = torch_module.in_proj_weight
        torch_weights = (
            torch_mha_weights[0:emb_size, :],
            torch_mha_weights[emb_size : 2 * emb_size, :],
            torch_mha_weights[2 * emb_size : 3 * emb_size, :],
            torch_module.out_proj.weight
        )
        torch_weights = tuple(LinearState(jnp.array(w.detach().numpy()), None) for w in torch_weights)
        return MultiHeadAttentionState(*torch_weights)
        
    else:
        raise NotImplementedError(f"to_jax_state not implemented for {type(torch_module)}")

