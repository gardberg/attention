import jax.numpy as jnp
import jax
from jax import random, vmap
from typing import Callable, Tuple, NamedTuple, Union
from log_utils import logger
from act import *
from states import *
from typing import NamedTuple, TypeVar, Type
import torch


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


# (context_len, batch_size, emb_dim)
class LayerNorm:
    def __init__(self, norm_dims: Union[Tuple[int, ...], int], eps=1e-5):
        assert isinstance(
            norm_dims, (tuple, int)
        ), f"norm_dims must be tuple or int, got {type(norm_dims)}"
        self.norm_dims = (norm_dims) if isinstance(norm_dims, int) else norm_dims
        # tuple of dims to normalize over
        # Example: x.shape = (2, 3, 1), norm_dims = (3, 1)
        # => Normalize over last 2 dims

        # x.shape = (2, 3), norm_dims = (3,)
        # => Normalize over last dim

        # For example, for x: (context_len, batch_size, embed_dim)
        # you'd normally normalize over embed_dim:
        # LayerNorm((embed_dim))

        self.eps = eps

    def init_state(self, rng: jax.Array = None):
        return LayerNormState(
            gamma=jnp.ones(self.norm_dims),
            beta=jnp.zeros(self.norm_dims),
        )

    # Do we need to pass the state as output?
    def forward(self, state: LayerNormState, x: jax.Array) -> jax.Array:
        assert (
            x.shape[-len(self.norm_dims) :] == self.norm_dims
        ), f"Input shape {x.shape} must have last dimension matching norm_dims: {self.norm_dims}"
        # x.shape: (*, *norm_dims), e.g. (context_len, batch_size, emb_size)

        # compute mean over last n = len(self.norm_dims) dimensions
        axes_to_reduce = tuple(range(-len(self.norm_dims), 0))  # (-n, ..., -1)
        means = jnp.mean(x, axis=axes_to_reduce, keepdims=True)

        vars = jnp.var(x, axis=axes_to_reduce, keepdims=True)
        x_norm = (x - means) / jnp.sqrt(vars + self.eps)
        return state.gamma * x_norm + state.beta

    def __call__(self, state: LayerNormState, x: jax.Array) -> jax.Array:
        # TODO: Vectorize with vmap
        return self.forward(state, x)


class RMSNorm:
    
    def __init__(self, norm_dims: int, eps=1e-5):
        self.norm_dim = norm_dims # size of last dim to normalize over
        self.eps = eps
        
    def init_state(self, rng: jax.Array = None):
        return RMSNormState(
            gamma=jnp.ones(self.norm_dim),
        )

    def forward(self, state: RMSNormState, x: jax.Array) -> jax.Array:
        rms = jnp.sqrt(jnp.mean(x**2, axis=-1, keepdims=True))
        return x / (rms + self.eps) * state.gamma

    def __call__(self, state: RMSNormState, x: jax.Array) -> jax.Array:
        return self.forward(state, x)
        

class Linear:
    def __init__(self, n_in: int, n_out: int, bias: bool = True, batch_dim: int = 0):
        self.n_in = n_in
        self.n_out = n_out
        self.bias = bias
        self.batch_dim = batch_dim

    def init_state(self, rng: jax.Array) -> LinearState:
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


class FeedForward:
    def __init__(
        self, n_in: int, d_ff: int, act: Callable = relu, dropout: float = 0.0
    ):
        self.layer1 = Linear(n_in, d_ff)
        self.layer2 = Linear(d_ff, n_in)
        self.act = act
        self.dropout = dropout

    def init_state(self, rng: jax.Array) -> FeedForwardState:
        rng1, rng2 = random.split(rng)
        return FeedForwardState(
            self.layer1.init_state(rng1),
            self.layer2.init_state(rng2),
        )

    def __call__(
        self, state: FeedForwardState, x: jax.Array, rng: jax.Array
    ) -> jax.Array:
        x = self.act(self.layer1(state.linear1_state, x))
        x = dropout(x, self.dropout, rng)
        return self.layer2(state.linear2_state, x)


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
        self, emb_size: int, n_heads: int, out_bias: bool = False, v_bias: bool = True
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

        self.out = Linear(emb_size, emb_size, bias=out_bias)

        self.debug_states = dict()

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
        # mask[i, j, ...] = true -> i can not attend to j
        base_mask = jnp.tril(jnp.ones((context_len, context_len), dtype=bool), k=0)

        # tile into shape (context_len, context_len, batch_size, n_heads)
        mask = jnp.tile(base_mask[:, :, None, None], (1, 1, batch_size, self.n_heads))
        return mask

    def forward(
        self,
        state: MultiHeadAttentionState,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        mask: jax.Array = None,
    ) -> jax.Array:
        # q, k, v shape: (context_len, batch_size, emb_size)

        self.debug_states["input_query"] = q

        context_len, batch_size, emb_size = q.shape

        query = self.query_fn(state.query_state, q)
        key = self.key_fn(state.key_state, k)
        value = self.value_fn(state.value_state, v)

        self.debug_states["query"] = query
        self.debug_states["key"] = key
        self.debug_states["value"] = value

        # shape: (context_len, batch_size, n_heads, d_k)
        # calc q * k^T = s with shape (contex_len, context_len, batch_size, n_heads)

        scores = jnp.einsum("cbhd,Cbhd->cCbh", query, key)

        self.debug_states["scores"] = scores

        assert scores.shape == (
            context_len,
            context_len,
            batch_size,
            self.n_heads,
        ), f"Expected shape {(context_len, context_len, batch_size, self.n_heads)}, got {scores.shape}"

        scaled_scores = scores * (1 / jnp.sqrt(self.d_k))

        self.debug_states["scaled_scores"] = scaled_scores
        self.debug_states["mask"] = mask

        # TODO: At the moment False => mask. Pytorch default: True => mask. Switch?
        if mask is not None:
            assert (
                mask.shape == scores.shape
            ), f"Mask shape {mask.shape} must match scores shape {scores.shape}. To create a mask, use MultiHeadAttention.get_causal_mask()"
            # replace values in scores with float('-inf') where mask is false
            scaled_scores = jnp.where(mask, scaled_scores, float("-inf"))

        self.debug_states["masked_scaled_scores"] = scaled_scores

        s2 = softmax(scaled_scores, dim=1)

        self.debug_states["softmax"] = s2

        attn = jnp.einsum("cCbh,Cbhd->cbhd", s2, value)
        # attn.shape: (context_len, batch_size, n_heads, d_k)

        self.debug_states["scaled_values"] = attn

        # concat heads
        attn = attn.reshape((context_len, batch_size, emb_size))

        self.debug_states["concat_heads"] = attn

        # out = jnp.einsum("cbd,Dd->cbD", attn, state.output_state.weights)
        out = self.out(state.output_state, attn)

        self.debug_states["out"] = out

        # out.shape: (context_len, batch_size, emb_dim)
        return out

    def __call__(
        self,
        state: MultiHeadAttentionState,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        mask: jax.Array = None,
    ) -> jax.Array:
        return self.forward(state, q, k, v, mask)


class PositionalEncoding:
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

    def __call__(
        self, x: jax.Array, rng: jax.Array, training: bool = True
    ) -> Tuple[jax.Array, jax.Array]:
        # x.shape: (context_len, batch_size, embed_dim)
        assert len(x.shape) == 3
        x = x + self.embeds[: x.shape[0]]
        return dropout(x, self.dropout, rng, training)


# TODO: Move into separate file
# Requires torch import, which is a bit heavy
NamedTupleSubclass = TypeVar("NamedTupleSubclass", bound=NamedTuple)


def to_jax_state(torch_module: torch.nn.Module) -> Type[NamedTupleSubclass]:
    if isinstance(torch_module, torch.nn.MultiheadAttention):
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

    elif isinstance(torch_module, torch.nn.Linear):
        weight, bias = (
            jnp.array(torch_module.weight.detach()),
            jnp.array(torch_module.bias.detach())
            if torch_module.bias is not None
            else None,
        )
        return LinearState(weight, bias)

    elif isinstance(torch_module, torch.nn.LayerNorm):
        return LayerNormState(
            jnp.array(torch_module.weight.detach()),
            jnp.array(torch_module.bias.detach()),
        )

    else:
        raise NotImplementedError(
            f"to_jax_state not implemented for {type(torch_module)}"
        )
