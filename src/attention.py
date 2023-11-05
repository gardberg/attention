import jax.numpy as jnp
import jax
import numpy as np
from jax import random, vmap
from typing import List, Callable, Tuple, NamedTuple
from utils import LOG_LEVEL, get_logger

logger = get_logger()

rng = random.PRNGKey(0)


# Naive
def softmax(x: jax.Array, dim: int = 1) -> jax.Array:
    xe = jnp.exp(x)
    return xe / jnp.sum(xe, axis=dim - 2, keepdims=True)


# Off by one
def softmax_one(x: jax.Array, dim: int = 1) -> jax.Array:
    xe = jnp.exp(x)
    return xe / (jnp.sum(xe, axis=dim - 2, keepdims=True) + 1)


# Stable
def softmax_stable(x: jax.Array, dim: int = 1) -> jax.Array:
    maxes = jnp.max(x, axis=dim - 2, keepdims=True)
    xm = jnp.exp(x - maxes)
    return xm / jnp.sum(xm, axis=dim - 2, keepdims=True)

    
def sigmoid(x: jax.Array) -> jax.Array:
    return 1 / (1 + jnp.exp(-x))


class DenseState(NamedTuple):
    weights: jax.Array
    bias: jax.Array


class Dense:
    def __init__(self, n_in: int = None, n_out: int = None):
        # Setting network shape optional since an arbitrary state can be passed via call
        self.n_in = n_in
        self.n_out = n_out
        self.call_batch_fn = vmap(self.forward, in_axes=(None, 0))

    def init_state(self, rng: jax.Array, scale=1e-2) -> DenseState:
        w_key, b_key = random.split(rng)
        w = scale * random.normal(w_key, (self.n_out, self.n_in))
        b = scale * random.normal(b_key, (self.n_out,))
        return DenseState(w, b)

    def __call__(self, state: DenseState, x: jax.Array) -> jax.Array:
        # Batch if x has a batch dim
        if x.ndim > 2:
            raise ValueError(f"Input dim must be 1 or 2, got {x.ndim}")
        elif x.ndim == 2:
            return self.call_batch_fn(state, x)
        else:
            return self.forward(state, x)

    def forward(self, state: DenseState, x: jax.Array) -> jax.Array:
        # Linear forward pass of non-batched input
        return jnp.dot(state.weights, x) + state.bias


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
