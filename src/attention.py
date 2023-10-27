import jax.numpy as jnp
import jax
import numpy as np
from jax import random
from typing import List, Callable, Tuple, NamedTuple
from utils import LOG_LEVEL, get_logger

logger = get_logger()

key = random.PRNGKey(0)


def allow_numpy(func: Callable) -> Callable:
    def wrapper(x: jnp.ndarray | np.ndarray, *args, **kwargs):
        if not isinstance(x, jnp.ndarray):
            x = jax.device_put(x)

        result = func(x, *args, **kwargs)

        logger.log(LOG_LEVEL, get_debug_string(func, result))

        return result

    return wrapper


def get_debug_string(f: Callable, result: jnp.ndarray):
    return f"{f.__name__}:\n{result}\nSum: {jnp.sum(result, axis=-1)}"


# Naive
@allow_numpy
def softmax(x: jnp.ndarray, dim: int = 1) -> jnp.ndarray:
    xe = jnp.exp(x)
    return xe / jnp.sum(xe, axis=dim - 2, keepdims=True)


# Off by one
@allow_numpy
def softmax_one(x: jnp.ndarray, dim: int = 1) -> jnp.ndarray:
    xe = jnp.exp(x)
    return xe / (jnp.sum(xe, axis=dim - 2, keepdims=True) + 1)


# Stable
@allow_numpy
def softmax_stable(x: jnp.ndarray, dim: int = 1) -> jnp.ndarray:
    maxes = jnp.max(x, axis=dim - 2, keepdims=True)
    xm = jnp.exp(x - maxes)
    return xm / jnp.sum(xm, axis=dim - 2, keepdims=True)


class LinearState(NamedTuple):
    weights: jnp.ndarray
    bias: jnp.ndarray


def linear(x: jnp.ndarray, state: LinearState) -> jnp.ndarray:
    # TODO: Add batching

    """
    :param x.shape:         (n_in, )
    :param weights.shape:   (n_out, n_in)
    :param bias.shape:      (n_out, )
    :return shape:          (n_out, )
    """

    return jnp.dot(state.weights, x) + state.bias


def relu(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(x, 0)


class BatchNormState(NamedTuple):
    """
    :param
    """

    mean: jnp.ndarray = 0
    var: jnp.ndarray = 1
    gamma: jnp.ndarray = 1
    beta: jnp.ndarray = 0
    momentum: jnp.float32 = 0.1
    eps: jnp.float32 = 1e-5


def batchnorm_1d(
    x: jnp.ndarray, state: BatchNormState, training: bool = True
) -> Tuple[jnp.ndarray, BatchNormState]:
    """
    :param jnp.ndarray x:           (B, N) or (B, N, L), B batch size, N input dim, L input length
    :param BatchNormState state:    NamedTuple with mean, var, gamma, beta
    """
    # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d

    if training:
        # Only update running mean and var during training
        mean = jnp.mean(x, axis=0)
        var = jnp.var(x, axis=0)  # ddof = 0, biased

        state.mean = (1 - state.momentum) * state.mean + state.momentum * mean
        state.var = (1 - state.momentum) * state.var + state.momentum * jnp.var(
            x, axis=0, ddof=1
        )  # unbiased

        x_norm = (x - mean) / jnp.sqrt(var + state.eps)
    else:
        x_norm = (x - state.mean) / jnp.sqrt(state.var + state.eps)

    return state.gamma * x_norm + state.beta, state
