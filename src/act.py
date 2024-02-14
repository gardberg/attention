import jax.numpy as jnp
import jax
from typing import Tuple, Union
from states import SnakeState


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


# aka silu
def swish(x: jax.Array, beta: float = 1) -> jax.Array:
    return x * sigmoid(beta * x)


def gelu(x: jax.Array) -> jax.Array:
    return x * sigmoid(1.702 * x)


# componentwise prod of sigmoid(L1) and L2. L1, L2 indep. affine transforms of x
def glu(x: jax.Array, dim=-1) -> jax.Array:
    """
    x.shape: (..., 2d, ...)
    dim: dimension to split on
    output.shape: (..., d, ...)
    """
    assert x.shape[dim] % 2 == 0, f"Dimension {dim} must be even, got {x.shape[dim]}"
    mid = x.shape[dim] // 2
    x1, x2 = jnp.split(x, [mid], axis=dim)
    return x1 * sigmoid(x2)


def swiglu(x: jax.Array, dim=-1) -> jax.Array:
    assert x.shape[dim] % 2 == 0, f"Dimension {dim} must be even, got {x.shape[dim]}"
    mid = x.shape[dim] // 2
    x1, x2 = jnp.split(x, [mid], axis=dim)
    return x1 * swish(x2)


def dropout(
    x: jax.Array, prob: float, rng: jax.Array, training: bool = True
) -> Tuple[jax.Array, jax.Array]:
    prob = float(prob)

    # p: probability of dropout
    assert 0 <= prob < 1, f"Probability must be in [0, 1), got {prob}"
    if not training:
        return x, rng
    rng, rng_input = jax.random.split(rng)
    mask = jax.random.bernoulli(rng_input, 1 - prob, x.shape)

    # Need to return new rng state?
    return x * mask / (1 - prob), rng


class Snake:
    def __init__(self, n_in: int, trainable: bool = False):
        self.n_in = n_in
        self.trainable = trainable

    def init_state(self, rng: jax.Array) -> SnakeState:
        # TODO: Proper initialization
        a = 0.1 * jax.random.exponential(rng, (self.n_in,))
        return SnakeState(a)

    def __call__(self, state: jax.Array, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        return self._forward(x, state.a)

    def _forward(self, x: jax.Array, a: jax.Array) -> jax.Array:
        return x + (1.0 / a) * jnp.power(jnp.sin(a * x), 2)
