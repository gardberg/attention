import jax.numpy as jnp
import jax
from typing import Tuple


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


def dropout(
    x: jax.Array, prob: float, rng: jax.Array, training: bool = True
) -> Tuple[jax.Array, jax.Array]:
    # p: probability of dropout
    assert 0 <= prob < 1, f"Probability must be in [0, 1), got {prob}"
    if not training:
        return x, rng
    rng, rng_input = jax.random.split(rng)
    mask = jax.random.bernoulli(rng_input, 1 - prob, x.shape)

    # Need to return new rng state?
    return x * mask / (1 - prob), rng
