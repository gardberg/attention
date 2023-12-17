import jax.numpy as jnp
import jax

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

