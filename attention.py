import jax.numpy as jnp
from jax import random
from typing import List, Callable

key = random.PRNGKey(0)
N = 2
D = 1 # Axis along which to normalize with softmax

def test():
    x = random.normal(key, (N,N))
    fs = [softmax, softmax_one, softmax_stable]
    printfs(fs, x)
    
def printfs(fs: List[Callable], x: jnp.ndarray):
    for f in fs:
        fx = f(x, D)
        print(f"{f.__name__}:")
        print(fx)
        print(f"Sum: {jnp.sum(fx, axis=D-1)}")
        print()

# Naive
def softmax(x: jnp.ndarray, dim: int = 1) -> jnp.ndarray:
    xe = jnp.exp(x)
    return xe / jnp.sum(xe, axis=dim-1, keepdims=True)

# Off by one
def softmax_one(x: jnp.ndarray, dim: int = 1) -> jnp.ndarray:
    xe = jnp.exp(x)
    return xe / (jnp.sum(xe, axis=dim-1, keepdims=True) + 1)

# Stable
def softmax_stable(x: jnp.ndarray, dim: int = 1) -> jnp.ndarray:
    maxes = jnp.max(x, axis=dim-1, keepdims=True)
    xm = jnp.exp(x - maxes)
    return xm / jnp.sum(xm, axis=dim-1, keepdims=True)


if __name__ == "__main__":
    test()
