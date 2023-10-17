import jax.numpy as jnp
import jax
import numpy as np
from jax import random
from typing import List, Callable
from utils import logger

key = random.PRNGKey(0)
N = 2
D = 1 # Axis along which to normalize with softmax

def allow_numpy(func: Callable) -> Callable:
    def wrapper(x: jnp.ndarray | np.ndarray, *args, **kwargs):
        if not isinstance(x, jnp.ndarray):
            x = jax.device_put(x)
            
        result = func(x, *args, **kwargs)

        logger.debug(get_debug_string(func, result))
        
        return result
    return wrapper
            

def get_debug_string(f: Callable, result: jnp.ndarray):
    return f"{f.__name__}:\n{result}\nSum: {jnp.sum(result, axis=D-2)}"


# Naive
@allow_numpy
def softmax(x: jnp.ndarray, dim: int = 1) -> jnp.ndarray:
    xe = jnp.exp(x)
    return xe / jnp.sum(xe, axis=dim-2, keepdims=True)


# Off by one
@allow_numpy
def softmax_one(x: jnp.ndarray, dim: int = 1) -> jnp.ndarray:
    xe = jnp.exp(x)
    return xe / (jnp.sum(xe, axis=dim-2, keepdims=True) + 1)


# Stable
@allow_numpy
def softmax_stable(x: jnp.ndarray, dim: int = 1) -> jnp.ndarray:
    maxes = jnp.max(x, axis=dim-2, keepdims=True)
    xm = jnp.exp(x - maxes)
    return xm / jnp.sum(xm, axis=dim-2, keepdims=True)
