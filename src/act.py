import jax.numpy as jnp
import jax
from states import SnakeState
from base import BaseModule, Array


# Naive
def softmax(x: Array, dim: int) -> Array:
    xe = jnp.exp(x)
    return xe / jnp.sum(xe, axis=dim, keepdims=True)


# Off by one
def softmax_one(x: Array, dim: int) -> Array:
    xe = jnp.exp(x)
    return xe / (jnp.sum(xe, axis=dim, keepdims=True) + 1)


# Stable
def softmax_stable(x: Array, dim: int) -> Array:
    maxes = jnp.max(x, axis=dim, keepdims=True)
    xm = jnp.exp(x - maxes)
    return xm / jnp.sum(xm, axis=dim, keepdims=True)


def sigmoid(x: Array) -> Array:
    return 1 / (1 + jnp.exp(-x))


def relu(x: Array) -> Array:
    return jnp.maximum(x, 0)


# aka silu
def swish(x: Array, beta: float = 1) -> Array:
    return x * sigmoid(beta * x)


def gelu(x: Array) -> Array:
    return x * sigmoid(1.702 * x)


# componentwise prod of sigmoid(L1) and L2. L1, L2 indep. affine transforms of x
def glu(x: Array, dim=-1) -> Array:
    """
    x.shape: (..., 2d, ...)
    dim: dimension to split on
    output.shape: (..., d, ...)
    """
    assert x.shape[dim] % 2 == 0, f"Dimension {dim} must be even, got {x.shape[dim]}"
    mid = x.shape[dim] // 2
    x1, x2 = jnp.split(x, [mid], axis=dim)
    return x1 * sigmoid(x2)


# GPT2 GELU: https://arxiv.org/abs/1606.08415
def gelu_new(x: Array) -> Array:
    return (
        0.5
        * x
        * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (x + 0.044715 * jnp.power(x, 3.0))))
    )


def swiglu(x: Array, dim=-1) -> Array:
    assert x.shape[dim] % 2 == 0, f"Dimension {dim} must be even, got {x.shape[dim]}"
    mid = x.shape[dim] // 2
    x1, x2 = jnp.split(x, [mid], axis=dim)
    return x1 * swish(x2)


def dropout(x: Array, prob: float, rng: Array, training: bool = True) -> Array:
    prob = float(prob)

    # p: probability of dropout
    assert 0 <= prob < 1, f"Probability must be in [0, 1), got {prob}"
    if not training:
        return x
    rng, rng_input = jax.random.split(rng)
    mask = jax.random.bernoulli(rng_input, 1 - prob, x.shape)

    return x * mask / (1 - prob)


# https://arxiv.org/abs/2006.08195
class Snake(BaseModule):
    def __init__(self, n_in: int, training: bool = False):
        super().__init__()
        self.n_in = n_in
        self.training = training

    def init_state(self, rng: Array) -> SnakeState:
        # TODO: Proper initialization
        a = 0.1 * jax.random.exponential(rng, (self.n_in,))
        return SnakeState(a)

    def forward(self, state: SnakeState, x: Array) -> Array:
        return x + (1.0 / state.a) * jnp.power(jnp.sin(state.a * x), 2)
