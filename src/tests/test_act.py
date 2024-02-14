import torch
import pytest
import numpy as np

from act import *
from testing_utils import *
from log_utils import logger


@pytest.mark.parametrize("p", [0.1, 0.5, 0.0, 0.8])
def test_dropout_train(p: float):
    x = torch.ones(4, 2, 3, requires_grad=False)

    # Torch
    with torch.no_grad():
        y_torch = torch.nn.functional.dropout(x, p=p, training=True).numpy()

    # Jax
    rng = jax.random.PRNGKey(0)
    y_jax, rng = dropout(jnp.array(x), p, rng, training=True)

    assert y_torch.shape == y_jax.shape, f"Got {y_jax.shape}, expected {y_torch.shape}"
    # assert y_jax is not all ones
    if p == 0.0:
        assert np.allclose(y_jax, x), f"y_torch = {y_torch}, y = {y_jax}"
    else:
        assert not np.allclose(y_jax, x), f"y_torch = {y_torch}, y = {y_jax}"


@pytest.mark.parametrize("p", [0.1, 0.5, 0.0, 0.8])
def test_dropout_eval(p: float):
    x = torch.ones(4, 2, 3, requires_grad=False)

    # Torch
    with torch.no_grad():
        y_torch = torch.nn.functional.dropout(x, p=p, training=False).numpy()

    # Jax
    rng = jax.random.PRNGKey(0)
    y_jax, rng = dropout(jnp.array(x), p, rng, training=False)

    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"


@pytest.mark.parametrize("shape", [(4, 4), (4, 4, 4)])
def test_softmax(shape: Tuple[int, ...]):
    x = torch.randn(shape)
    y_torch = torch.softmax(x, dim=1).numpy()
    y = jax.device_get(softmax(jnp.array(x), dim=1))

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"


@pytest.mark.parametrize("shape", [(4, 4), (4, 4, 4)])
def test_softmax_stable(shape: Tuple[int, ...]):
    x = torch.randn(shape)
    y_torch = torch.softmax(x, dim=1).numpy()
    y = jax.device_get(softmax_stable(jnp.array(x), dim=1))

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"


@pytest.mark.parametrize("shape", [(1,), (4,), (1, 1), (4, 4)])
def test_relu(shape: Tuple[int, ...]):
    x = torch.randn(*shape)
    y_torch = torch.relu(x).numpy()
    y = relu(jnp.array(x))

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"


@pytest.mark.parametrize("shape, dim", [((4, 4), -1), ((4, 4, 4), -2)])
def test_glu(shape: Tuple[int, ...], dim: int):
    x = torch.randn(shape)
    y_torch = torch.nn.functional.glu(x, dim=dim).numpy()
    logger.debug(f"y_torch.shape = {y_torch.shape}")

    y = glu(jnp.array(x), dim=dim)
    logger.debug(f"y.shape = {y.shape}")

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"


@pytest.mark.parametrize("shape, dim", [((4, 4), -1), ((4, 4, 4), -2)])
def test_swiglu(shape: Tuple[int, ...], dim: int):
    x = torch.randn(shape)
    y_torch = SwiGLU()(x, dim=dim).numpy()
    logger.debug(f"y_torch.shape = {y_torch.shape}")

    y = swiglu(jnp.array(x), dim=dim)
    logger.debug(f"y.shape = {y.shape}")

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"


@pytest.mark.parametrize("n_in, shape", [(1, 1), (2, 2), (4, 4)])
def test_snake(n_in: int, shape: Tuple[int, ...]):
    a = 2

    x = torch.randn(shape)
    torch_snake = TorchSnake(n_in, a, trainable=False)
    with torch.no_grad():
        y_torch = torch_snake(x).numpy()

    # jax
    jax_snake = Snake(n_in, trainable=False)
    state = SnakeState(jnp.array(a))
    y_jax = jax_snake(state, jnp.array(x))

    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"


def test_snake_trainable():
    # TODO
    pass
