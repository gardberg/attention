import numpy as np
import jax
import jax.numpy as jnp
import torch
from attention import *
import logging
from utils import LOG_LEVEL, get_logger
import pytest
from typing import Tuple

logger = get_logger()

np.random.seed(1337)

TOL = 1e-6
SHAPE = (4, 4)
x = np.random.randn(*SHAPE)


# Compare pytorch linear network to custom implementation
@pytest.mark.parametrize("n_in, n_out", [(1, 1), (4, 1), (1, 4), (4, 4)])
def test_linear(n_in: int, n_out: int):
    x_in = torch.randn(n_in)
    w = torch.randn(n_out, n_in)
    b = torch.randn(n_out)

    torch_linear = torch.nn.Linear(n_in, n_out)
    torch_linear.weight = torch.nn.Parameter(w)
    torch_linear.bias = torch.nn.Parameter(b)

    with torch.no_grad():
        y_torch = torch_linear(x_in).numpy()

    # Jax
    state = LinearState(jnp.array(w), jnp.array(b))
    y_jax = linear(jnp.array(x_in), state)

    logger.log(LOG_LEVEL, f"Diff: {np.linalg.norm(y_torch - y_jax):.2e}")
    assert np.allclose(
        y_torch, y_jax, atol=TOL
    ), f"y_torch = {y_torch}, y = {y_jax}"


def test_linear_square():
    n_in = (4, 4)
    n_out = 1

    x_in = torch.randn(*n_in)
    w = torch.randn(n_out, np.prod(n_in))
    b = torch.randn(n_out)

    torch_linear = torch.nn.Linear(np.prod(n_in), np.prod(n_out))
    torch_linear.weight = torch.nn.Parameter(w)
    torch_linear.bias = torch.nn.Parameter(b)

    with torch.no_grad():
        y_torch = torch_linear(x_in.flatten()).numpy()

    # Jax
    state = LinearState(jnp.array(w), jnp.array(b))
    y_jax = linear(jnp.array(x_in.flatten()), state)
    logger.log(LOG_LEVEL, f"Diff: {np.linalg.norm(y_torch - y_jax):.2e}")
    assert np.allclose(
        y_torch, y_jax, atol=TOL
    ), f"y_torch = {y_torch}, y = {y_jax}"


def test_softmax():
    y_torch = torch.softmax(torch.from_numpy(x), dim=1).numpy()
    y = jax.device_get(softmax(x, dim=1))

    logger.log(LOG_LEVEL, f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"


def test_softmax_stable():
    y_torch = torch.softmax(torch.from_numpy(x), dim=1).numpy()
    y = jax.device_get(softmax_stable(x, dim=1))

    logger.log(LOG_LEVEL, f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"


@pytest.mark.parametrize("shape", [(1,), (4,), (1, 1), (4, 4)])
def test_relu(shape: Tuple[int, ...]):
    x = torch.randn(*shape)
    y_torch = torch.relu(x).numpy()
    y = relu(jnp.array(x))

    logger.log(LOG_LEVEL, f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"
    

def test_batchnorm_1d_inference():
    x = torch.randn(1, 2, 2)
    torch_bn = torch.nn.BatchNorm1d(2)
    torch_bn.eval()

    with torch.no_grad():
        y_torch = torch_bn(x).numpy()
    
    # Jax
    state = BatchNormState()
    y, _state = batchnorm_1d(jnp.array(x), state, training=False)

    logger.log(LOG_LEVEL, f"y_torch = {y_torch}, y = {y}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"
    