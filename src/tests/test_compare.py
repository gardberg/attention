import numpy as np
import jax
import jax.numpy as jnp
import torch
import pytest
from typing import Tuple

from utils import LOG_LEVEL, get_logger
from attention import *
from act import *

logger = get_logger()

np.random.seed(1337)
rng = jax.random.PRNGKey(0)
torch.manual_seed(1337)

TOL = 1e-6


# Compare pytorch linear network to custom implementation
@pytest.mark.parametrize("n_in, n_out", [(1, 1), (4, 1), (1, 4), (4, 4)])
def test_dense(n_in: int, n_out: int):
    x_in = torch.randn(n_in)

    torch_linear = torch.nn.Linear(n_in, n_out)

    with torch.no_grad():
        y_torch = torch_linear(x_in).numpy()

    # Jax
    dense = Linear(n_in, n_out)
    state = to_jax_state(torch_linear)
    y_jax = dense(state, jnp.array(x_in))

    logger.log(LOG_LEVEL, f"Diff: {np.linalg.norm(y_torch - y_jax):.2e}")
    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"


@pytest.mark.parametrize("n_in, n_out, batch_size", [(1, 1, 1), (4, 4, 4)])
def test_dense_batch(n_in, n_out, batch_size):
    x_in = torch.randn(batch_size, n_in)
    logger.log(LOG_LEVEL, f"x_in: {x_in.shape}")

    torch_linear = torch.nn.Linear(n_in, n_out)

    with torch.no_grad():
        y_torch = torch_linear(x_in).numpy()

    # Jax
    dense = Linear(n_in, n_out)
    state = to_jax_state(torch_linear)
    y_jax = dense(state, jnp.array(x_in))
    logger.log(LOG_LEVEL, f"y_jax: {y_jax.shape}")

    logger.log(LOG_LEVEL, f"Diff: {np.linalg.norm(y_torch - y_jax):.2e}")
    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"


@pytest.mark.parametrize("n_in, n_out, batch_size", [(1, 1, 1), (4, 4, 4)])
def test_dense_batch_no_bias(n_in, n_out, batch_size):
    x_in = torch.randn(batch_size, n_in)
    logger.log(LOG_LEVEL, f"x_in: {x_in.shape}")

    torch_linear = torch.nn.Linear(n_in, n_out, bias=False)

    with torch.no_grad():
        y_torch = torch_linear(x_in).numpy()

    # Jax
    dense = Linear(n_in, n_out, bias=False)
    state = to_jax_state(torch_linear)
    y_jax = dense(state, jnp.array(x_in))
    logger.log(LOG_LEVEL, f"y_jax: {y_jax.shape}")

    logger.log(LOG_LEVEL, f"Diff: {np.linalg.norm(y_torch - y_jax):.2e}")
    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"


def test_dense_square():
    n_in = (4, 4)
    n_out = 1

    x_in = torch.randn(*n_in)

    torch_linear = torch.nn.Linear(np.prod(n_in), np.prod(n_out))

    with torch.no_grad():
        y_torch = torch_linear(x_in.flatten()).numpy()

    # Jax
    dense = Linear(n_in, n_out)
    state = to_jax_state(torch_linear)
    y_jax = dense(state, jnp.array(x_in.flatten()))

    logger.log(LOG_LEVEL, f"Diff: {np.linalg.norm(y_torch - y_jax):.2e}")
    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"


@pytest.mark.parametrize("shape", [(4, 4), (4, 4, 4)])
def test_softmax(shape: Tuple[int, ...]):
    x = torch.randn(shape)
    y_torch = torch.softmax(x, dim=1).numpy()
    y = jax.device_get(softmax(jnp.array(x), dim=1))

    logger.log(LOG_LEVEL, f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"


@pytest.mark.parametrize("shape", [(4, 4), (4, 4, 4)])
def test_softmax_stable(shape: Tuple[int, ...]):
    x = torch.randn(shape)
    y_torch = torch.softmax(x, dim=1).numpy()
    y = jax.device_get(softmax_stable(jnp.array(x), dim=1))

    logger.log(LOG_LEVEL, f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"


@pytest.mark.parametrize("shape", [(1,), (4,), (1, 1), (4, 4)])
def test_relu(shape: Tuple[int, ...]):
    x = torch.randn(*shape)
    y_torch = torch.relu(x).numpy()
    y = relu(jnp.array(x))

    logger.log(LOG_LEVEL, f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"


@pytest.mark.parametrize("B, N", [(1, 2), (2, 1), (2, 3)])
def test_batchnorm_1d_inference_small(B: int, N: int):
    x = torch.randn(B, N)
    torch_bn = torch.nn.BatchNorm1d(N)
    torch_bn.eval()

    with torch.no_grad():
        y_torch = torch_bn(x).numpy()

    # Jax
    state = BatchNormState()
    y, _state = batchnorm_1d(jnp.array(x), state, training=False)

    logger.log(LOG_LEVEL, f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"


@pytest.mark.parametrize("B, N, L", [(1, 2, 2), (2, 2, 2), (1, 1, 1)])
def test_batchnorm_1d_inference(B: int, N: int, L: int):
    x = torch.randn(B, N, L)
    torch_bn = torch.nn.BatchNorm1d(N)
    torch_bn.eval()

    with torch.no_grad():
        y_torch = torch_bn(x).numpy()

    # Jax
    state = BatchNormState()
    y, _state = batchnorm_1d(jnp.array(x), state, training=False)

    logger.log(LOG_LEVEL, f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"


@pytest.mark.parametrize("B, N", [(2, 2), (2, 1), (2, 3)])
def test_batchnorm_1d_train(B: int, N: int):
    x = torch.randn(B, N)
    torch_bn = torch.nn.BatchNorm1d(N)

    with torch.no_grad():
        y_torch = torch_bn(x).numpy()

    # Jax
    state = BatchNormState()
    y, _state = batchnorm_1d(jnp.array(x), state, training=True)

    logger.log(LOG_LEVEL, f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"

@pytest.mark.parametrize("norm_dims", [(3,), (2, 3)])
def test_layernorm(norm_dims: tuple):
    # assume input: (context_len, batch_size, emb_dim)
    x = torch.randn(4, 2, 3, requires_grad=False) * 10

    torch_ln = torch.nn.LayerNorm(norm_dims)
    with torch.no_grad():
        y_torch = torch_ln(x).numpy()

    # Jax
    jax_ln = LayerNorm(norm_dims)
    state = to_jax_state(torch_ln)

    y_jax = jax_ln(state, jnp.array(x))
    
    logger.log(LOG_LEVEL, f"Diff: {np.linalg.norm(y_torch - y_jax):.2e}")
    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"
