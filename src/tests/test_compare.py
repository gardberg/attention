import numpy as np
import jax
import jax.numpy as jnp
import torch
from attention import *
from utils import LOG_LEVEL, get_logger
import pytest
from typing import Tuple

logger = get_logger()

np.random.seed(1337)
rng = jax.random.PRNGKey(0)
torch.manual_seed(1337)

TOL = 1e-6


# Compare pytorch linear network to custom implementation
@pytest.mark.parametrize("n_in, n_out", [(1, 1), (4, 1), (1, 4), (4, 4)])
def test_dense(n_in: int, n_out: int):
    x_in = torch.randn(n_in)
    w = torch.randn(n_out, n_in)
    b = torch.randn(n_out)

    torch_linear = torch.nn.Linear(n_in, n_out)
    torch_linear.weight = torch.nn.Parameter(w)
    torch_linear.bias = torch.nn.Parameter(b)

    with torch.no_grad():
        y_torch = torch_linear(x_in).numpy()

    # Jax
    dense = Dense(n_in, n_out)
    state = DenseState(jnp.array(w), jnp.array(b))
    y_jax = dense(state, jnp.array(x_in))

    logger.log(LOG_LEVEL, f"Diff: {np.linalg.norm(y_torch - y_jax):.2e}")
    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"


@pytest.mark.parametrize("n_in, n_out, batch_size", [(1, 1, 1), (4, 4, 4)])
def test_dense_batch(n_in, n_out, batch_size):
    x_in = torch.randn(batch_size, n_in)
    logger.log(LOG_LEVEL, f"x_in: {x_in.shape}")
    w = torch.randn(n_out, n_in)
    b = torch.randn(n_out)

    torch_linear = torch.nn.Linear(n_in, n_out)
    torch_linear.weight = torch.nn.Parameter(w)
    torch_linear.bias = torch.nn.Parameter(b)

    with torch.no_grad():
        y_torch = torch_linear(x_in).numpy()

    # Jax
    dense = Dense(n_in, n_out)
    state = DenseState(jnp.array(w), jnp.array(b))
    y_jax = dense(state, jnp.array(x_in))
    logger.log(LOG_LEVEL, f"y_jax: {y_jax.shape}")

    logger.log(LOG_LEVEL, f"Diff: {np.linalg.norm(y_torch - y_jax):.2e}")
    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"


@pytest.mark.parametrize("n_in, n_out, batch_size", [(1, 1, 1), (4, 4, 4)])
def test_dense_batch_no_bias(n_in, n_out, batch_size):
    x_in = torch.randn(batch_size, n_in)
    logger.log(LOG_LEVEL, f"x_in: {x_in.shape}")
    w = torch.randn(n_out, n_in)

    torch_linear = torch.nn.Linear(n_in, n_out, bias=False)
    torch_linear.weight = torch.nn.Parameter(w)

    with torch.no_grad():
        y_torch = torch_linear(x_in).numpy()

    # Jax
    dense = Dense(n_in, n_out, bias=False)
    state = DenseState(jnp.array(w), None)
    y_jax = dense(state, jnp.array(x_in))
    logger.log(LOG_LEVEL, f"y_jax: {y_jax.shape}")

    logger.log(LOG_LEVEL, f"Diff: {np.linalg.norm(y_torch - y_jax):.2e}")
    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"


def test_dense_square():
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
    dense = Dense(n_in, n_out)
    state = DenseState(jnp.array(w), jnp.array(b))
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


@pytest.mark.parametrize(
    "emb_size, n_heads, use_bias",
    [(1, 1, False), (1, 1, True), (4, 4, False), (4, 4, True)],
)
def test_pre_attention(emb_size, n_heads, use_bias):
    # No batch
    seq_len = 10
    d_k = emb_size // n_heads
    x = torch.randn(seq_len, emb_size, requires_grad=False)
    linear = torch.nn.Linear(emb_size, n_heads * d_k, bias=use_bias)

    with torch.no_grad():
        head_shape = x.shape[:-1]
        y_torch = linear(x)
        y_torch = y_torch.view(*head_shape, n_heads, d_k)

    # Jax
    weight = jnp.array(linear.weight.detach())
    bias = jnp.array(linear.bias.detach()) if use_bias else None
    state = DenseState(weight, bias)

    preattn = PreAttention(emb_size=emb_size, n_heads=n_heads, d_k=d_k, bias=use_bias)
    y_jax = preattn(state, jnp.array(x))

    logger.log(
        LOG_LEVEL, f"y_torch.shape = {y_torch.shape}, y_jax.shape = {y_jax.shape}"
    )
    assert np.allclose(
        y_torch, y_jax, atol=TOL
    ), f"y_torch = {y_torch}, y_jax = {y_jax}"


def test_multihead_attention():
    # TODO: add test
    pass