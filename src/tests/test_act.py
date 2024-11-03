import torch
import pytest
import numpy as np

from act import *
from testing_utils import *
from log_utils import logger
from loss import MSELoss


@pytest.mark.parametrize("shape", [(4, 4), (4, 4, 4)])
def test_softmax(shape: tuple[int, ...]):
    x = torch.randn(shape)
    y_torch = torch.softmax(x, dim=1).numpy()
    y = jax.device_get(softmax(jnp.array(x), dim=1))

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"


def test_softmax_matching():
    x = torch.randn(1, 50257)  # vocab size
    torch_probs = torch.nn.functional.softmax(x, dim=-1)

    jax_probs = softmax(jnp.array(x), dim=-1)

    assert jnp.allclose(jax_probs, torch_probs.numpy(), atol=1e-5)


@pytest.mark.parametrize("shape", [(4, 4), (4, 4, 4)])
def test_softmax_stable(shape: tuple[int, ...]):
    x = torch.randn(shape)
    y_torch = torch.softmax(x, dim=1).numpy()
    y = jax.device_get(softmax_stable(jnp.array(x), dim=1))

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"


@pytest.mark.parametrize("shape", [(1,), (4,), (1, 1), (4, 4)])
def test_relu(shape: tuple[int, ...]):
    x = torch.randn(*shape)
    y_torch = torch.relu(x).numpy()
    y = relu(jnp.array(x))

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"


@pytest.mark.parametrize("shape, dim", [((4, 4), -1), ((4, 4, 4), -2)])
def test_glu(shape: tuple[int, ...], dim: int):
    x = torch.randn(shape)
    y_torch = torch.nn.functional.glu(x, dim=dim).numpy()
    logger.debug(f"y_torch.shape = {y_torch.shape}")

    y = glu(jnp.array(x), dim=dim)
    logger.debug(f"y.shape = {y.shape}")

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"


@pytest.mark.parametrize("shape, dim", [((4, 4), -1), ((4, 4, 4), -2)])
def test_swiglu(shape: tuple[int, ...], dim: int):
    x = torch.randn(shape)
    y_torch = SwiGLU()(x, dim=dim).numpy()
    logger.debug(f"y_torch.shape = {y_torch.shape}")

    y = swiglu(jnp.array(x), dim=dim)
    logger.debug(f"y.shape = {y.shape}")

    logger.debug(f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"


@pytest.mark.parametrize("shape, approximate", [((4, 4), "none"), ((4, 4), "tanh")])
def test_gelu(shape, approximate):
    x = torch.randn(shape)
    y_torch = torch.nn.functional.gelu(x, approximate=approximate).numpy()
    y = gelu(jnp.array(x), approximate=approximate)

    print(f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"


def test_gelu_sigmoid():
    TOL = 5e-2  # Higher tolerance for sigmoid approximation
    x = torch.randn(4, 4)
    y_torch = torch.nn.functional.gelu(x, approximate="none").numpy()
    y = gelu(jnp.array(x), approximate="sigmoid")

    print(f"Diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}, y_torch = {y_torch}"


@pytest.mark.parametrize("n_in, shape", [(1, 1), (2, 2), (4, 4)])
def test_snake(n_in: int, shape: tuple[int, ...]):
    a = 2

    x = torch.randn(shape)
    torch_snake = TorchSnake(n_in, a, trainable=False)
    with torch.no_grad():
        y_torch = torch_snake(x).numpy()

    # jax
    jax_snake = Snake(n_in, training=False)
    state = SnakeState(jnp.array(a))
    y_jax = jax_snake(state, jnp.array(x))

    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"


@pytest.mark.parametrize("n_in, shape", [(1, 1), (2, 2), (4, 4)])
def test_snake_trainable(n_in: int, shape: tuple[int, ...]):
    x = torch.randn(shape)
    y = torch.randn(shape)

    torch_snake = TorchSnake(n_in, trainable=True)

    a_jax = jnp.array(torch_snake.a.detach().numpy())

    y_torch = torch_snake(x)
    torch_mse_loss = torch.nn.MSELoss()
    torch_loss = torch_mse_loss(y_torch, y)

    torch_loss.backward()
    grads_pytorch = torch_snake.a.grad.numpy()

    # Jax
    jax_snake = Snake(n_in, training=True)
    state = SnakeState(a_jax)
    mse_loss = MSELoss()

    def loss(state, x, y):
        y_pred = jax_snake(state, x)
        return mse_loss(y_pred, y)

    grads = jax.grad(loss, argnums=0)(state, jnp.array(x), jnp.array(y))

    logger.debug(f"grad_jax: {grads[0]}, grad_pytorch: {grads_pytorch}")
    assert np.allclose(
        grads[0], grads_pytorch, atol=TOL
    ), f"grads = {grads[0]}, grads_pytorch = {grads_pytorch}"


@pytest.mark.parametrize("p", [0.1, 0.5, 0.0, 0.8])
def test_dropout_train(p: float):
    x = torch.ones(4, 2, 3, requires_grad=False)

    # Torch
    with torch.no_grad():
        y_torch = torch.nn.functional.dropout(x, p=p, training=True).numpy()

    # Jax
    rng = jax.random.PRNGKey(0)
    y_jax = dropout(jnp.array(x), p, rng, training=True)

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
    y_jax = dropout(jnp.array(x), p, rng, training=False)

    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"
