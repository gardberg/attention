import numpy as np
import jax
import jax.numpy as jnp
import torch
from attention import *
from utils import LOG_LEVEL, get_logger
import pytest
from typing import Tuple
from loss import MSELoss

logger = get_logger()

np.random.seed(1337)
rng = jax.random.PRNGKey(0)

TOL = 1e-6


# Check that computed gradients are the same for a linear layer
@pytest.mark.parametrize("n_in, n_out, batch_size", [(1, 1, 1), (2, 2, 2)])
def test_dense_grad(n_in: int, n_out: int, batch_size: int):
    x_in = torch.randn(batch_size, n_in)
    w = torch.randn(n_out, n_in, requires_grad=True)
    b = torch.randn(n_out, requires_grad=True)

    y = torch.randn(batch_size, n_out)

    torch_linear = torch.nn.Linear(n_in, n_out)

    with torch.no_grad():
        torch_linear.weight = torch.nn.Parameter(w)
        torch_linear.bias = torch.nn.Parameter(b)

    assert torch_linear.weight.grad is None
    assert torch_linear.bias.grad is None

    # With gradients on
    y_torch = torch_linear(x_in)

    torch_mse_loss = torch.nn.MSELoss()
    print(f"torch_mse_loss: {torch_mse_loss}")
    torch_loss = torch_mse_loss(y_torch, y)

    torch_loss.backward()
    logger.log(LOG_LEVEL, torch_loss)
    grads_pytorch = [
        torch_linear.weight.grad.numpy(),
        torch_linear.bias.grad.numpy(),
    ]

    # Jax
    dense = Dense(n_in, n_out)
    weights_jnp = jnp.array(torch_linear.weight.detach().numpy())
    bias_jnp = jnp.array(torch_linear.bias.detach().numpy())
    state = DenseState(weights_jnp, bias_jnp)

    mse_loss = MSELoss()

    def loss(state, x, y):
        y_pred = dense(state, x)
        return mse_loss(y_pred, y)

    # get gradients with respect to the state (arg 0)
    grads = jax.grad(loss, argnums=0)(state, jnp.array(x_in), jnp.array(y))

    for grad_jax, grad_pytorch in zip(grads, grads_pytorch):
        logger.log(LOG_LEVEL, f"grad_jax: {grad_jax}, grad_pytorch: {grad_pytorch}")
        assert np.allclose(
            grad_jax, grad_pytorch, atol=TOL
        ), f"grad_jax = {grad_jax}, grad_pytorch = {grad_pytorch}"
