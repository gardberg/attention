import numpy as np
import jax
import jax.numpy as jnp
import torch
from attention import *
import logging
from utils import LOG_LEVEL, get_logger

logger = get_logger()

np.random.seed(1337)

TOL = 1e-6
SHAPE = (4, 4)
x = np.random.randn(*SHAPE)

# Compare pytorch linear network to custom implementation
def test_linear_flat():
    n_in = 4
    n_out = 1
    
    x_in = torch.randn(n_in)
    w = torch.randn(n_out, n_in)
    b = torch.randn(n_out)
    
    torch_linear = torch.nn.Linear(n_in, n_out)
    torch_linear.weight = torch.nn.Parameter(w)
    torch_linear.bias = torch.nn.Parameter(b)
    
    with torch.no_grad():
        y_torch = torch_linear(x_in).numpy()
        
    logger.log(LOG_LEVEL, f"y_torch = {y_torch}")
    # Jax
    y_jax = linear(jnp.array(x_in), jnp.array(w), jnp.array(b))
    logger.log(LOG_LEVEL, f"y_jax = {y_jax}")
    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}\ny_jax = {y_jax}"

def test_softmax():
    y_torch = torch.softmax(torch.from_numpy(x), dim=1).numpy()
    y = jax.device_get(softmax(x, dim=1))

    logger.log(LOG_LEVEL, f"Softmax diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}\ny_torch = {y_torch}"

def test_softmax_grad():
    xt = torch.from_numpy(x)
    logger.log(LOG_LEVEL, f"xt = {xt}")

def test_softmax_stable():
    y_torch = torch.softmax(torch.from_numpy(x), dim=1).numpy()
    y = jax.device_get(softmax_stable(x, dim=1))

    logger.log(LOG_LEVEL, f"Softmax stable diff: {np.linalg.norm(y_torch - y):.2e}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}\ny_torch = {y_torch}"


if __name__ == "__main__":
    test_softmax()
