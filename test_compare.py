import numpy as np
import jax
import jax.numpy as jnp
import torch
from attention import *
import logging

logging.basicConfig(level=logging.INFO)

# set numpy random seed
np.random.seed(1337)

TOL = 1e-6
SHAPE = (4, 4)
x = np.random.randn(*SHAPE)

def test_softmax():
    y_torch = torch.softmax(torch.from_numpy(x), dim=1).numpy()
    y = jax.device_get(softmax(x, dim=1))

    logging.info(f"||y_torch - y|| = {np.linalg.norm(y_torch - y)}")
    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}\ny_torch = {y_torch}"


def test_softmax_stable():
    y_torch = torch.softmax(torch.from_numpy(x), dim=1).numpy()
    y = jax.device_get(softmax_stable(x, dim=1))

    assert np.allclose(y, y_torch, atol=TOL), f"y = {y}\ny_torch = {y_torch}"


if __name__ == "__main__":
    test_softmax()
