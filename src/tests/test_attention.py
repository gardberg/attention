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
torch.manual_seed(0)

TOL = 1e-6
CONTEXT_LEN = 3


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
    # weight = jnp.array(linear.weight.detach())
    # bias = jnp.array(linear.bias.detach()) if use_bias else None
    # state = LinearState(weight, bias)

    state = to_jax_state(linear)

    preattn = PreAttention(n_heads=n_heads, emb_size=emb_size, d_k=d_k, bias=use_bias)
    y_jax = preattn(state, jnp.array(x))

    logger.log(
        LOG_LEVEL, f"y_torch.shape = {y_torch.shape}, y_jax.shape = {y_jax.shape}"
    )
    assert np.allclose(
        y_torch, y_jax, atol=TOL
    ), f"y_torch = {y_torch}, y_jax = {y_jax}"


@pytest.mark.parametrize("n_heads, emb_size, batch_size", [(1, 2, 3), (2, 4, 3), (8, 16, 8)])
def test_multihead_attn(n_heads, emb_size, batch_size):
    bias = False

    x = torch.randn(CONTEXT_LEN, batch_size, emb_size, requires_grad=False)

    # Torch
    torch_mha = torch.nn.MultiheadAttention(emb_size, n_heads, bias=bias)

    with torch.no_grad():
        y_torch = torch_mha(x, x, x, need_weights=False)[0].detach().numpy()

    # Jax
    jax_mha = MultiHeadAttention(emb_size, n_heads, bias=bias, v_bias=False)

    jax_mha_state = to_jax_state(torch_mha)

    x_jnp = jnp.array(x.detach().numpy())

    print(f"Calling jax_mha.forward with x_jnp.shape = {x_jnp.shape} and type = {type(x_jnp)}")
    y_jax = jax_mha.forward(jax_mha_state, x_jnp, x_jnp, x_jnp)
    
    print(f"y_torch.shape = {y_torch.shape}, y_jax.shape = {y_jax.shape}")
    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"


@pytest.mark.parametrize("n_heads, emb_size, batch_size", [(1, 2, 3)])
def test_attention_with_mask(n_heads, emb_size, batch_size):
    pass
