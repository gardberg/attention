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

    torch_in_proj_weight = torch_mha.in_proj_weight
    torch_weights = (
        torch_in_proj_weight[0:emb_size, :],
        torch_in_proj_weight[emb_size : 2 * emb_size, :],
        torch_in_proj_weight[2 * emb_size : 3 * emb_size, :],
        torch_mha.out_proj.weight
    )

    torch_weights = tuple(LinearState(jnp.array(w.detach().numpy()), None) for w in torch_weights)

    jax_mha_state = MultiHeadAttentionState(*torch_weights)
    x_jnp = jnp.array(x.detach().numpy())

    print(f"Calling jax_mha.forward with x_jnp.shape = {x_jnp.shape} and type = {type(x_jnp)}")
    y_jax = jax_mha.forward(jax_mha_state, x_jnp, x_jnp, x_jnp)
    
    print(f"y_torch.shape = {y_torch.shape}, y_jax.shape = {y_jax.shape}")
    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"
