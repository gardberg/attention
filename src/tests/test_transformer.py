import jax
import jax.numpy as jnp
from torch.nn import TransformerEncoderLayer
import torch
import numpy as np
from testing_utils import *

from transformer import *
from states import to_jax_state

CONTEXT_LEN = 3


def test_transformer_encoder_layer():
    emb_size = 4
    n_heads = 1
    d_ff = 8
    dropout = 0.0
    batch_size = 2

    x = torch.randn(CONTEXT_LEN, batch_size, emb_size, requires_grad=False)
    x_jnp = jnp.array(x.detach().numpy())
    logger.debug(f"x_jnp.shape = {x_jnp.shape}")

    # Torch
    torch_encoder_layer = TransformerEncoderLayer(
        emb_size, n_heads, dim_feedforward=d_ff, dropout=dropout, norm_first=True
    )

    with torch.no_grad():
        y_torch = torch_encoder_layer(x).detach().numpy()

    # Jax
    encoder_layer = EncoderLayer(emb_size, n_heads, d_ff, dropout)
    # encoder_state = encoder_layer.init_state(jax.random.PRNGKey(0))
    encoder_state = to_jax_state(torch_encoder_layer)

    y = encoder_layer(encoder_state, x_jnp, mask=None, rng=jax.random.PRNGKey(0))

    logger.debug(f"y_jax.shape = {y.shape}")
    logger.debug(f"y_torch.shape = {y_torch.shape}")

    assert np.allclose(y_torch, y, atol=TOL), f"y_torch = {y_torch}, y = {y}"
