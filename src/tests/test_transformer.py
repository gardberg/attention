import jax
import jax.numpy as jnp
from jax import Array
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer
import torch
import numpy as np
from testing_utils import TOL
import pytest

from transformer import EncoderLayer, DecoderLayer
from states import to_jax_state
from log_utils import logger

CONTEXT_LEN = 3


def test_transformer_encoder_layer_init():
    x = jnp.ones((CONTEXT_LEN, 2, 2))

    encoder_layer = EncoderLayer(emb_size=2, n_heads=2, d_ff=2, dropout=0.1)
    state = encoder_layer.init_state(jax.random.PRNGKey(0))
    out = encoder_layer(state, x, jax.random.PRNGKey(0))

def test_transformer_decoder_layer_init():
    x = jnp.ones((CONTEXT_LEN, 2, 2))

    decoder_layer = EncoderLayer(emb_size=2, n_heads=2, d_ff=2, dropout=0.1)
    state = decoder_layer.init_state(jax.random.PRNGKey(0))
    out = decoder_layer(state, x, jax.random.PRNGKey(0))
    

# TODO: Look at docs for transformer: https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
# Add use mask
@pytest.mark.parametrize("use_mask", [False])
def test_transformer_encoder_layer(use_mask):
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
    # jax_mask = encoder_layer.self_attn.get_causal_mask(CONTEXT_LEN, batch_size) if use_mask else None
    jax_mask = None

    encoder_state = to_jax_state(torch_encoder_layer)
    y = encoder_layer(encoder_state, x_jnp, mask=jax_mask, rng=jax.random.PRNGKey(0))

    logger.debug(f"y_jax.shape = {y.shape}")
    logger.debug(f"y_torch.shape = {y_torch.shape}")

    assert np.allclose(y_torch, y, atol=TOL), f"y_torch = {y_torch}, y = {y}"


@pytest.mark.parametrize("use_mask", [False, True])
def test_transformer_decoder_layer(use_mask: bool):    

    emb_size = 4
    n_heads = 1
    d_ff = 8
    dropout = 0.0
    batch_size = 2    

    x = torch.randn(CONTEXT_LEN, batch_size, emb_size, requires_grad=False)
    x_jnp = jnp.array(x.detach().numpy())
    logger.debug(f"x_jnp.shape = {x_jnp.shape}")
    
    # Torch
    torch_decoder_layer = TransformerDecoderLayer(
        emb_size, n_heads, dim_feedforward=d_ff, dropout=dropout, norm_first=True
    )

    with torch.no_grad():
        y_torch = torch_decoder_layer(x, x).detach().numpy()

    # Jax
    decoder_layer = DecoderLayer(emb_size, n_heads, d_ff, dropout)
    decoder_state = to_jax_state(torch_decoder_layer)

    y = decoder_layer(decoder_state, x_jnp, x_jnp, mask=None, rng=jax.random.PRNGKey(0))

    logger.debug(f"y_jax.shape = {y.shape}")
    logger.debug(f"y_torch.shape = {y_torch.shape}")

    assert np.allclose(y_torch, y, atol=TOL), f"y_torch = {y_torch}, y = {y}"
