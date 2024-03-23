import jax
import jax.numpy as jnp
from jax import Array

import torch
import numpy as np
from testing_utils import TOL
import pytest

from transformer import EncoderLayer, DecoderLayer, Encoder, Decoder, Transformer
from states import to_jax_state
from log_utils import logger
from attention import create_causal_mask, LayerNorm

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


def test_transformer_encoder_init():
    x = jnp.ones((CONTEXT_LEN, 2, 2))

    encoder_layer = EncoderLayer(emb_size=2, n_heads=2, d_ff=2, dropout=0.1)
    encoder = Encoder(encoder_layer, 2)
    state = encoder.init_state(jax.random.PRNGKey(0))
    out = encoder(state, x, jax.random.PRNGKey(0))


def test_transformer_decoder_init():
    x = jnp.ones((CONTEXT_LEN, 2, 2))

    decoder_layer = DecoderLayer(emb_size=2, n_heads=2, d_ff=2, dropout=0.1)
    decoder = Decoder(decoder_layer, 2)
    state = decoder.init_state(jax.random.PRNGKey(0))
    out = decoder(state, x, x, jax.random.PRNGKey(0))


# https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
@pytest.mark.parametrize("use_mask", [False, True])
def test_transformer_encoder_layer(use_mask):
    emb_size = 4
    n_heads = 1
    d_ff = 8
    dropout = 0.0
    batch_size = 2

    x = torch.randn(CONTEXT_LEN, batch_size, emb_size, requires_grad=False)
    x_jnp = jnp.array(x.detach().numpy())
    logger.debug(f"x_jnp.shape = {x_jnp.shape}")

    jax_mask = None
    torch_mask = None
    if use_mask:
        jax_mask = create_causal_mask(CONTEXT_LEN, CONTEXT_LEN)
        logger.debug(f"jax_mask = {jax_mask}")
        torch_mask = torch.from_numpy(np.array(jax_mask))
        logger.debug(f"torch_mask = {torch_mask}")

    # Torch
    torch_encoder_layer = torch.nn.TransformerEncoderLayer(
        emb_size, n_heads, dim_feedforward=d_ff, dropout=dropout, norm_first=True
    )

    with torch.no_grad():
        y_torch = torch_encoder_layer(x, src_mask=torch_mask).detach().numpy()

    # Jax
    encoder_layer = EncoderLayer(emb_size, n_heads, d_ff, dropout)

    encoder_state = to_jax_state(torch_encoder_layer)
    y = encoder_layer(encoder_state, x_jnp, src_mask=jax_mask, rng=jax.random.PRNGKey(0))

    logger.debug(f"y_jax.shape = {y.shape}")
    logger.debug(f"y_torch.shape = {y_torch.shape}")

    assert np.allclose(y_torch, y, atol=TOL), f"y_torch = {y_torch}, y = {y}"


@pytest.mark.parametrize("use_mask", [False, True])
def test_transformer_encoder(use_mask):
    emb_size = 4
    n_heads = 1
    d_ff = 8
    dropout = 0.0
    batch_size = 2

    N_LAYERS = 2

    x = torch.randn(CONTEXT_LEN, batch_size, emb_size, requires_grad=False)
    x_jnp = jnp.array(x.detach().numpy())
    logger.debug(f"x_jnp.shape = {x_jnp.shape}")

    jax_mask = None
    torch_mask = None
    if use_mask:
        jax_mask = create_causal_mask(CONTEXT_LEN, CONTEXT_LEN)
        logger.debug(f"jax_mask = {jax_mask}")
        torch_mask = torch.from_numpy(np.array(jax_mask))
        logger.debug(f"torch_mask = {torch_mask}")

    # Torch
    torch_encoder_layer = torch.nn.TransformerEncoderLayer(
        emb_size, n_heads, dim_feedforward=d_ff, dropout=dropout, norm_first=True
    )
    torch_norm = torch.nn.LayerNorm(emb_size)
    torch_encoder = torch.nn.TransformerEncoder(torch_encoder_layer, N_LAYERS, norm=torch_norm)

    with torch.no_grad():
        y_torch = torch_encoder(x, mask=torch_mask).detach().numpy()

    # Jax
    encoder_layer = EncoderLayer(emb_size, n_heads, d_ff, dropout)
    norm = LayerNorm(emb_size)

    encoder_state = to_jax_state(torch_encoder)
    encoder = Encoder(encoder_layer, N_LAYERS, norm=norm)
    y = encoder(encoder_state, x_jnp, src_mask=jax_mask, rng=jax.random.PRNGKey(0))

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

    tgt = torch.randn(CONTEXT_LEN, batch_size, emb_size, requires_grad=False)
    tgt_jnp = jnp.array(tgt.detach().numpy())
    src = torch.randn(CONTEXT_LEN + 1, batch_size, emb_size, requires_grad=False)
    src_jnp = jnp.array(src.detach().numpy())
    logger.debug(f"tgt_jnp.shape = {tgt_jnp.shape}")
    logger.debug(f"src_jnp.shape = {src_jnp.shape}")

    tgt_mask = None
    src_mask = None
    tgt_mask_torch = None
    src_mask_torch = None
    if use_mask:
        tgt_mask = create_causal_mask(CONTEXT_LEN, CONTEXT_LEN)
        logger.debug(f"tgt_mask = {tgt_mask}")
        src_mask = create_causal_mask(CONTEXT_LEN, CONTEXT_LEN + 1)
        logger.debug(f"src_mask = {src_mask}")

        tgt_mask_torch = torch.from_numpy(np.array(tgt_mask))
        src_mask_torch = torch.from_numpy(np.array(src_mask))

    # Torch
    torch_decoder_layer = torch.nn.TransformerDecoderLayer(
        emb_size, n_heads, dim_feedforward=d_ff, dropout=dropout, norm_first=True
    )

    with torch.no_grad():
        y_torch = (
            torch_decoder_layer(
                tgt=tgt, memory=src, tgt_mask=tgt_mask_torch, memory_mask=src_mask_torch
            )
            .detach()
            .numpy()
        )

    # Jax
    decoder_layer = DecoderLayer(emb_size, n_heads, d_ff, dropout)
    decoder_state = to_jax_state(torch_decoder_layer)

    y = decoder_layer(
        decoder_state,
        tgt=tgt_jnp,
        memory=src_jnp,
        tgt_mask=tgt_mask,
        memory_mask=src_mask,
        rng=jax.random.PRNGKey(0),
    )

    logger.debug(f"y_jax.shape = {y.shape}")
    logger.debug(f"y_torch.shape = {y_torch.shape}")

    assert np.allclose(y_torch, y, atol=TOL), f"y_torch = {y_torch}, y = {y}"


@pytest.mark.parametrize("use_mask", [False, True])
def test_transformer_decoder(use_mask):
    emb_size = 4
    n_heads = 1
    d_ff = 8
    dropout = 0.0
    batch_size = 2

    tgt = torch.randn(CONTEXT_LEN, batch_size, emb_size, requires_grad=False)
    tgt_jnp = jnp.array(tgt.detach().numpy())
    memory = torch.randn(CONTEXT_LEN + 1, batch_size, emb_size, requires_grad=False)
    memory_jnp = jnp.array(memory.detach().numpy())

    tgt_mask = None
    memory_mask = None
    tgt_mask_torch = None
    memory_mask_torch = None
    if use_mask:
        tgt_mask = create_causal_mask(CONTEXT_LEN, CONTEXT_LEN)
        memory_mask = create_causal_mask(CONTEXT_LEN, CONTEXT_LEN + 1)

        tgt_mask_torch = torch.from_numpy(np.array(tgt_mask))
        memory_mask_torch = torch.from_numpy(np.array(memory_mask))

    # Torch
    torch_decoder_layer = torch.nn.TransformerDecoderLayer(
        emb_size, n_heads, dim_feedforward=d_ff, dropout=dropout, norm_first=True
    )
    torch_norm = torch.nn.LayerNorm(emb_size)
    torch_decoder = torch.nn.TransformerDecoder(torch_decoder_layer, 2, norm=torch_norm)

    with torch.no_grad():
        y_torch = (
            torch_decoder(
                tgt, memory, tgt_mask=tgt_mask_torch, memory_mask=memory_mask_torch
            )
            .detach()
            .numpy()
        )

    # Jax
    decoder_layer = DecoderLayer(emb_size, n_heads, d_ff, dropout)
    norm = LayerNorm(emb_size)

    decoder_state = to_jax_state(torch_decoder)
    decoder = Decoder(decoder_layer, 2, norm=norm)
    y = decoder(
        decoder_state, tgt_jnp, memory_jnp, jax.random.PRNGKey(0), tgt_mask, memory_mask
    )

    logger.debug(f"y_jax.shape = {y.shape}")
    logger.debug(f"y_torch.shape = {y_torch.shape}")

    assert np.allclose(y_torch, y, atol=TOL), f"y_torch = {y_torch}, y = {y}"


@pytest.mark.parametrize("use_mask", [False, True])
def test_transformer(use_mask):
    # TODO: add test with causal masks

    batch_size = 2
    emb_size = 4
    dropout = 0.0

    n_heads = 2
    n_layers = 2
    d_ff = 4

    target_len = CONTEXT_LEN + 1

    src = torch.randn(CONTEXT_LEN, batch_size, emb_size, requires_grad=False)
    tgt = torch.randn(target_len, batch_size, emb_size, requires_grad=False)

    # Torch
    torch_transformer = torch.nn.Transformer(
        d_model=emb_size,
        nhead=n_heads,
        num_encoder_layers=n_layers,
        num_decoder_layers=n_layers,
        dim_feedforward=d_ff,
        norm_first=True,
        dropout=dropout,
    )

    with torch.no_grad():
        y_torch = torch_transformer(src, tgt).detach().numpy()

    # Jax
    src_jnp = jnp.array(src.detach().numpy())
    tgt_jnp = jnp.array(tgt.detach().numpy())

    jax_transformer = Transformer(
        emb_size=emb_size,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
    )

    state = to_jax_state(torch_transformer)

    y_jax = jax_transformer(state, src_jnp, tgt_jnp, jax.random.PRNGKey(0))
    logger.debug(f"y_jax.shape = {y_jax.shape}")
    logger.debug(f"y_jax: {y_jax}")

    logger.debug(f"y_torch.shape = {y_torch.shape}")
    logger.debug(f"y_torch: {y_torch}")

    assert np.allclose(y_torch, y_jax, atol=TOL), f"y_torch = {y_torch}, y = {y_jax}"
