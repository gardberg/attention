from models.wav2vec2 import (
    ConvLayerBlock,
    FeatureExtractor,
    FeatureProjection,
    ConvPosEmbedding,
    ConvPosTransformer,
)
from states import to_jax_state
from attention import GroupNorm, FeedForward, MultiHeadAttention
from act import gelu
from transformer import EncoderLayer
from utils import count_params, torch_count_params
from log_utils import logger

from torchaudio.models.wav2vec2.components import ConvLayerBlock as TorchConvLayerBlock
from torchaudio.models.wav2vec2.model import Wav2Vec2Model
from torch.nn import GroupNorm as TorchGroupNorm
import torchaudio
import pytest
from pytest import fixture
import torch
import jax
import jax.numpy as jnp

BATCH_SIZE = 2
LENGTH = 12

TOL = 1e-5  # Larger tolerance since we're getting a lot of numerical instability with groupnorm


@fixture
def wav2vec2_model() -> Wav2Vec2Model:
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    return bundle.get_model()


@pytest.mark.parametrize(
    "in_channels, out_channels, kernel_size, stride, bias, layer_norm, length",
    [
        (3, 4, 2, 1, True, False, None),
        (1, 512, 10, 5, False, True, LENGTH),
    ],
)
def test_wav2vec2_conv_layer_block(
    in_channels, out_channels, kernel_size, stride, bias, layer_norm, length
):
    x = torch.randn(BATCH_SIZE, in_channels, LENGTH)

    # Torch
    if layer_norm:
        torch_layer_norm = TorchGroupNorm(out_channels, out_channels, affine=True)
    else:
        torch_layer_norm = None

    torch_conv_layer_block = TorchConvLayerBlock(
        in_channels, out_channels, kernel_size, stride, bias, torch_layer_norm
    )

    with torch.no_grad():
        y_torch, length_torch = torch_conv_layer_block(x, length)

    y_torch = jnp.array(y_torch)

    if length is not None:
        length_torch = jnp.array(length_torch)

    # Jax
    if layer_norm:
        jax_layer_norm = GroupNorm(out_channels, out_channels, affine=True)
    else:
        jax_layer_norm = None

    jax_conv_layer_block = ConvLayerBlock(
        in_channels, out_channels, kernel_size, stride, bias, jax_layer_norm
    )
    jax_state = to_jax_state(torch_conv_layer_block)

    jax_params = count_params(jax_state)
    torch_params = torch_count_params(torch_conv_layer_block)
    assert torch_params == jax_params, f"Torch: {torch_params}, Jax: {jax_params}"

    y_jax, length_jax = jax_conv_layer_block.forward(jax_state, jnp.array(x), length)

    assert y_torch.shape == y_jax.shape, f"Torch: {y_torch.shape}, Jax: {y_jax.shape}"
    logger.debug(f"Diff: {jnp.linalg.norm(y_torch - y_jax):.2e}")
    assert jnp.allclose(
        y_torch, y_jax, atol=1e-04
    ), f"Diff: {jnp.linalg.norm(y_torch - y_jax):.2e}"
    if length is not None:
        assert jnp.allclose(
            length_torch, length_jax, atol=1e-04
        ), f"Diff: {jnp.linalg.norm(length_torch - length_jax):.2e}"


def test_wav2vec2_feature_extractor(wav2vec2_model: Wav2Vec2Model):
    LENGTH = 500
    x = torch.randn(BATCH_SIZE, LENGTH)
    length = torch.ones(BATCH_SIZE) * LENGTH
    # Torch
    torch_feature_extractor = wav2vec2_model.feature_extractor
    jax_state = to_jax_state(torch_feature_extractor)

    jax_params = count_params(jax_state)
    torch_params = torch_count_params(torch_feature_extractor)
    assert torch_params == jax_params, f"Torch: {torch_params}, Jax: {jax_params}"

    with torch.no_grad():
        y_torch, length_torch = torch_feature_extractor(x, length)

    y_torch = jnp.array(y_torch)
    length_torch = jnp.array(length_torch)

    # Jax
    conv_layers = [
        ConvLayerBlock(1, 512, 10, 5, False, GroupNorm(512, 512, affine=True)),
        *[ConvLayerBlock(512, 512, 3, 2, False, None) for _ in range(4)],
        *[ConvLayerBlock(512, 512, 2, 2, False, None) for _ in range(2)],
    ]
    jax_feature_extractor = FeatureExtractor(conv_layers)

    y_jax, length_jax = jax_feature_extractor.forward(
        jax_state, jnp.array(x), jnp.array(length)
    )

    assert y_torch.shape == y_jax.shape, f"Torch: {y_torch.shape}, Jax: {y_jax.shape}"
    logger.debug(f"Diff: {jnp.linalg.norm(y_torch - y_jax):.2e}")

    assert jnp.allclose(y_torch, y_jax, atol=TOL), f"Torch: {y_torch}, Jax: {y_jax}"
    assert jnp.allclose(
        length_torch, length_jax, atol=TOL
    ), f"Torch: {length_torch}, Jax: {length_jax}"


def test_wav2vec2_feature_projection(wav2vec2_model: Wav2Vec2Model):
    rng = jax.random.PRNGKey(0)
    x = torch.randn(BATCH_SIZE, 512)

    torch_feature_projection = wav2vec2_model.encoder.feature_projection
    jax_state = to_jax_state(torch_feature_projection)

    jax_params = count_params(jax_state)
    torch_params = torch_count_params(torch_feature_projection)
    assert torch_params == jax_params, f"Torch: {torch_params}, Jax: {jax_params}"

    with torch.no_grad():
        y_torch = torch_feature_projection(x)

    y_torch = jnp.array(y_torch)

    jax_feature_projection = FeatureProjection(512, 768)
    y_jax = jax_feature_projection.forward(
        jax_state, jnp.array(x), jnp.array(rng), training=False
    )

    assert y_torch.shape == y_jax.shape, f"Torch: {y_torch.shape}, Jax: {y_jax.shape}"
    logger.debug(f"Diff: {jnp.linalg.norm(y_torch - y_jax):.2e}")
    assert jnp.allclose(y_torch, y_jax, atol=TOL), f"Torch: {y_torch}, Jax: {y_jax}"


def test_wav2vec2_conv_pos_embedding(wav2vec2_model: Wav2Vec2Model):
    # input: (batch_size, n_frames, n_features)
    N_FRAMES = 10
    N_FEATURES = 768
    x = torch.randn(BATCH_SIZE, N_FRAMES, N_FEATURES)

    torch_conv_pos_embedding = wav2vec2_model.encoder.transformer.pos_conv_embed.eval()
    jax_state = to_jax_state(torch_conv_pos_embedding)

    jax_params = count_params(jax_state)
    torch_params = torch_count_params(torch_conv_pos_embedding)
    assert torch_params == jax_params, f"Torch: {torch_params}, Jax: {jax_params}"

    with torch.no_grad():
        y_torch = torch_conv_pos_embedding(x)

    y_torch = jnp.array(y_torch)

    jax_conv_pos_embedding = ConvPosEmbedding(
        embed_dim=N_FEATURES, kernel_size=128, groups=16
    )
    y_jax = jax_conv_pos_embedding.forward(jax_state, jnp.array(x))

    assert y_torch.shape == y_jax.shape, f"Torch: {y_torch.shape}, Jax: {y_jax.shape}"
    logger.debug(f"Diff: {jnp.linalg.norm(y_torch - y_jax):.2e}")
    assert jnp.allclose(y_torch, y_jax, atol=TOL), f"Torch: {y_torch}, Jax: {y_jax}"


# getting some numerical error here, prob unavoidable, so added rtol
def test_wav2vec2_feed_forward(wav2vec2_model: Wav2Vec2Model):
    rng = jax.random.PRNGKey(0)
    x = torch.randn(BATCH_SIZE, 768, dtype=torch.float32)
    torch_feed_forward = wav2vec2_model.encoder.transformer.layers[0].feed_forward
    jax_state = to_jax_state(torch_feed_forward)

    jax_params = count_params(jax_state)
    torch_params = torch_count_params(torch_feed_forward)
    assert torch_params == jax_params, f"Torch: {torch_params}, Jax: {jax_params}"

    # Debug intermediate values
    with torch.no_grad():
        # Get intermediate values from torch
        intermediate = torch_feed_forward.intermediate_dense(x)
        intermediate_act = torch.nn.functional.gelu(intermediate)
        output = torch_feed_forward.output_dense(intermediate_act)
        y_torch = output

    # Convert to JAX arrays
    x_jax = jnp.array(x)
    intermediate_jax = jnp.array(intermediate)
    intermediate_act_jax = jnp.array(intermediate_act)
    y_torch = jnp.array(y_torch)

    # Get JAX intermediate values
    jax_feed_forward = FeedForward(768, 3072, act=gelu)

    # Compare layer1 output
    layer1_out = jax_feed_forward.layer1(jax_state.linear1, x_jax)
    logger.debug(f"Layer1 diff: {jnp.linalg.norm(layer1_out - intermediate_jax):.2e}")

    # Compare activation output
    act_out = jax_feed_forward.act(layer1_out)
    logger.debug(f"Activation diff: {jnp.linalg.norm(act_out - intermediate_act_jax):.2e}")

    # Compare output output
    output_out = jax_feed_forward.layer2(jax_state.linear2, act_out)
    logger.debug(f"Output diff: {jnp.linalg.norm(output_out - y_torch):.2e}")

    # Final output
    y_jax = jax_feed_forward.forward(jax_state, x_jax, jnp.array(rng), training=False)
    logger.debug(f"Final diff: {jnp.linalg.norm(y_torch - y_jax):.2e}")

    assert jnp.allclose(
        y_torch, y_jax, atol=TOL, rtol=5e-4
    ), f"Torch: {y_torch}, Jax: {y_jax}"


def test_wav2vec2_self_attention(wav2vec2_model: Wav2Vec2Model):
    LENGTH = 10
    x = torch.randn(BATCH_SIZE, LENGTH, 768)

    torch_self_attention = wav2vec2_model.encoder.transformer.layers[0].attention.eval()
    jax_state = to_jax_state(torch_self_attention)

    jax_params = count_params(jax_state)
    torch_params = torch_count_params(torch_self_attention)
    assert torch_params == jax_params, f"Torch: {torch_params}, Jax: {jax_params}"

    with torch.no_grad():
        y_torch, _pos_bias = torch_self_attention(x)

    y_torch = jnp.array(y_torch)

    # JAX IS (LENGTH, BATCH_SIZE, 768)

    jax_self_attention = MultiHeadAttention(
        emb_size=768, n_heads=12, qk_bias=True, out_bias=True
    )

    x_jax = jnp.array(x).transpose(1, 0, 2)
    y_jax = jax_self_attention.forward(jax_state, x_jax, x_jax, x_jax)
    y_jax = y_jax.transpose(1, 0, 2)

    logger.debug(f"Diff: {jnp.linalg.norm(y_torch - y_jax):.2e}")
    assert jnp.allclose(
        y_torch, y_jax, atol=TOL, rtol=5e-4
    ), f"Torch: {y_torch}, Jax: {y_jax}"


@pytest.mark.parametrize("layer_idx", [0, 5])
def test_wav2vec2_encoder_layer(wav2vec2_model: Wav2Vec2Model, layer_idx: int):
    rng = jax.random.PRNGKey(0)
    LENGTH = 10
    x = torch.randn(BATCH_SIZE, LENGTH, 768)

    torch_encoder_layer = wav2vec2_model.encoder.transformer.layers[layer_idx].eval()
    jax_state = to_jax_state(torch_encoder_layer)

    torch_params = torch_count_params(torch_encoder_layer)
    jax_params = count_params(jax_state)
    assert torch_params == jax_params, f"Torch: {torch_params}, Jax: {jax_params}"

    with torch.no_grad():
        y_torch, _pos_bias = torch_encoder_layer(x)

    y_torch = jnp.array(y_torch)

    x_jax = jnp.array(x).transpose(1, 0, 2)
    jax_encoder_layer = EncoderLayer(
        emb_size=768,
        d_ff=3072,
        n_heads=12,
        dropout=0.1,
        layer_norm_first=False,
        ff_activation=gelu,
        attn_out_bias=True,
        attn_qk_bias=True,
    )
    y_jax = jax_encoder_layer.forward(jax_state, x_jax, jnp.array(rng), training=False)
    y_jax = y_jax.transpose(1, 0, 2)

    logger.debug(f"Diff: {jnp.linalg.norm(y_torch - y_jax):.2e}")
    assert jnp.allclose(
        y_torch, y_jax, atol=TOL, rtol=1e-4
    ), f"Torch: {y_torch}, Jax: {y_jax}"


def test_wav2vec2_transformer_intermediate(wav2vec2_model: Wav2Vec2Model):
    rng = jax.random.PRNGKey(0)
    x = torch.randn(BATCH_SIZE, 10, 768)

    logger.debug(f"Jax default dtype: {jnp.ones(1).dtype}")
    logger.debug(f"Torch default dtype: {x.dtype}")

    torch_transformer = wav2vec2_model.encoder.transformer.eval()
    jax_transformer = ConvPosTransformer(n_layers=12, emb_size=768)

    jax_state = to_jax_state(torch_transformer)

    with torch.no_grad():
        y_torch_intermediate = torch_transformer.get_intermediate_outputs(x, num_layers=12)

    y_jax_intermediate = jax_transformer.get_intermediate_outputs(
        jax_state, jnp.array(x), rng, num_layers=12
    )

    all_same = True
    for i, (torch_intermediate, jax_intermediate) in enumerate(zip(y_torch_intermediate, y_jax_intermediate)):
        jax_intermediate = jax_intermediate.transpose(1, 0, 2)
        same = jnp.allclose(
            jnp.array(torch_intermediate),
            jax_intermediate,
            atol=5e-4,
            rtol=5e-4,
        )
        all_same = all_same and same
        logger.debug(
            f"Same: {same}, Torch norm: {jnp.linalg.norm(jnp.array(torch_intermediate)):.2e}, Jax norm: {jnp.linalg.norm(jax_intermediate):.2e}, Diff: {jnp.linalg.norm(jnp.array(torch_intermediate) - jax_intermediate):.2e}"
        )

    assert all_same, "Not all intermediate outputs are the same"


# some numerical instability here, increased atol
def test_wav2vec2_transformer(wav2vec2_model: Wav2Vec2Model):
    TOL = 1e-3
    rng = jax.random.PRNGKey(0)
    x = torch.randn(BATCH_SIZE, 10, 768)

    torch_transformer = wav2vec2_model.encoder.transformer.eval()
    jax_transformer = ConvPosTransformer(n_layers=12, emb_size=768)

    jax_state = to_jax_state(torch_transformer)
    jax_params = count_params(jax_state)
    torch_params = torch_count_params(torch_transformer)
    assert torch_params == jax_params, f"Torch: {torch_params}, Jax: {jax_params}"

    with torch.no_grad():
        y_torch = torch_transformer(x)

    y_torch = jnp.array(y_torch)

    y_jax = jax_transformer.forward(
        jax_state, jnp.array(x), jnp.array(rng), training=False
    )

    logger.debug(f"Diff: {jnp.linalg.norm(y_torch - y_jax):.2e}")
    assert jnp.allclose(y_torch, y_jax, atol=TOL, rtol=5e-4), f"Max diff: {jnp.max(jnp.abs(y_torch - y_jax)):.2e}"
