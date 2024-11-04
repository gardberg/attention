from models.wav2vec2 import ConvLayerBlock, FeatureExtractor, FeatureProjection
from states import to_jax_state
from attention import GroupNorm

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

    y_jax, length_jax = jax_conv_layer_block.forward(jax_state, jnp.array(x), length)

    assert y_torch.shape == y_jax.shape, f"Torch: {y_torch.shape}, Jax: {y_jax.shape}"
    print(f"Diff: {jnp.linalg.norm(y_torch - y_jax):.2e}")
    assert jnp.allclose(
        y_torch, y_jax, atol=TOL
    ), f"Diff: {jnp.linalg.norm(y_torch - y_jax):.2e}"
    if length is not None:
        assert jnp.allclose(
            length_torch, length_jax, atol=TOL
        ), f"Diff: {jnp.linalg.norm(length_torch - length_jax):.2e}"


def test_wav2vec2_feature_extractor(wav2vec2_model: Wav2Vec2Model):
    LENGTH = 500
    x = torch.randn(BATCH_SIZE, LENGTH)
    length = torch.ones(BATCH_SIZE) * LENGTH
    # Torch
    torch_feature_extractor = wav2vec2_model.feature_extractor
    jax_state = to_jax_state(torch_feature_extractor)

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

    y_jax, length_jax = jax_feature_extractor.forward(jax_state, jnp.array(x), jnp.array(length))

    assert y_torch.shape == y_jax.shape, f"Torch: {y_torch.shape}, Jax: {y_jax.shape}"
    print(f"Diff: {jnp.linalg.norm(y_torch - y_jax):.2e}")

    assert jnp.allclose(y_torch, y_jax, atol=TOL), f"Torch: {y_torch}, Jax: {y_jax}"
    assert jnp.allclose(length_torch, length_jax, atol=TOL), f"Torch: {length_torch}, Jax: {length_jax}"


def test_wav2vec2_feature_projection(wav2vec2_model: Wav2Vec2Model):
    rng = jax.random.PRNGKey(0)
    x = torch.randn(BATCH_SIZE, 512)

    torch_feature_projection = wav2vec2_model.encoder.feature_projection
    jax_state = to_jax_state(torch_feature_projection)

    with torch.no_grad():
        y_torch = torch_feature_projection(x)

    y_torch = jnp.array(y_torch)

    jax_feature_projection = FeatureProjection(512, 768)
    y_jax = jax_feature_projection.forward(jax_state, jnp.array(x), jnp.array(rng), training=False)

    assert y_torch.shape == y_jax.shape, f"Torch: {y_torch.shape}, Jax: {y_jax.shape}"
    print(f"Diff: {jnp.linalg.norm(y_torch - y_jax):.2e}")
    assert jnp.allclose(y_torch, y_jax, atol=TOL), f"Torch: {y_torch}, Jax: {y_jax}"


def test_wav2vec2_transformer(wav2vec2_model: Wav2Vec2Model):
    pass
