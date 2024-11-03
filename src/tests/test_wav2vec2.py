from models.wav2vec2 import ConvLayerBlock
from states import to_jax_state
from attention import GroupNorm

from torchaudio.models.wav2vec2.components import ConvLayerBlock as TorchConvLayerBlock
from torch.nn import GroupNorm as TorchGroupNorm
import pytest
import torch
import jax.numpy as jnp

BATCH_SIZE = 2
LENGTH = 12

TOL = 1e-4  # Larger tolerance since we're getting a lot of numerical instability with groupnorm


@pytest.mark.parametrize(
    "in_channels, out_channels, kernel_size, stride, bias, layer_norm, length",
    [
        (3, 4, 2, 1, True, False, None),
        (1, 512, 10, 5, False, True, LENGTH),
    ],
)
def test_conv_layer_block(
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
