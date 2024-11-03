from conv import Conv1d
from states import to_jax_state

from torch.nn.modules.conv import Conv1d as TorchConv1d
import torch
import pytest
import jax.numpy as jnp

BATCH_SIZE = 2
LENGTH = 20

torch.manual_seed(3)


@pytest.mark.parametrize(
    "in_channels, out_channels, kernel_size, stride, bias",
    [
        (3, 4, 2, 1, True),
        (3, 4, 3, 1, False),
        (3, 4, 2, 2, True),
        (3, 4, 3, 2, False),
        (1, 1, 1, 1, True),
        (1, 1, 1, 1, False),
        (1, 512, 10, 5, False),
    ],
)
def test_conv1d(in_channels, out_channels, kernel_size, stride, bias):
    x = torch.randn(BATCH_SIZE, in_channels, LENGTH)

    torch_conv = TorchConv1d(in_channels, out_channels, kernel_size, stride, bias=bias)
    with torch.no_grad():
        torch_out = torch_conv(x)
    torch_out = jnp.array(torch_out)

    jax_conv = Conv1d(in_channels, out_channels, kernel_size, stride, bias=bias)
    jax_state = to_jax_state(torch_conv)
    jax_out = jax_conv.forward(jax_state, jnp.array(x))

    assert (
        torch_out.shape == jax_out.shape
    ), f"Torch: {torch_out.shape}, Jax: {jax_out.shape}"
    print(f"Diff: {jnp.linalg.norm(torch_out - jax_out):.2e}")
    assert jnp.allclose(torch_out, jax_out), f"Torch: {torch_out}, Jax: {jax_out}"
