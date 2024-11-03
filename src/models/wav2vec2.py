from base import BaseModule, Array
from conv import Conv1d
from states import ConvLayerBlockState
from typing import Optional, Tuple
from act import gelu
import jax.numpy as jnp


class Wav2Vec2(BaseModule):
    def __init__(self):
        super().__init__()

        self.sample_rate = 16000

    def forward(self):
        pass

        # Raw waveform ->

        # CNN Feature Extractor

        # Transformer Encoder

        # Quantization (?)

        # -> Embeddings

        # Classification Head

        # -> Logits


class ConvLayerBlock(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool,
        layer_norm: Optional[BaseModule] = None,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv = Conv1d(in_channels, out_channels, kernel_size, stride, bias=bias)
        self.layer_norm = layer_norm

    def forward(
        self,
        state: ConvLayerBlockState,
        x: Array["batch_size, in_channels, in_length"],
        length: Optional[Array["batch_size,"]] = None,  # length of each sequence
    ) -> Tuple[
        Array["batch_size, out_channels, out_length"], Optional[Array["batch_size,"]]
    ]:

        x = self.conv(state.conv, x)
        if self.layer_norm is not None:
            x = self.layer_norm(state.layer_norm, x)
        x = gelu(x)

        # calculate resulting length
        if length is not None:
            length = (length - self.kernel_size) // self.stride + 1
            length = jnp.maximum(
                length, jnp.zeros_like(length)
            )  # make sure we dont get negative lengths

        return x, length
