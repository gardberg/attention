from base import BaseModule, Array
from states import Conv1dState
import jax.numpy as jnp
import jax.random as random


# TODO: Add padding, dilation, groups
class Conv1d(BaseModule):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias

        self.sqrt_k = jnp.sqrt(1 / (self.in_channels * self.kernel_size))
        self.weight_shape = (self.out_channels, self.in_channels, self.kernel_size)

    def forward(
        self, state: Conv1dState, x: Array["batch_size, in_channels, in_length"]
    ) -> Array["batch_size, out_channels, out_length"]:
        # Calculate output length
        out_length = (x.shape[-1] - self.kernel_size) // self.stride + 1

        # Extract strided windows (batch_size, out_length, in_channels, kernel_size)
        indices = jnp.arange(out_length) * self.stride
        windows = jnp.stack(
            [x[:, :, i : i + self.kernel_size] for i in indices], axis=1
        )

        # Compute convolution using einsum
        # windows: (batch_size, out_length, in_channels, kernel_size)
        # weight: (out_channels, in_channels, kernel_size)
        # output: (batch_size, out_channels, out_length)
        output = jnp.einsum("boik,cik->bco", windows, state.weight)

        return output + state.bias[None, :, None] if self.bias else output

    def init_state(self, rng: Array) -> Conv1dState:
        # pytorch initialization
        rngs = random.split(rng, 2)
        return Conv1dState(
            weight=random.uniform(
                rngs[0], self.weight_shape, minval=-self.sqrt_k, maxval=self.sqrt_k
            ),
            bias=(
                random.uniform(
                    rngs[1],
                    (self.out_channels,),
                    minval=-self.sqrt_k,
                    maxval=self.sqrt_k,
                )
                if self.bias
                else None
            ),
        )
