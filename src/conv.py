from base import BaseModule, Array
from states import Conv1dState
import jax.numpy as jnp
import jax.random as random
import jax


# NOTE: No dilation
class Conv1d(BaseModule):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups # Number of blocked connections from input channels to output channels
        self.bias = bias

        assert self.in_channels % self.groups == 0, "in_channels must be divisible by groups"
        assert self.out_channels % self.groups == 0, "out_channels must be divisible by groups"

        self.sqrt_k = jnp.sqrt(self.groups / (self.in_channels * self.kernel_size))
        self.weight_shape = (self.out_channels, self.in_channels // self.groups, self.kernel_size)

    def forward(
        self, state: Conv1dState, x: Array["batch_size, in_channels, in_length"]
    ) -> Array["batch_size, out_channels, out_length"]:
        batch_size, in_channels, in_length = x.shape

        if self.padding > 0:
            pad_width = [(0, 0), (0, 0), (self.padding, self.padding)]
            x = jnp.pad(x, pad_width, mode='constant', constant_values=0)

        # added length by padding is here included in x.shape[-1]
        out_length = (x.shape[-1] - self.kernel_size) // self.stride + 1
        
        indices = jnp.arange(out_length) * self.stride
        windows = jnp.stack(
            [x[:, :, i : i + self.kernel_size] for i in indices], axis=1
        )
        # windows shape: (batch_size, out_length, in_channels, kernel_size)

        # Split windows into groups along the channel dimension
        channels_per_group = self.in_channels // self.groups
        windows = windows.reshape(batch_size, out_length, self.groups, channels_per_group, self.kernel_size)
        
        # Split weights into groups
        out_channels_per_group = self.out_channels // self.groups
        weights = state.weight.reshape(self.groups, out_channels_per_group, channels_per_group, self.kernel_size)

        # Define the convolution for a single group
        def conv_group(group_w, group_x):
            return jnp.einsum('boik,cik->bco', group_x, group_w)

        # Apply convolution to each group using vmap
        # Map over dimension 2 of windows (groups) and dimension 0 of weights (groups)
        grouped_output = jax.vmap(conv_group, in_axes=(0, 2))(weights, windows)
        # grouped_output shape: (groups, batch_size, out_channels_per_group, out_length)

        # Reshape and combine group outputs
        output = grouped_output.transpose(1, 0, 2, 3).reshape(batch_size, self.out_channels, out_length)

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
