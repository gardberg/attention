import torch
import math
from torch import nn, Tensor, sin, pow
import torch.nn.functional as F
from torch.nn import Parameter
from torch.distributions.exponential import Exponential
from typing import NamedTuple
from utils import count_params, torch_count_params
from log_utils import logger

# Code for usage in tests
TOL = 1e-5


def get_nbr_params(
    jax_state: NamedTuple, torch_model: torch.nn.Module, debug=False
) -> tuple:
    jax_params = count_params(jax_state)
    if debug:
        logger.debug(f"jax_params: {jax_params}")
    torch_params = torch_count_params(torch_model)
    if debug:
        logger.debug(f"torch_params: {torch_params}")
    return jax_params, torch_params


# https://github.com/EdwardDixon/snake/blob/master/snake/activations.py
class TorchSnake(nn.Module):
    def __init__(self, in_features, a=None, trainable=True):
        """
        Initialization.
        Args:
            in_features: shape of the input
            a: trainable parameter
            trainable: sets `a` as a trainable parameter

            `a` is initialized to 1 by default, higher values = higher-frequency,
            5-50 is a good starting point if you already think your data is periodic,
            consider starting lower e.g. 0.5 if you think not, but don't worry,
            `a` will be trained along with the rest of your model
        """
        super(TorchSnake, self).__init__()
        self.in_features = (
            in_features if isinstance(in_features, list) else [in_features]
        )

        # Initialize `a`
        if a is not None:
            self.a = Parameter(
                torch.ones(self.in_features) * a
            )  # create a tensor out of alpha
        else:
            m = Exponential(torch.tensor([0.1]))
            self.a = Parameter(
                (m.rsample(self.in_features)).squeeze()
            )  # random init = mix of frequencies

        self.a.requiresGrad = trainable  # set the training of `a` to true

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a* sin^2 (xa)
        """
        return x + (1.0 / self.a) * pow(sin(x * self.a), 2)


class TorchPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class SwiGLU(nn.Module):
    def forward(self, x: Tensor, dim: int) -> Tensor:
        x, gate = x.chunk(2, dim=dim)
        return F.silu(gate) * x


# https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py
class TRMSNorm(nn.Module):
    def __init__(self, d, p=-1.0, eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(TRMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed
