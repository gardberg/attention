import jax.numpy as jnp
from typing import Callable
from base import Array


class Loss:
    def __init__(self, reduction: str = "mean"):
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Reduction must be one of 'mean', 'sum', or 'none', got {reduction}"
            )
        self.reduction = reduction
        self.reduction_fn = self.get_reduction_fn()

    def get_reduction_fn(self) -> Callable:
        if self.reduction == "mean":
            return jnp.mean
        elif self.reduction == "sum":
            return jnp.sum
        else:
            return lambda x: x


class MSELoss(Loss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)

    def __call__(self, input: Array, target: Array):
        return self.reduction_fn(jnp.square(input - target))


class BCELoss(Loss):
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction)

    def __call__(self, input: Array, target: Array):
        return self.reduction_fn(
            -(target * jnp.log(input) + (1 - target) * jnp.log(1 - input))
        )
