from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Callable, Dict, Any, Tuple
from jax.numpy import ndarray

class Array(ndarray, Generic[TypeVar("Shape")]):
    ...


class BaseModule(ABC):
    def __init__(self):
        self.use_jit = False
        self._forward_hooks: List[Callable[..., None]] = []

    def set_use_jit(self, use_jit: bool):
        self.use_jit = use_jit

    @abstractmethod
    def forward(self, *args, **kwargs) -> Array:
        ...

    def register_forward_hook(
        self, hook: Callable[[Any, Tuple[tuple, Dict[str, Any]], Array], None]
    ) -> int:
        r"""
        Register a forward hook that will be called after the forward pass.
        The hook should have the signature:

            hook(module, input, output) -> None

        Returns the hook id which can be used to remove the hook later.
        """
        self._forward_hooks.append(hook)
        return len(self._forward_hooks) - 1

    def remove_forward_hook(self, hook_id: int):
        if hook_id < 0 or hook_id >= len(self._forward_hooks):
            raise ValueError(f"Invalid hook_id: {hook_id}")

        self._forward_hooks.pop(hook_id)

    def clear_forward_hooks(self):
        self._forward_hooks.clear()

    def _run_forward_hooks(self, args: tuple, output: Array):
        hook_input = args[0] if len(args) == 1 else args
        for hook in self._forward_hooks:
            if hook is not None:
                hook(self, hook_input, output)

    def __call__(self, *args, **kwargs) -> Array:
        if hasattr(self, "_jit_forward") and self.use_jit:
            output = self._jit_forward(*args, **kwargs)
        else:
            output = self.forward(*args, **kwargs)

        self._run_forward_hooks(args, output)
        return output
