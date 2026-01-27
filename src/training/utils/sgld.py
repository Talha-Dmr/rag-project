"""
Stochastic Gradient Langevin Dynamics (SGLD) optimizer.
"""

from __future__ import annotations

from typing import Iterable, Optional

import math
import torch


class SGLD(torch.optim.Optimizer):
    """
    SGLD optimizer with optional weight decay and configurable noise scale.

    Update:
      theta <- theta - lr * (grad + weight_decay * theta) + N(0, 2 * lr * noise_scale)
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        noise_scale: float = 1.0
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if noise_scale < 0.0:
            raise ValueError(f"Invalid noise_scale: {noise_scale}")
        defaults = dict(lr=lr, weight_decay=weight_decay, noise_scale=noise_scale)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            noise_scale = group["noise_scale"]

            noise_std = math.sqrt(2.0 * lr * noise_scale)

            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.grad.is_sparse:
                    raise RuntimeError("SGLD does not support sparse gradients")

                grad = param.grad
                if weight_decay != 0.0:
                    grad = grad.add(param, alpha=weight_decay)

                noise = torch.randn_like(param) * noise_std
                param.add_(grad, alpha=-lr)
                param.add_(noise)

        return loss
