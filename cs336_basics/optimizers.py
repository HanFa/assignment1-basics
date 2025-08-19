from typing import Optional, Callable, overload

import torch
import math


class SGD(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data

                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss


class AdamW(torch.optim.Optimizer):

    def __init__(self, params: torch.nn.Parameter,
                 lr: float = 1e-3, betas: tuple[float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }

        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:

                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:  # state initialization
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["step"] = 1

                m = state.get("m")
                v = state.get("v")
                step = state.get("step")

                g = p.grad

                m = betas[0] * m + (1 - betas[0]) * g
                v = betas[1] * v + (1 - betas[1]) * g * g

                a_t = lr * math.sqrt(1 - betas[1] ** step) / (1 - betas[0] ** step)

                p.data -= a_t * m / (torch.sqrt(v) + eps)

                p.data = p.data * (1 - lr * weight_decay)

                # update state
                state["m"] = m
                state["v"] = v
                state["step"] = step + 1

        return loss
