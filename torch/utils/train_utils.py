"""Training-time helpers: LR schedule + EMA shadow."""

import copy
import math
from typing import Dict, Iterable

import torch
import torch.nn as nn


class LRSchedule:
    """Linear warmup -> (constant | cosine) decay.

    Step semantics are *optimizer* steps (i.e., what `optimizer.step()` is called with),
    not micro-batch steps.
    """

    def __init__(self, num_train_steps: int, num_warmup_steps: int, lr: float,
                 schedule: str = "constant", min_lr: float = 0.0):
        self.num_train_steps = num_train_steps
        self.num_warmup_steps = max(0, int(num_warmup_steps))
        self.lr = lr
        self.schedule = schedule
        self.min_lr = min_lr

    def __call__(self, step: int) -> float:
        if step < self.num_warmup_steps and self.num_warmup_steps > 0:
            return self.lr * step / self.num_warmup_steps
        if self.schedule == "constant":
            return self.lr
        if self.schedule == "cosine":
            decay_steps = max(1, self.num_train_steps - self.num_warmup_steps)
            t = (step - self.num_warmup_steps) / decay_steps
            t = min(max(t, 0.0), 1.0)
            return self.min_lr + 0.5 * (self.lr - self.min_lr) * (1.0 + math.cos(math.pi * t))
        raise ValueError(f"Unknown schedule {self.schedule}")


class ParamEMA:
    """Exponential moving average kept on the same device as the source params.

    Mirrors `state.ema_params1` in the JAX TrainState. Use `update()` only on
    optimizer-step boundaries, never on micro-batch grad-accumulation steps.
    """

    def __init__(self, params: Iterable[nn.Parameter], decay: float = 0.9999):
        self.decay = decay
        self.shadow: Dict[int, torch.Tensor] = {}
        for p in params:
            if p.requires_grad:
                self.shadow[id(p)] = p.detach().clone()
        self._params_ref = [p for p in params if p.requires_grad]

    @torch.no_grad()
    def update(self, params: Iterable[nn.Parameter]):
        for p in params:
            if not p.requires_grad:
                continue
            buf = self.shadow.get(id(p))
            if buf is None:
                continue
            buf.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {"decay": self.decay, "shadow": [v.detach().cpu() for v in self.shadow.values()],
                "param_order_ids": list(self.shadow.keys())}

    def load_state_dict(self, state: dict, params: Iterable[nn.Parameter]):
        # Load by positional order — IDs change across processes.
        params = [p for p in params if p.requires_grad]
        shadows = state["shadow"]
        if len(params) != len(shadows):
            raise ValueError(
                f"EMA state has {len(shadows)} tensors but model has {len(params)} params"
            )
        self.decay = state.get("decay", self.decay)
        self.shadow = {id(p): s.to(p.device, dtype=p.dtype).detach().clone()
                       for p, s in zip(params, shadows)}

    def swap_into(self, params: Iterable[nn.Parameter]):
        """Replace `params.data` with EMA values; return originals for restore_into()."""
        params = [p for p in params if p.requires_grad]
        backup = [p.detach().clone() for p in params]
        for p in params:
            buf = self.shadow.get(id(p))
            if buf is not None:
                p.data.copy_(buf)
        return backup

    def restore_into(self, params: Iterable[nn.Parameter], backup):
        params = [p for p in params if p.requires_grad]
        for p, b in zip(params, backup):
            p.data.copy_(b)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def clip_grad_norm_(params, max_norm: float):
    return torch.nn.utils.clip_grad_norm_(params, max_norm)


def maybe_unwrap_ddp(model: nn.Module) -> nn.Module:
    """Strip DDP wrapper for direct attribute / forward access."""
    if hasattr(model, "module"):
        return model.module
    return model
