from __future__ import annotations

import math
from typing import Iterable

import torch


def zeropower_via_newtonschulz5(update: torch.Tensor, steps: int = 5) -> torch.Tensor:
    original_shape = update.shape
    compute_dtype = torch.bfloat16 if update.device.type == "cuda" else torch.float32
    matrix = update.reshape(update.shape[0], -1).to(dtype=compute_dtype)
    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.T
    matrix = matrix / (matrix.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        gram = matrix @ matrix.T
        matrix = a * matrix + (b * gram + c * gram @ gram) @ matrix
    if transposed:
        matrix = matrix.T
    return matrix.reshape(original_shape).to(dtype=update.dtype)


class Muon(torch.optim.Optimizer):
    """Small self-contained Muon optimizer for matrix-like hidden weights.

    Biases, norms, embeddings, and heads should usually be optimized by AdamW
    in a separate parameter group. This implementation is intentionally compact
    and works with DDP because gradients have already been all-reduced by the
    time ``step`` runs.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        nesterov: bool = True,
        ns_steps: int = 5,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if weight_decay != 0:
                    param.mul_(1.0 - lr * weight_decay)
                grad = param.grad
                state = self.state[param]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad)
                update = grad.add(buf, alpha=momentum) if group["nesterov"] else buf
                if update.ndim >= 2:
                    update = zeropower_via_newtonschulz5(update, steps=group["ns_steps"])
                    rows = update.reshape(update.shape[0], -1).shape[0]
                    cols = update.reshape(update.shape[0], -1).shape[1]
                    update = update * math.sqrt(max(1.0, rows / max(cols, 1)))
                param.add_(update, alpha=-lr)
        return loss


def split_muon_adamw_params(
    model: torch.nn.Module,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    adamw_excludes = (
        "proj_kernel",
        "unembed",
        "t_embedder",
        "self_cond_cfg_embedder",
    )
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 2 and not any(fragment in name for fragment in adamw_excludes):
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params


def build_optimizers(
    model: torch.nn.Module,
    *,
    optimizer: str,
    lr: float,
    weight_decay: float,
    adam_b1: float,
    adam_b2: float,
) -> list[torch.optim.Optimizer]:
    if optimizer == "adamw":
        return [
            torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=lr,
                weight_decay=weight_decay,
                betas=(adam_b1, adam_b2),
            )
        ]
    if optimizer == "muon":
        muon_params, adamw_params = split_muon_adamw_params(model)
        optimizers: list[torch.optim.Optimizer] = []
        if muon_params:
            optimizers.append(Muon(muon_params, lr=lr, weight_decay=weight_decay))
        if adamw_params:
            optimizers.append(
                torch.optim.AdamW(
                    adamw_params,
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=(adam_b1, adam_b2),
                )
            )
        return optimizers
    raise ValueError(f"Unknown optimizer: {optimizer}. Choose 'adamw' or 'muon'.")


def set_optimizer_lr(optimizers: list[torch.optim.Optimizer], lr: float) -> None:
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            group["lr"] = lr


def get_lr(
    step: int,
    *,
    total_steps: int,
    warmup_steps: int,
    learning_rate: float,
    schedule: str = "constant",
    min_lr: float = 0.0,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return learning_rate * float(step + 1) / float(warmup_steps)
    if schedule == "cosine":
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (learning_rate - min_lr) * cosine
    return learning_rate
