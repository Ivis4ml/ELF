"""Muon optimizer (Newton-Schulz orthogonalized momentum, K. Jordan 2024).

This is a self-contained PyTorch implementation tuned for DDP:

  * Newton-Schulz is run identically on every rank. With DDP all-reducing
    gradients in fp32 we are guaranteed bit-identical inputs and therefore
    bit-identical updates, so we do not need the sharded distributed Muon.
  * Parameters with `ndim != 2` (biases, norms, embeddings, the unembedding
    head, the time-token embeddings) fall through to AdamW under the hood.

The "muon" group of params is everything else (the matmul weights inside the
transformer blocks).
"""

from typing import Iterable, List, Tuple

import torch
from torch.optim import AdamW


@torch.no_grad()
def _newton_schulz_5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Quintic Newton-Schulz iteration; returns ~U @ V^T from SVD(G).

    Coefficients (3.4445, -4.7750, 2.0315) from K. Jordan's reference. Operates
    in bf16 for speed; the input is normalized to unit Frobenius norm so the
    iteration stays in its convergence basin.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(torch.bfloat16)
    # Normalize so that ||X||_2 <= 1.
    X = X / (X.norm() + eps)
    # The iteration converges faster on the "tall" orientation.
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T
    return X.to(G.dtype)


def _is_muon_param(p: torch.Tensor) -> bool:
    """Use Muon on 2D matmul weights only (transformer attention / FFN / bottleneck etc.)."""
    return p.ndim == 2 and p.requires_grad


def split_muon_params(model: torch.nn.Module,
                      muon_excludes: Tuple[str, ...] = (
                          "dec_proj", "dec_unembed", "t_embedder", "self_cond_cfg_embedder",
                      )) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """Partition parameters into (muon_params, adamw_params).

    Anything 2D goes to Muon by default, with a small exclusion list for the
    output head (decoder unembedding) and the timestep MLPs — heads get a
    sharper-than-ideal update from Muon's spectral normalization in practice,
    so we hand them to AdamW like the original blog post recommends.
    """
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_excluded = any(tag in name for tag in muon_excludes)
        if _is_muon_param(p) and not is_excluded:
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw


class Muon(torch.optim.Optimizer):
    """Muon with Nesterov-momentum + Newton-Schulz orthogonalization."""

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 0.02,
                 momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5,
                 weight_decay: float = 0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                if g.ndim != 2:
                    raise ValueError(
                        "Muon group received a non-2D parameter. Put 1D/Nd "
                        "params in a separate AdamW group via split_muon_params()."
                    )

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    update = g.add(buf, alpha=momentum)
                else:
                    update = buf

                update = _newton_schulz_5(update, steps=ns_steps)

                # Rectangular scale correction so that ||update||_op ~ 1.
                update = update * max(1.0, update.size(0) / update.size(1)) ** 0.5

                if wd != 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.data.add_(update, alpha=-lr)

        return loss


def build_optimizer(config, model: torch.nn.Module):
    """Return (optimizer, [param_groups]) per config.optimizer."""
    if config.optimizer == "muon":
        muon_params, adamw_params = split_muon_params(model)
        muon_opt = Muon(muon_params, lr=config.lr, momentum=0.95,
                        weight_decay=config.weight_decay)
        adamw_opt = AdamW(adamw_params, lr=config.lr,
                          betas=(config.adam_b1, config.adam_b2),
                          weight_decay=config.weight_decay)
        return _MultiOptim([muon_opt, adamw_opt])
    if config.optimizer == "adamw":
        return AdamW(model.parameters(), lr=config.lr,
                     betas=(config.adam_b1, config.adam_b2),
                     weight_decay=config.weight_decay)
    raise ValueError(f"Unknown optimizer: {config.optimizer}")


class _MultiOptim:
    """Compose multiple torch optimizers behind one `.step()` / `.zero_grad()` interface.

    Same LR can be applied across all inner optimizers via `set_lr`.
    """

    def __init__(self, optimizers: List[torch.optim.Optimizer]):
        self.optimizers = optimizers

    @property
    def param_groups(self):
        groups = []
        for opt in self.optimizers:
            groups.extend(opt.param_groups)
        return groups

    def state_dict(self):
        return {f"opt_{i}": opt.state_dict() for i, opt in enumerate(self.optimizers)}

    def load_state_dict(self, state):
        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(state[f"opt_{i}"])

    def step(self):
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def set_lr(self, lr: float):
        for opt in self.optimizers:
            for g in opt.param_groups:
                g["lr"] = lr
