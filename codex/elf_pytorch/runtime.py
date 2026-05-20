from __future__ import annotations

import os
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


def init_distributed() -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def log0(message: str) -> None:
    if is_main_process():
        print(message, flush=True)


def seed_everything(seed: int, rank: int = 0) -> None:
    # Per-rank RNGs should differ for data noise/dropout. Decisions that change
    # the global autograd graph, such as denoiser-vs-decoder branch selection,
    # must be broadcast separately.
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def broadcast_decoder_branch(
    decoder_prob: float,
    device: torch.device,
    generator: torch.Generator,
) -> bool:
    """Sample a decoder/denoiser branch on rank 0 and broadcast it to all ranks."""
    flag = torch.zeros(1, dtype=torch.uint8, device=device)
    if is_main_process():
        flag.fill_(1 if torch.rand((), generator=generator, device=device).item() < decoder_prob else 0)
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(flag, src=0)
    return bool(flag.item())


def reduce_metrics(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    reduced = {}
    for key, value in metrics.items():
        tensor = value.detach().float()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
        reduced[key] = float(tensor.item())
    return reduced


class EMA:
    def __init__(
        self,
        model: torch.nn.Module,
        decay: float,
        *,
        device: torch.device | str,
    ) -> None:
        self.decay = decay
        self.device = torch.device(device) if isinstance(device, str) else device
        self.shadow = [
            param.detach().to(self.device).clone()
            for param in model.parameters()
            if param.requires_grad
        ]

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        params = [param for param in model.parameters() if param.requires_grad]
        if len(params) != len(self.shadow):
            raise ValueError(f"EMA size mismatch: {len(params)} params vs {len(self.shadow)} buffers.")
        for target, param in zip(self.shadow, params):
            source = param.detach().to(device=target.device, dtype=target.dtype)
            target.mul_(self.decay).add_(source, alpha=1.0 - self.decay)

    def state_dict(self) -> dict[str, Any]:
        return {"decay": self.decay, "shadow": [tensor.cpu() for tensor in self.shadow]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if "shadow" in state:
            loaded = state["shadow"]
            self.decay = float(state.get("decay", self.decay))
        else:
            # Backward compatibility with the original name-keyed EMA checkpoints.
            loaded = list(state.values())
        if len(loaded) != len(self.shadow):
            raise ValueError(f"EMA size mismatch: checkpoint has {len(loaded)} buffers, model has {len(self.shadow)}.")
        self.shadow = [tensor.to(self.device) for tensor in loaded]

    @contextmanager
    def swapped(self, model: torch.nn.Module):
        params = [param for param in model.parameters() if param.requires_grad]
        if len(params) != len(self.shadow):
            raise ValueError(f"EMA size mismatch: {len(params)} params vs {len(self.shadow)} buffers.")
        backup = [param.detach().clone() for param in params]
        try:
            for param, shadow in zip(params, self.shadow):
                param.data.copy_(shadow.to(device=param.device, dtype=param.dtype))
            yield
        finally:
            for param, saved in zip(params, backup):
                param.data.copy_(saved.to(device=param.device, dtype=param.dtype))


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizers: list[torch.optim.Optimizer],
    ema: EMA | None,
    epoch: int,
    step: int,
    config: Any,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": unwrap_model(model).state_dict(),
        "optimizers": [optimizer.state_dict() for optimizer in optimizers],
        "ema": ema.state_dict() if ema is not None else None,
        "epoch": epoch,
        "step": step,
        "config": config,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizers: list[torch.optim.Optimizer] | None = None,
    ema: EMA | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[int, int]:
    payload = torch.load(path, map_location=map_location)
    unwrap_model(model).load_state_dict(payload["model"])
    if optimizers is not None and "optimizers" in payload:
        for optimizer, state in zip(optimizers, payload["optimizers"]):
            optimizer.load_state_dict(state)
    if ema is not None and payload.get("ema") is not None:
        ema.load_state_dict(payload["ema"])
    return int(payload.get("epoch", 0)), int(payload.get("step", 0))
