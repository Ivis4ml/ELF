from __future__ import annotations

import torch


def add_noise(
    x0: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
    denoiser_noise_scale: float,
    cond_seq_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    t_expanded = t.view(-1, 1, 1)
    z = t_expanded * x0 + (1.0 - t_expanded) * noise * denoiser_noise_scale
    if cond_seq_mask is not None:
        z = cond_seq_mask * x0 + (1.0 - cond_seq_mask) * z
    return z


def sample_timesteps(
    batch_size: int,
    *,
    device: torch.device,
    p_mean: float = -0.8,
    p_std: float = 0.8,
    time_schedule: str = "logit_normal",
) -> torch.Tensor:
    if time_schedule == "logit_normal":
        z = torch.randn(batch_size, device=device) * p_std + p_mean
        return torch.sigmoid(z)
    if time_schedule == "uniform":
        return torch.rand(batch_size, device=device)
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def get_sampling_steps(
    n_steps: int,
    *,
    device: torch.device,
    time_schedule: str = "logit_normal",
    p_mean: float = -0.8,
    p_std: float = 0.8,
) -> torch.Tensor:
    if time_schedule == "uniform":
        return torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    if time_schedule == "logit_normal":
        steps = sample_timesteps(
            max(n_steps - 1, 0),
            device=device,
            p_mean=p_mean,
            p_std=p_std,
            time_schedule=time_schedule,
        )
        return torch.cat(
            [torch.zeros(1, device=device), steps.sort().values, torch.ones(1, device=device)]
        )
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def sample_cfg_scale(
    batch_size: int,
    *,
    device: torch.device,
    cfg_min: float = 0.0,
    cfg_max: float = 3.0,
) -> torch.Tensor:
    u = torch.rand(batch_size, device=device)
    a = torch.tensor(1.0 + cfg_min, device=device)
    b = torch.tensor(1.0 + cfg_max, device=device)
    return a * torch.exp(u * torch.log(b / a)) - 1.0


def restore_cond(
    z_updated: torch.Tensor,
    cond_seq: torch.Tensor,
    cond_seq_mask: torch.Tensor,
) -> torch.Tensor:
    mask = cond_seq_mask
    while mask.ndim < max(z_updated.ndim, cond_seq.ndim):
        mask = mask.unsqueeze(-1)
    return torch.where(mask > 0, cond_seq, z_updated)


def restore_vx(
    v: torch.Tensor,
    x: torch.Tensor,
    cond_seq: torch.Tensor | None,
    cond_seq_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cond_seq is not None and cond_seq_mask is not None:
        x = restore_cond(x, cond_seq, cond_seq_mask)
        v = restore_cond(v, torch.zeros_like(cond_seq), cond_seq_mask)
    return v, x


def net_out_to_v_x(
    net_out: torch.Tensor,
    z: torch.Tensor,
    t: torch.Tensor,
    t_eps: float = 5e-2,
) -> tuple[torch.Tensor, torch.Tensor]:
    t_reshaped = t.view(-1, 1, 1)
    x = net_out
    v = (x - z) / torch.clamp(1.0 - t_reshaped, min=t_eps)
    return v, x
