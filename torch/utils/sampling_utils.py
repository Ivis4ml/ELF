"""Noise schedules, ODE/SDE samplers, and the CFG-aware forward passes.

Mirrors src/utils/sampling_utils.py.
"""

from typing import Optional, Tuple

import torch


def sample_timesteps(B: int, P_mean: float, P_std: float,
                     time_schedule: str = "logit_normal",
                     device: Optional[torch.device] = None,
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
    if time_schedule == "logit_normal":
        z = torch.randn(B, device=device, generator=generator) * P_std + P_mean
        return torch.sigmoid(z)
    if time_schedule == "uniform":
        return torch.rand(B, device=device, generator=generator)
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def get_sampling_steps(n_steps: int, time_schedule: str = "logit_normal",
                       P_mean: float = -1.5, P_std: float = 0.8,
                       device: Optional[torch.device] = None,
                       generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Length-(n_steps+1) tensor of t-values in [0, 1].

    - "uniform": linspace(0, 1, n_steps+1)
    - "logit_normal": n_steps-1 sorted logit-normal samples, with 0 and 1 prepended/appended.
    """
    if time_schedule == "uniform":
        return torch.linspace(0.0, 1.0, n_steps + 1, device=device)
    if time_schedule == "logit_normal":
        mid = sample_timesteps(
            n_steps - 1, P_mean=P_mean, P_std=P_std,
            time_schedule="logit_normal", device=device, generator=generator,
        ).sort().values
        ends = torch.tensor([0.0, 1.0], device=device, dtype=mid.dtype)
        return torch.cat([ends[:1], mid, ends[1:]])
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def sample_cfg_scale(B: int, cfg_min: float = 0.5, cfg_max: float = 5.0,
                     device: Optional[torch.device] = None,
                     generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Log-uniform on [1+cfg_min, 1+cfg_max] minus 1 — biased toward smaller scales."""
    u = torch.rand(B, device=device, generator=generator)
    a = torch.tensor(1.0 + cfg_min, device=device, dtype=u.dtype)
    b = torch.tensor(1.0 + cfg_max, device=device, dtype=u.dtype)
    return a * torch.exp(u * torch.log(b / a)) - 1.0


def add_noise(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor, config,
              cond_seq_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    t_exp = t.reshape(-1, 1, 1)
    z = t_exp * x0 + (1.0 - t_exp) * noise * config.denoiser_noise_scale
    if cond_seq_mask is not None:
        z = cond_seq_mask * x0 + (1.0 - cond_seq_mask) * z
    return z


def restore_cond(z_updated: torch.Tensor, cond_seq: torch.Tensor,
                 cond_seq_mask: torch.Tensor) -> torch.Tensor:
    mask = cond_seq_mask
    target_ndim = max(z_updated.ndim, cond_seq.ndim)
    while mask.ndim < target_ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask > 0, cond_seq, z_updated)


def restore_vx(v: torch.Tensor, x: torch.Tensor,
               cond_seq: Optional[torch.Tensor],
               cond_seq_mask: Optional[torch.Tensor]):
    if cond_seq is not None:
        x = restore_cond(x, cond_seq, cond_seq_mask)
        v = restore_cond(v, torch.zeros_like(cond_seq), cond_seq_mask)
    return v, x


def net_out_to_v_x(net_out, z: torch.Tensor, t: torch.Tensor, t_eps: float = 5e-2):
    if isinstance(net_out, tuple):
        net_out = net_out[0]
    t_exp = t.reshape(-1, 1, 1)
    x = net_out
    v = (x - z) / torch.clamp(1.0 - t_exp, min=t_eps)
    return v, x


# ---------------- inference helpers (no grad) ----------------

@torch.no_grad()
def _forward_sample_self_cond(
    model, z, t_batch, x_pred_prev, config, self_cond_cfg_scale,
    cond_seq, cond_seq_mask,
):
    t_eps = config.t_eps
    self_cond_prob = config.self_cond_prob

    if config.num_self_cond_cfg_tokens > 0:
        if x_pred_prev is None:
            x_pred_prev = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
        z_in = torch.cat([z, x_pred_prev], dim=-1)
        sc_scale = torch.full((z.shape[0],), float(self_cond_cfg_scale),
                              device=z.device, dtype=z.dtype)
        net_out = model(z_in, t_batch, self_cond_cfg_scale=sc_scale)
        v, x = net_out_to_v_x(net_out, z, t_batch, t_eps)
        return restore_vx(v, x, cond_seq, cond_seq_mask)

    # No CFG conditioning token.
    if self_cond_prob == 0:
        net_out = model(z, t_batch)
        v, x = net_out_to_v_x(net_out, z, t_batch, t_eps)
        return restore_vx(v, x, cond_seq, cond_seq_mask)

    if self_cond_cfg_scale != 1 or x_pred_prev is None:
        z_uncond = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
        z_in_uncond = torch.cat([z, z_uncond], dim=-1)
        net_out_uncond = model(z_in_uncond, t_batch)
        v_uncond, x_uncond = net_out_to_v_x(net_out_uncond, z, t_batch, t_eps)
        v_uncond, x_uncond = restore_vx(v_uncond, x_uncond, cond_seq, cond_seq_mask)
        if self_cond_cfg_scale == 0.0 or x_pred_prev is None:
            return v_uncond, x_uncond

    z_in_cond = torch.cat([z, x_pred_prev], dim=-1)
    net_out_cond = model(z_in_cond, t_batch)
    v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, t_eps)
    v_cond, x_cond = restore_vx(v_cond, x_cond, cond_seq, cond_seq_mask)
    if self_cond_cfg_scale == 1:
        return v_cond, x_cond

    v_out = v_uncond + self_cond_cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + self_cond_cfg_scale * (x_cond - x_uncond)
    return restore_vx(v_out, x_out, cond_seq, cond_seq_mask)


@torch.no_grad()
def _forward_sample(
    model, z, t_batch, x_pred_prev, config,
    cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask,
):
    v_cond, x_cond = _forward_sample_self_cond(
        model, z, t_batch, x_pred_prev, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    if cfg_scale == 1.0:
        return v_cond, x_cond

    z_uncond = restore_cond(z, torch.zeros_like(z), cond_seq_mask)
    x_pred_prev_uncond = (
        None if x_pred_prev is None
        else restore_cond(x_pred_prev, torch.zeros_like(x_pred_prev), cond_seq_mask)
    )
    v_uncond, x_uncond = _forward_sample_self_cond(
        model, z_uncond, t_batch, x_pred_prev_uncond, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=torch.zeros_like(cond_seq), cond_seq_mask=cond_seq_mask,
    )

    v_out = v_uncond + cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + cfg_scale * (x_cond - x_uncond)
    return restore_vx(v_out, x_out, cond_seq, cond_seq_mask)


@torch.no_grad()
def ode_step(model, z, t, t_next, x_pred_prev, config,
             cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask):
    t_batch = torch.full((z.shape[0],), float(t), device=z.device, dtype=z.dtype)
    v_pred, x_pred = _forward_sample(
        model, z, t_batch, x_pred_prev, config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    return z + (t_next - t) * v_pred, x_pred


@torch.no_grad()
def sde_step(model, z, t, t_next, x_pred_prev, config,
             cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask,
             gamma: float, generator: Optional[torch.Generator] = None):
    """Re-inject noise, jump back to t_back = alpha*t with alpha = 1 - gamma*h."""
    h = t_next - t
    alpha = max(0.0, min(1.0, 1.0 - gamma * h))
    t_back = alpha * t
    eps = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator) * config.denoiser_noise_scale
    z_back = restore_cond(alpha * z + (1.0 - alpha) * eps, cond_seq, cond_seq_mask)
    t_batch = torch.full((z.shape[0],), float(t_back), device=z.device, dtype=z.dtype)
    v_pred, x_pred = _forward_sample(
        model, z_back, t_batch, x_pred_prev, config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    return z_back + (t_next - t_back) * v_pred, x_pred
