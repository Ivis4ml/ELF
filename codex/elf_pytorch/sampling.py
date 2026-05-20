from __future__ import annotations

import torch

from .config import TrainingConfig
from .flow import get_sampling_steps, net_out_to_v_x, restore_cond, restore_vx


@torch.no_grad()
def forward_sample_self_cond(
    model: torch.nn.Module,
    z: torch.Tensor,
    t_batch: torch.Tensor,
    x_pred_prev: torch.Tensor | None,
    config: TrainingConfig,
    *,
    self_cond_cfg_scale: float,
    cond_seq: torch.Tensor,
    cond_seq_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if config.num_self_cond_cfg_tokens > 0:
        if x_pred_prev is None:
            x_pred_prev = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
        z_input = torch.cat([z, x_pred_prev], dim=-1)
        scale_batch = torch.full((z.shape[0],), self_cond_cfg_scale, device=z.device)
        out = model(
            z_input,
            t_batch,
            deterministic=True,
            self_cond_cfg_scale=scale_batch,
        ).denoised
        v, x = net_out_to_v_x(out, z, t_batch, config.t_eps)
        return restore_vx(v, x, cond_seq, cond_seq_mask)

    if config.self_cond_prob == 0:
        out = model(z, t_batch, deterministic=True).denoised
        v, x = net_out_to_v_x(out, z, t_batch, config.t_eps)
        return restore_vx(v, x, cond_seq, cond_seq_mask)

    if self_cond_cfg_scale != 1.0 or x_pred_prev is None:
        z_uncond = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
        z_input_uncond = torch.cat([z, z_uncond], dim=-1)
        out_uncond = model(z_input_uncond, t_batch, deterministic=True).denoised
        v_uncond, x_uncond = net_out_to_v_x(out_uncond, z, t_batch, config.t_eps)
        v_uncond, x_uncond = restore_vx(v_uncond, x_uncond, cond_seq, cond_seq_mask)
        if self_cond_cfg_scale == 0.0 or x_pred_prev is None:
            return v_uncond, x_uncond

    z_input_cond = torch.cat([z, x_pred_prev], dim=-1)
    out_cond = model(z_input_cond, t_batch, deterministic=True).denoised
    v_cond, x_cond = net_out_to_v_x(out_cond, z, t_batch, config.t_eps)
    v_cond, x_cond = restore_vx(v_cond, x_cond, cond_seq, cond_seq_mask)
    if self_cond_cfg_scale == 1.0:
        return v_cond, x_cond

    v_out = v_uncond + self_cond_cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + self_cond_cfg_scale * (x_cond - x_uncond)
    return restore_vx(v_out, x_out, cond_seq, cond_seq_mask)


@torch.no_grad()
def forward_sample(
    model: torch.nn.Module,
    z: torch.Tensor,
    t_batch: torch.Tensor,
    x_pred_prev: torch.Tensor | None,
    config: TrainingConfig,
    *,
    cfg_scale: float,
    self_cond_cfg_scale: float,
    cond_seq: torch.Tensor,
    cond_seq_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    v_cond, x_cond = forward_sample_self_cond(
        model,
        z,
        t_batch,
        x_pred_prev,
        config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq,
        cond_seq_mask=cond_seq_mask,
    )
    if cfg_scale == 1.0:
        return v_cond, x_cond

    z_uncond = restore_cond(z, torch.zeros_like(z), cond_seq_mask)
    x_prev_uncond = (
        None
        if x_pred_prev is None
        else restore_cond(x_pred_prev, torch.zeros_like(x_pred_prev), cond_seq_mask)
    )
    v_uncond, x_uncond = forward_sample_self_cond(
        model,
        z_uncond,
        t_batch,
        x_prev_uncond,
        config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=torch.zeros_like(cond_seq),
        cond_seq_mask=cond_seq_mask,
    )
    v_out = v_uncond + cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + cfg_scale * (x_cond - x_uncond)
    return restore_vx(v_out, x_out, cond_seq, cond_seq_mask)


@torch.no_grad()
def sample_tokens(
    model: torch.nn.Module,
    shape: tuple[int, int, int],
    config: TrainingConfig,
    *,
    num_steps: int = 64,
    cfg_scale: float = 1.0,
    self_cond_cfg_scale: float = 1.0,
    sampler: str = "ode",
    sde_gamma: float = 0.0,
    cond_seq: torch.Tensor | None = None,
    cond_seq_mask: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    z = torch.randn(shape, device=device) * config.denoiser_noise_scale
    if cond_seq is None:
        cond_seq = torch.zeros_like(z)
    if cond_seq_mask is None:
        cond_seq_mask = torch.zeros(shape[0], shape[1], 1, device=device)
    z = restore_cond(z, cond_seq, cond_seq_mask)
    x_pred_prev = torch.zeros_like(z)
    steps = get_sampling_steps(
        num_steps,
        device=device,
        time_schedule=config.time_schedule,
        p_mean=config.denoiser_p_mean,
        p_std=config.denoiser_p_std,
    )
    for i in range(len(steps) - 1):
        t = steps[i]
        t_next = steps[i + 1]
        if sampler == "sde" and sde_gamma > 0:
            h = t_next - t
            alpha = torch.clamp(1.0 - sde_gamma * h, min=0.0, max=1.0)
            t_eval = alpha * t
            eps = torch.randn_like(z) * config.denoiser_noise_scale
            z_eval = restore_cond(alpha * z + (1.0 - alpha) * eps, cond_seq, cond_seq_mask)
        else:
            t_eval = t
            z_eval = z
        t_batch = torch.full((shape[0],), float(t_eval), device=device)
        v_pred, x_pred_prev = forward_sample(
            model,
            z_eval,
            t_batch,
            x_pred_prev,
            config,
            cfg_scale=cfg_scale,
            self_cond_cfg_scale=self_cond_cfg_scale,
            cond_seq=cond_seq,
            cond_seq_mask=cond_seq_mask,
        )
        z = z_eval + (t_next - t_eval) * v_pred
        z = restore_cond(z, cond_seq, cond_seq_mask)

    final_t = torch.ones(shape[0], device=device)
    decoder_input = (
        torch.cat([z, torch.zeros_like(z)], dim=-1) if config.self_cond_prob > 0 else z
    )
    scale_batch = (
        torch.full((shape[0],), self_cond_cfg_scale, device=device)
        if config.num_self_cond_cfg_tokens > 0
        else None
    )
    logits = model(
        decoder_input,
        final_t,
        deterministic=True,
        self_cond_cfg_scale=scale_batch,
        decoder_step_active=True,
    ).decoder_logits
    if was_training:
        model.train()
    return logits.argmax(dim=-1)
