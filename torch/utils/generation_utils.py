"""Inference scaffolding: prefix stripping, EOS masking, single-batch ODE/SDE drivers."""

from typing import Optional, Tuple

import torch

from utils.sampling_utils import (
    get_sampling_steps, ode_step, sde_step, restore_cond,
)


def mask_after_eos(predicted_ids: torch.Tensor, eos_token_id: int,
                   pad_token_id: int) -> torch.Tensor:
    """Replace everything at and after the first EOS in each row with pad."""
    eos = predicted_ids == eos_token_id
    keep = torch.cumsum(eos.long(), dim=1) == 0
    return torch.where(keep, predicted_ids, torch.full_like(predicted_ids, pad_token_id))


def shift_left(x: torch.Tensor, shift_per_sample: torch.Tensor, pad_value: int = 0,
               axis: int = 1) -> torch.Tensor:
    """Per-row left shift; positions that fall off the right get pad_value.

    Used to strip the condition prefix from a generated (B, L, ...) sequence so the
    target tokens line up at the left.
    """
    if x.ndim < 2:
        raise ValueError("x must have at least batch and sequence dims")
    if axis == 0:
        raise ValueError("axis=0 is the batch axis")
    if axis != 1:
        x = x.movedim(axis, 1)
    seq_len = x.shape[1]
    base = torch.arange(seq_len, device=x.device).unsqueeze(0)
    gather_idx = shift_per_sample.long().unsqueeze(1) + base
    valid = gather_idx < seq_len
    gather_idx = gather_idx.clamp(0, seq_len - 1)
    if x.ndim == 2:
        shifted = torch.gather(x, 1, gather_idx)
        shifted = torch.where(valid, shifted, torch.full_like(shifted, pad_value))
    else:
        idx = gather_idx
        for _ in range(x.ndim - 2):
            idx = idx.unsqueeze(-1)
        idx = idx.expand(-1, -1, *x.shape[2:])
        shifted = torch.gather(x, 1, idx)
        valid_b = valid
        for _ in range(x.ndim - 2):
            valid_b = valid_b.unsqueeze(-1)
        valid_b = valid_b.expand_as(shifted)
        shifted = torch.where(valid_b, shifted, torch.full_like(shifted, pad_value))
    if axis != 1:
        shifted = shifted.movedim(1, axis)
    return shifted


# ---------------- Drivers ----------------

@torch.no_grad()
def run_sampling_single_batch(
    model, z: torch.Tensor, t_steps: torch.Tensor,
    cond_seq: Optional[torch.Tensor], cond_seq_mask: Optional[torch.Tensor],
    config, sampling_config,
    cfg_scale: float, self_cond_cfg_scale: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Drive ODE or SDE sampling up to (but not including) the final time step.

    Returns the latent at t = t_steps[-1] (clean embedding regime). The decoder
    head is applied separately via decode_batch().
    """
    method = sampling_config.sampling_method
    B, L, C = z.shape
    device = z.device

    if cond_seq is None:
        cond_seq = torch.zeros((B, L, C), device=device, dtype=z.dtype)
        cond_seq_mask = torch.zeros((B, L), device=device, dtype=z.dtype)

    z = restore_cond(z, cond_seq, cond_seq_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)

    gamma = float(getattr(sampling_config, "sde_gamma", 0.0))

    # All but the last step.
    for i in range(t_steps.numel() - 2):
        t = float(t_steps[i].item())
        t_next = float(t_steps[i + 1].item())
        if method == "sde":
            z, x_pred = sde_step(
                model, z, t, t_next, x_pred, config,
                cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
                gamma=gamma, generator=generator,
            )
        elif method == "ode":
            z, x_pred = ode_step(
                model, z, t, t_next, x_pred, config,
                cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
                cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
            )
        else:
            raise ValueError(f"Unknown sampling_method: {method}")

    # Final step always with ODE (the paper's prescription).
    t = float(t_steps[-2].item())
    t_next = float(t_steps[-1].item())
    z, x_pred = ode_step(
        model, z, t, t_next, x_pred, config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
    )
    return z


@torch.no_grad()
def decode_batch(model, z: torch.Tensor, t_final_val: float, config,
                 self_cond_cfg_scale: float) -> torch.Tensor:
    """Run the decoder head on the final z and return argmax tokens."""
    B = z.shape[0]
    t_final = torch.full((B,), float(t_final_val), device=z.device, dtype=z.dtype)
    sc = (
        torch.full((B,), float(self_cond_cfg_scale), device=z.device, dtype=z.dtype)
        if config.num_self_cond_cfg_tokens > 0 else None
    )
    z_in = torch.cat([z, torch.zeros_like(z)], dim=-1) if config.self_cond_prob > 0 else z
    _, logits = model(z_in, t_final, self_cond_cfg_scale=sc, decoder_step_active=True)
    return logits.argmax(dim=-1)


def build_run_name(sampling_method: str, num_sampling_steps: int, cfg_scale, self_cond_cfg_scale,
                   time_schedule: str, sde_gamma: float, suffix: str) -> str:
    ts_str = f"-ts_{time_schedule}"
    sccfg_str = f"-sccfg{self_cond_cfg_scale}" if self_cond_cfg_scale != 1.0 else ""
    sde_str = f"-gamma{sde_gamma}" if sampling_method == "sde" else ""
    return f"{sampling_method}-steps{num_sampling_steps}-cfg{cfg_scale}{sccfg_str}{ts_str}{sde_str}-{suffix}"
