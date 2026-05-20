from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .config import TrainingConfig
from .flow import (
    add_noise,
    net_out_to_v_x,
    restore_cond,
    sample_cfg_scale,
    sample_timesteps,
)


@dataclass
class LossOutput:
    loss: torch.Tensor
    l2_loss: torch.Tensor
    ce_loss: torch.Tensor


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def encode_text(
    encoder: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    latent_mean: float,
    latent_std: float,
) -> torch.Tensor:
    if attention_mask is not None and attention_mask.ndim == 3:
        encoder_dtype = next(encoder.parameters()).dtype
        keep = attention_mask.to(dtype=encoder_dtype)
        attention_mask = ((1.0 - keep) * torch.finfo(encoder_dtype).min).unsqueeze(1)
    with torch.no_grad():
        outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, "last_hidden_state"):
            latents = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)):
            latents = outputs[0]
        else:
            latents = outputs
    return (latents - latent_mean) / latent_std


def reduce_token_loss(per_token_loss: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    loss_mask = loss_mask.to(dtype=per_token_loss.dtype)
    safe_loss = torch.where(loss_mask > 0, per_token_loss, torch.zeros_like(per_token_loss))
    return (safe_loss * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)


def compute_elf_loss(
    model: torch.nn.Module,
    encoder: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    config: TrainingConfig,
    *,
    decoder_step_active: bool,
) -> LossOutput:
    input_ids = batch["input_ids"]
    device = input_ids.device
    batch_size, seq_length = input_ids.shape
    cond_seq_mask = batch["cond_seq_mask"].unsqueeze(-1)
    attention_mask = batch["attention_mask"]

    label_drop_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
    encoder_attention_mask = batch["encoder_attention_mask"]
    if config.label_drop_prob > 0:
        label_drop_mask = torch.rand(batch_size, device=device) < config.label_drop_prob
        drop = label_drop_mask[:, None, None].to(encoder_attention_mask.dtype)
        cond_mask = batch["cond_seq_mask"]
        block_mask = (1.0 - cond_mask)[:, :, None] * cond_mask[:, None, :]
        encoder_attention_mask = encoder_attention_mask * (1.0 - drop * block_mask)

    x0 = encode_text(
        encoder,
        input_ids,
        encoder_attention_mask,
        latent_mean=config.latent_mean,
        latent_std=config.latent_std,
    )

    t = sample_timesteps(
        batch_size,
        device=device,
        p_mean=config.denoiser_p_mean,
        p_std=config.denoiser_p_std,
        time_schedule=config.time_schedule,
    )
    noise = torch.randn_like(x0)
    denoiser_z = add_noise(
        x0,
        noise,
        t,
        config.denoiser_noise_scale,
        cond_seq_mask=cond_seq_mask,
    )

    if config.pad_token == "pad":
        loss_mask = attention_mask
    else:
        loss_mask = torch.ones_like(attention_mask)
    loss_mask = loss_mask * (1.0 - batch["cond_seq_mask"])

    if config.label_drop_prob > 0:
        drop = label_drop_mask[:, None, None]
        is_cond = cond_seq_mask > 0
        denoiser_z = torch.where(drop & is_cond, torch.zeros_like(denoiser_z), denoiser_z)
        x0 = torch.where(drop & is_cond, torch.zeros_like(x0), x0)

    decoder_lambda = torch.sigmoid(
        torch.randn(batch_size, seq_length, 1, device=device) * config.decoder_p_std
        + config.decoder_p_mean
    )
    decoder_noise = torch.randn_like(x0) * config.decoder_noise_scale
    decoder_z = decoder_lambda * x0 + (1.0 - decoder_lambda) * decoder_noise

    t_expanded = t.view(-1, 1, 1)
    v_target = (x0 - denoiser_z) / torch.clamp(1.0 - t_expanded, min=config.t_eps)

    if config.self_cond_prob > 0:
        use_self_cond_mask = (
            torch.rand(batch_size, device=device) < config.self_cond_prob
        ).view(-1, 1, 1)
    else:
        use_self_cond_mask = torch.zeros(batch_size, 1, 1, device=device, dtype=torch.bool)

    if config.num_self_cond_cfg_tokens > 0:
        self_cond_cfg_scale = sample_cfg_scale(
            batch_size,
            device=device,
            cfg_min=config.self_cond_cfg_min,
            cfg_max=config.self_cond_cfg_max,
        )
    else:
        self_cond_cfg_scale = None

    def get_z_input(
        z: torch.Tensor,
        t_input: torch.Tensor,
        self_cond_cfg_input: torch.Tensor | None,
        x_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if config.self_cond_prob == 0:
            return z
        z_uncond = restore_cond(torch.zeros_like(z), x_tokens, cond_seq_mask)
        z_with_zeros = torch.cat([z, z_uncond], dim=-1)
        with torch.no_grad():
            net_out_init = model(
                z_with_zeros,
                t_input,
                deterministic=True,
                self_cond_cfg_scale=self_cond_cfg_input,
            ).denoised
            _, x_pred_init = net_out_to_v_x(net_out_init, z, t_input, config.t_eps)
            x_pred_init = restore_cond(x_pred_init, x_tokens, cond_seq_mask)
            x_pred_cond = x_pred_init * use_self_cond_mask.to(dtype=z.dtype)
            x_pred_cond = restore_cond(x_pred_cond, x_tokens, cond_seq_mask)
        return torch.cat([z, x_pred_cond], dim=-1)

    def get_sc_cond_and_uncond(
        z: torch.Tensor,
        t_input: torch.Tensor,
        cond_mask: torch.Tensor,
        x_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kwargs = {
            "self_cond_cfg_scale": self_cond_cfg_scale,
            "deterministic": True,
        }
        if config.self_cond_prob == 0:
            net_out_uncond = model(z, t_input, **kwargs).denoised
            v_uncond, _ = net_out_to_v_x(net_out_uncond, z, t_input, config.t_eps)
            return v_uncond, v_uncond

        z_uncond = restore_cond(torch.zeros_like(z), x_tokens, cond_mask)
        z_input_uncond = torch.cat([z, z_uncond], dim=-1)
        net_out_uncond = model(z_input_uncond, t_input, **kwargs).denoised
        v_uncond, x_uncond = net_out_to_v_x(net_out_uncond, z, t_input, config.t_eps)
        x_uncond = restore_cond(x_uncond, x_tokens, cond_mask)

        z_input_cond = torch.cat([z, x_uncond], dim=-1)
        net_out_cond = model(z_input_cond, t_input, **kwargs).denoised
        v_cond, _ = net_out_to_v_x(net_out_cond, z, t_input, config.t_eps)
        return v_cond, v_uncond

    def get_v_target(
        z: torch.Tensor,
        t_input: torch.Tensor,
        base_v_target: torch.Tensor,
        x_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if config.num_self_cond_cfg_tokens <= 0:
            return base_v_target
        with torch.no_grad():
            v_cond, v_uncond = get_sc_cond_and_uncond(
                z, t_input, cond_mask=cond_seq_mask, x_tokens=x_tokens
            )
            sc_w = self_cond_cfg_scale.view(batch_size, 1, 1)
            guidance = (1.0 - 1.0 / sc_w) * (v_cond - v_uncond)
            guidance = torch.where(use_self_cond_mask, guidance, torch.zeros_like(guidance))
            return base_v_target + guidance

    zero = torch.zeros((), device=device, dtype=x0.dtype)
    if decoder_step_active:
        decoder_t = torch.ones_like(t)
        decoder_input = (
            torch.cat([decoder_z, torch.zeros_like(decoder_z)], dim=-1)
            if config.self_cond_prob > 0
            else decoder_z
        )
        logits = model(
            decoder_input,
            decoder_t,
            deterministic=False,
            self_cond_cfg_scale=self_cond_cfg_scale,
            decoder_step_active=True,
        ).decoder_logits
        if logits is None:
            raise RuntimeError("decoder logits were not produced in decoder branch.")
        ce = F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            input_ids.reshape(-1),
            reduction="none",
        ).view(batch_size, seq_length)
        ce_loss = reduce_token_loss(ce, loss_mask)
        loss = ce_loss
        l2_loss = zero
    else:
        denoiser_input = get_z_input(
            denoiser_z,
            t,
            self_cond_cfg_input=self_cond_cfg_scale,
            x_tokens=x0,
        )
        net_out = model(
            denoiser_input,
            t,
            deterministic=False,
            self_cond_cfg_scale=self_cond_cfg_scale,
            decoder_step_active=False,
        ).denoised
        v_pred, _ = net_out_to_v_x(net_out, denoiser_z, t, config.t_eps)
        final_target = get_v_target(denoiser_z, t, base_v_target=v_target, x_tokens=x0)
        per_token_l2 = (v_pred - final_target).pow(2).mean(dim=-1)
        l2_loss = reduce_token_loss(per_token_l2, loss_mask)
        loss = l2_loss
        ce_loss = zero

    decoder_prob = torch.tensor(config.decoder_prob, device=device, dtype=loss.dtype)
    denoiser_prob = torch.tensor(1.0 - config.decoder_prob, device=device, dtype=loss.dtype)
    logged_ce = torch.where(
        decoder_prob > 0, ce_loss / decoder_prob.clamp_min(1e-12), torch.zeros_like(ce_loss)
    )
    logged_l2 = torch.where(
        denoiser_prob > 0, l2_loss / denoiser_prob.clamp_min(1e-12), torch.zeros_like(l2_loss)
    )
    return LossOutput(loss=loss, l2_loss=logged_l2.detach(), ce_loss=logged_ce.detach())
