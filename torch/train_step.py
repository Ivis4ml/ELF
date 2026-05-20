"""ELF training step in PyTorch.

Mirrors src/train_step.py. For each micro-batch we sample a single Bernoulli
trial to pick either the *decoder* branch (CE on tokens at t=1) or the *denoiser*
branch (MSE on flow velocity at random t). The denoiser branch may run up to
4 forward passes if both self-conditioning and CFG-on-self-cond are enabled:

  1) detached forward with [z, 0] -> initial x_pred used as the self-cond signal
  2) trained forward with [z, sg(x_pred)] -> conditional velocity prediction
  3) detached uncond CFG target forward with [z, 0]
  4) detached cond  CFG target forward with [z, sg(x_pred_uncond)]

Passes 1 and 3 are mathematically identical in deterministic mode and could be
shared for ~25% throughput; we keep them separate for parity with the JAX
reference and easier debugging.
"""

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from utils.sampling_utils import add_noise, sample_cfg_scale, sample_timesteps, net_out_to_v_x
from utils.train_utils import maybe_unwrap_ddp


def _restore_cond_2d_or_3d(z_updated, cond_seq, cond_seq_mask):
    """cond_seq_mask: (B, S) -> broadcast to z dims; substitute clean cond_seq where mask>0."""
    mask = cond_seq_mask
    while mask.ndim < z_updated.ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask > 0, cond_seq, z_updated)


def _mask_reduce_loss(per_token_loss: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Safe masked mean: avoids dividing by zero and avoids letting NaN in masked positions
    poison the result. per_token_loss: (B, S); loss_mask: (B, S) {0, 1}."""
    loss_mask = loss_mask.to(per_token_loss.dtype)
    safe = torch.where(loss_mask > 0, per_token_loss, torch.zeros_like(per_token_loss))
    denom = loss_mask.sum().clamp(min=1.0)
    return (safe * loss_mask).sum() / denom


def train_step(
    model,
    encoder,
    batch: Dict[str, torch.Tensor],
    config,
    autocast_dtype: torch.dtype,
    is_decoder_step: bool,
) -> Tuple[torch.Tensor, Dict[str, float], bool]:
    """One forward+backward. Returns (loss, metrics, is_decoder_step).

    `is_decoder_step` MUST be agreed across DDP ranks (decide it once on rank 0
    and broadcast, or derive it from a deterministic global-step rng). If ranks
    disagree, the decoder/denoiser branches run different sub-modules and DDP's
    all-reduce will hang on the first parameter only one branch touched.

    The caller is responsible for `loss.backward()`, `optimizer.step()`, EMA, and
    `optimizer.zero_grad()` — keeping those in the train.py loop makes grad
    accumulation, DDP no_sync, and gradient clipping easier to coordinate.
    """
    device = batch["input_ids"].device
    inner_model = maybe_unwrap_ddp(model)

    # -------- encoder (no grad) --------
    encoder_attn = batch["encoder_attention_mask"]
    label_drop_mask = batch.get("label_drop_mask", None)
    cond_seq_mask = batch["cond_seq_mask"]
    attention_mask = batch["attention_mask"]

    if config.label_drop_prob > 0 and label_drop_mask is not None and label_drop_mask.any():
        # Zero out the cond->target attention rows for dropped samples (so target tokens
        # are encoded purely from themselves). cond->cond stays intact.
        drop = label_drop_mask.float()[:, None, None]
        cm = cond_seq_mask
        block_mask = (1 - cm)[:, :, None] * cm[:, None, :]
        encoder_attn = encoder_attn * (1.0 - drop * block_mask)

    with torch.no_grad():
        x0 = encoder(input_ids=batch["input_ids"], attention_mask=encoder_attn)
        x0 = (x0 - config.latent_mean) / config.latent_std

    B, L, C = x0.shape
    t_eps = config.t_eps

    # -------- sample noise schedule, time, CFG --------
    t = sample_timesteps(
        B, P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        time_schedule=config.time_schedule, device=device,
    )

    noise = torch.randn_like(x0)
    cond_seq_mask3 = cond_seq_mask.unsqueeze(-1)

    loss_mask = attention_mask if config.pad_token == "pad" else torch.ones_like(attention_mask)
    loss_mask = loss_mask * (1.0 - cond_seq_mask)

    denoiser_z = add_noise(x0, noise, t, config, cond_seq_mask=cond_seq_mask3)

    if config.label_drop_prob > 0 and label_drop_mask is not None:
        drop3 = label_drop_mask[:, None, None]
        zero_mask = drop3 & (cond_seq_mask3 > 0)
        denoiser_z = torch.where(zero_mask, torch.zeros_like(denoiser_z), denoiser_z)
        x0 = torch.where(zero_mask, torch.zeros_like(x0), x0)

    # -------- decoder-branch z: per-token logit-normal corruption around clean --------
    p = torch.randn(B * L, device=device) * config.decoder_p_std + config.decoder_p_mean
    lambda_t = torch.sigmoid(p).view(B, L, 1)
    dec_noise = torch.randn_like(x0) * config.decoder_noise_scale
    decoder_z = lambda_t * x0 + (1.0 - lambda_t) * dec_noise

    # -------- targets and per-sample knobs --------
    t_exp = t.view(-1, 1, 1)
    v_target = (x0 - denoiser_z) / torch.clamp(1.0 - t_exp, min=t_eps)

    if config.self_cond_prob > 0:
        use_self_cond_mask = (torch.rand(B, device=device) < config.self_cond_prob).float().view(B, 1, 1)
    else:
        use_self_cond_mask = None

    if config.num_self_cond_cfg_tokens > 0:
        sc_cfg_scale = sample_cfg_scale(
            B, cfg_min=config.self_cond_cfg_min, cfg_max=config.self_cond_cfg_max, device=device,
        )
    else:
        sc_cfg_scale = None

    # -------- branch chooser (decided by caller; same across DDP ranks) --------
    if is_decoder_step:
        return _decoder_branch(
            model, inner_model, decoder_z, sc_cfg_scale, batch["input_ids"], loss_mask,
            autocast_dtype, config,
        )
    return _denoiser_branch(
        model, inner_model, denoiser_z, t, x0, v_target, use_self_cond_mask,
        sc_cfg_scale, cond_seq_mask3, loss_mask, autocast_dtype, config,
    )


# ===================================================================
# Decoder branch (CE)
# ===================================================================

def _decoder_branch(model, inner_model, decoder_z, sc_cfg_scale, target_ids, loss_mask,
                    autocast_dtype, config):
    B, L, C = decoder_z.shape
    device = decoder_z.device
    t_one = torch.ones(B, device=device, dtype=decoder_z.dtype)
    dec_in = (
        torch.cat([decoder_z, torch.zeros_like(decoder_z)], dim=-1)
        if config.self_cond_prob > 0 else decoder_z
    )
    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype != torch.float32):
        _, logits = model(
            dec_in, t_one,
            self_cond_cfg_scale=sc_cfg_scale,
            decoder_step_active=True,
        )
    log_probs = F.log_softmax(logits.float(), dim=-1)
    nll = -log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    ce_loss = _mask_reduce_loss(nll, loss_mask)
    metrics = {"loss": ce_loss.detach(), "l2_loss": torch.zeros((), device=device),
               "ce_loss": ce_loss.detach()}
    return ce_loss, metrics, True


# ===================================================================
# Denoiser branch (MSE on velocity, with training-time CFG target)
# ===================================================================

def _denoiser_branch(model, inner_model, denoiser_z, t, x0, v_target,
                     use_self_cond_mask, sc_cfg_scale, cond_seq_mask3, loss_mask,
                     autocast_dtype, config):
    device = denoiser_z.device
    t_eps = config.t_eps
    has_sc = config.self_cond_prob > 0

    # --- step 1: initial self-cond signal (detached) -----------------------
    if has_sc:
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=autocast_dtype != torch.float32):
                z_in0 = torch.cat(
                    [denoiser_z, _restore_cond_2d_or_3d(torch.zeros_like(denoiser_z), x0, cond_seq_mask3)],
                    dim=-1,
                )
                net_out0 = model(z_in0, t, self_cond_cfg_scale=sc_cfg_scale,
                                 decoder_step_active=False)
                _, x_pred_init = net_out_to_v_x(net_out0, denoiser_z, t, t_eps)
        x_pred_init = _restore_cond_2d_or_3d(x_pred_init, x0, cond_seq_mask3)
        x_pred_cond = x_pred_init * use_self_cond_mask.to(x_pred_init.dtype)
        x_pred_cond = _restore_cond_2d_or_3d(x_pred_cond, x0, cond_seq_mask3)
        z_in = torch.cat([denoiser_z, x_pred_cond], dim=-1)
    else:
        z_in = denoiser_z

    # --- step 2: trained forward (THE pass that gets gradients) ------------
    with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                        enabled=autocast_dtype != torch.float32):
        net_out = model(z_in, t, self_cond_cfg_scale=sc_cfg_scale,
                        decoder_step_active=False)
    v_pred, _ = net_out_to_v_x(net_out, denoiser_z, t, t_eps)

    # --- training-time CFG target (two detached forwards) -----------------
    v_final_target = v_target
    if config.num_self_cond_cfg_tokens > 0:
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=autocast_dtype != torch.float32):
                if has_sc:
                    z_in_uncond = torch.cat(
                        [denoiser_z, _restore_cond_2d_or_3d(torch.zeros_like(denoiser_z), x0, cond_seq_mask3)],
                        dim=-1,
                    )
                else:
                    z_in_uncond = denoiser_z
                net_out_uncond = model(z_in_uncond, t, self_cond_cfg_scale=sc_cfg_scale,
                                       decoder_step_active=False)
                v_uncond, x_uncond = net_out_to_v_x(net_out_uncond, denoiser_z, t, t_eps)
                if has_sc:
                    x_uncond = _restore_cond_2d_or_3d(x_uncond, x0, cond_seq_mask3)
                    z_in_cond = torch.cat([denoiser_z, x_uncond], dim=-1)
                else:
                    z_in_cond = denoiser_z
                net_out_cond = model(z_in_cond, t, self_cond_cfg_scale=sc_cfg_scale,
                                     decoder_step_active=False)
                v_cond, _ = net_out_to_v_x(net_out_cond, denoiser_z, t, t_eps)

        sc_w = sc_cfg_scale.view(-1, 1, 1).to(v_cond.dtype)
        sc_guidance = (1.0 - 1.0 / sc_w) * (v_cond - v_uncond)
        if has_sc and use_self_cond_mask is not None:
            sc_guidance = torch.where(
                use_self_cond_mask.bool(), sc_guidance, torch.zeros_like(sc_guidance),
            )
        v_final_target = (v_target + sc_guidance).detach()

    per_dim_loss = (v_pred - v_final_target).float().pow(2)
    per_token_loss = per_dim_loss.mean(dim=-1)
    l2_loss = _mask_reduce_loss(per_token_loss, loss_mask)
    metrics = {"loss": l2_loss.detach(), "l2_loss": l2_loss.detach(),
               "ce_loss": torch.zeros((), device=device)}
    return l2_loss, metrics, False
