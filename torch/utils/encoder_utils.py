"""Encoder forward + (collator-side) self-attention mask builder."""

from typing import Optional

import numpy as np
import torch


@torch.no_grad()
def encode_text(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    encoder,
    latent_mean: float,
    latent_std: float,
) -> torch.Tensor:
    """Frozen-encoder forward + channel normalization (mean/std from the dataset)."""
    latents = encoder(input_ids=input_ids, attention_mask=attention_mask)
    return (latents - latent_mean) / latent_std


def build_self_attn_cond_masks(is_cond, is_valid, xp=np):
    """Replicates src/utils/encoder_utils.py::build_self_attn_cond_masks.

    Returns:
        encoder_attention_mask: (B, L, L) float32 — cond rows see cond cols, target rows
          see all valid cols. Used by the frozen encoder to prevent target tokens
          from leaking into condition encodings.
        attention_mask: (B, L) float32 — global "this position is real" mask.
        cond_seq_mask: (B, L) float32 — 1 at condition positions.
    """
    is_cond_b = is_cond.astype(bool) if hasattr(is_cond, "astype") else is_cond.bool()
    is_valid_b = is_valid.astype(bool) if hasattr(is_valid, "astype") else is_valid.bool()

    if xp is np:
        encoder_attention_mask = (
            (is_cond_b[:, :, None] & is_cond_b[:, None, :])
            | (~is_cond_b[:, :, None] & is_valid_b[:, None, :])
        ).astype(np.float32)
        attention_mask = is_valid_b.astype(np.float32)
        cond_seq_mask = is_cond_b.astype(np.float32)
    else:
        encoder_attention_mask = (
            (is_cond_b[:, :, None] & is_cond_b[:, None, :])
            | (~is_cond_b[:, :, None] & is_valid_b[:, None, :])
        ).to(torch.float32)
        attention_mask = is_valid_b.to(torch.float32)
        cond_seq_mask = is_cond_b.to(torch.float32)

    return encoder_attention_mask, attention_mask, cond_seq_mask
