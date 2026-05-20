"""PyTorch ELF transformer (DiT-style with in-context conditioning)."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from modules.layers import (
    Attention, BottleneckTextProj, FinalLayer, RMSNorm, SwiGLUFFN,
    TextRotaryEmbedding, TimestepEmbedder,
    normal_002_, xavier_uniform_, zeros_,
)


class ELFBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(hidden_size, num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden, drop=proj_drop)

    def forward(self, x, rope, attention_mask=None):
        x = x + self.attn(self.norm1(x), rope=rope, attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class ELF(nn.Module):
    """The denoiser/decoder shared network.

    Forward:
        x: (B, S, C) with C = text_encoder_dim, or (B, S, 2C) if self-conditioning
           (the JAX code's `self_cond_proj` lives inside this module).
        t: (B,) floats in [0, 1] for the denoise mode, exactly 1 for the decode mode.
        attention_mask: (B, S) 1-keep, or (B, S, S) for prefix-attention masks.
        self_cond_cfg_scale: (B,) floats in [self_cond_cfg_min, self_cond_cfg_max].
        decoder_step_active: scalar bool. Gates the mode tokens and the unembedding head.

    Returns (x_pred, decoder_logits). decoder_logits is None unless decoder_step_active is True.
    """

    def __init__(
        self,
        text_encoder_dim: int,
        max_length: int,
        vocab_size: int,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bottleneck_dim: int = 128,
        num_time_tokens: int = 4,
        num_self_cond_cfg_tokens: int = 4,
        num_model_mode_tokens: int = 0,
        self_cond: bool = True,
        activation_checkpointing: bool = False,
    ):
        super().__init__()
        if num_time_tokens <= 0:
            raise ValueError("num_time_tokens must be positive for prefix time conditioning.")
        self.text_encoder_dim = text_encoder_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.bottleneck_dim = bottleneck_dim
        self.num_time_tokens = num_time_tokens
        self.num_self_cond_cfg_tokens = num_self_cond_cfg_tokens
        self.num_model_mode_tokens = num_model_mode_tokens
        self.self_cond = self_cond
        self.vocab_size = vocab_size
        self.activation_checkpointing = activation_checkpointing

        # Self-conditioning projection: [z, x_pred_prev] (2C) -> C.
        if self_cond:
            self.self_cond_proj = nn.Linear(2 * text_encoder_dim, text_encoder_dim, bias=True)
            xavier_uniform_(self.self_cond_proj.weight)
            zeros_(self.self_cond_proj.bias)

        self.text_proj = BottleneckTextProj(text_encoder_dim, hidden_size, bottleneck_dim)

        # Learned prefix tokens.
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_emb_tokens = nn.Parameter(torch.empty(1, num_time_tokens, hidden_size))
        normal_002_(self.t_emb_tokens)

        if num_self_cond_cfg_tokens > 0:
            self.self_cond_cfg_embedder = TimestepEmbedder(hidden_size)
            self.self_cond_cfg_tokens = nn.Parameter(
                torch.empty(1, num_self_cond_cfg_tokens, hidden_size)
            )
            normal_002_(self.self_cond_cfg_tokens)

        if num_model_mode_tokens > 0:
            self.mode_tokens = nn.Parameter(torch.empty(1, num_model_mode_tokens, hidden_size))
            normal_002_(self.mode_tokens)

        head_dim = hidden_size // num_heads
        # The empty-token offset has to cover the full prefix (mode + time + cfg tokens).
        num_prefix = num_model_mode_tokens + num_time_tokens + (
            num_self_cond_cfg_tokens if num_self_cond_cfg_tokens > 0 else 0
        )
        self.rope = TextRotaryEmbedding(
            dim=head_dim, pt_seq_len=max_length, num_empty_token=num_prefix,
        )

        self.blocks = nn.ModuleList([
            ELFBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if (depth // 4) <= i < (depth // 4 * 3) else 0.0,
                proj_drop=proj_drop if (depth // 4) <= i < (depth // 4 * 3) else 0.0,
            )
            for i in range(depth)
        ])

        # Factored decoder unembedding: hidden -> text_encoder_dim -> vocab.
        self.dec_proj = nn.Linear(hidden_size, text_encoder_dim, bias=True)
        self.dec_unembed = nn.Linear(text_encoder_dim, vocab_size, bias=True)
        xavier_uniform_(self.dec_proj.weight); zeros_(self.dec_proj.bias)
        xavier_uniform_(self.dec_unembed.weight); zeros_(self.dec_unembed.bias)

        self.final_layer = FinalLayer(hidden_size, text_encoder_dim)

    def _build_prefix(self, B: int, t: torch.Tensor,
                      self_cond_cfg_scale: Optional[torch.Tensor], dtype: torch.dtype):
        prefixes = []
        time_emb = self.t_embedder(t).to(dtype)
        prefixes.append(self.t_emb_tokens.to(dtype).expand(B, -1, -1) + time_emb.unsqueeze(1))
        if self.num_self_cond_cfg_tokens > 0 and self_cond_cfg_scale is not None:
            sc_emb = self.self_cond_cfg_embedder(self_cond_cfg_scale).to(dtype)
            prefixes.append(
                self.self_cond_cfg_tokens.to(dtype).expand(B, -1, -1) + sc_emb.unsqueeze(1)
            )
        return torch.cat(prefixes, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        self_cond_cfg_scale: Optional[torch.Tensor] = None,
        decoder_step_active: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = x.shape[0]

        # Self-conditioning channel concat -> back to encoder dim.
        if self.self_cond and x.shape[-1] == 2 * self.text_encoder_dim:
            x = self.self_cond_proj(x)
        elif self.self_cond and x.shape[-1] != self.text_encoder_dim:
            raise ValueError(
                f"Self-cond model expects last dim {self.text_encoder_dim} or "
                f"{2 * self.text_encoder_dim}, got {x.shape[-1]}"
            )

        x = self.text_proj(x)

        prefix_len = 0
        # Mode tokens are zero when not in decoder mode; this keeps the residual stream
        # identical between the two modes outside of the gated head.
        if self.num_model_mode_tokens > 0:
            gate = float(bool(decoder_step_active))
            mode = self.mode_tokens.to(x.dtype).expand(B, -1, -1) * gate
            x = torch.cat([mode, x], dim=1)
            prefix_len += self.num_model_mode_tokens
            if attention_mask is not None:
                attention_mask = self._extend_mask(attention_mask, self.num_model_mode_tokens, B)

        ctx = self._build_prefix(B, t, self_cond_cfg_scale, x.dtype)
        ctx_len = ctx.shape[1]
        x = torch.cat([ctx, x], dim=1)
        prefix_len += ctx_len
        if attention_mask is not None:
            attention_mask = self._extend_mask(attention_mask, ctx_len, B)

        for block in self.blocks:
            if self.activation_checkpointing and self.training:
                x = ckpt.checkpoint(block, x, self.rope, attention_mask, use_reentrant=False)
            else:
                x = block(x, rope=self.rope, attention_mask=attention_mask)

        x = x[:, prefix_len:]

        decoder_logits = None
        if decoder_step_active:
            h = self.dec_unembed(F.gelu(self.dec_proj(x)))
            decoder_logits = h

        return self.final_layer(x), decoder_logits

    @staticmethod
    def _extend_mask(attention_mask: torch.Tensor, n_prefix: int, B: int) -> torch.Tensor:
        if attention_mask.dim() == 2:
            prefix = torch.ones(B, n_prefix, dtype=attention_mask.dtype, device=attention_mask.device)
            return torch.cat([prefix, attention_mask], dim=1)
        if attention_mask.dim() == 3:
            # (B, S, S) with rows=queries, cols=keys. Pad rows AND cols with all-ones.
            S = attention_mask.shape[1]
            new_S = S + n_prefix
            new_mask = torch.ones(B, new_S, new_S, dtype=attention_mask.dtype,
                                  device=attention_mask.device)
            new_mask[:, n_prefix:, n_prefix:] = attention_mask
            return new_mask
        raise ValueError(f"Unsupported attention_mask ndim {attention_mask.dim()}")


def ELF_B(**kw): return ELF(depth=12, hidden_size=768, num_heads=12, **kw)
def ELF_M(**kw): return ELF(depth=24, hidden_size=1056, num_heads=16, **kw)
def ELF_L(**kw): return ELF(depth=32, hidden_size=1280, num_heads=16, **kw)

ELF_models = {"ELF-B": ELF_B, "ELF-M": ELF_M, "ELF-L": ELF_L}
