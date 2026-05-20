from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ELFConfig


def _init_xavier(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1 = x[..., 0]
    x2 = x[..., 1]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (hidden_states.to(dtype) * self.weight).to(dtype)


class TextRotaryEmbeddingFast(nn.Module):
    def __init__(
        self,
        dim: int,
        max_length: int = 512,
        max_empty_tokens: int = 0,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        pos = torch.arange(max_length, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        freqs = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        self.register_buffer("main_cos", freqs.cos(), persistent=False)
        self.register_buffer("main_sin", freqs.sin(), persistent=False)
        self.register_buffer("empty_cos", torch.ones(max_empty_tokens, dim), persistent=False)
        self.register_buffer("empty_sin", torch.zeros(max_empty_tokens, dim), persistent=False)
        self.dim = dim
        self.max_length = max_length

    def _cos_sin(
        self,
        seq_len: int,
        num_empty_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        main_len = seq_len - num_empty_tokens
        if main_len < 0:
            raise ValueError(
                f"num_empty_tokens={num_empty_tokens} exceeds sequence length {seq_len}."
            )
        if main_len > self.main_cos.shape[0]:
            raise ValueError(
                f"RoPE sequence length {main_len} exceeds configured max_length={self.max_length}."
            )
        cos = self.main_cos[:main_len].to(device=device)
        sin = self.main_sin[:main_len].to(device=device)
        if num_empty_tokens > 0:
            if num_empty_tokens <= self.empty_cos.shape[0]:
                empty_cos = self.empty_cos[:num_empty_tokens].to(device=device)
                empty_sin = self.empty_sin[:num_empty_tokens].to(device=device)
            else:
                empty_cos = torch.ones(num_empty_tokens, self.dim, device=device, dtype=cos.dtype)
                empty_sin = torch.zeros(num_empty_tokens, self.dim, device=device, dtype=sin.dtype)
            cos = torch.cat([empty_cos, cos], dim=0)
            sin = torch.cat([empty_sin, sin], dim=0)
        return cos.to(dtype=dtype)[None, None, :, :], sin.to(dtype=dtype)[None, None, :, :]

    def forward(self, x: torch.Tensor, num_empty_tokens: int = 0) -> torch.Tensor:
        cos, sin = self._cos_sin(x.shape[-2], num_empty_tokens, x.device, x.dtype)
        return x * cos + rotate_half(x) * sin


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.float()[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class BottleneckTextProj(nn.Module):
    def __init__(self, text_encoder_dim: int, hidden_size: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.proj1 = nn.Linear(text_encoder_dim, bottleneck_dim, bias=False)
        self.proj2 = nn.Linear(bottleneck_dim, hidden_size, bias=True)
        self.apply(_init_xavier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.proj1(x))


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"hidden size {dim} must be divisible by heads {num_heads}.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_drop = attn_drop
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(_init_xavier)

    def forward(
        self,
        x: torch.Tensor,
        rope: TextRotaryEmbeddingFast | None = None,
        num_rope_empty_tokens: int = 0,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope is not None:
            q = rope(q, num_empty_tokens=num_rope_empty_tokens)
            k = rope(k, num_empty_tokens=num_rope_empty_tokens)
        attn_mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attn_mask = attention_mask[:, None, None, :].to(torch.bool)
            elif attention_mask.ndim == 3:
                attn_mask = attention_mask[:, None, :, :].to(torch.bool)
            else:
                attn_mask = attention_mask.to(torch.bool)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        x = x.transpose(1, 2).contiguous().view(bsz, seq_len, channels)
        return self.proj_drop(self.proj(x))


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.drop = nn.Dropout(drop)
        self.apply(_init_xavier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.drop(F.silu(x1) * x2))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(x))


class ELFBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(
            hidden_size,
            num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    def forward(
        self,
        x: torch.Tensor,
        rope: TextRotaryEmbeddingFast,
        num_rope_empty_tokens: int,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            rope=rope,
            num_rope_empty_tokens=num_rope_empty_tokens,
            attention_mask=attention_mask,
        )
        x = x + self.mlp(self.norm2(x))
        return x


@dataclass(frozen=True)
class ELFOutput:
    denoised: torch.Tensor
    decoder_logits: torch.Tensor | None


class ELF(nn.Module):
    def __init__(self, config: ELFConfig) -> None:
        super().__init__()
        self.config = config
        self.text_encoder_dim = config.text_encoder_dim
        self.max_length = config.max_length
        self.hidden_size = config.hidden_size
        self.depth = config.depth
        self.num_heads = config.num_heads
        self.num_time_tokens = config.num_time_tokens
        self.num_self_cond_cfg_tokens = config.num_self_cond_cfg_tokens
        self.num_model_mode_tokens = config.num_model_mode_tokens
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing = config.gradient_checkpointing

        self.self_cond_proj = nn.Linear(2 * config.text_encoder_dim, config.text_encoder_dim)
        self.text_proj = BottleneckTextProj(
            config.text_encoder_dim, config.hidden_size, config.bottleneck_dim
        )
        self.t_embedder = TimestepEmbedder(config.hidden_size)
        self.self_cond_cfg_embedder = TimestepEmbedder(config.hidden_size)
        self.t_emb_tokens = nn.Parameter(
            torch.empty(1, config.num_time_tokens, config.hidden_size)
        )
        nn.init.normal_(self.t_emb_tokens, std=0.02)
        if config.num_self_cond_cfg_tokens > 0:
            self.self_cond_cfg_tokens = nn.Parameter(
                torch.empty(1, config.num_self_cond_cfg_tokens, config.hidden_size)
            )
            nn.init.normal_(self.self_cond_cfg_tokens, std=0.02)
        else:
            self.register_parameter("self_cond_cfg_tokens", None)
        if config.num_model_mode_tokens > 0:
            self.mode_tokens = nn.Parameter(
                torch.empty(1, config.num_model_mode_tokens, config.hidden_size)
            )
            nn.init.normal_(self.mode_tokens, std=0.02)
        else:
            self.register_parameter("mode_tokens", None)

        q1, q3 = config.depth // 4, config.depth // 4 * 3
        blocks = []
        for i in range(config.depth):
            in_drop_range = q3 > i >= q1
            blocks.append(
                ELFBlock(
                    config.hidden_size,
                    config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    attn_drop=config.attn_dropout if in_drop_range else 0.0,
                    proj_drop=config.proj_dropout if in_drop_range else 0.0,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.rope = TextRotaryEmbeddingFast(
            dim=config.hidden_size // config.num_heads,
            max_length=config.max_length,
            max_empty_tokens=(
                config.num_model_mode_tokens
                + config.num_time_tokens
                + config.num_self_cond_cfg_tokens
            ),
        )

        self.proj_kernel = nn.Linear(config.hidden_size, config.text_encoder_dim)
        self.unembed = nn.Linear(config.text_encoder_dim, config.vocab_size)
        self.final_layer = FinalLayer(config.hidden_size, config.text_encoder_dim)
        self.self_cond_proj.apply(_init_xavier)
        self.proj_kernel.apply(_init_xavier)
        self.unembed.apply(_init_xavier)

    def _prefix_from_embedding(
        self,
        embedding: torch.Tensor,
        token_param: torch.Tensor,
        n_tokens: int,
    ) -> torch.Tensor:
        return token_param.expand(embedding.shape[0], n_tokens, -1) + embedding[:, None, :]

    def build_context(
        self,
        t: torch.Tensor,
        self_cond_cfg_scale: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if self.num_time_tokens <= 0:
            raise ValueError("num_time_tokens must be positive for prefix time conditioning.")
        prefix_tokens = [
            self._prefix_from_embedding(
                self.t_embedder(t), self.t_emb_tokens, self.num_time_tokens
            )
        ]
        if self_cond_cfg_scale is not None and self.num_self_cond_cfg_tokens > 0:
            prefix_tokens.append(
                self._prefix_from_embedding(
                    self.self_cond_cfg_embedder(self_cond_cfg_scale),
                    self.self_cond_cfg_tokens,
                    self.num_self_cond_cfg_tokens,
                )
            )
        return prefix_tokens

    @staticmethod
    def _decoder_gate(
        decoder_step_active: bool | torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if decoder_step_active is None:
            return torch.zeros((), device=device, dtype=dtype)
        gate = torch.as_tensor(decoder_step_active, device=device)
        gate = gate.to(dtype=dtype)
        if gate.ndim == 0:
            return gate
        return gate.view(batch_size, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        deterministic: bool | None = None,
        self_cond_cfg_scale: torch.Tensor | None = None,
        decoder_step_active: bool | torch.Tensor | None = None,
    ) -> ELFOutput:
        batch_size = x.shape[0]
        if x.shape[-1] == 2 * self.text_encoder_dim:
            x = self.self_cond_proj(x)
        elif x.shape[-1] != self.text_encoder_dim:
            raise ValueError(
                f"Expected input dim {self.text_encoder_dim} or {2 * self.text_encoder_dim}; "
                f"got {x.shape[-1]}."
            )

        x = self.text_proj(x)
        model_mode_offset = 0
        if self.num_model_mode_tokens > 0:
            mode_tokens = self.mode_tokens.expand(batch_size, -1, -1)
            mode_tokens = mode_tokens * self._decoder_gate(
                decoder_step_active, batch_size, x.device, x.dtype
            )
            x = torch.cat([mode_tokens, x], dim=1)
            model_mode_offset = self.num_model_mode_tokens
            if attention_mask is not None:
                mode_mask = torch.ones(
                    batch_size,
                    self.num_model_mode_tokens,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                )
                attention_mask = torch.cat([mode_mask, attention_mask], dim=1)

        context_prefix_tokens = self.build_context(t, self_cond_cfg_scale)
        prefix_tokens = torch.cat(context_prefix_tokens, dim=1)
        prefix_len = prefix_tokens.shape[1]
        x = torch.cat([prefix_tokens, x], dim=1)
        if attention_mask is not None:
            prefix_mask = torch.ones(
                batch_size,
                prefix_len,
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        num_rope_empty_tokens = prefix_len + model_mode_offset
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    lambda y, b=block: b(y, self.rope, num_rope_empty_tokens, attention_mask),
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(x, self.rope, num_rope_empty_tokens, attention_mask)

        x = x[:, prefix_len + model_mode_offset :]
        decoder_logits = None
        if decoder_step_active is not None:
            active = bool(decoder_step_active) if not torch.is_tensor(decoder_step_active) else bool(decoder_step_active.item())
            if active:
                decoder_logits = self.unembed(F.gelu(self.proj_kernel(x)))
            else:
                decoder_logits = torch.zeros(
                    *x.shape[:2],
                    self.vocab_size,
                    device=x.device,
                    dtype=x.dtype,
                )
        output = self.final_layer(x)
        return ELFOutput(output, decoder_logits)


def ELF_B(**kwargs: object) -> ELF:
    config = ELFConfig(depth=12, hidden_size=768, num_heads=12, **kwargs)
    return ELF(config)


def ELF_M(**kwargs: object) -> ELF:
    config = ELFConfig(depth=24, hidden_size=1056, num_heads=16, **kwargs)
    return ELF(config)


def ELF_L(**kwargs: object) -> ELF:
    config = ELFConfig(depth=32, hidden_size=1280, num_heads=16, **kwargs)
    return ELF(config)


ELF_MODELS = {"ELF-B": ELF_B, "ELF-M": ELF_M, "ELF-L": ELF_L}
