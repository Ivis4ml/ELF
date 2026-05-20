"""Building blocks for the PyTorch ELF transformer.

Layout intentionally tracks src/modules/layers.py one-to-one (RMSNorm, RoPE
with an empty-token offset, SwiGLU FFN, qk-norm SDPA attention, bottleneck
projection, sinusoidal time-step embedder, zero-init final layer) so weights
and the JAX implementation can be cross-checked behaviorally.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# Init defaults match the JAX/Flax version: xavier_uniform on Dense kernels,
# zero biases, normal(0.02) for learned tokens / time-step MLPs, zero init on
# the final projection (so the residual starts at the identity).


def xavier_uniform_(t: torch.Tensor) -> torch.Tensor:
    nn.init.xavier_uniform_(t)
    return t


def normal_002_(t: torch.Tensor) -> torch.Tensor:
    nn.init.normal_(t, mean=0.0, std=0.02)
    return t


def zeros_(t: torch.Tensor) -> torch.Tensor:
    nn.init.zeros_(t)
    return t


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class TextRotaryEmbedding(nn.Module):
    """1D rotary positional embedding with an empty-token prefix that gets no rotation.

    Empty positions are the in-context conditioning tokens (time, CFG, mode);
    they should be position-agnostic so we hand them cos=1 / sin=0.
    """

    def __init__(self, dim: int, pt_seq_len: int = 512, ft_seq_len: Optional[int] = None,
                 theta: float = 10000.0, num_empty_token: int = 0):
        super().__init__()
        ft_seq_len = ft_seq_len if ft_seq_len is not None else pt_seq_len

        # Inverse frequencies on the rotary half-dim.
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32)[: dim // 2] / dim))
        pos = torch.arange(ft_seq_len, dtype=torch.float32) / ft_seq_len * pt_seq_len
        freqs_main = torch.einsum("i,f->if", pos, freqs)
        freqs_main = repeat(freqs_main, "... n -> ... (n r)", r=2)

        D = freqs_main.shape[-1]
        cos_parts, sin_parts = [], []
        if num_empty_token > 0:
            cos_parts.append(torch.ones(num_empty_token, D, dtype=torch.float32))
            sin_parts.append(torch.zeros(num_empty_token, D, dtype=torch.float32))
        cos_parts.append(torch.cos(freqs_main))
        sin_parts.append(torch.sin(freqs_main))

        self.register_buffer("freqs_cos", torch.cat(cos_parts, dim=0), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(sin_parts, dim=0), persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        cos = self.freqs_cos[: t.shape[-2]].to(t.dtype)
        sin = self.freqs_sin[: t.shape[-2]].to(t.dtype)
        return t * cos + rotate_half(t) * sin


class RMSNorm(nn.Module):
    """RMSNorm computed in fp32; result cast back to the input dtype."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x32 = x.float()
        variance = x32.pow(2).mean(dim=-1, keepdim=True)
        x32 = x32 * torch.rsqrt(variance + self.eps)
        return (self.weight * x32).to(input_dtype)


class BottleneckTextProj(nn.Module):
    """Project encoder embeddings into the transformer hidden size through a
    low-rank bottleneck (default rank 128, see Sec. C.2 of the paper)."""

    def __init__(self, text_encoder_dim: int, hidden_size: int, bottleneck_dim: int):
        super().__init__()
        self.proj1 = nn.Linear(text_encoder_dim, bottleneck_dim, bias=False)
        self.proj2 = nn.Linear(bottleneck_dim, hidden_size, bias=True)
        xavier_uniform_(self.proj1.weight)
        xavier_uniform_(self.proj2.weight)
        zeros_(self.proj2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.proj1(x))


class TimestepEmbedder(nn.Module):
    """Sinusoidal scalar embedding -> 2-layer MLP, used to encode time and CFG scales."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp_0 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.mlp_2 = nn.Linear(hidden_size, hidden_size, bias=True)
        normal_002_(self.mlp_0.weight); zeros_(self.mlp_0.bias)
        normal_002_(self.mlp_2.weight); zeros_(self.mlp_2.bias)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float()[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.timestep_embedding(t, self.frequency_embedding_size).to(self.mlp_0.weight.dtype)
        return self.mlp_2(F.silu(self.mlp_0(t_emb)))


class Attention(nn.Module):
    """Multi-head self-attention with qk-norm. Uses torch SDPA so we get
    FlashAttention 2 / cuDNN backends automatically on H200.

    attention_mask is a boolean / 0-1 tensor of shape (B, S) or (B, S, S); 1=keep.
    """

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 qk_norm: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop_p = attn_drop
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        xavier_uniform_(self.qkv.weight)
        if qkv_bias:
            zeros_(self.qkv.bias)
        xavier_uniform_(self.proj.weight)
        zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, rope: Optional[TextRotaryEmbedding] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, N, Hd)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rope is not None:
            q = rope(q)
            k = rope(k)

        attn_mask = None
        if attention_mask is not None:
            # SDPA wants a True-keeps boolean mask broadcastable to (B, H, N, S).
            if attention_mask.dim() == 2:
                attn_mask = attention_mask[:, None, None, :].bool()
            elif attention_mask.dim() == 3:
                attn_mask = attention_mask[:, None, :, :].bool()
            else:
                attn_mask = attention_mask.bool()

        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_p if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward block sized to ~hidden_dim * 2/3 like LLaMA."""

    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0, bias: bool = True):
        super().__init__()
        inner = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * inner, bias=bias)
        self.w3 = nn.Linear(inner, dim, bias=bias)
        self.drop = nn.Dropout(drop)
        xavier_uniform_(self.w12.weight)
        xavier_uniform_(self.w3.weight)
        if bias:
            zeros_(self.w12.bias)
            zeros_(self.w3.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(self.drop(F.silu(x1) * x2))


class FinalLayer(nn.Module):
    """RMSNorm + zero-init linear projection back to the encoder dim."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        zeros_(self.linear.weight)
        zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(x))
