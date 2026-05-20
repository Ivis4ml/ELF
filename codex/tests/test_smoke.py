from __future__ import annotations

from types import SimpleNamespace

import torch

from elf_pytorch.config import ELFConfig, TrainingConfig
from elf_pytorch.model import ELF
from elf_pytorch.sampling import sample_tokens
from elf_pytorch.training import compute_elf_loss, encode_text


class TinyEncoder(torch.nn.Module):
    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, dim)

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


class MaskCheckingEncoder(TinyEncoder):
    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__(vocab_size, dim)
        self.last_mask = None

    def forward(self, input_ids, attention_mask=None):
        self.last_mask = attention_mask
        return super().forward(input_ids, attention_mask=attention_mask)


def make_batch(batch_size: int = 2, seq_len: int = 8, vocab_size: int = 32):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    cond_seq_mask = torch.zeros(batch_size, seq_len)
    encoder_attention_mask = torch.ones(batch_size, seq_len, seq_len)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "cond_seq_mask": cond_seq_mask,
        "encoder_attention_mask": encoder_attention_mask,
    }


def make_model(vocab_size: int = 32, dim: int = 8):
    cfg = ELFConfig(
        text_encoder_dim=dim,
        max_length=8,
        hidden_size=32,
        depth=2,
        num_heads=4,
        bottleneck_dim=4,
        vocab_size=vocab_size,
        gradient_checkpointing=False,
    )
    return ELF(cfg)


def make_config(decoder_prob: float) -> TrainingConfig:
    return TrainingConfig(
        max_length=8,
        decoder_prob=decoder_prob,
        self_cond_prob=0.5,
        num_self_cond_cfg_tokens=4,
        num_model_mode_tokens=4,
        denoiser_noise_scale=2.0,
        decoder_noise_scale=5.0,
    )


def test_forward_shapes():
    model = make_model()
    x = torch.randn(2, 8, 16)
    t = torch.rand(2)
    out = model(
        x,
        t,
        self_cond_cfg_scale=torch.ones(2),
        decoder_step_active=True,
    )
    assert out.denoised.shape == (2, 8, 8)
    assert out.decoder_logits is not None
    assert out.decoder_logits.shape == (2, 8, 32)


def test_denoiser_and_decoder_training_losses_backward():
    torch.manual_seed(0)
    for decoder_prob in (0.0, 1.0):
        model = make_model()
        encoder = TinyEncoder(vocab_size=32, dim=8)
        encoder.eval().requires_grad_(False)
        batch = make_batch()
        loss_out = compute_elf_loss(
            model,
            encoder,
            batch,
            make_config(decoder_prob),
            decoder_step_active=bool(decoder_prob),
        )
        assert torch.isfinite(loss_out.loss)
        loss_out.loss.backward()
        grad_norm = sum(
            p.grad.detach().float().norm().item()
            for p in model.parameters()
            if p.grad is not None
        )
        assert grad_norm > 0


def test_sampling_returns_token_ids():
    model = make_model()
    cfg = make_config(decoder_prob=0.0)
    tokens = sample_tokens(
        model,
        (2, 8, 8),
        cfg,
        num_steps=2,
        self_cond_cfg_scale=1.0,
        device=torch.device("cpu"),
    )
    assert tokens.shape == (2, 8)
    assert tokens.dtype == torch.long


def test_encode_text_converts_3d_keep_mask_to_4d_additive_mask():
    encoder = MaskCheckingEncoder(vocab_size=32, dim=8)
    ids = torch.randint(0, 32, (2, 8))
    keep = torch.ones(2, 8, 8)
    keep[:, 0, 1] = 0
    latents = encode_text(
        encoder,
        ids,
        keep,
        latent_mean=0.0,
        latent_std=1.0,
    )
    assert latents.shape == (2, 8, 8)
    assert encoder.last_mask is not None
    assert encoder.last_mask.shape == (2, 1, 8, 8)
    assert encoder.last_mask[:, :, 0, 1].lt(-1e20).all()
