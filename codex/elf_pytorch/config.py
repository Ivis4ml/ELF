from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ELFConfig:
    text_encoder_dim: int = 512
    max_length: int = 1024
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    bottleneck_dim: int = 128
    num_time_tokens: int = 4
    num_self_cond_cfg_tokens: int = 4
    num_model_mode_tokens: int = 4
    vocab_size: int = 32128
    gradient_checkpointing: bool = True


@dataclass
class TrainingConfig:
    # Dataset/tokenizer/encoder
    data_path: str | None = None
    eval_data_path: str | None = None
    dataset_cache_dir: str | None = None
    max_length: int = 1024
    max_input_length: int | None = None
    pad_token: str = "pad"
    tokenizer_name: str | None = None
    encoder_model_name: str = "google-t5/t5-small"
    latent_mean: float = 0.0
    latent_std: float = 0.2

    # Model
    model: str = "ELF-B"
    bottleneck_dim: int = 128
    num_time_tokens: int = 4
    num_self_cond_cfg_tokens: int = 4
    num_model_mode_tokens: int = 4
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0
    gradient_checkpointing: bool = True
    torch_compile: bool = False

    # Denoiser objective
    denoiser_p_mean: float = -1.5
    denoiser_p_std: float = 0.8
    denoiser_noise_scale: float = 2.0
    t_eps: float = 5e-2
    time_schedule: str = "logit_normal"

    # Decoder objective
    decoder_prob: float = 0.2
    decoder_noise_scale: float = 5.0
    decoder_p_mean: float = 0.8
    decoder_p_std: float = 0.8

    # Conditioning / CFG
    label_drop_prob: float = 0.0
    self_cond_prob: float = 0.5
    self_cond_cfg_min: float = 0.5
    self_cond_cfg_max: float = 5.0

    # Training
    epochs: int = 5
    warmup_epochs: float | None = 0.5
    warmup_steps: int = -1
    global_batch_size: int = 512
    micro_batch_size: int | None = None
    lr: float | None = None
    blr: float = 0.001
    min_lr: float = 0.0
    lr_schedule: str = "constant"
    weight_decay: float = 0.0
    optimizer: str = "muon"
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    precision: str = "bf16"

    # EMA/checkpoint/logging
    ema_decay: float = 0.9999
    ema_device: str = "cuda"
    log_freq: int = 100
    save_freq: float = 1.0
    output_dir: str = "outputs/elf_b-owt-pytorch"
    resume: str | None = None
    use_wandb: bool = False
    wandb_project: str = "elf"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None

    # Runtime
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4
    persistent_workers: bool = True
    allow_tf32: bool = True

    def to_elf_config(self, text_encoder_dim: int, vocab_size: int) -> ELFConfig:
        depth, hidden_size, num_heads = {
            "ELF-B": (12, 768, 12),
            "ELF-M": (24, 1056, 16),
            "ELF-L": (32, 1280, 16),
        }[self.model]
        return ELFConfig(
            text_encoder_dim=text_encoder_dim,
            max_length=self.max_length,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            attn_dropout=self.attn_dropout,
            proj_dropout=self.proj_dropout,
            bottleneck_dim=self.bottleneck_dim,
            num_time_tokens=self.num_time_tokens,
            num_self_cond_cfg_tokens=self.num_self_cond_cfg_tokens,
            num_model_mode_tokens=self.num_model_mode_tokens,
            vocab_size=vocab_size,
            gradient_checkpointing=self.gradient_checkpointing,
        )


def _coerce_value(value: str, current: Any) -> Any:
    if value.lower() == "none":
        return None
    if isinstance(current, bool):
        return value.lower() in {"1", "true", "yes", "y"}
    if isinstance(current, int) and not isinstance(current, bool):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


def load_config(path: str | None = None, overrides: list[str] | None = None) -> TrainingConfig:
    cfg = TrainingConfig()
    valid = {field.name for field in fields(TrainingConfig)}
    if path:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        unknown = sorted(set(data) - valid)
        if unknown:
            raise ValueError(f"Unknown config field(s): {unknown}")
        for key, value in data.items():
            setattr(cfg, key, value)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}; expected key=value.")
        key, value = override.split("=", 1)
        key = key.strip()
        if key not in valid:
            raise ValueError(f"Unknown config field: {key}")
        setattr(cfg, key, _coerce_value(value.strip(), getattr(cfg, key)))
    return cfg


def dump_config(config: TrainingConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(asdict(config), handle, sort_keys=False)
