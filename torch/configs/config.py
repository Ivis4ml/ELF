"""Config dataclass and YAML loader for the PyTorch ELF port.

Mirrors src/configs/config.py one-to-one so YAMLs can be shared with the JAX
version with minor edits (encoder_checkpoint is unused here; we load T5 from
the HuggingFace hub directly).
"""

import os
import yaml


class SamplingConfig:
    """A single sampler sweep entry."""
    sampling_method: str = "ode"
    num_sampling_steps: list = [50]
    cfgs: list = [1]
    self_cond_cfg_scales: list = [1.0]
    time_schedule: str = "logit_normal"
    sde_gamma: float = 0.0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        fields = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        for k in self.__class__.__annotations__:
            if k not in fields:
                fields[k] = getattr(self, k, None)
        items = ", ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"SamplingConfig({items})"


class Config:
    # Dataset
    data_path: str = None
    eval_data_path: str = None
    max_length: int = 128
    max_input_length: int = None
    pad_token: str = "pad"

    # Tokenizer / encoder
    tokenizer_name: str = None
    encoder_model_name: str = "google-t5/t5-small"
    encoder_checkpoint: str = None  # unused in the PyTorch port; we pull T5 from HF
    latent_mean: float = 0.0
    latent_std: float = 1.0

    # Model architecture
    model: str = "ELF-B"
    bottleneck_dim: int = 128
    num_time_tokens: int = 4
    num_self_cond_cfg_tokens: int = 4
    num_model_mode_tokens: int = 0
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0

    # Denoiser objective
    denoiser_p_mean: float = 0.8
    denoiser_p_std: float = 0.8
    denoiser_noise_scale: float = 1.0
    t_eps: float = 5e-2
    time_schedule: str = "logit_normal"

    # Decoder objective
    decoder_prob: float = 0.5
    decoder_noise_scale: float = 1.0
    decoder_p_mean: float = 0.8
    decoder_p_std: float = 0.8

    # Conditioning / CFG
    label_drop_prob: float = 0.0
    self_cond_prob: float = 0.5
    self_cond_cfg_min: float = 0.5
    self_cond_cfg_max: float = 5.0

    # Training (optimizer + schedule)
    epochs: int = 200
    warmup_epochs: float = None
    warmup_steps: int = 5000
    batch_size: int = None
    global_batch_size: int = 512
    lr: float = None
    blr: float = 5e-5
    min_lr: float = 0.0
    lr_schedule: str = "constant"
    weight_decay: float = 0.0
    optimizer: str = "muon"
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    grad_accum_steps: int = 1
    grad_clip: float = 1.0

    # Mixed precision: bf16 forward is the only sensible default on H200.
    autocast_dtype: str = "bfloat16"  # "bfloat16", "float16", or "float32"
    activation_checkpointing: bool = False

    # EMA
    ema_decay1: float = 0.9999

    # Sampling
    sampling_configs_path: str = None
    sampling_configs: list = [SamplingConfig()]
    num_samples: int = 100

    # PPL evaluation
    online_eval: bool = True
    eval_ppl_model: str = "gpt2-large"
    eval_ppl_batch_size: int = 8
    eval_ppl_max_length: int = 1024

    # Logging / checkpointing
    log_freq: int = 100
    eval_freq: int = 10
    save_freq: float = 1
    keep_last_n: int = 10

    # Output
    output_dir: str = "./output_dir"
    hf_repo_id: str = None
    resume: str = None

    # Wandb
    use_wandb: bool = False
    wandb_project: str = "ELF"
    wandb_entity: str = None
    wandb_run_name: str = None
    wandb_tag: str = None
    wandb_resume: str = "allow"

    # Misc
    seed: int = 0
    num_workers: int = 0


def load_config_from_yaml(path: str) -> Config:
    config = Config()
    if not path or not os.path.isfile(path):
        return config

    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f) or {}

    for key, value in cfg_dict.items():
        if key == "sampling_configs":
            continue
        if hasattr(config, key):
            setattr(config, key, value)

    if config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)

    return config


def apply_config_overrides(config: Config, overrides: list) -> Config:
    if not overrides:
        return config

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: '{override}'. Expected 'field_name=value'")

        field_name, value_str = override.split("=", 1)
        field_name = field_name.strip()
        value_str = value_str.strip()

        if not hasattr(config, field_name):
            raise ValueError(f"Config has no field named '{field_name}'")

        original_value = getattr(config, field_name)
        original_type = type(original_value)

        if value_str.lower() == "none":
            setattr(config, field_name, None)
            continue

        if original_value is None:
            annotated_type = config.__annotations__.get(field_name)
            if annotated_type == int:
                converted = int(value_str)
            elif annotated_type == float:
                converted = float(value_str)
            elif annotated_type == bool:
                converted = value_str.lower() in ("true", "1", "yes")
            else:
                converted = value_str
        elif original_type == bool:
            converted = value_str.lower() in ("true", "1", "yes")
        elif original_type == int:
            converted = int(value_str)
        elif original_type == float:
            converted = float(value_str)
        elif original_type == str:
            converted = value_str
        else:
            converted = value_str

        setattr(config, field_name, converted)

    return config


def load_sampling_configs(path: str):
    with open(path, "r") as f:
        entries = yaml.safe_load(f)
    return [SamplingConfig(**entry) for entry in entries]
