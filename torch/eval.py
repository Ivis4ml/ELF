#!/usr/bin/env python
"""Evaluate a trained PyTorch ELF checkpoint: generates samples and reports PPL/BLEU/ROUGE.

Launch:
    torchrun --nproc_per_node=8 torch/eval.py \
        --config torch/configs/training_configs/train_owt_ELF-B.yml \
        --checkpoint_path outputs/elf_b-owt/checkpoint_19000.pt
"""

import argparse
import logging
import os
import sys

# Ensure the `torch/` folder itself (NOT the repo root) is on sys.path so imports
# look like `from modules.model import ELF_models` exactly as the JAX source does —
# critically, do NOT add the repo root, since that would let `import torch` resolve
# to our directory and shadow the PyTorch package.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch
import torch.distributed as dist
from transformers import AutoTokenizer

from configs.config import apply_config_overrides, load_config_from_yaml, load_sampling_configs
from generation import run_generation
from modules.model import ELF_models
from modules.t5_encoder import FrozenT5Encoder
from utils.checkpoint_utils import load_checkpoint
from utils.data_utils import get_pad_token_id, load_dataset_split, load_jsonl_dataset
from utils.logging_utils import is_main_process, log_for_0
from utils.metrics_utils import PerplexityEvaluator
from utils.train_utils import ParamEMA


logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--config_override", action="append", default=[])
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated list of seeds to evaluate")
    return p.parse_args()


def maybe_init_distributed():
    if "LOCAL_RANK" not in os.environ:
        return False
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return True


def main():
    args = parse_args()
    maybe_init_distributed()

    config = load_config_from_yaml(args.config)
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)

    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.seed)

    log_for_0(f"Loading tokenizer {config.tokenizer_name or config.encoder_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    log_for_0(f"Pad token id: {pad_token_id}")

    eval_dataset = None
    if config.eval_data_path is not None:
        if config.eval_data_path.endswith(".jsonl"):
            eval_dataset = load_jsonl_dataset(config.eval_data_path, tokenizer)
        else:
            eval_dataset = load_dataset_split(config.eval_data_path)
        log_for_0(f"Eval size: {len(eval_dataset)}")

    log_for_0(f"Loading encoder {config.encoder_model_name}")
    encoder = FrozenT5Encoder(config.encoder_model_name, dtype=torch.float32).to(device)

    vocab_size = len(tokenizer)
    log_for_0(f"Vocab: {vocab_size}")

    model_fn = ELF_models[config.model]
    model = model_fn(
        text_encoder_dim=encoder.d_model, max_length=config.max_length,
        vocab_size=vocab_size, bottleneck_dim=config.bottleneck_dim,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        num_model_mode_tokens=config.num_model_mode_tokens,
        attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
        self_cond=config.self_cond_prob > 0,
        activation_checkpointing=False,
    ).to(device)

    ema = ParamEMA(model.parameters(), decay=config.ema_decay1)
    step, epoch = load_checkpoint(args.checkpoint_path, model, optimizer=None, ema=ema, map_location=device)
    log_for_0(f"Loaded checkpoint at step={step} epoch={epoch}")

    if config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)

    seed_list = [int(s.strip()) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    ppl_evaluator = None
    if config.online_eval and eval_dataset is None and is_main_process():
        ppl_evaluator = PerplexityEvaluator(
            model_name=config.eval_ppl_model,
            batch_size=config.eval_ppl_batch_size,
            context_size=config.eval_ppl_max_length,
            device=device,
        )

    original_out = config.output_dir
    for seed_val in seed_list:
        torch.manual_seed(seed_val)
        if len(seed_list) > 1:
            log_for_0(f"\n=== Seed {seed_val} ===")
            config.output_dir = os.path.join(original_out, f"seed_{seed_val}")

        run_generation(
            model=model, ema=ema, encoder=encoder, eval_dataset=eval_dataset,
            tokenizer=tokenizer, config=config, epoch=epoch, step=step,
            ppl_evaluator=ppl_evaluator,
        )
        config.output_dir = original_out

    log_for_0("Eval complete.")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
