#!/usr/bin/env python
"""ELF training entry point (PyTorch, DDP-only).

Launch on a single 8xH200 node:

    cd torch
    torchrun --standalone --nproc_per_node=8 train.py \
        --config configs/training_configs/train_owt_ELF-B.yml

Notes
-----
* This repo's `torch/` directory must NOT be added to `sys.path` because the
  name shadows the PyTorch package. We only prepend `torch/` itself, and run
  the script from inside it (or via `torchrun torch/train.py`).
* Mixed precision: bf16 forward via autocast, fp32 params + fp32 gradients
  through DDP so all-reduce is bit-identical across ranks. This keeps Muon's
  Newton-Schulz output bit-identical without needing a distributed Muon impl.
* Decoder/denoiser branch decision is rank-broadcast each step to keep DDP's
  gradient bucketing consistent.
"""

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from typing import Optional

# CRITICAL: only the `torch/` directory itself goes on sys.path. Do NOT add the
# repo root; that would let the local `torch/` dir shadow the PyTorch package.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer

from configs.config import (
    apply_config_overrides, load_config_from_yaml, load_sampling_configs, SamplingConfig,
)
from modules.model import ELF_models
from modules.t5_encoder import FrozenT5Encoder
from utils.checkpoint_utils import find_latest_checkpoint, load_checkpoint, save_checkpoint
from utils.data_utils import get_dataloader, get_pad_token_id, load_dataset, prepare_batch
from utils.logging_utils import is_main_process, log_for_0
from utils.muon import build_optimizer
from utils.train_utils import (
    LRSchedule, ParamEMA, clip_grad_norm_, count_parameters, maybe_unwrap_ddp,
)
from train_step import train_step
from generation import run_generation
from utils.metrics_utils import PerplexityEvaluator


logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--config_override", action="append", default=[])
    return p.parse_args()


# ============================================
# Distributed init
# ============================================
def setup_distributed() -> int:
    """Return local_rank (0 if single process). Initializes NCCL when launched by torchrun."""
    if "LOCAL_RANK" not in os.environ:
        return 0
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank


# ============================================
# Branch sync (decoder vs denoiser) across ranks
# ============================================
def broadcast_branch(decoder_prob: float, device: torch.device, generator: torch.Generator) -> bool:
    """Sample the Bernoulli on rank 0 and broadcast so all ranks agree."""
    flip = torch.zeros(1, dtype=torch.uint8, device=device)
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        flip.fill_(1 if torch.rand((), generator=generator, device=device).item() < decoder_prob else 0)
    if dist.is_initialized():
        dist.broadcast(flip, src=0)
    return bool(flip.item())


# ============================================
# Main training loop
# ============================================
def run_training(config):
    local_rank = setup_distributed()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    log_for_0("=" * 60)
    log_for_0("ELF (PyTorch) — DDP training")
    log_for_0("=" * 60)
    log_for_0(f"Model: {config.model}  Encoder: {config.encoder_model_name}")
    log_for_0(f"Data: {config.data_path}  Max length: {config.max_length}")
    log_for_0(f"World size: {world_size}  Device: {device}")
    log_for_0("=" * 60)

    autocast_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                      "float32": torch.float32}[config.autocast_dtype]

    # -------- wandb --------
    if config.use_wandb and is_main_process():
        import wandb
        wandb_tags = config.wandb_tag.split(",") if config.wandb_tag else None
        wandb.init(
            project=config.wandb_project, entity=config.wandb_entity,
            name=config.wandb_run_name, id=config.wandb_run_name,
            resume=config.wandb_resume, tags=wandb_tags,
            config={k: getattr(config, k) for k in dir(config) if not k.startswith("_")},
            dir="/tmp",
        )

    torch.manual_seed(config.seed + rank)

    # -------- tokenizer + data --------
    log_for_0(f"Loading tokenizer {config.tokenizer_name or config.encoder_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    log_for_0(f"Pad token id: {pad_token_id}")

    train_dataset, eval_dataset = load_dataset(config)

    # -------- encoder (frozen) --------
    log_for_0(f"Loading encoder {config.encoder_model_name}")
    encoder = FrozenT5Encoder(config.encoder_model_name, dtype=torch.float32).to(device)
    log_for_0(f"Encoder d_model = {encoder.d_model}")

    # -------- model --------
    vocab_size = len(tokenizer)
    log_for_0(f"Vocab size for CE head: {vocab_size}")
    model_fn = ELF_models[config.model]
    model = model_fn(
        text_encoder_dim=encoder.d_model,
        max_length=config.max_length,
        vocab_size=vocab_size,
        bottleneck_dim=config.bottleneck_dim,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        num_model_mode_tokens=config.num_model_mode_tokens,
        attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
        self_cond=config.self_cond_prob > 0,
        activation_checkpointing=config.activation_checkpointing,
    ).to(device)
    log_for_0(f"ELF parameters: {count_parameters(model):,}")

    # DDP wrap. find_unused_parameters=True because the decode branch touches the
    # unembedding head and mode tokens that the denoise branch doesn't touch.
    if world_size > 1:
        model = DDP(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, gradient_as_bucket_view=True,
            broadcast_buffers=False,
        )

    # -------- batch sizes --------
    total_bs = config.global_batch_size
    local_bs = total_bs // world_size
    if local_bs * world_size != total_bs:
        raise ValueError(f"global_batch_size={total_bs} not divisible by world_size={world_size}")
    config.batch_size = local_bs
    steps_per_epoch = len(train_dataset) // total_bs
    num_train_steps = steps_per_epoch * config.epochs

    grad_accum = max(1, config.grad_accum_steps)
    num_optimizer_steps = num_train_steps // grad_accum

    if config.warmup_steps and config.warmup_steps >= 0:
        num_warmup_steps = config.warmup_steps
    elif config.warmup_epochs is not None:
        num_warmup_steps = int(config.warmup_epochs * steps_per_epoch)
    else:
        num_warmup_steps = 0
    num_warmup_optimizer_steps = num_warmup_steps // grad_accum

    if config.lr is None or config.lr <= 0:
        config.lr = config.blr * (total_bs * grad_accum) / 256
    log_for_0(
        f"local_bs={local_bs} total_bs={total_bs} | steps/epoch={steps_per_epoch} | "
        f"total_steps={num_train_steps} | warmup={num_warmup_steps} | lr={config.lr:.2e}"
    )

    # -------- optimizer + LR sched --------
    optimizer = build_optimizer(config, maybe_unwrap_ddp(model))
    lr_schedule = LRSchedule(
        num_train_steps=num_optimizer_steps, num_warmup_steps=num_warmup_optimizer_steps,
        lr=config.lr, schedule=config.lr_schedule, min_lr=config.min_lr,
    )

    # -------- EMA --------
    ema = ParamEMA(maybe_unwrap_ddp(model).parameters(), decay=config.ema_decay1)

    # -------- resume --------
    start_step, start_epoch = 0, 0.0
    if not config.resume:
        auto = find_latest_checkpoint(config.output_dir)
        if auto:
            config.resume = config.output_dir
            log_for_0(f"Auto-resuming from {auto}")
    if config.resume:
        try:
            start_step, start_epoch = load_checkpoint(
                config.resume, model, optimizer, ema, map_location=device,
            )
            log_for_0(f"Resumed at step={start_step} epoch={start_epoch:.2f}")
        except Exception as e:
            log_for_0(f"Resume failed ({e}); training from scratch.")
            start_step, start_epoch = 0, 0.0

    # -------- output dir + config dump --------
    os.makedirs(config.output_dir, exist_ok=True)
    if is_main_process():
        cfg_dump = {
            k: ([vars(sc) for sc in v]
                if isinstance(v, list) and v and isinstance(v[0], SamplingConfig)
                else v)
            for k, v in vars(config).items()
        }
        with open(os.path.join(config.output_dir, "config.yml"), "w") as f:
            yaml.dump(cfg_dump, f, default_flow_style=False, sort_keys=False)
        log_for_0("Config saved.")

    if config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)

    # -------- data loader --------
    train_loader = get_dataloader(
        train_dataset, batch_size=local_bs, shuffle=True,
        num_workers=config.num_workers, drop_last=True,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length,
        distributed=world_size > 1, seed=config.seed,
    )

    # PPL evaluator only on rank 0 (eats GPU memory; the eval set is tiny).
    ppl_evaluator = None
    if config.online_eval and eval_dataset is None and is_main_process():
        ppl_evaluator = PerplexityEvaluator(
            model_name=config.eval_ppl_model,
            batch_size=config.eval_ppl_batch_size,
            context_size=config.eval_ppl_max_length,
            device=device,
        )

    # -------- training loop --------
    global_step = start_step
    last_log_step = global_step
    train_metrics = []
    last_log_time = time.time()
    last_save_epoch = start_epoch

    branch_gen = torch.Generator(device=device).manual_seed(config.seed * 7919 + 1)

    int_start_epoch = int(math.floor(start_epoch))
    for epoch in range(int_start_epoch, config.epochs):
        log_for_0(f"\nEpoch {epoch + 1}/{config.epochs}")
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)

        # Skip already-processed batches when resuming mid-epoch.
        steps_to_skip = max(0, start_step - epoch * steps_per_epoch) if epoch == int_start_epoch else 0

        # Branch decision is decided ONCE per grad-accumulation window, not per
        # micro-batch. With per-micro-batch decisions, a window like
        # [decode, decode, denoise] would leave `dec_proj.grad` accumulated
        # locally (different per rank) when the sync micro-batch reports
        # `dec_proj` as unused; PyTorch DDP's behavior for prior-accumulated
        # grads on currently-unused params with find_unused_parameters=True is
        # implementation-defined, and the safest assumption is that the
        # already-accumulated grad won't be all-reduced — ranks then drift.
        # Per-window decisions sidestep the issue: every micro-batch in the
        # window uses the same param set, so the sync sees a consistent
        # used-param set across ranks.
        current_branch = False  # initialized; will be re-decided at each window boundary

        model.train()
        for step_in_epoch, batch in enumerate(train_loader):
            if epoch == int_start_epoch and step_in_epoch < steps_to_skip:
                continue

            # Sample a fresh branch at the start of each accumulation window
            # (i.e. when zero_grad was just called, or at the very start).
            if step_in_epoch % grad_accum == 0:
                current_branch = broadcast_branch(config.decoder_prob, device, branch_gen)

            batch = prepare_batch(batch, config)

            # Gradient accumulation: only the LAST micro-batch in the window all-reduces.
            is_last_micro = ((step_in_epoch + 1) % grad_accum == 0)
            ctx = model.no_sync() if (world_size > 1 and not is_last_micro) else _nullctx()
            with ctx:
                loss, metrics, _ = train_step(
                    model=model, encoder=encoder, batch=batch, config=config,
                    autocast_dtype=autocast_dtype, is_decoder_step=current_branch,
                )
                (loss / grad_accum).backward()

            if is_last_micro:
                if config.grad_clip and config.grad_clip > 0:
                    clip_grad_norm_(maybe_unwrap_ddp(model).parameters(), config.grad_clip)

                current_lr = lr_schedule(global_step // grad_accum)
                if hasattr(optimizer, "set_lr"):
                    optimizer.set_lr(current_lr)
                else:
                    for g in optimizer.param_groups:
                        g["lr"] = current_lr

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # EMA update only on optimizer-step boundaries.
                ema.update(maybe_unwrap_ddp(model).parameters())

            # All-reduce metric tensors before host transfer so the logged values
            # reflect the global batch's loss, not just rank 0's slice. Without
            # this the wandb loss curve wobbles by per-rank batch variance —
            # JAX's reference does the equivalent via `lax.pmean` in train_step.
            if dist.is_initialized() and world_size > 1:
                for k, v in metrics.items():
                    if isinstance(v, torch.Tensor):
                        dist.all_reduce(v, op=dist.ReduceOp.AVG)
            train_metrics.append({k: v.detach().float().cpu().item() if isinstance(v, torch.Tensor) else float(v)
                                   for k, v in metrics.items()})
            global_step += 1

            if global_step % config.log_freq == 0:
                avg = {k: sum(m[k] for m in train_metrics) / len(train_metrics) for k in train_metrics[0]}
                # Re-scale per-branch losses so each branch reports its conditional mean.
                if config.decoder_prob > 0:
                    avg["ce_loss"] /= config.decoder_prob
                if (1 - config.decoder_prob) > 0:
                    avg["l2_loss"] /= (1.0 - config.decoder_prob)

                now = time.time()
                sps = (global_step - last_log_step) / max(now - last_log_time, 1e-8)
                cur_lr = lr_schedule(max(0, (global_step - 1) // grad_accum))
                log_for_0(
                    f"Step {global_step}: loss={avg['loss']:.4f} "
                    f"l2={avg['l2_loss']:.4f} ce={avg['ce_loss']:.4f} "
                    f"lr={cur_lr:.2e} sps={sps:.2f}"
                )
                if config.use_wandb and is_main_process():
                    import wandb
                    wandb.log({
                        "train_loss": avg["loss"], "train_l2_loss": avg["l2_loss"],
                        "train_ce_loss": avg["ce_loss"], "lr": cur_lr,
                        "epoch": epoch + (step_in_epoch + 1) / steps_per_epoch,
                        "step": global_step,
                    }, step=global_step)
                train_metrics.clear()
                last_log_step = global_step
                last_log_time = now

            # Intra-epoch fractional saves.
            if 0 < config.save_freq < 1:
                progress = epoch + (step_in_epoch + 1) / steps_per_epoch
                if progress - last_save_epoch >= config.save_freq:
                    save_checkpoint(
                        model, optimizer, ema, global_step, progress,
                        config.output_dir, keep_last_n=config.keep_last_n,
                        hf_repo_id=config.hf_repo_id,
                    )
                    last_save_epoch = progress

            if step_in_epoch + 1 >= steps_per_epoch:
                break

        current_epoch = epoch + 1
        if config.save_freq >= 1 and current_epoch % int(config.save_freq) == 0:
            save_checkpoint(
                model, optimizer, ema, global_step, float(current_epoch),
                config.output_dir, keep_last_n=config.keep_last_n,
                hf_repo_id=config.hf_repo_id,
            )

        if config.eval_freq >= 1 and current_epoch % int(config.eval_freq) == 0:
            run_generation(
                model=model, ema=ema, encoder=encoder, eval_dataset=eval_dataset,
                tokenizer=tokenizer, config=config,
                epoch=current_epoch, step=global_step,
                ppl_evaluator=ppl_evaluator,
            )
            last_log_time = time.time()
            last_log_step = global_step

    log_for_0("Training complete; writing final checkpoint.")
    save_checkpoint(
        model, optimizer, ema, global_step, float(config.epochs),
        config.output_dir, keep_last_n=config.keep_last_n,
        hf_repo_id=config.hf_repo_id,
    )
    if config.use_wandb and is_main_process():
        import wandb
        wandb.finish()
    if dist.is_initialized():
        dist.destroy_process_group()


class _nullctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def main():
    args = parse_args()
    config = load_config_from_yaml(args.config)
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)
        log_for_0(f"Applied {len(args.config_override)} override(s)")
    run_training(config)


if __name__ == "__main__":
    main()
