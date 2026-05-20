from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, T5EncoderModel

from .config import TrainingConfig, dump_config, load_config
from .data import create_dataloader, get_pad_token_id, load_train_eval_datasets
from .model import ELF
from .optim import build_optimizers, get_lr, set_optimizer_lr
from .runtime import (
    EMA,
    barrier,
    broadcast_decoder_branch,
    cleanup_distributed,
    init_distributed,
    is_main_process,
    load_checkpoint,
    log0,
    save_checkpoint,
    seed_everything,
    unwrap_model,
)
from .training import compute_elf_loss, move_batch_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ELF with pure PyTorch/DDP.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--config_override",
        action="append",
        default=[],
        help="Override config values as key=value. Can be repeated.",
    )
    return parser.parse_args()


def autocast_context(config: TrainingConfig, device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    if config.precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if config.precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def configure_torch(config: TrainingConfig) -> None:
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.allow_tf32
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)


def load_tokenizer_and_encoder(config: TrainingConfig, device: torch.device):
    tokenizer_name = config.tokenizer_name or config.encoder_model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encoder = T5EncoderModel.from_pretrained(config.encoder_model_name)
    encoder.eval().requires_grad_(False)
    encoder.to(device)
    return tokenizer, encoder


def make_model(config: TrainingConfig, encoder: T5EncoderModel, vocab_size: int, device: torch.device):
    elf_config = config.to_elf_config(
        text_encoder_dim=int(encoder.config.d_model),
        vocab_size=vocab_size,
    )
    model = ELF(elf_config).to(device)
    if config.torch_compile:
        model = torch.compile(model)  # type: ignore[assignment]
    return model


def run_training(config: TrainingConfig) -> None:
    rank, world_size, local_rank, device = init_distributed()
    configure_torch(config)
    seed_everything(config.seed, rank)

    try:
        if rank == 0:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)
            dump_config(config, Path(config.output_dir) / "config.yml")

        log0("=" * 72)
        log0("ELF PyTorch training")
        log0("=" * 72)
        log0(f"model={config.model} encoder={config.encoder_model_name}")
        log0(f"world_size={world_size} device={device} precision={config.precision}")

        # Let rank 0 populate HF caches first on shared filesystems.
        if rank == 0:
            load_tokenizer_and_encoder(config, device=torch.device("cpu"))
        barrier()
        tokenizer, encoder = load_tokenizer_and_encoder(config, device)
        pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
        vocab_size = len(tokenizer)

        train_dataset, _ = load_train_eval_datasets(config)
        if config.micro_batch_size is None:
            if config.global_batch_size % world_size != 0:
                raise ValueError(
                    f"global_batch_size={config.global_batch_size} must divide world_size={world_size}."
                )
            local_batch_size = config.global_batch_size // world_size
            global_forward_batch = config.global_batch_size
        else:
            local_batch_size = config.micro_batch_size
            global_forward_batch = local_batch_size * world_size
        effective_batch = global_forward_batch * config.grad_accum_steps
        steps_per_epoch = len(train_dataset) // global_forward_batch
        total_micro_steps = steps_per_epoch * config.epochs
        total_optimizer_steps = max(total_micro_steps // config.grad_accum_steps, 1)
        if config.warmup_steps >= 0:
            warmup_micro_steps = config.warmup_steps
        elif config.warmup_epochs is not None:
            warmup_micro_steps = int(config.warmup_epochs * steps_per_epoch)
        else:
            warmup_micro_steps = 0
        warmup_optimizer_steps = warmup_micro_steps // config.grad_accum_steps
        if config.lr is None or config.lr <= 0:
            config.lr = config.blr * effective_batch / 256

        log0(
            "batch: "
            f"local={local_batch_size} global_forward={global_forward_batch} "
            f"grad_accum={config.grad_accum_steps} effective={effective_batch}"
        )
        log0(
            f"steps_per_epoch={steps_per_epoch} optimizer_steps={total_optimizer_steps} "
            f"warmup_optimizer_steps={warmup_optimizer_steps} lr={config.lr:.3e}"
        )

        model = make_model(config, encoder, vocab_size, device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log0(f"ELF trainable parameters: {trainable_params:,}")
        if world_size > 1:
            model = DDP(
                model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
                broadcast_buffers=False,
            )

        optimizers = build_optimizers(
            unwrap_model(model),
            optimizer=config.optimizer,
            lr=config.lr,
            weight_decay=config.weight_decay,
            adam_b1=config.adam_b1,
            adam_b2=config.adam_b2,
        )
        ema_device = device if config.ema_device == "cuda" and device.type == "cuda" else torch.device("cpu")
        ema = EMA(unwrap_model(model), config.ema_decay, device=ema_device)

        start_epoch = 0
        global_micro_step = 0
        optimizer_step = 0
        if config.resume:
            start_epoch, global_micro_step = load_checkpoint(
                config.resume,
                model=model,
                optimizers=optimizers,
                ema=ema,
                map_location=device,
            )
            optimizer_step = global_micro_step // config.grad_accum_steps
            log0(f"Resumed from {config.resume}: epoch={start_epoch} step={global_micro_step}")

        train_loader = create_dataloader(
            train_dataset,
            batch_size=local_batch_size,
            rank=rank,
            world_size=world_size,
            shuffle=True,
            num_workers=config.num_workers,
            max_seq_length=config.max_length,
            pad_token_id=pad_token_id,
            max_input_seq_length=config.max_input_length,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_factor,
            persistent_workers=config.persistent_workers,
        )

        wandb_run = None
        if config.use_wandb and is_main_process():
            import wandb

            wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.wandb_run_name,
                config=config.__dict__,
            )

        metric_sums = torch.zeros(3, device=device, dtype=torch.float32)
        metric_count = 0
        last_log_time = time.time()
        branch_gen = (
            torch.Generator(device=device).manual_seed(config.seed * 7919 + 1)
            if device.type == "cuda"
            else torch.Generator().manual_seed(config.seed * 7919 + 1)
        )
        current_branch_is_decoder = False
        for epoch in range(start_epoch, config.epochs):
            train_loader.sampler.set_epoch(epoch)
            log0(f"Epoch {epoch + 1}/{config.epochs}")
            for batch in train_loader:
                if global_micro_step >= total_micro_steps:
                    break
                if global_micro_step % config.grad_accum_steps == 0:
                    current_branch_is_decoder = broadcast_decoder_branch(
                        config.decoder_prob, device, branch_gen
                    )
                model.train()
                batch = move_batch_to_device(batch, device)
                sync_grads = (global_micro_step + 1) % config.grad_accum_steps == 0
                ddp_no_sync = (
                    model.no_sync()
                    if isinstance(model, DDP) and not sync_grads
                    else nullcontext()
                )
                with ddp_no_sync:
                    with autocast_context(config, device):
                        losses = compute_elf_loss(
                            model,
                            encoder,
                            batch,
                            config,
                            decoder_step_active=current_branch_is_decoder,
                        )
                        loss_for_backward = losses.loss / config.grad_accum_steps
                    loss_for_backward.backward()

                metric_sums += torch.stack(
                    [
                        losses.loss.detach().float(),
                        losses.l2_loss.detach().float(),
                        losses.ce_loss.detach().float(),
                    ]
                )
                metric_count += 1

                global_micro_step += 1
                if sync_grads:
                    lr = get_lr(
                        optimizer_step,
                        total_steps=total_optimizer_steps,
                        warmup_steps=warmup_optimizer_steps,
                        learning_rate=config.lr,
                        schedule=config.lr_schedule,
                        min_lr=config.min_lr,
                    )
                    set_optimizer_lr(optimizers, lr)
                    if config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            unwrap_model(model).parameters(), config.max_grad_norm
                        )
                    for optimizer in optimizers:
                        optimizer.step()
                    for optimizer in optimizers:
                        optimizer.zero_grad(set_to_none=True)
                    ema.update(unwrap_model(model))
                    optimizer_step += 1

                if global_micro_step % config.log_freq == 0 and metric_count > 0:
                    now = time.time()
                    dt = max(now - last_log_time, 1e-8)
                    steps_per_sec = config.log_freq / dt
                    metric_packet = torch.cat(
                        [
                            metric_sums,
                            torch.tensor([metric_count], device=device, dtype=torch.float32),
                        ]
                    )
                    if dist.is_available() and dist.is_initialized():
                        dist.all_reduce(metric_packet, op=dist.ReduceOp.SUM)
                    avg_values = metric_packet[:3] / metric_packet[3].clamp_min(1.0)
                    avg = {
                        "loss": float(avg_values[0].item()),
                        "l2_loss": float(avg_values[1].item()),
                        "ce_loss": float(avg_values[2].item()),
                    }
                    current_lr = optimizers[0].param_groups[0]["lr"]
                    log0(
                        f"step={global_micro_step} opt_step={optimizer_step} "
                        f"loss={avg['loss']:.4f} l2={avg['l2_loss']:.4f} "
                        f"ce={avg['ce_loss']:.4f} lr={current_lr:.2e} "
                        f"micro_steps/s={steps_per_sec:.2f}"
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "train_loss": avg["loss"],
                                "train_l2_loss": avg["l2_loss"],
                                "train_ce_loss": avg["ce_loss"],
                                "lr": current_lr,
                                "epoch": epoch + global_micro_step / max(total_micro_steps, 1),
                            },
                            step=global_micro_step,
                        )
                    metric_sums.zero_()
                    metric_count = 0
                    last_log_time = now

                if 0 < config.save_freq < 1:
                    progress = (global_micro_step % max(steps_per_epoch, 1)) / max(
                        steps_per_epoch, 1
                    )
                    save_every = max(1, int(config.save_freq * steps_per_epoch))
                    if global_micro_step % save_every == 0 and is_main_process():
                        ckpt_path = Path(config.output_dir) / f"checkpoint_{global_micro_step}.pt"
                        save_checkpoint(
                            ckpt_path,
                            model=model,
                            optimizers=optimizers,
                            ema=ema,
                            epoch=epoch,
                            step=global_micro_step,
                            config=config.__dict__,
                        )
                        log0(f"Saved checkpoint at epoch progress {epoch + progress:.2f}: {ckpt_path}")

            if config.save_freq >= 1 and (epoch + 1) % int(config.save_freq) == 0:
                if is_main_process():
                    ckpt_path = Path(config.output_dir) / f"checkpoint_{global_micro_step}.pt"
                    save_checkpoint(
                        ckpt_path,
                        model=model,
                        optimizers=optimizers,
                        ema=ema,
                        epoch=epoch + 1,
                        step=global_micro_step,
                        config=config.__dict__,
                    )
                    log0(f"Saved checkpoint: {ckpt_path}")
                barrier()

        if is_main_process():
            final_path = Path(config.output_dir) / "checkpoint_final.pt"
            save_checkpoint(
                final_path,
                model=model,
                optimizers=optimizers,
                ema=ema,
                epoch=config.epochs,
                step=global_micro_step,
                config=config.__dict__,
            )
            log0(f"Final checkpoint saved: {final_path}")
        if wandb_run is not None:
            wandb_run.finish()
    finally:
        cleanup_distributed()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.config_override)
    run_training(config)


if __name__ == "__main__":
    main()
