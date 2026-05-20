"""Test-time generation entry points: unconditional + conditional.

All ranks generate sample shards; only rank 0 writes the file and computes
PPL/BLEU/ROUGE on the full set. Sample text is gathered via simple Gloo
all_gather_object, which is fine at the scale of 1K-5K samples.
"""

import itertools
import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from configs.config import Config, SamplingConfig
from utils.checkpoint_utils import upload_output_dir_to_hf
from utils.data_utils import get_dataloader, get_pad_token_id
from utils.encoder_utils import encode_text
from utils.generation_utils import (
    build_run_name, decode_batch, mask_after_eos, run_sampling_single_batch, shift_left,
)
from utils.logging_utils import is_main_process, log_for_0
from utils.metrics_utils import PerplexityEvaluator, compute_bleu, compute_rouge
from utils.sampling_utils import get_sampling_steps
from utils.train_utils import maybe_unwrap_ddp


def _gather_objects(obj):
    if not (dist.is_available() and dist.is_initialized()):
        return [obj]
    out = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(out, obj)
    return out


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def run_generation(model, ema, encoder, eval_dataset, tokenizer, config: Config,
                   epoch: int, step: int, ppl_evaluator: Optional[PerplexityEvaluator] = None):
    """Loop over sampling_configs and run the matching test fn."""
    # Switch to EMA params for generation, restore after.
    inner = maybe_unwrap_ddp(model)
    backup = ema.swap_into(inner.parameters()) if ema is not None else None
    inner.eval()
    try:
        for sc_idx, sc in enumerate(config.sampling_configs):
            if len(config.sampling_configs) > 1:
                log_for_0(f"\n--- Sampling config {sc_idx + 1}/{len(config.sampling_configs)} ---")

            local_bs = max(1, config.global_batch_size // _world_size())
            kwargs = dict(
                model=inner, tokenizer=tokenizer, config=config, sampling_config=sc,
                batch_size=local_bs, num_samples=config.num_samples,
                epoch=epoch, step=step,
            )
            if eval_dataset is None:
                test_generation_uncond(ppl_evaluator=ppl_evaluator, **kwargs)
            else:
                test_generation_cond(encoder=encoder, dataset=eval_dataset, **kwargs)
    finally:
        if backup is not None:
            ema.restore_into(inner.parameters(), backup)
        inner.train()


# ============================================
# Unconditional
# ============================================

@torch.no_grad()
def test_generation_uncond(model, tokenizer, config: Config, sampling_config: SamplingConfig,
                           epoch: int, step: int, num_samples: int, batch_size: int,
                           ppl_evaluator: Optional[PerplexityEvaluator] = None):
    sampling_method = sampling_config.sampling_method
    time_schedule = sampling_config.time_schedule
    log_for_0(f"Config: {sampling_config}")

    device = next(model.parameters()).device
    pad_token_id = get_pad_token_id(tokenizer)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    cfg_list = [1]
    steps_list = sampling_config.num_sampling_steps
    sccfg_list = sampling_config.self_cond_cfg_scales

    encoder_d = config_encoder_d(model)

    for num_steps, cfg_scale, sccfg in itertools.product(steps_list, cfg_list, sccfg_list):
        log_for_0(f"--- {sampling_method} steps={num_steps} cfg={cfg_scale} sccfg={sccfg} ---")

        # Each rank generates a non-overlapping shard.
        per_rank_samples = num_samples // _world_size()
        leftover = num_samples - per_rank_samples * _world_size()
        if _rank() < leftover:
            per_rank_samples += 1

        num_batches = (per_rank_samples + batch_size - 1) // batch_size
        all_local = []
        gen_t, dec_t = 0.0, 0.0

        for bi in tqdm(range(num_batches), desc="generate", disable=not is_main_process()):
            cur = min(batch_size, per_rank_samples - len(all_local))
            if cur <= 0:
                break
            gen = torch.Generator(device=device).manual_seed(
                config.seed * 9973 + (epoch * 10007 + bi) * _world_size() + _rank()
            )
            t_steps = get_sampling_steps(
                n_steps=num_steps, time_schedule=time_schedule,
                P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                device=device, generator=gen,
            )
            z = torch.randn(cur, config.max_length, encoder_d, device=device, generator=gen) \
                * config.denoiser_noise_scale

            gs = time.time()
            latent = run_sampling_single_batch(
                model, z, t_steps, cond_seq=None, cond_seq_mask=None,
                config=config, sampling_config=sampling_config,
                cfg_scale=cfg_scale, self_cond_cfg_scale=sccfg, generator=gen,
            )
            gen_t += time.time() - gs

            ds = time.time()
            t_final = float(t_steps[-1].item())
            ids = decode_batch(model, latent, t_final, config, sccfg)
            dec_t += time.time() - ds

            ids = mask_after_eos(ids, eos_token_id, pad_token_id).cpu().numpy()
            for row in ids:
                text = tokenizer.decode(row, skip_special_tokens=True)
                all_local.append(text)

        log_for_0(f"Generation: {gen_t:.2f}s | Decode: {dec_t:.2f}s")

        # Gather across ranks (rank 0 keeps full list).
        gathered = _gather_objects(all_local)
        if is_main_process():
            full = [t for shard in gathered for t in shard][:num_samples]

            name = build_run_name(
                sampling_method, num_steps, cfg_scale, sccfg, time_schedule,
                getattr(sampling_config, "sde_gamma", 0.0), suffix="uncond",
            )
            out_dir = os.path.join(config.output_dir, name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"all_generated_{epoch}_{step}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for tid, text in enumerate(full):
                    f.write(json.dumps({"id": tid, "generated": text}, ensure_ascii=False) + "\n")
            log_for_0(f"Saved {len(full)} samples to {out_path}")

            if config.online_eval and ppl_evaluator is not None:
                nonempty = [s for s in full if isinstance(s, str) and s.strip()]
                if not nonempty:
                    log_for_0("All samples empty; skipping PPL.")
                else:
                    res = ppl_evaluator.evaluate(nonempty, max_length=config.eval_ppl_max_length)
                    log_for_0(f"PPL: {res['ppl']:.4f}  H: {res['mean_entropy']:.4f}")
                    with open(os.path.join(out_dir, "metrics.jsonl"), "a", encoding="utf-8") as f:
                        f.write(json.dumps({"epoch": epoch, "step": step,
                                            "ppl": res["ppl"],
                                            "mean_entropy": res["mean_entropy"]},
                                           ensure_ascii=False) + "\n")
            upload_output_dir_to_hf(config.output_dir, config.hf_repo_id, reason="uncond gen")

        if dist.is_available() and dist.is_initialized():
            dist.barrier()


# ============================================
# Conditional
# ============================================

@torch.no_grad()
def test_generation_cond(model, encoder, tokenizer, config: Config,
                         sampling_config: SamplingConfig, dataset,
                         epoch: int, step: int, num_samples: int, batch_size: int):
    sampling_method = sampling_config.sampling_method
    time_schedule = sampling_config.time_schedule
    log_for_0(f"Config: {sampling_config}")

    device = next(model.parameters()).device
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    eos_token_id = tokenizer.eos_token_id

    loader = get_dataloader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length, distributed=False,
    )

    cfg_list = sampling_config.cfgs
    steps_list = sampling_config.num_sampling_steps
    sccfg_list = sampling_config.self_cond_cfg_scales
    encoder_d = config_encoder_d(model)

    for num_steps, cfg_scale, sccfg in itertools.product(steps_list, cfg_list, sccfg_list):
        log_for_0(f"--- {sampling_method} steps={num_steps} cfg={cfg_scale} sccfg={sccfg} ---")
        gen_t, dec_t = 0.0, 0.0
        all_records = []  # (orig, gen, ctx)
        processed = 0
        for bi, batch in enumerate(tqdm(loader, desc="cond gen", disable=not is_main_process())):
            if processed >= num_samples:
                break
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            B = batch["input_ids"].shape[0]

            cond_seq_mask = batch["cond_seq_mask"]
            input_ids = batch["input_ids"]
            encoder_attn = batch["encoder_attention_mask"]

            cond_seq = encode_text(
                input_ids=input_ids, attention_mask=encoder_attn,
                encoder=encoder, latent_mean=config.latent_mean, latent_std=config.latent_std,
            )
            gen = torch.Generator(device=device).manual_seed(
                config.seed * 9973 + (epoch * 10007 + bi) * _world_size() + _rank()
            )
            z = torch.randn(B, config.max_length, encoder_d, device=device, generator=gen) \
                * config.denoiser_noise_scale
            t_steps = get_sampling_steps(
                n_steps=num_steps, time_schedule=time_schedule,
                P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                device=device, generator=gen,
            )
            gs = time.time()
            latent = run_sampling_single_batch(
                model, z, t_steps, cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
                config=config, sampling_config=sampling_config,
                cfg_scale=cfg_scale, self_cond_cfg_scale=sccfg, generator=gen,
            )
            gen_t += time.time() - gs

            gen_length = config.max_length - (config.max_input_length or 0)
            cond_len = cond_seq_mask.long().sum(dim=1)

            ds = time.time()
            t_final = float(t_steps[-1].item())
            ids = decode_batch(model, latent, t_final, config, sccfg)
            ids = shift_left(ids, cond_len, pad_value=0)[:, :gen_length]
            ids = mask_after_eos(ids, eos_token_id, pad_token_id).cpu().numpy()
            dec_t += time.time() - ds

            origs = batch.get("target", [""] * B)
            ctxs = batch.get("input", [""] * B)
            for i, row in enumerate(ids):
                if processed >= num_samples:
                    break
                text = tokenizer.decode(row, skip_special_tokens=True)
                all_records.append((origs[i], text, ctxs[i]))
                processed += 1

        log_for_0(f"Generation: {gen_t:.2f}s | Decode: {dec_t:.2f}s")

        if is_main_process():
            name = build_run_name(
                sampling_method, num_steps, cfg_scale, sccfg, time_schedule,
                getattr(sampling_config, "sde_gamma", 0.0), suffix="cond",
            )
            out_dir = os.path.join(config.output_dir, name)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"all_generated_{epoch}_{step}.jsonl")
            with open(out_path, "w", encoding="utf-8") as f:
                for tid, (orig, gen_t_text, ctx) in enumerate(all_records):
                    f.write(json.dumps({"id": tid, "generated": gen_t_text},
                                       ensure_ascii=False) + "\n")
            log_for_0(f"Saved {len(all_records)} generations to {out_path}")

            if config.online_eval and all_records:
                hyps = [g for _, g, _ in all_records]
                refs = [o for o, _, _ in all_records]
                bleu = compute_bleu(hyps, refs)
                rouges = compute_rouge(hyps, refs)
                log_for_0(f"BLEU={bleu:.2f} R1={rouges['rouge1']:.2f} "
                          f"R2={rouges['rouge2']:.2f} RL={rouges['rougeL']:.2f}")
                with open(os.path.join(out_dir, "metrics.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({"epoch": epoch, "step": step,
                                        "bleu": bleu, **rouges},
                                       ensure_ascii=False) + "\n")
            upload_output_dir_to_hf(config.output_dir, config.hf_repo_id, reason="cond gen")
        if dist.is_available() and dist.is_initialized():
            dist.barrier()


def config_encoder_d(model) -> int:
    inner = maybe_unwrap_ddp(model)
    return inner.text_encoder_dim
