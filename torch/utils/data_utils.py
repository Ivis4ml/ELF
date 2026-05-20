"""Dataset loading + PyTorch DataLoader matching the JAX collator one-to-one.

Each batch carries:
    input_ids: (B, L) int64
    encoder_attention_mask: (B, L, L) float32 — for the frozen T5 encoder
    attention_mask: (B, L) float32 — global "real token" mask
    cond_seq_mask: (B, L) float32 — 1 at the conditioning prefix
    label_drop_mask: (B,) bool — sampled per-step from label_drop_prob
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from datasets import DatasetDict, load_dataset as hf_load_dataset, load_from_disk

from utils.encoder_utils import build_self_attn_cond_masks
from utils.logging_utils import log_for_0


def get_pad_token_id(tokenizer, pad_token: str = "pad") -> int:
    token_id = tokenizer.eos_token_id if pad_token == "eos" else tokenizer.pad_token_id
    if token_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id.")
    return token_id


def pad_and_truncate(ids_list: List[np.ndarray], target_len: int, pad_token_id: int):
    padded, lengths = [], []
    for ids in ids_list:
        orig_len = min(len(ids), target_len)
        ids = ids[:target_len]
        if orig_len < target_len:
            ids = np.concatenate([ids, np.full(target_len - orig_len, pad_token_id, dtype=ids.dtype)])
        padded.append(ids)
        lengths.append(orig_len)
    return np.stack(padded), np.array(lengths)


def make_collate_fn(max_seq_length: int, pad_token_id: int,
                    max_input_seq_length: Optional[int] = None):
    def collate(batch_list: List[Dict]):
        input_ids_list = [np.asarray(item["input_ids"]) for item in batch_list]

        if "condition_input_ids" in batch_list[0]:
            seq_list, cond_lens = [], []
            for item in batch_list:
                cond = np.asarray(item["condition_input_ids"])[:max_input_seq_length]
                inp = np.asarray(item["input_ids"])
                seq_list.append(np.concatenate([cond, inp]))
                cond_lens.append(len(cond))
            cond_lens = np.asarray(cond_lens, dtype=np.int64)
        else:
            seq_list = input_ids_list
            cond_lens = np.zeros(len(input_ids_list), dtype=np.int64)

        ids, total_lens = pad_and_truncate(seq_list, max_seq_length, pad_token_id)
        pos = np.arange(max_seq_length)[None, :]
        is_cond = pos < cond_lens[:, None]
        is_valid = pos < total_lens[:, None]
        enc_mask, attn_mask, cond_mask = build_self_attn_cond_masks(is_cond, is_valid, xp=np)

        out = {
            "input_ids": torch.as_tensor(ids, dtype=torch.long),
            "encoder_attention_mask": torch.as_tensor(enc_mask, dtype=torch.float32),
            "attention_mask": torch.as_tensor(attn_mask, dtype=torch.float32),
            "cond_seq_mask": torch.as_tensor(cond_mask, dtype=torch.float32),
        }
        for key in ("index", "input", "target"):
            if key in batch_list[0]:
                out[key] = [item[key] for item in batch_list]
        return out

    return collate


def prepare_batch(batch: Dict, config) -> Dict:
    """Move tensors to GPU and sample per-sample label_drop_mask on host."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    B = out["input_ids"].shape[0]
    if config.label_drop_prob > 0:
        out["label_drop_mask"] = (torch.rand(B, device=device) < config.label_drop_prob)
    else:
        out["label_drop_mask"] = torch.zeros(B, dtype=torch.bool, device=device)
    return out


def get_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    max_seq_length: int = 512,
    pad_token_id: int = 0,
    max_input_seq_length: Optional[int] = None,
    distributed: bool = True,
    seed: int = 0,
):
    collate = make_collate_fn(max_seq_length, pad_token_id, max_input_seq_length)
    common = dict(
        batch_size=batch_size, num_workers=num_workers, collate_fn=collate,
        drop_last=drop_last, persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    if distributed and dist.is_available() and dist.is_initialized():
        sampler = DistributedSampler(
            dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(),
            shuffle=shuffle, drop_last=drop_last, seed=seed,
        )
        return DataLoader(dataset, sampler=sampler, **common)
    return DataLoader(dataset, shuffle=shuffle, **common)


def load_jsonl_dataset(path: str, tokenizer, input_key: str = "input", output_key: str = "output"):
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append({
                "index": i,
                "input": data[input_key],
                "target": data[output_key],
                "condition_input_ids": tokenizer(data[input_key], add_special_tokens=False)["input_ids"],
                "input_ids": tokenizer(data[output_key], add_special_tokens=False)["input_ids"],
            })
    return examples


def _looks_like_save_to_disk_arrow(ds) -> bool:
    return (
        len(ds) == 1
        and any(c.startswith("_") for c in ds.column_names)
        and not any(not c.startswith("_") for c in ds.column_names)
    )


def load_dataset_split(path: str, dataset_cache_dir: Optional[str] = None):
    ds = None
    try:
        ds = hf_load_dataset(path, cache_dir=dataset_cache_dir)
    except Exception:
        ds = load_from_disk(path)

    if isinstance(ds, DatasetDict):
        splits = list(ds.keys())
        if len(splits) != 1:
            raise ValueError(f"Expected dataset at {path!r} to have a single split, got {splits}.")
        ds = ds[splits[0]]

    if _looks_like_save_to_disk_arrow(ds):
        from huggingface_hub import snapshot_download
        log_for_0(
            f"Dataset at {path!r} looks like a save_to_disk-format HF repo; "
            f"re-downloading via snapshot_download + load_from_disk."
        )
        local_dir = snapshot_download(repo_id=path, repo_type="dataset", cache_dir=dataset_cache_dir)
        ds = load_from_disk(local_dir)
        if isinstance(ds, DatasetDict):
            splits = list(ds.keys())
            if len(splits) != 1:
                raise ValueError(f"Expected single split, got {splits}.")
            ds = ds[splits[0]]

    ds.set_format(type="numpy", columns=ds.column_names)
    return ds


def load_dataset(config, dataset_cache_dir: Optional[str] = None):
    log_for_0(f"Loading dataset from {config.data_path}...")
    train_dataset = load_dataset_split(config.data_path, dataset_cache_dir)
    log_for_0(f"Train size: {len(train_dataset)}")

    eval_dataset = None
    if config.eval_data_path:
        eval_dataset = load_dataset_split(config.eval_data_path, dataset_cache_dir)
        log_for_0(f"Eval size: {len(eval_dataset)}")
    else:
        log_for_0("No eval dataset")

    return train_dataset, eval_dataset
