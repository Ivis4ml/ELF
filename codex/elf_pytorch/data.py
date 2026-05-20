from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict, load_dataset as hf_load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import TrainingConfig


def get_pad_token_id(tokenizer: Any, pad_token: str = "pad") -> int:
    token_id = tokenizer.eos_token_id if pad_token == "eos" else tokenizer.pad_token_id
    if token_id is None:
        raise ValueError(f"Tokenizer has no token id for pad_token={pad_token!r}.")
    return int(token_id)


def build_self_attn_cond_masks(
    is_cond: np.ndarray,
    is_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    encoder_attention_mask = (
        (is_cond[:, :, None] & is_cond[:, None, :])
        | (~is_cond[:, :, None] & is_valid[:, None, :])
    ).astype(np.float32)
    attention_mask = is_valid.astype(np.float32)
    cond_seq_mask = is_cond.astype(np.float32)
    return encoder_attention_mask, attention_mask, cond_seq_mask


def pad_and_truncate(
    ids_list: list[np.ndarray],
    target_len: int,
    pad_token_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    padded = []
    lengths = []
    for ids in ids_list:
        ids = np.asarray(ids, dtype=np.int64)
        orig_len = min(len(ids), target_len)
        ids = ids[:target_len]
        if orig_len < target_len:
            ids = np.concatenate(
                [ids, np.full(target_len - orig_len, pad_token_id, dtype=np.int64)]
            )
        padded.append(ids)
        lengths.append(orig_len)
    return np.stack(padded), np.asarray(lengths, dtype=np.int64)


def load_jsonl_dataset(path: str, tokenizer: Any, input_key: str = "input", output_key: str = "output"):
    examples = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append(
                {
                    "index": index,
                    "input": data[input_key],
                    "target": data[output_key],
                    "condition_input_ids": tokenizer(
                        data[input_key], add_special_tokens=False
                    )["input_ids"],
                    "input_ids": tokenizer(data[output_key], add_special_tokens=False)[
                        "input_ids"
                    ],
                }
            )
    return examples


def _looks_like_save_to_disk_arrow(ds: Any) -> bool:
    return (
        len(ds) == 1
        and any(column.startswith("_") for column in ds.column_names)
        and not any(not column.startswith("_") for column in ds.column_names)
    )


def load_dataset_split(path: str, dataset_cache_dir: str | None = None):
    try:
        ds = hf_load_dataset(path, cache_dir=dataset_cache_dir)
    except Exception:
        ds = load_from_disk(path)

    if isinstance(ds, DatasetDict):
        splits = list(ds.keys())
        if len(splits) != 1:
            raise ValueError(f"Expected one split at {path!r}, got {splits}.")
        ds = ds[splits[0]]

    if _looks_like_save_to_disk_arrow(ds):
        from huggingface_hub import snapshot_download

        local_dir = snapshot_download(
            repo_id=path,
            repo_type="dataset",
            cache_dir=dataset_cache_dir,
        )
        ds = load_from_disk(local_dir)
        if isinstance(ds, DatasetDict):
            splits = list(ds.keys())
            if len(splits) != 1:
                raise ValueError(f"Expected one split at {path!r}, got {splits}.")
            ds = ds[splits[0]]

    ds.set_format(type="numpy", columns=ds.column_names)
    return ds


def load_train_eval_datasets(config: TrainingConfig):
    if config.data_path is None:
        raise ValueError("data_path must be set for training.")
    train_dataset = load_dataset_split(config.data_path, config.dataset_cache_dir)
    eval_dataset = (
        load_dataset_split(config.eval_data_path, config.dataset_cache_dir)
        if config.eval_data_path
        else None
    )
    return train_dataset, eval_dataset


def create_collate_fn(
    *,
    max_seq_length: int,
    pad_token_id: int,
    max_input_seq_length: int | None = None,
):
    def collate_fn(batch_list: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids_list = [np.asarray(item["input_ids"], dtype=np.int64) for item in batch_list]

        if "condition_input_ids" in batch_list[0]:
            seq_list = []
            cond_lens = []
            input_limit = max_input_seq_length or max_seq_length
            for item in batch_list:
                cond = np.asarray(item["condition_input_ids"], dtype=np.int64)[:input_limit]
                inp = np.asarray(item["input_ids"], dtype=np.int64)
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
        encoder_attn, attn, pred = build_self_attn_cond_masks(is_cond, is_valid)
        result = {
            "input_ids": torch.from_numpy(ids).long(),
            "encoder_attention_mask": torch.from_numpy(encoder_attn).float(),
            "attention_mask": torch.from_numpy(attn).float(),
            "cond_seq_mask": torch.from_numpy(pred).float(),
        }
        for key in ("index", "input", "target"):
            if key in batch_list[0]:
                result[key] = [item[key] for item in batch_list]
        return result

    return collate_fn


def create_dataloader(
    dataset: Any,
    *,
    batch_size: int,
    rank: int,
    world_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    max_seq_length: int = 512,
    pad_token_id: int = 0,
    max_input_seq_length: int | None = None,
    pin_memory: bool = True,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
) -> DataLoader:
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    kwargs: dict[str, Any] = {}
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["persistent_workers"] = persistent_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=create_collate_fn(
            max_seq_length=max_seq_length,
            pad_token_id=pad_token_id,
            max_input_seq_length=max_input_seq_length,
        ),
        drop_last=drop_last,
        pin_memory=pin_memory,
        **kwargs,
    )
