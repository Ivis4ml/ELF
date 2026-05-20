"""Checkpoint save/load and HuggingFace mirror upload."""

import logging
import os
import re
from typing import Optional

import torch

from utils.logging_utils import log_for_0


def _local_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def upload_output_dir_to_hf(output_dir: str, hf_repo_id: Optional[str], reason: str = "artifacts"):
    if not hf_repo_id:
        return
    from utils.logging_utils import is_main_process
    if not is_main_process():
        return

    folder = _local_path(output_dir)
    if not os.path.isdir(folder):
        log_for_0(f"HF upload skipped; missing dir: {folder}", level=logging.WARNING)
        return
    try:
        from huggingface_hub import HfApi
        repo_id = hf_repo_id.strip("/")
        api = HfApi()
        api.create_repo(repo_id, repo_type="model", exist_ok=True)
        log_for_0(f"Uploading {reason} to HF: {repo_id}")
        api.upload_folder(repo_id=repo_id, folder_path=folder, repo_type="model")
    except Exception as e:
        log_for_0(f"HF upload failed ({e})", level=logging.WARNING)


def _checkpoint_step(checkpoint_name: str) -> int:
    m = re.search(r"(\d+)$", checkpoint_name)
    return int(m.group(1)) if m else -1


def find_all_checkpoints(ckpt_dir: str, prefix: str = "checkpoint_"):
    ckpt_dir = _local_path(ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        return []
    names = sorted(
        [f for f in os.listdir(ckpt_dir) if f.startswith(prefix) and f.endswith(".pt")],
        key=_checkpoint_step,
    )
    return [os.path.join(ckpt_dir, n) for n in names]


def find_latest_checkpoint(ckpt_dir: str, prefix: str = "checkpoint_"):
    files = find_all_checkpoints(ckpt_dir, prefix)
    return files[-1] if files else None


def save_checkpoint(model, optimizer, ema, step: int, epoch: float, output_dir: str,
                    keep_last_n: int = 10, hf_repo_id: Optional[str] = None,
                    extra: Optional[dict] = None):
    """Rank-0-only checkpoint write. Always saves an `ema_params1` shadow."""
    from utils.logging_utils import is_main_process
    if not is_main_process():
        return

    out = _local_path(output_dir)
    os.makedirs(out, exist_ok=True)

    inner = model.module if hasattr(model, "module") else model
    state = {
        "params": inner.state_dict(),
        "ema_params1": ema.state_dict() if ema is not None else None,
        "opt_state": optimizer.state_dict(),
        "step": int(step),
        "epoch": float(epoch),
    }
    if extra:
        state.update(extra)

    path = os.path.join(out, f"checkpoint_{step}.pt")
    log_for_0(f"Saving checkpoint to {path}")
    torch.save(state, path)

    # Rotate.
    all_ckpts = find_all_checkpoints(out)
    for old in all_ckpts[:-keep_last_n]:
        try:
            os.remove(old)
        except OSError:
            pass

    upload_output_dir_to_hf(output_dir, hf_repo_id, reason="checkpoint")


def load_checkpoint(checkpoint_path: str, model, optimizer=None, ema=None, map_location="cpu"):
    """Load checkpoint into model/optimizer/ema in place. Returns (step, epoch).

    Accepts either a .pt file or a directory (uses latest .pt inside).
    """
    path = _local_path(checkpoint_path)
    if os.path.isdir(path):
        latest = find_latest_checkpoint(path)
        if latest is None:
            raise FileNotFoundError(f"No checkpoint in directory {path}")
        path = latest
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    log_for_0(f"Loading checkpoint from {path}")
    state = torch.load(path, map_location=map_location)

    inner = model.module if hasattr(model, "module") else model
    inner.load_state_dict(state["params"], strict=True)
    if optimizer is not None and state.get("opt_state") is not None:
        try:
            optimizer.load_state_dict(state["opt_state"])
        except Exception as e:
            log_for_0(f"Optimizer state load failed ({e}); continuing with fresh optimizer.")
    if ema is not None and state.get("ema_params1") is not None:
        try:
            ema.load_state_dict(state["ema_params1"], inner.parameters())
        except Exception as e:
            log_for_0(f"EMA state load failed ({e}); reinitializing EMA from current params.")

    return int(state.get("step", 0)), float(state.get("epoch", 0))
