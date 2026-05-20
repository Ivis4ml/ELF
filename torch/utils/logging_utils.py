"""Distributed-aware logging: log only on rank 0."""

import inspect
import logging
import os

import torch.distributed as dist


def is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0


def log_for_0(msg, *args, level=logging.INFO):
    if not is_main_process():
        return
    caller_module = inspect.currentframe().f_back.f_globals.get("__name__", __name__)
    logging.getLogger(caller_module).log(level, msg, *args)
