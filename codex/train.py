#!/usr/bin/env python
"""DDP launcher entrypoint for the pure PyTorch ELF implementation.

Run this file path directly (for example with ``torchrun codex/train.py``)
so Python resolves the real PyTorch package instead of the repository's
unrelated top-level ``torch/`` directory.
"""

from pathlib import Path
import sys


CODEX_DIR = Path(__file__).resolve().parent
if str(CODEX_DIR) not in sys.path:
    sys.path.insert(0, str(CODEX_DIR))

from elf_pytorch.train import main


if __name__ == "__main__":
    main()
