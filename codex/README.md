# Pure PyTorch ELF

This directory contains a clean PyTorch port of the JAX/Flax ELF code in `src/`, based on the ELF paper in `docs/ELF.pdf`. It deliberately does not depend on the repository's top-level `torch/` directory.

## What Is Implemented

- ELF Transformer with RMSNorm, qk-norm attention, RoPE, SwiGLU, bottleneck text projection, in-context time / self-conditioning CFG / mode tokens, and shared denoiser-decoder weights.
- Flow-matching denoiser objective and final-step decoder cross-entropy objective.
- Training-time self-conditioning CFG target from the paper/source implementation.
- Frozen PyTorch `T5EncoderModel` encoder, HF dataset loader, conditional-prefix masks, label-drop support, DDP training, bf16 autocast, EMA, checkpointing, AdamW, and a self-contained Muon optimizer.
- ODE/SDE-style sampling utilities for checkpoint smoke checks and downstream evaluation scripts.

## 8xH200 Launch

Install the PyTorch-side requirements in the environment you will use on the H200 node:

```bash
pip install -r codex/requirements-pytorch.txt
```

Launch from the repository root by executing the file path directly. This avoids Python resolving the unrelated top-level `torch/` directory as the PyTorch package.

```bash
torchrun --standalone --nproc_per_node=8 \
  codex/train.py \
  --config codex/configs/train_owt_elf_b_8xh200.yaml
```

Larger configs are also included:

```bash
torchrun --standalone --nproc_per_node=8 codex/train.py --config codex/configs/train_owt_elf_m_8xh200.yaml
torchrun --standalone --nproc_per_node=8 codex/train.py --config codex/configs/train_owt_elf_l_8xh200.yaml
```

The configs keep the effective batch at 512 with gradient accumulation:

| Model | Local micro-batch | Accum | Effective global batch |
| --- | ---: | ---: | ---: |
| ELF-B | 16 | 4 | 512 |
| ELF-M | 8 | 8 | 512 |
| ELF-L | 4 | 16 | 512 |

Outputs and checkpoints are written under `codex/outputs/`.

## Quick CPU Smoke Test

Run tests from inside `codex/` so imports cannot collide with the repository root:

```bash
cd codex
PYTHONPATH=. pytest tests/test_smoke.py
```

## Notes

- The original code trains with a JAX T5 encoder checkpoint. This port uses Hugging Face's PyTorch `T5EncoderModel` (`google-t5/t5-small` by default) as the frozen contextual embedder, which keeps the training stack pure PyTorch.
- The model follows the source implementation's step-level branch sampling: each micro-step is either a denoiser step or a decoder step according to `decoder_prob`.
- `torch_compile` is available in config but disabled by default; turn it on only after the first DDP run is stable on the target machine.
