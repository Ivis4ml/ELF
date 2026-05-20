# ELF — PyTorch port (8xH200 training)

A faithful PyTorch port of the JAX reference implementation under `src/`. The
file layout under `torch/` mirrors `src/` one-to-one so the two can be diffed
side-by-side; algorithmically, the training step, sampling, and CFG/self-cond
logic are line-for-line transcriptions of `src/train_step.py` and
`src/utils/sampling_utils.py`.

## Installation

```bash
pip install -r torch/requirements.txt
# Optional: huggingface_hub login if you plan to mirror checkpoints
huggingface-cli login
# Optional: wandb login if you plan to track experiments
wandb login YOUR_WANDB_API_KEY
```

PyTorch 2.4+ on CUDA 12.x is recommended for FlashAttention 2 / cuDNN SDPA
support on H200. The training loop uses bf16 autocast for the forward pass
and fp32 for parameters/gradients so that DDP all-reduce is bit-identical
across ranks (a hard requirement for Muon — see "Muon and DDP" below).

## Layout

```
torch/
├── train.py                # DDP entry point
├── eval.py                 # Generation + metric reporting from a checkpoint
├── generation.py           # Uncond / cond generation drivers
├── train_step.py           # Single forward+backward, 4-pass denoiser branch
├── configs/                # Config dataclass + YAML training/sampling configs
├── modules/                # Model, building-block layers, T5 encoder wrapper
└── utils/                  # Sampling, Muon, data, EMA, checkpoint, metrics
```

## Quick start: 8xH200, unconditional generation on OWT (ELF-B)

```bash
cd torch
torchrun --standalone --nproc_per_node=8 train.py \
    --config configs/training_configs/train_owt_ELF-B.yml
```

The default config matches the paper: global batch 512, Muon `blr=0.001`
(effective `lr = 0.001 * 512 / 256 = 0.002`), bf16 autocast, 5 epochs over
OpenWebText (~95K steps), EMA decay 0.9999, save and eval every epoch. Each
optimizer step at `max_length=1024` on 8xH200 holds 64 sequences / GPU; ELF-B
fits comfortably without activation checkpointing.

ELF-M (342M) and ELF-L (652M) configs enable activation checkpointing by
default so the larger blocks fit at the same batch size.

## Conditional generation (WMT14 De-En, XSum)

```bash
# Translation
torchrun --standalone --nproc_per_node=8 train.py \
    --config configs/training_configs/train_de-en_ELF-B.yml

# Summarization (note global_batch_size=64 due to L=1088)
torchrun --standalone --nproc_per_node=8 train.py \
    --config configs/training_configs/train_xsum_ELF-B.yml
```

## Evaluation from a checkpoint

```bash
torchrun --standalone --nproc_per_node=8 eval.py \
    --config configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint_path outputs/elf_b-owt/checkpoint_19000.pt
```

`eval.py` switches the model to EMA weights before sampling, sweeps over the
`sampling_configs/*.yml` entries (32-step SDE + 64-step SDE for uncond by
default), and computes Gen. PPL under a frozen `gpt2-large` plus unigram
entropy.

## What the paper says, where it lives in the code

| Paper section / Algorithm | File / function |
| --- | --- |
| Eq. 1 — MSE on velocity | `train_step.py::_denoiser_branch` (per-dim L2 → masked mean) |
| Eq. 2 — token CE at t=1 | `train_step.py::_decoder_branch` |
| Alg. 1 — joint training, two-branch | `train_step.py::train_step` (Bernoulli + broadcast) |
| Alg. 2 — ODE sampling | `utils/sampling_utils.py::ode_step`, `utils/generation_utils.py::run_sampling_single_batch` |
| Alg. 3 — training-time CFG via self-cond | `train_step.py::_denoiser_branch` (passes 3 and 4) |
| Alg. 4 — decoder-branch corruption | `train_step.py::train_step` (`decoder_z = lambda*x0 + (1-lambda)*eps*scale`) |
| Alg. 5 — inference with conditioning + CFG | `utils/sampling_utils.py::_forward_sample` (CFG over uncond+cond passes) |
| Alg. 6 — ODE / SDE one-step updates | `utils/sampling_utils.py::ode_step` / `sde_step` |
| Sec. B.1 — encoder normalization | `utils/encoder_utils.py::encode_text` (`(x - mean) / std`) |
| Sec. D.1 — DiT block w/ RMSNorm, SwiGLU, RoPE, qk-norm | `modules/layers.py`, `modules/model.py::ELFBlock` |
| Sec. D.2 — model dims (Tab. 3) | `modules/model.py::ELF_B/M/L` |
| Sec. D.2 — sampling configs (uncond/cond) | `configs/sampling_configs/*.yml` |

## Muon and DDP

Muon orthogonalizes 2D parameter updates with a quintic Newton-Schulz
iteration in bf16. To stay correct under DDP without spinning up the
sharded distributed Muon variant, we keep parameters and gradients in fp32 so
that NCCL ring all-reduce gives bit-identical gradients on every rank; each
rank then runs the same NS iterations and arrives at the same update.

Embeddings, biases, RMSNorm scales, the decoder unembedding head, and the
timestep MLPs all fall through to AdamW (Keller Jordan's recommendation):
they are 1D or sit on the model output / time embedding paths where Muon's
spectral normalization hurts more than it helps.

## Subtleties worth knowing

1. **Branch decision is broadcast from rank 0 every step.** If different
   ranks flip different branch coins, the denoise branch touches more
   parameters than the decode branch and DDP's `find_unused_parameters` path
   will hang.
2. **EMA updates only on optimizer-step boundaries.** With grad accumulation
   `K`, naively updating EMA every micro-batch makes the effective decay
   `decay^K`. We guard with `if (step+1) % grad_accum == 0`.
3. **Self-conditioning is 4 forward passes per denoiser step.** This is the
   dominant compute cost. Two of the four are mathematically identical in
   deterministic mode (`[z, 0]` uncond) and could be shared for ~25% speedup
   if you want to deviate from the reference.
4. **Loss masking uses safe division.** Padded and condition-token positions
   contribute zero and are not counted in the denominator.
5. **Logit-normal CFG sampling is log-uniform on `[1+min, 1+max]` minus 1.**
   Defaults `[0.5, 5]` -> sample on `[1.5, 6]` log-uniformly.

## Out of scope

* **Loading the released JAX checkpoints into the PyTorch model is not
  supported.** Train from scratch with the configs above. The JAX and
  PyTorch parameter layouts differ at every level (Flax dicts vs.
  `state_dict`); a checkpoint converter would be a separate project.
* **Cluster-aware Muon.** Single-node DDP only. For multi-node training,
  either reuse this implementation (the NS work is replicated) or wire in
  the Moonshot distributed Muon for compute savings.
