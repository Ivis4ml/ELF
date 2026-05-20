# ELF 训练分步指南 · 8×H200

按阶段一步一步来。每个阶段都有明确的"成功标志"，达成了再进下一阶段，不要跳。

> 总耗时预估：准备 30 分钟 → 烟测 30 分钟 → 短训 1.5 小时 → 全量 16-24 小时 → 评估 30 分钟

---

## 目录

- [Phase 0 · 准备（一次性）](#phase-0--准备一次性)
- [Phase 1 · 烟测（10-30 分钟）](#phase-1--烟测10-30-分钟)
- [Phase 2 · 短训（1-2 小时）](#phase-2--短训1-2-小时)
- [Phase 3 · 全量训练（5 epochs）](#phase-3--全量训练5-epochs)
- [Phase 4 · 评估](#phase-4--评估)
- [Phase 5 · 训练变种](#phase-5--训练变种)
- [Appendix A · 故障排查](#appendix-a--故障排查)
- [Appendix B · 关键监控指标速查](#appendix-b--关键监控指标速查)

---

## Phase 0 · 准备（一次性）

### 0.1 硬件检查

```bash
nvidia-smi
```

成功标志：
- 8 张 H200，每张 141 GB HBM
- CUDA 驱动 ≥ 535
- 所有卡的 `Persistence-M: On`（如果是 Off，跑一次 `sudo nvidia-smi -pm 1`）

确认 NVLink 拓扑：

```bash
nvidia-smi topo -m
```

期望看到大量 `NV18` / `NV12` 而不是 `PHB` / `PIX`，说明 8 张卡是全互联 NVLink。

### 0.2 Python 环境

推荐 Python 3.11 或 3.12，CUDA 12.4+。

```bash
python --version
nvcc --version
```

### 0.3 装依赖

```bash
cd /path/to/ELF

# PyTorch 单独装（用官方 wheel 拉对 CUDA 版本）
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124

# 其他依赖
pip install -r torch/requirements.txt
```

验证 PyTorch 看到 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

成功标志：`True 8`

### 0.4 HuggingFace 登录

需要拉取 T5 编码器和 pre-tokenized 的 OWT / WMT / XSum 数据集。

```bash
huggingface-cli login
# 粘贴个人 token (hf_xxx)
huggingface-cli whoami
```

### 0.5 W&B 登录（可选）

```bash
wandb login
```

如果不用 W&B，在配置 YAML 里把 `use_wandb` 设为 `false`，跳过这步。

### 0.6 仓库结构验证

```bash
cd torch
ls
```

应该看到：`configs/`、`modules/`、`utils/`、`train.py`、`train_step.py`、`eval.py`、`generation.py`、`README.md`、`requirements.txt`、`ELF.html`。

---

## Phase 1 · 烟测（10-30 分钟）

目标：确认代码能跑通，**不动真格**。在烧 8 GPU 几十小时之前，先用 10 分钟把所有路径走一遍。

### 1.1 单 GPU 模型烟测

只检查模型能初始化 + 一次前向反向不崩。

```bash
cd torch
python -c "
import sys, os
sys.path.insert(0, '.')
import torch
from modules.model import ELF_models

m = ELF_models['ELF-B'](
    text_encoder_dim=512, max_length=128, vocab_size=32128,
    bottleneck_dim=128, num_time_tokens=4,
    num_self_cond_cfg_tokens=4, num_model_mode_tokens=4,
).cuda()
print(f'ELF-B params: {sum(p.numel() for p in m.parameters()):,}')

x = torch.randn(2, 128, 1024).cuda()
t = torch.rand(2).cuda()
sc = torch.rand(2).cuda() * 4 + 0.5
out, _ = m(x, t, self_cond_cfg_scale=sc, decoder_step_active=False)
print(f'denoise out shape: {tuple(out.shape)}')

out, logits = m(x, t, self_cond_cfg_scale=sc, decoder_step_active=True)
print(f'decode logits shape: {tuple(logits.shape)}')
print('OK')
"
```

成功标志：
```
ELF-B params: 104,594,304
denoise out shape: (2, 128, 512)
decode logits shape: (2, 128, 32128)
OK
```

参数数对不上就是模型架构装配出问题，先停下查。

### 1.2 T5 与数据集可达

```bash
python -c "
from transformers import T5EncoderModel, AutoTokenizer
from datasets import load_dataset

tok = AutoTokenizer.from_pretrained('google-t5/t5-small')
m = T5EncoderModel.from_pretrained('google-t5/t5-small')
print('T5 ok, d_model =', m.config.d_model)

ds = load_dataset('embedded-language-flows/openwebtext-t5', split='train', streaming=True)
sample = next(iter(ds))
print(f'Dataset ok, sample keys: {list(sample.keys())}, input_ids len: {len(sample[\"input_ids\"])}')
"
```

成功标志：T5 拉下来 + OWT 第一条样本能读出 `input_ids` 字段。

### 1.3 写一个烟测配置

复制基础配置，改成 mini 版：

```bash
cd torch
cp configs/training_configs/train_owt_ELF-B.yml configs/training_configs/train_smoke.yml
```

编辑 `train_smoke.yml`，覆盖以下字段：

```yaml
max_length: 128                # 短序列, 加速
global_batch_size: 8           # 跑得过单 GPU
epochs: 1
warmup_steps: 100
log_freq: 5
save_freq: 99                  # 烟测不存 checkpoint
eval_freq: 99                  # 不跑评估
use_wandb: false
output_dir: outputs/smoke
```

### 1.4 单 GPU 跑 10 步训练

```bash
cd torch
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config configs/training_configs/train_smoke.yml \
    2>&1 | head -80
```

成功标志：
- 看到 `ELF parameters: 104,594,304`
- 第 5 步出现 `Step 5: loss=... l2=... ce=... lr=...`
- loss 是有限数（不是 NaN）
- 程序在 50 步左右自然停止（1 epoch 的 OWT 数据集很小一部分）

跑不过去看 [Appendix A](#appendix-a--故障排查)。

### 1.5 多 GPU DDP 烟测

确认 NCCL 通信不会 hang。先用 2 GPU 跑：

```bash
cd torch
torchrun --standalone --nproc_per_node=2 train.py \
    --config configs/training_configs/train_smoke.yml \
    2>&1 | head -80
```

成功标志：
- 看到 `World size: 2`
- 第一步 loss 在 60 秒内出现
- 两个 rank 的 loss 数值在 fp32 epsilon 内一致（grep `Step 5`）

如果第一步卡住超过 60 秒，看 [A.1](#a1-nccl-hang)。

### 1.6 全机 DDP 烟测

确认 8 GPU 都能起来：

```bash
torchrun --standalone --nproc_per_node=8 train.py \
    --config configs/training_configs/train_smoke.yml \
    2>&1 | head -80
```

成功标志：
- `World size: 8`
- 看到 8 张卡都进入训练循环
- `nvidia-smi` 上 8 张卡都有进程

跑通就说明所有路径都通了，可以进 Phase 2。

---

## Phase 2 · 短训（1-2 小时）

目标：用真实配置跑 1 个 epoch，看 loss 是不是真的在下降。

### 2.1 启动短训

```bash
cd torch
mkdir -p outputs/short-elf-b

torchrun --standalone --nproc_per_node=8 train.py \
    --config configs/training_configs/train_owt_ELF-B.yml \
    --config_override "epochs=1" "output_dir=outputs/short-elf-b" \
                       "use_wandb=false" \
    2>&1 | tee outputs/short-elf-b/train.log
```

ELF-B 在 8×H200 上 1 epoch ≈ 19K 步 ≈ 1.5-2 小时。

### 2.2 实时监控

新开一个 terminal：

```bash
# 显存 + 利用率
watch -n 2 nvidia-smi

# loss 曲线
tail -f outputs/short-elf-b/train.log | grep "Step "
```

### 2.3 健康检查（前 200 步）

成功标志：

| 指标 | 健康范围 |
|---|---|
| `l2_loss` 起点 | 1.5 - 2.5 |
| `l2_loss` 第 200 步 | 0.8 - 1.2 |
| `ce_loss` 起点 | 9 - 11 (≈ log(vocab)) |
| `ce_loss` 第 200 步 | 6 - 8 |
| `sps`（steps/sec） | > 2.0 |
| 显存占用 / 卡 | 30 - 60 GB |
| `lr` 第 200 步 | 接近 2e-3（warmup 完成）|

不健康信号：
- `loss=nan`：[A.3](#a3-loss-nan)
- `loss` 1000 步还不动：[A.4](#a4-loss-不动)
- 某张卡显存 130 GB+：[A.5](#a5-oom)
- 某张卡掉队：[A.6](#a6-某张卡掉队)

### 2.4 1 epoch 后看生成样本

短训结束后会自动跑一次评估。看一眼：

```bash
ls outputs/short-elf-b/
# 应该有 sde-steps32-... 和 sde-steps64-... 两个子目录

head -3 outputs/short-elf-b/sde-steps32-*/all_generated_1_*.jsonl
```

1 epoch 的生成质量必然很差（PPL 可能 50-100），但能看出"句子结构"就证明训练方向正确。如果生成全是重复字符或乱码，停下排查。

### 2.5 决策点

跑完短训后：

- **loss 曲线漂亮 + 生成有句子结构** → 进 Phase 3 全量训练
- **loss 不下降 / NaN / 生成全乱** → 不要进 Phase 3，先排查（Appendix A）

---

## Phase 3 · 全量训练（5 epochs）

目标：跑出论文目标 Gen. PPL ≈ 24（ELF-B, 32-step SDE）。

### 3.1 启动全量训练

```bash
cd torch
mkdir -p outputs/elf-b-owt-full

nohup torchrun --standalone --nproc_per_node=8 train.py \
    --config configs/training_configs/train_owt_ELF-B.yml \
    --config_override "output_dir=outputs/elf-b-owt-full" \
                       "use_wandb=true" \
                       "wandb_run_name=elf-b-full-$(date +%Y%m%d)" \
    > outputs/elf-b-owt-full/train.log 2>&1 &

TRAIN_PID=$!
echo "Train PID: $TRAIN_PID" > outputs/elf-b-owt-full/pid
```

`nohup ... &` 让训练在 ssh 断线后继续跑。

总耗时预估：5 epoch × ~19K 步 / 2.5 sps ≈ 16-22 小时。

### 3.2 验证训练在跑

```bash
# 看进程
ps -p $(cat outputs/elf-b-owt-full/pid) -o pid,etime,cmd

# 看最近几个 step
tail -20 outputs/elf-b-owt-full/train.log | grep "Step "

# 看显存
nvidia-smi
```

### 3.3 监控曲线（W&B）

打开 W&B run 页面，盯三条曲线：

| 曲线 | 期望走势 |
|---|---|
| `train_loss` | 单调下降（带噪声）|
| `train_l2_loss` | denoise 分支 MSE，单调下降 |
| `train_ce_loss` | decode 分支 CE，单调下降 |

每个 epoch 结束会自动保存 checkpoint + 跑评估。期望 PPL 进展：

| Epoch | 期望 Gen PPL（32-step SDE）| 期望 entropy |
|---|---|---|
| 1 | 80-120 | 5.0-5.3 |
| 2 | 50-70 | 5.1-5.2 |
| 3 | 35-45 | 5.13-5.18 |
| 4 | 28-32 | 5.14-5.17 |
| 5 | 24-26（论文目标）| 5.14-5.16 |

如果第 2 个 epoch 还 > 100，训练有问题，停下排查。

### 3.4 中途断电 / 抢占 / 手动停止

直接重启同一条 `torchrun` 命令即可。auto-resume 会扫 `outputs/elf-b-owt-full/checkpoint_*.pt`，自动从最新一份恢复。

强制重新开始：

```bash
rm outputs/elf-b-owt-full/checkpoint_*.pt
# 或换 output_dir
```

### 3.5 训练完成

最后一行日志：

```
Training complete; writing final checkpoint.
```

`outputs/elf-b-owt-full/` 下应该有：
- `checkpoint_XXXXX.pt`（最新 10 份）
- `config.yml`（实际跑的完整配置）
- 多个 `sde-stepsXX-.../` 子目录（每 epoch 评估结果）

---

## Phase 4 · 评估

### 4.1 单 seed 评估

```bash
cd torch
torchrun --standalone --nproc_per_node=8 eval.py \
    --config configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint_path outputs/elf-b-owt-full
```

`--checkpoint_path` 给目录时自动用目录里最新的那份。会切到 EMA 权重，跑 `sampling_configs/uncond_sampling_configs.yml` 里的所有 sweep：
- 32-step SDE，γ=1.5，self_cond_cfg=3
- 64-step SDE，γ=1.0，self_cond_cfg=3

### 4.2 看结果

```bash
cat outputs/elf-b-owt-full/sde-steps32-cfg1-sccfg3-ts_logit_normal-gamma1.5-uncond/metrics.jsonl
```

期望（与论文 Tab. 6 对照）：

```json
{"epoch": 5, "step": ...,
 "ppl": 24.08,
 "mean_entropy": 5.15}
```

`ppl` 在 24-26 之间、`mean_entropy` 在 5.10-5.20 之间就算复现成功。

### 4.3 多 seed 评估（论文报告值）

论文表 6 是 6 seed 平均：

```bash
torchrun --standalone --nproc_per_node=8 eval.py \
    --config configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint_path outputs/elf-b-owt-full \
    --seeds "0,1,2,3,4,5"
```

每个 seed 的结果写到 `outputs/elf-b-owt-full/seed_X/.../metrics.jsonl`。手动取均值：

```bash
python -c "
import json, glob, statistics
ppls = []
for f in glob.glob('outputs/elf-b-owt-full/seed_*/sde-steps32-*/metrics.jsonl'):
    with open(f) as fp:
        for line in fp:
            d = json.loads(line)
            ppls.append(d['ppl'])
print(f'mean PPL: {statistics.mean(ppls):.2f} ± {statistics.stdev(ppls):.2f}')
"
```

期望接近 24.08 ± 0.16。

---

## Phase 5 · 训练变种

### 5.1 ELF-M（342M）/ ELF-L（652M）

```bash
# ELF-M
torchrun --standalone --nproc_per_node=8 train.py \
    --config configs/training_configs/train_owt_ELF-M.yml

# ELF-L
torchrun --standalone --nproc_per_node=8 train.py \
    --config configs/training_configs/train_owt_ELF-L.yml
```

M 和 L 默认开启 `activation_checkpointing`。如果显存还不够，加 `grad_accum_steps=2`。

期望 Gen PPL：
- ELF-M: ~21-22（论文 Tab. 7）
- ELF-L: ~21-23

### 5.2 WMT14 De-En 翻译

```bash
torchrun --standalone --nproc_per_node=8 train.py \
    --config configs/training_configs/train_de-en_ELF-B.yml
```

100 epochs ≈ 880K 步，长。1-2 天。

期望最终 BLEU ≈ 26-27（论文 26.4）。

### 5.3 XSum 摘要

```bash
torchrun --standalone --nproc_per_node=8 train.py \
    --config configs/training_configs/train_xsum_ELF-B.yml
```

`global_batch_size: 64`（max_length=1088，显存吃紧）。

期望 ROUGE-1 ≈ 35-36，ROUGE-L ≈ 27-28。

---

## Appendix A · 故障排查

### A.1 NCCL hang

**症状**：`torchrun` 起来后第一步过了 60 秒没有 loss 输出。

**诊断**：

```bash
NCCL_DEBUG=INFO torchrun --standalone --nproc_per_node=8 train.py ... 2>&1 | grep -i "nccl\|ib\|ring"
```

**常见原因**：
- NVLink 没起来：`nvidia-smi topo -m` 确认拓扑
- Multi-node 时 hostfile / IB 配置错：设置 `NCCL_SOCKET_IFNAME=eth0`（或对应网卡名）
- shared memory 不够：`mount -o remount,size=64G /dev/shm`

### A.2 `import torch` 被本目录遮蔽

**症状**：启动报 `ImportError: cannot import name 'XXX' from 'torch'` 或 `AttributeError` 怪错。

**确认**：是不是在仓库根目录（包含 `torch/` 文件夹的地方）直接跑了 `python torch/train.py`？这会让 Python 把本地 `torch/` 目录当成 PyTorch 包。

**修复**：进入 `torch/` 目录再跑：

```bash
cd torch
torchrun ... train.py ...
```

或用 `torchrun` 而不是 `python`（`torchrun` 内部走 `__main__` 不会触发遮蔽）。

### A.3 Loss NaN

**可能原因**：

1. **lr 太大**：把 yaml 里 `blr` 从 1e-3 降到 5e-4
2. **bf16 数值不稳**：在 yaml 改 `autocast_dtype: float32` 看是否复现
3. **encoder normalization 错**：检查 `latent_mean=0.0`、`latent_std=0.2` 是不是配上了
4. **数据集 token 越界**：跑 `python -c "from datasets import load_dataset; ds = load_dataset(..., split='train[:100]'); print(max(max(s['input_ids']) for s in ds))"` 看 max id 是否超过 vocab

### A.4 Loss 不动

**正常情况**：前 50-100 步 `l2_loss` 几乎不变，因为 `FinalLayer` 是零初始化的（扩散标配）。

**不正常**：1000 步后 `l2_loss` 还在 1.5+ 不下降。

**排查**：
- 检查 `lr` 是不是 0（warmup_steps 配错？）
- 检查数据：随机抽 batch 看 `input_ids` 不全是 pad token
- 检查 EMA 没有反向把权重往回拽：临时禁用 EMA 看是否复现

### A.5 OOM

**预算大致**：

| 模型 | 单卡 batch | 不开 ckpt | 开 ckpt |
|---|---|---|---|
| ELF-B (105M, max_len 1024) | 64 | ~45 GB | ~25 GB |
| ELF-M (342M, max_len 1024) | 64 | ~85 GB | ~40 GB |
| ELF-L (652M, max_len 1024) | 64 | ~115 GB | ~55 GB |

如果 OOM：
1. 在 yaml 设 `activation_checkpointing: true`
2. 降 `global_batch_size`（代价是更慢，需要更多 epochs 才能达到同样 token 数）
3. 用 `grad_accum_steps: 2`（维持有效 batch，单步显存减半）

### A.6 某张卡掉队

**症状**：W&B 上 `train_loss` 周期性突刺，或 `nvidia-smi` 看到某张卡利用率明显低于其他。

**排查**：

```bash
# 看 ECC 错误
nvidia-smi --query-gpu=index,ecc.errors.uncorrected.aggregate.total --format=csv

# 看温度
nvidia-smi --query-gpu=index,temperature.gpu --format=csv
```

ECC 非零的卡有硬件问题，停止训练换卡。温度长期 90°C+ 是散热问题。

### A.7 训练慢于预期

**期望 sps**（8×H200，ELF-B max_len 1024）：

- 无 activation_checkpointing：2.5-3.5 sps
- 开 activation_checkpointing：1.5-2.0 sps

实测明显低于这个：

1. 检查 PyTorch 版本：`python -c "import torch; print(torch.__version__)"` 应该 ≥ 2.4
2. 确认 FlashAttention 2 在跑：

```bash
python -c "
from torch.backends.cuda import sdp_kernel
import torch.nn.functional as F
import torch
with sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
    q = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.bfloat16)
    out = F.scaled_dot_product_attention(q, q, q)
    print('flash works')
"
```

3. 检查数据 I/O 不是瓶颈：`config.num_workers` 配 4-8

### A.8 评估生成全是 pad / EOS

**症状**：生成样本几乎全是 `</s>` 或 `<pad>`。

**原因**：模型还没训练好，或 EMA 没切换上。

**确认**：

```python
# eval.py 应该在 EMA swap 之后跑
grep "swap_into" torch/eval.py
```

如果是早期 epoch（< 3）正常现象。后期还出现说明训练失败。

---

## Appendix B · 关键监控指标速查

### B.1 训练时（每 100 步）

```
Step 1500: loss=0.8421 l2=0.8421 ce=0.0000 lr=2.00e-03 sps=2.83
Step 1600: loss=7.2156 l2=0.0000 ce=7.2156 lr=2.00e-03 sps=2.81
```

- `loss`：当前 micro-batch 的 loss，是 l2 或 ce 中的一个（取决于该步是哪个分支）
- `l2`：denoise 分支的 MSE（已按 `1 - decoder_prob` 归一化）
- `ce`：decode 分支的 CE（已按 `decoder_prob` 归一化）
- `lr`：当前学习率
- `sps`：steps per second，包含 forward + backward + step

### B.2 GPU 监控

```bash
# 持续显存 + 利用率
nvidia-smi dmon -s pucvmet -i 0,1,2,3,4,5,6,7

# 一次性看温度
nvidia-smi --query-gpu=index,temperature.gpu,power.draw --format=csv
```

### B.3 W&B 推荐 dashboard

| Panel | 内容 |
|---|---|
| `train_loss` line | 应该单调下降 |
| `train_l2_loss` line | denoise 分支 |
| `train_ce_loss` line | decode 分支 |
| `lr` line | warmup 后保持常数 0.002 |
| `epoch` step | 进度，到 5.0 训练结束 |
| `generation/*/ppl` (epoch 结束) | 每 epoch 后的评估 PPL |
| `generation/*/mean_entropy` | 评估时的 unigram entropy |

### B.4 训练结束后磁盘

```bash
du -sh outputs/elf-b-owt-full/
```

期望 ~3-5 GB：10 份 checkpoint × ~250 MB（ELF-B 含 params + EMA + optimizer state）+ 各 epoch 的生成 jsonl。

---

## 下一步

- 复现成功后想改进什么：先读 `ELF.html` 的 §5 (CFG) 和 §6 (Sampling)，sweep `sampling_configs/*.yml`
- 想训练条件生成：跑 `train_de-en_ELF-B.yml` 或 `train_xsum_ELF-B.yml`
- 想理解代码细节：`ELF.html` 的 §8 代码导览里每个文件都有关键代码片段

复现遇到与本指南不符的情况，记得在 W&B 或 git issue 里反馈，方便后人少走弯路。
