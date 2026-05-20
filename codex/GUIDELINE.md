# ELF PyTorch 运行指南

这份指南说明如何一步步运行 `codex/` 下的纯 PyTorch ELF。命令默认从仓库根目录 `/Users/xinyu/Code/ELF` 执行。

## 0. 先确认目录约定

本实现都在 `codex/` 下：

- `codex/train.py`：推荐训练入口。
- `codex/elf_pytorch/`：PyTorch 模型、训练、数据、采样和运行工具。
- `codex/configs/`：8xH200 配置。
- `codex/outputs/`：默认 checkpoint 输出目录。
- `codex/elf_explainer.html`：ELF 工作方式解释页。

仓库根目录有一个叫 `torch/` 的文件夹。不要把它加入实现依赖，也不要从里面读代码。训练时请运行文件路径 `codex/train.py`，不要使用 `python -m ...` 从仓库根目录导入包。

## 1. 准备 Python 环境

在 H200 机器上建议使用已有 CUDA/PyTorch 环境，或新建 conda/venv。安装依赖：

```bash
pip install -r codex/requirements-pytorch.txt
```

如果集群要求指定 CUDA wheel，请先按集群文档安装匹配版本的 PyTorch，再安装其余依赖：

```bash
pip install transformers datasets huggingface-hub PyYAML numpy wandb
```

## 2. 准备 Hugging Face 和可选 WandB

默认配置会从 Hugging Face 读取数据集和 T5 encoder：

- `embedded-language-flows/openwebtext-t5`
- `google-t5/t5-small`

如需登录：

```bash
huggingface-cli login
```

如果打开 `use_wandb: true`，先登录 WandB：

```bash
wandb login
```

## 3. 本地 smoke test

先跑小模型单元测试，确认 PyTorch 实现的导入、前向、反向和采样都正常：

```bash
cd codex
PYTHONPATH=. pytest tests/test_smoke.py
cd ..
```

预期结果：

```text
3 passed
```

## 4. 单机 8xH200 启动 ELF-B

ELF-B 是推荐的第一条完整链路：

```bash
torchrun --standalone --nproc_per_node=8 \
  codex/train.py \
  --config codex/configs/train_owt_elf_b_8xh200.yaml
```

配置默认值：

| 项目 | 值 |
| --- | --- |
| model | ELF-B |
| sequence length | 1024 |
| local micro-batch | 16 |
| grad accum | 4 |
| effective global batch | 512 |
| precision | bf16 |
| optimizer | Muon |
| output dir | `codex/outputs/elf_b-owt-8xh200` |

## 5. 启动 ELF-M 或 ELF-L

确认 ELF-B 链路稳定后再跑更大模型：

```bash
torchrun --standalone --nproc_per_node=8 \
  codex/train.py \
  --config codex/configs/train_owt_elf_m_8xh200.yaml
```

```bash
torchrun --standalone --nproc_per_node=8 \
  codex/train.py \
  --config codex/configs/train_owt_elf_l_8xh200.yaml
```

默认显存策略：

| Model | local micro-batch | grad accum | effective global batch |
| --- | ---: | ---: | ---: |
| ELF-B | 16 | 4 | 512 |
| ELF-M | 8 | 8 | 512 |
| ELF-L | 4 | 16 | 512 |

## 6. 先做短步数试跑

如果只想快速检查 8 卡是否能正常开始训练，可以用 override 改短训练：

```bash
torchrun --standalone --nproc_per_node=8 \
  codex/train.py \
  --config codex/configs/train_owt_elf_b_8xh200.yaml \
  --config_override epochs=1 \
  --config_override log_freq=10 \
  --config_override save_freq=0.02 \
  --config_override output_dir=codex/outputs/debug_elf_b
```

看到日志里持续打印 `loss`、`l2`、`ce`、`lr`，且 `codex/outputs/debug_elf_b/` 下面出现 checkpoint，就说明主链路可用。

## 7. 恢复训练

从 checkpoint 恢复：

```bash
torchrun --standalone --nproc_per_node=8 \
  codex/train.py \
  --config codex/configs/train_owt_elf_b_8xh200.yaml \
  --config_override resume=codex/outputs/elf_b-owt-8xh200/checkpoint_10000.pt
```

也可以直接改 YAML 里的 `resume` 字段。

## 8. 调整显存和吞吐

如果 OOM，优先降低 `micro_batch_size`，同时提高 `grad_accum_steps` 来保持 effective batch：

```bash
--config_override micro_batch_size=8 \
--config_override grad_accum_steps=8
```

如果显存富余，可以逐步增加 `micro_batch_size`，并按比例降低 `grad_accum_steps`。目标是保持：

```text
effective global batch = micro_batch_size * 8 * grad_accum_steps = 512
```

## 9. 输出文件

训练输出默认包括：

- `config.yml`：保存实际训练配置。
- `checkpoint_*.pt`：周期 checkpoint。
- `checkpoint_final.pt`：最终 checkpoint。

默认路径在各 YAML 的 `output_dir` 字段中。

## 10. 常见问题

### `ModuleNotFoundError: elf_pytorch`

运行测试时需要在 `codex/` 目录设置 `PYTHONPATH=.`：

```bash
cd codex
PYTHONPATH=. pytest tests/test_smoke.py
```

训练时使用仓库根目录的文件入口：

```bash
torchrun --standalone --nproc_per_node=8 codex/train.py --config ...
```

### 误导入仓库根目录的 `torch/`

不要从仓库根目录执行直接依赖包导入的 one-liner，例如 `python -c "import torch"` 可能被本地 `torch/` 目录干扰。训练入口 `codex/train.py` 已经做了路径隔离。

### Hugging Face 下载失败

确认机器能访问 Hugging Face，或设置共享缓存目录：

```bash
export HF_HOME=/path/to/shared/hf_cache
```

然后重新启动训练。也可以把数据集保存到本地 Arrow 目录，再把 YAML 的 `data_path` 指向本地路径。

### `CUDA out of memory`

先降低 `micro_batch_size`，保持或增加 `grad_accum_steps`。ELF-L 建议从默认 `micro_batch_size=4` 开始。

### loss 只有 L2 或只有 CE

这是正常现象。当前实现和原 JAX 版本一样，每个 micro-step 随机选择 denoiser 或 decoder 分支。长期平均下，默认大约 80% 是 L2，20% 是 CE。

## 11. 查看 HTML 解释页

直接在浏览器打开：

```text
codex/elf_explainer.html
```

如果需要本地 HTTP 服务：

```bash
cd /Users/xinyu/Code/ELF
python3 -m http.server 8000
```

然后访问：

```text
http://localhost:8000/codex/elf_explainer.html
```
