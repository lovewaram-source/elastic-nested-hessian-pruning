# 安装与离线数据准备

当前 Hessian 实验使用 Linux、NVIDIA GPU、Python 3.10+ 和支持 SDPA 的 PyTorch。建议新建环境。

```bash
conda create -n nested-pruning python=3.10 -y
conda activate nested-pruning
# 根据服务器驱动选择 PyTorch CUDA 构建，再安装其余依赖。
python -m pip install -r requirements.txt
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())'
```

`requirements.txt` 是兼容范围，尚不是经过完整7B配对复跑验证的锁定环境。运行正式实验时，将实际版本另存为运行记录：

```bash
python -m pip freeze > results-environment.txt
nvidia-smi
```

推荐Python 3.10或3.11环境。当前代码面向标准Llama结构和Linux CUDA实验，Windows本地可阅读源码或运行环境允许的轻量检查，不作为已验证的7B运行平台。完整小模型数值测试尚需在具备可用PyTorch的干净环境复核；文档提供命令不等于该环境已经通过验证。

## 模型和数据

模型目录应包含已获授权的 Hugging Face Llama 模型、tokenizer 和配置文件。使用 `--model /absolute/path/to/model` 显式传入。当前 checkpoint runner 默认离线运行。

数据加载使用 `datasets.load_from_disk`，四个路径分别应指向含 `text` 列的 Dataset，而不是压缩包或原始 JSON 文件。请事先准备正确的 C4 train/validation、WikiText2 train/test 划分。

```bash
export WANDA_C4_TRAIN=/data/datasets/c4/train
export WANDA_C4_VALIDATION=/data/datasets/c4/validation
export WANDA_WIKITEXT2_TRAIN=/data/datasets/wikitext2/train
export WANDA_WIKITEXT2_TEST=/data/datasets/wikitext2/test
```

必须在启动 Python 前设置变量。未设置时保留原服务器默认路径 `/data/DataSet/llm_dataset/...`。环境变量只改变路径，不改变采样逻辑、拼接方式、随机种子或归一化。

四个路径都需要准备：当前C4加载器会读取validation，WikiText2评测加载器也会读取train。每个路径应直接对应一个Dataset，而不是带多个split的DatasetDict。可先检查：

```bash
python - <<'PY'
import os
from pathlib import Path
from datasets import load_from_disk
for key in ('WANDA_C4_TRAIN', 'WANDA_C4_VALIDATION',
            'WANDA_WIKITEXT2_TRAIN', 'WANDA_WIKITEXT2_TEST'):
    path = Path(os.environ[key])
    if not path.is_dir():
        raise FileNotFoundError(f'{key}: {path}')
    data = load_from_disk(str(path))
    if not hasattr(data, 'column_names') or 'text' not in data.column_names:
        raise ValueError(f'{key}: expected a Dataset with a text column')
    if len(data) == 0:
        raise ValueError(f'{key}: empty dataset')
    print(key, len(data), data.column_names, getattr(data, '_fingerprint', None))
PY
```

目前未提供自动下载和转换数据的脚本。请记录自己准备数据所用的公开来源、版本、split和转换步骤；只有相同名称或相同样本数量，不能保证复现原实验。C4校准文本需要有足够长的样本；若所有文本都短于seqlen，当前重试采样循环不会正常结束，应在正式运行前核对数据长度。

数据来源、版本、预处理和样本顺序会影响校准采样；仅有同样的路径名称不保证复现历史数值。正式实验需要同时记录数据来源/指纹。不要提交模型权重、数据集或访问凭据。

## 验证

```bash
python -m unittest discover -s tests -p 'test_*.py'
python scripts/run_checkpoint_token_isolation.py --help
```

测试含小型随机 Llama 模型，主要验证数值和数据流，不需要下载7B权重。正式 GPU 实验及其缓存需求见 [EXPERIMENTS.md](docs/EXPERIMENTS.md)。
