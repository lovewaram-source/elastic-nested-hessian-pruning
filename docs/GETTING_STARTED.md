# 同门接手指南

本项目研究多稀疏率激活统计如何参与嵌套剪枝补偿。第一次接手先跑通测试和小规模流程，再做正式PPL。全部操作在自己的输出和缓存目录中进行。

## 1. 先建立共同理解

按顺序阅读：[README](../README.md) → [METHOD](METHOD.md) → [EXPERIMENTS](EXPERIMENTS.md)。

需要先区分三件事：固定稀疏参考分支提供激活；Token策略构造统计；目标层从dense权重分阶段剪枝和补偿。私有Token表示部分H统计随分支变化，不是额外拟合私有权重ΔW。

历史common dense-H、历史完整对齐replay和当前单轨迹重建并非同一执行策略。先读结果页的来源说明，不把历史PPL差异当成严格配对的因果结论。

## 2. 下载与环境准备

```bash
git clone https://github.com/lovewaram-source/elastic-nested-hessian-pruning.git
cd elastic-nested-hessian-pruning
conda create -n elastic-hessian python=3.10 -y
conda activate elastic-hessian
# 先按 INSTALL.md 选择匹配驱动的PyTorch构建。
python -m pip install -r requirements.txt
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())'
```

实际验证过的服务器版本、设备和测试范围在 [VALIDATION.md](VALIDATION.md)。版本快照不能替代在新机器上重新安装和验证；不要直接修改他人的共享Conda环境。

## 3. 配置模型与数据

先准备已获授权的本地Hugging Face Llama模型，以及四个独立的`datasets.save_to_disk`数据目录。数据格式校验命令见 [INSTALL.md](../INSTALL.md)。下面路径都要改为自己的路径：

```bash
export MODEL_PATH=/data/models/Llama-2-7b-hf
export WANDA_C4_TRAIN=/data/datasets/c4/train
export WANDA_C4_VALIDATION=/data/datasets/c4/validation
export WANDA_WIKITEXT2_TRAIN=/data/datasets/wikitext2/train
export WANDA_WIKITEXT2_TEST=/data/datasets/wikitext2/test
```

四份数据都需要准备，且每份应是含`text`列的Dataset。记录来源、版本、split和指纹；相同seed不能抵消数据内容或样本顺序的变化。C4需要有分词后长于seqlen的文本。

## 4. 先运行测试

```bash
# 无GPU时，CUDA专用测试会跳过；有GPU时会执行小型随机Llama流程。
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/run_checkpoint_token_isolation.py --help
```

正常结束应看到`OK`；如果出现`FAILED`或Traceback，先解决问题再运行正式实验。数值测试使用随机小模型，不需要下载7B权重。通过这些测试不能证明新的7B PPL与历史记录相同。

## 5. 跑一次小规模实际流程

确认环境、模型和数据都能加载后执行：

```bash
python scripts/run_checkpoint_token_isolation.py \
  --model "$MODEL_PATH" --mode 2of4 --gpu 0 --device-map cpu \
  --nsamples 2 --seqlen 128 --seed 0 --eval-samples 2 \
  --rates 0.2,0.3,0.4,0.5 \
  --selection-rule cosine --shared-fusion anchor20 --private-fraction 0.1 \
  --structured-alpha 0.25 \
  --output results/my-smoke-private10 --cache cache/my-smoke-private10
```

这条命令仍会加载自己的完整模型并运行全部层，仅减少校准和评测Token，不能据此推断最低显存。逐层日志应出现`TRUE_CHECKPOINT_LAYER_COMPLETE`，成功评测的档位出现`RESULT`。完整运行后应有四个完成记录。

下面是结构示例，不是本次新PPL：

```text
TRUE_CHECKPOINT_LAYER_COMPLETE {"layer": 0, ...}
...
RESULT {"sparsity": "0.20", "ppl_wikitext2": "<实际值>", "status": "completed", ...}
```

## 6. 查看结果，再扩大实验

```bash
python - <<'PY'
import json
from pathlib import Path
rows = json.loads(Path('results/my-smoke-private10/results.json').read_text())
for row in sorted(rows, key=lambda x: float(x['sparsity'])):
    print(row['sparsity'], row['ppl_wikitext2'], row['status'], row['nested_mask_violations'])
PY
```

核对目标档位、`status=completed`、实际零权重预算及`nested_mask_violations=0`。正式实验另开目录，将nsamples恢复64、seqlen恢复4096，并移除`--eval-samples`；主方法的完整参数在 [EXPERIMENTS.md](EXPERIMENTS.md)。正式与调试结果分开保存。

## 7. 接着改实验时改哪里

| 想做的改动 | 从哪里开始 | 至少核验什么 |
|---|---|---|
| 10%改为50%隔离 | CLI `--private-fraction 0.5` | 位置比例、嵌套预算和前向路径 |
| 改余弦筛选指标 | `checkpoint_token_isolation_core.py` | 统一位置排名、稳定平局、直接H等价 |
| 改平均/私有统计 | 同上及`run_checkpoint_token_isolation.py` | 归一化、分支独立传播、H与直接矩阵乘法一致 |
| 改补偿系数 | CLI两个`*-alpha`参数 | 固定掩码与数据，避免同时改多个因素 |
| 改OBS核心 | `lib/elastic_sparsegpt.py` | 零权重不复活、前缀继承、逐阶段预算 |
| 改Token定位消融 | `module_token_isolation.py` | 同预算与随机/余弦控制组 |

源码阅读路径见 [CODE_MAP.md](CODE_MAP.md)。建议第一项新实验是同一执行策略下的0%/100%私有控制与候选方法对照，再增加seed和完整成本统计。

## 8. 交回结果需要带什么

每个实验保留完整命令、源码提交号、`SOURCE_MANIFEST.json`、依赖版本、模型和数据指纹，以及`protocol.json`、`comparison.tsv`、`results.json`、计时文件。报告是否使用完整PPL、是否中断，以及是否有掩码违例。

当前边界：checkpoint主入口无断点续跑、无永久模型导出，尚未统一标定最低显存/内存；CPU调度不代表PPL阶段不需要完整模型显存。不要删除失败运行目录或复用别人的非空缓存。
