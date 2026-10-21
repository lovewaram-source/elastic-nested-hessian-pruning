# 当前 Hessian 实验复现

先读[方法说明](METHOD.md)了解参考分支与目标重建的角色。下面的命令在源码根目录执行，先按 [INSTALL.md](../INSTALL.md) 设置四个数据路径，并把模型路径换成自己的本地模型目录。

## 第一次运行先做小规模检查

```bash
python scripts/run_checkpoint_token_isolation.py \
  --model /path/to/Llama-2-7b-hf \
  --mode 2of4 --gpu 0 --device-map cpu \
  --nsamples 2 --seqlen 128 --seed 0 --eval-samples 2 \
  --rates 0.2,0.3,0.4,0.5 \
  --selection-rule cosine --shared-fusion anchor20 --private-fraction 0.1 \
  --structured-alpha 0.25 \
  --output results/smoke-private10 --cache cache/smoke-private10
```

这只缩小校准和评测预算，仍加载原模型和运行全部层，不能用来估计最低模型显存。小型随机模型的CPU数值检查请运行`python -m unittest discover -s tests -p 'test_*.py'`。这里的短评测PPL不能与正式表格比较。

运行中应逐层出现`TRUE_CHECKPOINT_LAYER_COMPLETE`，评测成功的档位出现`RESULT`记录。全部结束后，应在输出目录看到四档`status=completed`、零权重预算符合目标、`nested_mask_violations=0`。异常中断时，目录中有部分文件并不表示整个实验完成。

## 正式实验和方法参数

以下配方统一使用 `scripts/run_checkpoint_token_isolation.py`。公共参数：`--model /path/to/model --mode 2of4 --gpu 0 --device-map cpu --nsamples 64 --seqlen 4096 --seed 0 --rates 0.2,0.3,0.4,0.5 --structured-alpha 0.25`。每种方法另外提供独立的 `--output results/<run>` 和全新 `--cache cache/<run>`。

| 方法 | 附加参数 |
|---|---|
| 10% Token 私有 | `--selection-rule cosine --shared-fusion anchor20 --private-fraction 0.1` |
| 50% Token 私有 | `--selection-rule cosine --shared-fusion anchor20 --private-fraction 0.5` |
| 全位置平均激活，再计算 H | `--selection-rule all_mean --shared-fusion isolated_mean_anchor20 --private-fraction 1` |
| 差异最大50%位置平均，其他位置用20%锚点 | `--selection-rule cosine --shared-fusion isolated_mean_anchor20 --private-fraction 0.5` |
| 全位置各分支 Gram 平均 | `--selection-rule all_mean --shared-fusion mean_hessian --private-fraction 1` |
| 替换误差≤5%用锚点，其他位置平均 | `--selection-rule relative_error --shared-fusion isolated_mean_anchor20 --share-error-threshold 0.05` |
| 替换误差≤10%用锚点，其他位置平均 | 上一项的阈值改为 `0.10` |

非结构化将 `--mode` 改为 `unstructured`，设置 `--unstructured-alpha 1`，其他条件保持一致。非结构化内部保留辅助10%阶段，10%/20%阶段使用 H20；当前 checkpoint 入口的默认报告范围为20%至50%。

例如，完整的全部平均表示对照为：

```bash
python scripts/run_checkpoint_token_isolation.py \
  --model /path/to/Llama-2-7b-hf \
  --mode 2of4 --gpu 0 --device-map cpu \
  --nsamples 64 --seqlen 4096 --seed 0 \
  --rates 0.2,0.3,0.4,0.5 \
  --selection-rule all_mean --shared-fusion isolated_mean_anchor20 --private-fraction 1 \
  --structured-alpha 0.25 \
  --output results/all-mean-2of4 --cache cache/all-mean-2of4
```

其他主对照以这条完整命令为模板，替换表中的三个方法参数并使用不同输出/缓存目录。平均H只将`--shared-fusion`改为`mean_hessian`，仍使用`--selection-rule all_mean`。

这里2:4表中的20/30/40/50%指全局零权重比例。按40/20/20/20%的列宏块依次开启2:4剪枝，尚未开启的区域保持稠密；因此中间档位是稠密区域与2:4区域混合，50%档位才覆盖全部宏块。

## 参数速查

| 参数 | 默认或推荐设置 | 含义 |
|---|---|---|
| `--model` | 显式传本地模型目录 | 默认路径属于原实验服务器，外部用户应覆盖 |
| `--gpu` | `0` | 单个物理GPU编号；主入口拒绝逗号分隔多卡列表 |
| `--device-map` | `cpu` | 校准/重建时模型驻CPU并逐层调入GPU；PPL时完整模型上GPU |
| `--mode` | 必填`2of4`或`unstructured` | 剪枝类型；全局稀疏率由`--rates`定义 |
| `--rates` | `0.2,0.3,0.4,0.5` | 目标分支集合，必须递增并从0.2开始；修改它会改变融合参与者 |
| `--private-fraction` | `0.1`或`0.5` | 余弦排名选中的位置比例；`all_mean`选全部位置 |
| `--selection-rule` | `cosine`、`relative_error`或`all_mean` | 按排名选位置、按误差门槛路由、全位置平均 |
| `--shared-fusion` | 见上方方法表 | 决定所选/未选位置使用何种激活统计 |
| `--share-error-threshold` | `0.05` | 相对误差门槛；不是Token比例 |
| `--structured-alpha` | `0.25` | 2:4补偿更新缩放系数 |
| `--unstructured-alpha` | `1` | 非结构化补偿更新缩放系数 |
| `--percdamp` | `0.01` | H分解前的相对对角阻尼，与α不同 |
| `--nsamples` / `--seqlen` | `64` / `4096` | C4校准序列数和每条序列长度 |
| `--eval-rates` | 默认评测全部目标率 | 只影响待评测档位，不减少融合所用参考分支 |
| `--eval-samples` | 默认不截短 | 调试用评测窗口上限；正式PPL不要设置 |
| `--output` / `--cache` | 必填不同的新目录 | 结果与临时层快照的位置，不能复用非空目录 |

CLI还保留`mean_token`这个不同的统计变体，不对应主表中的“全部平均”；复现主表请严格使用上方参数组合。`module_residual`、`module_random`、`module_cosine`是按模块的诊断消融，当前限定2of4、anchor20及10%隔离；修改实现路径前需通过数值一致性测试。

## 激活和补偿流程

先用普通嵌套基线产生20/30/40/50四个固定参考分支。每层使用各分支自己的参考权重和真实上游激活分别前向，收集实际线性输入；融合只影响 Hessian 统计，不覆盖四条真实前向路径。

余弦隔离在 block 入口比较同一位置与20%锚点的余弦距离，取30/40/50分支的最大值，按全部校准位置稳定排名。一层使用同一套隔离位置。第一层入口为相同 embedding，选择没有语义差异信号，受平局规则影响。

对私有集合 M，H_r 使用锚点非私有位置的 Gram，加上分支 r 私有位置的 Gram；按原校准样本数归一化。平均激活用各率逐位置均值再算 Gram；平均 H 则先算各分支 Gram 再平均。Q/K/V、gate/up 只有在输入确实相同的条件下复用统计。

构造本层 H 后，重建目标从原始 dense 层权重开始，分阶段使用对应 H 执行嵌套剪枝与补偿，并冻结继承前缀。参考路径与重建目标是两套角色；不对已经补偿的参考层重复执行目标 OBS。

## 完整对齐历史参考

```bash
python scripts/run_rate_aligned_nested.py \
  --model /path/to/Llama-2-7b-hf --mode 2of4 --gpu 0 --device-map auto \
  --nsamples 64 --seqlen 4096 --seed 0 --structured-alpha 0.25 \
  --output results/rate-aligned
```

这组包含 common dense-reference H 与每目标率完整对齐 replay。每个目标率从 dense 权重重放已有掩码，并用自己的 H 处理全部已开启片段；与当前 checkpoint 入口的单轨迹数值共享不同。历史 dense-H 不应直接改称“真实20%分支 H”，两者统计来源不同。

## 输出与核验

- `protocol.json`：参数、统计路由、执行策略与关键源码哈希。
- `comparison.tsv`、`results.json`：已完成的 PPL、预算、掩码违例和计时。只将 `status=completed` 的记录作为正式结果。
- `timings.json` / `preparation_timings.json`：包含参考生成的总成本与分项；PPL计时另计。
- Token 位置/误差统计与快照：以实际运行生成的文件为准。部分评测成功后会清理已消费目标快照；缓存不是长期模型交付格式。

可用标准库查看完成的PPL，不需要读取临时权重：

```bash
python - <<'PY'
import json
from pathlib import Path
rows = json.loads(Path('results/all-mean-2of4/results.json').read_text())
for row in sorted(rows, key=lambda x: float(x['sparsity'])):
    print(row['sparsity'], row['ppl_wikitext2'], row['status'], row['nested_mask_violations'])
PY
```

比较方法时同时核对模型、数据/样本顺序、seed、seqlen、alpha、评测窗口和重建策略。PPL没有固定的“必须等于某值”验收阈值，尤其不要把调试结果和[历史完整设置](RESULTS.md)混用。

输出/缓存路径非空时 checkpoint 入口会拒绝运行。调试时可将 `--nsamples`、`--seqlen`、`--eval-samples` 减小；正式结果必须恢复完整设置，并核对全部档位、零权重比例、`nested_mask_violations=0` 和全部参考前向计数。

当前入口会提前拒绝非法评测窗口数、非有限/非正阻尼、超出[0,1]的补偿系数，以及不支持的2:4档位。2:4的`--rates`必须是`0.2,0.3,0.4,0.5`的连续前缀（至少两档）；非结构化可使用其他整数百分比档位，避免按百分比命名的缓存碰撞。

`protocol.json.runtime`自动记录环境、实际加载的数据路径/指纹、模型config和运行源码依赖哈希。结果行的`evaluation_scope`区分`truncated_debug`与`full_available_windows`，`evaluation_windows`记录实际窗口数。独立排序实验使用[排序说明](SORTING.md)中的新入口，不与本页Token融合配方叠加。

## 常见问题与当前边界

- **为什么设了CPU调度还需要足够GPU显存？** 逐层调度用于校准和重建，最终PPL调用会把完整模型移动到所选GPU。
- **H20与历史common dense-H一样吗？** 不一样。H20来自真实20%参考分支输入，历史common来自dense参考；比较时应分别命名。
- **10%私有会额外剪掉10%的Token吗？** 不会。Token全部参与真实前向，比例只控制H统计的路由。
- **能在原目录断点续跑吗？** 当前checkpoint主入口不支持，失败后保留目录供核查，另开新目录重跑；历史rate-aligned入口的`--resume`是另一套机制。
- **`--save-final-target`会留下完整模型吗？** 不会。它只将最大稀疏率的临时层权重保留到该率评测成功，随后仍会清理；尚未提供永久模型导出接口。
- **预期多久、至少多少显存？** 尚未对统一硬件和当前实现做完整资源标定。比较时间时包含参考准备、数据流转、快照和重载，不只报告剪枝内核时间。

## 下一轮实验

先同口径复跑共用统计、100%自身统计和候选平均/隔离方法，保持重建策略一致，再比较PPL、不同seed和包括参考准备在内的总时间。当前历史完整对齐表可作描述性参照，不能代替这组严格控制实验。
