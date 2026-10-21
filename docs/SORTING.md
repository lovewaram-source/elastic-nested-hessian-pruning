# 历史排序工作：代码现状与阅读入口

## 检查结论

**现已提供独立排序入口 `scripts/run_sorting_nested.py`，历史旧CLI仍归档。** 新入口固定使用稠密参考路径的H，不融合Token，在单卡上按层重建、评测20%/30%/40%/50%四档。当前Hessian/Token主入口仍保持原始列顺序。

本页基于源码检查，不代表重新运行了排序PPL实验。现有Hessian结果表也不包含排序收益。

## 已保留的实现

主要实现位于 [lib/elastic_sparsegpt.py](../lib/elastic_sparsegpt.py) 的 `column_permutation()`。独立排序入口通过`--ordering`开放下面的部分静态模式；未开放的变体仍仅为内核API，不是Token主入口选项。

| 内核 `ordering` | 排序依据与范围 | 实际顺序 |
|---|---|---|
| `original` | 不排序 | 原始列序 |
| `wanda` | 全列Wanda分数求和，排列单列 | 低分在前 |
| `sparsegpt` | 全列SparseGPT敏感度求和，排列单列 | 低分在前 |
| `wanda_pruned_part` / `sparsegpt_pruned_part` | 先按给定全局稀疏率预测层级掩码，再按各列被预测剪掉部分的分数和排序 | 低分在前 |
| `wanda_group` | 每个原始四列组的完整Wanda分数和，整组移动 | 低分在前；不是宏块内降序 |
| `wanda_group_pruned_part` | 每行每四列预测剪掉两个，累加这些位置的Wanda分数，整组移动 | 低分在前 |
| `wanda_group_stage_desc` | 在固定40%/20%/20%/20%宏块内，对完整四列组评分 | **高分在前**；不跨宏块 |
| `wanda_group_pruned_part_stage_desc` | 同上，但只累加预测2:4剪掉位置的Wanda分数 | **高分在前**；不跨宏块 |
| `wanda_group_stage_quartile_interleave` | 每个宏块内单列按Wanda降序，切成四个等长区间，各取同一排名位置形成新四列组 | P&P式四分位交错；会改变组成员 |
| `wanda_group_stage_pnp_late_interleave` | 同一四分位交错，仅从40%稀疏率阶段开始 | 前两个宏块保持原序，后两个交错 |

另有组内列升/降序、预测剪枝部分的高低半区交错等变体，具体分支见 `column_permutation()`。**不能把所有排序都描述成“高敏感度在前”，也不能把保留四列组与重新分组混为一谈。**

### 宏块内、预测剪掉部分的评分

令Wanda元素分数为 $A_{ij}=|W_{ij}|a_j$，其中默认 $a_j=\sqrt{H_{jj}}$，在加阻尼前提取。对一个物理四列组 $g$，每行选分数最小的两个位置组成预测掩码 $P_g$：

$$
C_g=\sum_i\sum_{j\in g}A_{ij}(P_g)_{ij}.
$$

`wanda_group_pruned_part_stage_desc` 在每个宏块内按 $C_g$ 降序排列四列组。组内原始四列顺序不变。这是“用2:4 Wanda预测剪掉的部分来决定宏块内四列组顺序”的实现。

预测掩码用于**排序评分**；它不自动等于后续实际SparseGPT剪枝掩码。后者仍由重建阶段的掩码策略决定。

### 静态与动态不是一回事

`prepare_plan()` 通常在收集校准统计后生成一次排列。`reorder_next_2of4_stage_groups()` 则支持根据当前已更新权重，在下一阶段宏块内重排完整四列组，并重新分解排列后的逆Hessian；需要保留动态统计。

动态函数当前使用**整组Wanda分数**，不是预测剪掉部分的分数。它也不是每处理四列就重新排列一次，更不代表重新前向收集激活。

## 排列如何进入OBS

`ElasticSparseGPT.prepare_plan()` 生成排列 $\pi$，对逆Hessian的两轴应用相同排列后重新求上三角Cholesky因子。重建时使用 $W[:,\pi]$，结束后用逆排列恢复原始坐标。

不能只排列W而保持Hessian不变，也不能直接排列旧Cholesky因子冒充重新分解。排序改变处理顺序及分组；OBS仍在相应坐标系中传播剪枝误差。

## 如何调用已保留的内核

下面是单层API用法示例，不是历史7B实验的完整复现命令；本轮仅检查了接口与源码，未执行这个数值示例。

```python
import torch
from lib.elastic_sparsegpt import ElasticSparseGPT, apply_elastic_2of4_stage

layer = torch.nn.Linear(40, 8, bias=False)
pruner = ElasticSparseGPT(layer)
x = torch.randn(1, 16, 40)
with torch.no_grad():
    pruner.add_batch(x, layer(x))
plan = pruner.prepare_plan(
    ordering="wanda_group_pruned_part_stage_desc",
    percdamp=0.01,
)
apply_elastic_2of4_stage(
    layer, plan, sparsity=0.2, previous_sparsity=0.0,
    compensation_alpha=0.25,
)
```

完整模型实验还需要逐层校准、嵌套阶段调度、缓存、PPL评测和协议记录，不能用这段单层示例替代。

## RIA与历史入口的核对

当前 `column_permutation()` **没有 `ria` 排序分支**。本地归档中的RIA掩码代码、RIA+OBS代码不能当作RIA列排序的实现；已检查的 `ria_obs.py` 甚至明确拒绝非`none`列排序。历史聊天或表格中出现RIA，不足以证明当前发布包支持该排序实验。

本地可追查的归档位置（不在公开源码包中）：

- `.local_archive/legacy-source-20261020/main.py` 和 `lib/prune.py`：旧CLI及排序方法路由。
- 同目录下 `tests/test_wanda_group_ordering.py`：包含宏块内降序、预测剪枝部分和P&P交错的历史测试；它依赖旧`lib.prune`，未作为当前测试发布。
- 同目录下 `scripts/run_sorting_ablation_20_50.sh`、`run_sorting_ablation_20_50_global50.sh`、`aggregate_sorting_ablation.py`：非结构化排序调度与汇总脚本；含原机器路径，不是可移植入口。

本轮没有恢复整套旧入口，也没有将RIA排序或历史排序PPL标为已复现。新入口由小模型测试覆盖预算、排列和嵌套零权重；历史旧测试不直接依赖恢复。

## 独立排序对照怎么跑

先按[安装说明](../INSTALL.md)设置四个数据路径。下面是同一流水线中的2:4正式对照，使用一张卡逐个运行，各模式使用不同输出和缓存目录：

```bash
for ordering in original wanda_group_stage_desc wanda_group_pruned_part_stage_desc wanda_group_stage_quartile_interleave; do
  python scripts/run_sorting_nested.py \
    --model /path/to/Llama-2-7b-hf --mode 2of4 --gpu 0 \
    --ordering "$ordering" --structured-alpha 0.25 \
    --nsamples 64 --seqlen 4096 --seed 0 \
    --output "results/sorting-$ordering" --cache "cache/sorting-$ordering"
done
```

先调试时将校准改为`--nsamples 2 --seqlen 128`并增加`--eval-samples 2`；仍加载完整模型。非结构化支持`original`、`wanda`、`sparsegpt`、`wanda_pruned_part`、`sparsegpt_pruned_part`，预测全局掩码的稀疏率固定为50%。2:4只开放`original`与宏块内的四个静态组模式，晚期交错也可用；不开放任意单列排列、动态排序或RIA。

每层保存`permutations_layer_*.pt`；P&P重组后的2:4约束在排列坐标内检查，不保证逆排列回原矩阵后仍符合原始相邻物理四列组。部署或压缩时必须保留并正确使用这些排列，当前没有相应部署导出器。

结果写入`comparison.tsv`与`results.json`，协议记录H策略、排序时机、实际环境和数据指纹。每个成功评测档位才删除其已消费快照；本入口同样不提供断点续跑，也不改变现有Hessian结果表。

## 与当前Hessian工作的边界

`scripts/run_rate_aligned_nested.py` 的历史实验调用保持默认 `ordering="original"`；当前Token主入口使用 `scripts/token_isolation_core.py` 中的原序计划（单位排列）。因此已有Hessian/Token对照**不启用上述排序**；只有新独立入口显式传入排序选项。

排序回答“哪些列先处理、哪些列组成一组”；Hessian构造回答“使用哪些分支的激活统计来补偿”。两者可以作为不同实验维度，但开启排序后需要重新设置对照，不能直接沿用当前无排序PPL表。
