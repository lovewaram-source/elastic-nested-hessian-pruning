# 代码阅读地图

先读 [METHOD.md](METHOD.md) 对齐符号和数据流，再按下列顺序看代码。

## 推荐顺序

1. `scripts/run_checkpoint_token_isolation.py`：主实验。`prepare_reference_layer` 生成固定稀疏参考层；`reconstruct` 逐层收集各率真实输入、构造 H 并从原始稠密权重重建嵌套目标；`parse_args` 检查方法组合。
2. `scripts/checkpoint_token_isolation_core.py`：按 block 入口位置选 Token、挂钩真实线性输入、构造隔离/平均统计。`select_block_private_tokens` 对所有校准位置使用最坏分支余弦差异，一层一套位置，各投影共用。
3. `scripts/token_isolation_core.py`：Hessian 累加和分解。保留原始样本归一化、FP32 统计与 damping。
4. `lib/elastic_sparsegpt.py`、`lib/sparsegpt.py`：分阶段掩码与 OBS/SparseGPT 数值更新。主要算法内核保留在这里。
5. `scripts/run_rate_aligned_nested.py`：历史完整对齐 replay；每个目标率从 dense 权重重放继承掩码，使用自身分支激活。与当前嵌套单轨迹不同。

## 公共依赖

- `scripts/experiment_common.py` 集中提供模型输入捕获、PPL、结果写入、Llama输入分组、设备同步和数值一致性门槛。旧实验入口已移出，当前实验不依赖这些旧入口。
- `scripts/rate_aligned_src/__init__.py` 将实验包解析到项目 `lib/`。服务器过去可能复制内核到这个包；公开源码保留本地 fallback，导出时需要连同 `lib/` 一起提供。
- `lib/` 只保留 `data.py`、`nm.py`、`sparsegpt.py`、`elastic_sparsegpt.py`。原 `main.py`、整套基础方法路由及其他实验模块不属于当前代码入口。

## 修改位置

历史排序内核的函数、实际方向和旧入口归档情况见 [SORTING.md](SORTING.md)。排序主要在 `lib/elastic_sparsegpt.py` 的 `column_permutation()`、`_wanda_group_stage_desc_order()`、`_wanda_group_stage_quartile_interleave_order()` 和 `reorder_next_2of4_stage_groups()`；当前Hessian/Token主入口不启用它们。

| 要改什么 | 主要文件 |
|---|---|
| Block 级余弦排名、替换误差、全平均位置 | `checkpoint_token_isolation_core.py` |
| 单卡逐层载入、缓存、统计路由、评测阶段 | `run_checkpoint_token_isolation.py` |
| 按模块 residual/random/cosine 的隔离 | `module_token_isolation.py` |
| Token 误差诊断 | `token_residual_diagnostics.py`、`run_token_residual_diagnostics.py` |
| 补偿冲突诊断 | `compensation_conflict.py`、`run_compensation_conflict.py` |
| 数据集位置与采样 | `lib/data.py` |
| 精简发布范围 | `scripts/export_github.py` 的 `select_files` |

`run_*`、`benchmark_*`、`summarize_*` 以 `python scripts/文件名.py ...` 启动。辅助模块通常使用同目录导入；不要直接把单个文件复制到其他目录运行。

## 测试覆盖与实验边界

关键测试包括 `test_checkpoint_token_isolation.py`、`test_elastic_2of4.py`、`test_mean_hessian.py`、`test_module_token_isolation.py`。测试核验真实参考权重前向、直接 Gram 等价、四路独立传播、嵌套掩码和数值共享。公共函数另有不依赖PyTorch的 `test_experiment_common.py`。

“Token 私有”仅表示该位置的激活统计随分支变化。当前 checkpoint 实验从原始稠密层权重建立一条分阶段重建轨迹，使用 H20/H30/H40/H50，并继承冻结的数值前缀；没有单独拟合每率的参数残差 ΔW。
