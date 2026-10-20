# Hessian 对照记录

这些数值是已核对的历史运行汇总，不是本次代码整理重新运行的结果。模型 Llama-2-7B，C4 64×4096，seed=0，完整 WikiText2；非结构化 alpha=1，2:4 alpha=0.25。代码随后继续演进，复现时需要核对原记录中的源码哈希、数据和执行协议。当前导出包不包含服务器原始日志或临时权重。

## 非结构化 PPL

| 方法 | 20% | 30% | 40% | 50% |
|---|---:|---:|---:|---:|
| 历史 common dense-reference H | 5.56156042 | 5.90267799 | 6.22839916 | 6.60510868 |
| 历史各率完整对齐 H | 5.55390626 | 5.87659452 | 6.20337076 | 6.58272300 |
| 全位置平均激活→H | 5.54298120 | 5.86470670 | 6.21793234 | 6.59688037 |
| 50%位置平均激活→H | — | — | — | — |
| 10% Token 私有（checkpoint） | 5.55376972 | 5.87503266 | 6.22234310 | 6.59199453 |
| 50% Token 私有（checkpoint） | 5.55386925 | 5.87764920 | 6.21922068 | 6.59906307 |

## 2:4 PPL

| 方法 | 20% | 30% | 40% | 50% |
|---|---:|---:|---:|---:|
| 历史 common dense-reference H | 5.91052147 | 6.52338581 | 7.29507053 | 8.72178931 |
| 历史各率完整对齐 H | 5.90984287 | 6.50378078 | 7.24646935 | 8.61114292 |
| 全位置平均激活→H | 5.90432750 | 6.50485909 | 7.24290565 | 8.64300416 |
| 50%位置平均激活→H | 5.90766047 | 6.50751398 | 7.26464703 | 8.67728226 |
| 10% Token 私有（checkpoint） | 5.90946925 | 6.51422239 | 7.27105275 | 8.69575374 |
| 50% Token 私有（checkpoint） | 5.90949285 | 6.51477124 | 7.27451802 | 8.66845144 |
| 全位置平均各分支 H | 5.89861083 | 6.49984734 | 7.24896893 | 8.65199327 |

`—` 表示没有已核实的完成记录。50%平均是在差异最大的50%位置平均分支激活，其余位置使用20%锚点；50%私有在隔离位置使用各分支自身激活。

## 来源和可比性

来源运行目录标识：`checkpoint-token-isolation-20261016`、`checkpoint-token-isolation-private050-20261016`、`checkpoint-token-isolation-isolated-mean-20261016`、`checkpoint-all-mean-20261017`、`checkpoint-mean-hessian-20261017`。历史 dense-H/完整对齐 H 来源为第一组的 `summary.tsv`，候选方法来自各模式的 `results.json` / `comparison.tsv`。

## 方法、命令与运行记录对应

当前源码中的命令配方统一见 [EXPERIMENTS.md](EXPERIMENTS.md)。下表用于定位历史记录，不表示用当前源码再次得到过同样PPL。

| 表中方法 | 当前入口/方法参数 | 历史运行标识 |
|---|---|---|
| common dense-reference H / 各率完整对齐H | `run_rate_aligned_nested.py`；重建策略与当前主入口不同 | `checkpoint-token-isolation-20261016/summary.tsv`中引用的历史列 |
| 10%私有 | `cosine / anchor20 / 0.1` | `checkpoint-token-isolation-20261016` |
| 50%私有 | `cosine / anchor20 / 0.5` | `checkpoint-token-isolation-private050-20261016` |
| 全部平均表示 | `all_mean / isolated_mean_anchor20 / 1` | `checkpoint-all-mean-20261017` |
| 50%位置平均 | `cosine / isolated_mean_anchor20 / 0.5` | `checkpoint-token-isolation-isolated-mean-20261016/private050` |
| 平均H | `all_mean / mean_hessian / 1` | `checkpoint-mean-hessian-20261017` |

参数顺序为`selection-rule / shared-fusion / private-fraction`。历史运行应以自己的`protocol.json`内`source_sha256`识别源码，不用今天的Git提交号替代原版本。当前公开包尚未提供所有历史源码快照、数据指纹和原始日志；缺失的追溯材料应补齐后再宣称严格复现。当前小模型代码验证记录见 [VALIDATION.md](VALIDATION.md)。

完整对齐 H 是每个目标率从 dense 重放掩码的历史参考；当前 checkpoint 方法为单轨迹分阶段重建、继承数值前缀。虽然模型和校准配置相同，执行策略不同，以上不是严格同次配对实验。

此前展示的“收益恢复率”定义为 `(PPL_common - PPL_method) / (PPL_common - PPL_aligned)`。这是描述性对照；2:4的20%分母仅0.00067860，比例对微小PPL波动极为敏感。不要把大于100%的比例解释为稳定的倍数收益，也不要把历史 common dense-H 误标成真实20%分支 H。
