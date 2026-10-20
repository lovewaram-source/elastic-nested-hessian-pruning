# 验证记录

## 2026-10-20：整理后的小模型 CUDA 验证

在 Linux 服务器的既有环境中，对独立上传的精简源码运行完整测试集，未修改共享环境。使用一张 RTX 4090（24GB），GPU 编号通过 `CUDA_VISIBLE_DEVICES=0` 限制。

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m unittest discover -s tests -p 'test_*.py'
```

实际结果：`Ran 56 tests in 33.204s`，`OK`，退出码为 0。

测试包含随机小型 Llama 的分支前向、Token 选择与统计构造、非结构化和2:4重建、PPL数据流，以及掩码、数值等价和导出检查。不需要下载7B模型。

### 实际环境

| 项目 | 版本 |
|---|---|
| 系统 | Ubuntu 22.04；Linux 6.8.0-110-generic |
| Python | 3.10.20 |
| PyTorch | 2.3.1+cu121 |
| Transformers | 4.40.2 |
| datasets | 3.6.0 |
| accelerate | 0.29.3 |
| NumPy | 2.2.6 |
| sentencepiece | 0.2.2 |
| protobuf | 7.36.0 |

`requirements.txt`记录这些已测试的顶层依赖版本，不是包含所有传递依赖的锁文件；尚未验证从空环境安装的全过程。

### 本次修复

抽取公共函数后，`synchronize()`在CUDA路径缺少`torch`导入，首次测试出现36个错误。已改为CUDA分支内延迟导入，并增加CPU无操作、CUDA同步调用的回归测试，随后完整复跑通过。

服务器上已验证文件的SHA-256：

```text
scripts/experiment_common.py
95380a4dc15e39a7991dc1dad442d0c1b50de09290a9d71537177170f0f77a3e
tests/test_experiment_common.py
a39e65aaed37fa678b4d6bc8956225cf4c442d51cb76b397a797982659d571a4
```

### 不应推导出的结论

- 本轮没有重跑完整7B校准与WikiText2 PPL，不能把历史结果称为本版本重新复现。
- 小模型测试通过不证明所有模型结构、显存预算、数据预处理或硬件环境均受支持。
- 本轮没有验证断点续跑、部署或永久弹性模型导出；当前主入口不提供这些功能。

历史数值及来源限制见[结果记录](RESULTS.md)；接手者仍应在自己的环境执行测试和小规模检查。
