# SlimGPT Structured Pruning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a paper-derived SlimGPT structured-pruning baseline for LLaMA-2-7B.

**Architecture:** A focused `lib/slimgpt.py` owns OBS group selection, compensation, dependency-aware module slicing, and layer-ratio scheduling. `lib/prune.py` owns layer-wise calibration orchestration, while `main.py` exposes the CLI and standard evaluation/reporting flow.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers, unittest.

**Spec:** `docs/superpowers/specs/2026-08-24-slimgpt-structured-pruning-design.md`

## Global Constraints

- Do not alter existing pruning method behavior.
- Support standard LLaMA MHA only; reject GQA explicitly.
- Physically shrink Linear modules; zero masks alone are invalid.
- Use server-local LLaMA-2-7B and local C4/WikiText-2 assets.
- Put transient server logs and stage artifacts only under `/tmp`.

---

### Task 1: OBS structured group kernel

**Files:**
- Create: `lib/slimgpt.py`
- Create: `tests/test_slimgpt.py`

**Interfaces:**
- Produces: `obs_group_loss(weight, hessian_inverse, indices) -> Tensor`
- Produces: `remove_obs_group(weight, hessian_inverse, indices) -> tuple[Tensor, Tensor, Tensor]`

- [ ] Write tests with a two-row, four-column matrix that compare group loss and compensation to direct matrix formulas, verify removed indices, and verify the returned inverse Hessian has the survivor shape.
- [ ] Run `python -m unittest -v tests.test_slimgpt.OBSGroupTests` and confirm missing imports fail.
- [ ] Implement the two functions using `torch.linalg.solve`, the OBS update, and the Schur complement.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Ratio schedule and physical LLaMA slicing

**Files:**
- Modify: `lib/slimgpt.py`
- Modify: `tests/test_slimgpt.py`

**Interfaces:**
- Produces: `layer_pruning_ratios(layer_count, target_ratio, profile) -> list[float]`
- Produces: `calibrate_structural_ratios(model, target_ratio, profile) -> list[float]`
- Produces: `slice_llama_attention(attention, kept_heads) -> None`
- Produces: `slice_llama_mlp(mlp, kept_channels) -> None`

- [ ] Write tests that assert the log profile is increasing, the calibrated discrete schedule matches a global parameter target, attention projections retain coupled head rows/columns, FFN projections retain coupled neuron rows/columns, module attributes update, and GQA raises `ValueError`.
- [ ] Run `python -m unittest -v tests.test_slimgpt.StructuralSlicingTests tests.test_slimgpt.RatioScheduleTests` and confirm failures are due to missing APIs.
- [ ] Implement Linear slicing while preserving dtype, device, and bias; update attention head counts and the legacy module-local concatenated-head width without changing projection input or residual-stream widths, and update MLP `intermediate_size` where present.
- [ ] Run the focused tests and confirm they pass.

### Task 3: Greedy head and dynamic-batch FFN selection

**Files:**
- Modify: `lib/slimgpt.py`
- Modify: `tests/test_slimgpt.py`

**Interfaces:**
- Produces: `prune_attention_with_obs(attention, hessian_inverse, ratio) -> dict`
- Produces: `prune_mlp_with_obs(mlp, hessian_inverse, ratio, min_group_size=8) -> dict`

- [ ] Write deterministic tests showing the lowest-loss complete head is selected, the FFN reaches the exact requested channel count, and both methods reduce parameters physically.
- [ ] Run the focused tests and confirm the new APIs are absent.
- [ ] Implement greedy head removal with one grouped OBS update per head and FFN dynamic batches from 1024 down to `min_group_size`, recomputing scores after every batch.
- [ ] Run all `tests.test_slimgpt` tests and confirm they pass.

### Task 4: Layer-wise calibration orchestration and CLI

**Files:**
- Modify: `lib/prune.py`
- Modify: `main.py`
- Modify: `lib/reporting.py`
- Modify: `tests/test_slimgpt.py`
- Modify: `tests/test_reporting.py`

**Interfaces:**
- Produces: `prune_slimgpt(args, model, tokenizer, device) -> dict`
- CLI method: `--prune_method slimgpt`

- [ ] Write tests with a dummy LLaMA model confirming calibration hooks feed `o_proj` and `down_proj`, the CLI accepts SlimGPT flags, forward still runs after pruning, and reporting includes actual parameter reduction.
- [ ] Run the focused orchestration and reporting tests and confirm they fail for missing behavior.
- [ ] Implement one-layer-at-a-time activation capture using the existing `get_loaders` and model device-map conventions; construct damped inverse Hessians, prune, propagate updated activations, and return timing/parameter metadata.
- [ ] Wire `main.py` dispatch and reporting without changing existing method branches.
- [ ] Run `python -m unittest -v tests.test_slimgpt tests.test_reporting` and confirm they pass.

### Task 5: Verification, upload, and server experiment

**Files:**
- Verify: all modified files

**Interfaces:**
- Consumes: `--prune_method slimgpt` and all APIs above.
- Produces: one normalized `slimgpt-20` row in the server summary tables.

- [ ] Run the complete local unit suite and `python -m py_compile main.py lib/prune.py lib/slimgpt.py lib/reporting.py`.
- [ ] Run `git diff --check`, inspect the final diff, commit, and push `main`.
- [ ] Pull the commit on `~/projects/wanda`, verify `lib/data.py` uses `load_from_disk`, and verify GPU 0 is available.
- [ ] Run LLaMA-2-7B with `--prune_method slimgpt --sparsity_ratio 0.2 --nsamples 128` using `/tmp/slimgpt-20` for output and `/tmp/slimgpt-20.out` for stdout/stderr.
- [ ] On completion, append the normalized result to `results/ALL_RESULTS.tsv`, regenerate `ALL_RESULTS.txt`, and remove `/tmp/slimgpt-20*`.
