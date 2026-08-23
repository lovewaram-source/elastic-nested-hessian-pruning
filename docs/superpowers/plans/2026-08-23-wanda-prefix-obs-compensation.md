# Wanda Prefix OBS Compensation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `elastic_wanda_prefix_obs`: fixed Wanda column ordering, nested prefix deletion, and OBS compensation at every checkpoint.

**Architecture:** Reuse the current calibration and disk-plan machinery. Retain the dense-pass inverse Hessian, calculate one Wanda permutation per Linear layer, and progressively zero newly opened prefix columns while updating only the live suffix by a group OBS solve.

**Tech Stack:** Python, PyTorch, Transformers, existing Wanda/SparseGPT utilities, unittest.

**Spec:** `docs/designs/2026-08-23-wanda-prefix-obs-compensation.md`

## Global Constraints

- Use `column_permutation(..., ordering="wanda")`.
- Supported checkpoints remain 20%, 30%, 40%, 50%, 60%, 70%, 80%.
- Do not change existing elastic methods.
- Use `torch.linalg.solve`, never `torch.linalg.inv`.
- Restore original column coordinates before assigning the weight.
- This is not a physical architecture-shrinking method.

---

### Task 1: Add prefix-boundary helper and test

**Files:**
- Modify: `lib/elastic_sparsegpt.py`
- Test: `tests/test_elastic_sparsegpt.py`

**Interfaces:**
- Produces: `prefix_stage_indices(columns: int, previous_sparsity: float, sparsity: float) -> tuple[int, int]`.

- [ ] **Step 1: Write the failing test**

```python
def test_prefix_stage_indices_open_only_new_columns(self):
    from lib.elastic_sparsegpt import prefix_stage_indices
    self.assertEqual(prefix_stage_indices(10, 0.0, 0.2), (0, 2))
    self.assertEqual(prefix_stage_indices(10, 0.2, 0.3), (2, 3))
    self.assertEqual(prefix_stage_indices(10, 0.7, 0.8), (7, 8))
```

- [ ] **Step 2: Run the test**

```powershell
python -m unittest -v tests.test_elastic_sparsegpt.ElasticSparseGPTTest.test_prefix_stage_indices_open_only_new_columns
```

Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Implement the helper**

```python
def prefix_stage_indices(columns, previous_sparsity, sparsity):
    if columns < 1:
        raise ValueError("columns must be positive")
    if not 0.0 <= previous_sparsity < sparsity <= MAX_SPARSITY:
        raise ValueError("stage sparsities must satisfy 0 <= previous < current <= max")
    return round(previous_sparsity * columns), round(sparsity * columns)
```

- [ ] **Step 4: Re-run the test**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add lib/elastic_sparsegpt.py tests/test_elastic_sparsegpt.py
git commit -m "test: define Wanda prefix intervals"
```

### Task 2: Implement one OBS prefix stage with direct-reference test

**Files:**
- Modify: `lib/elastic_sparsegpt.py`
- Test: `tests/test_elastic_sparsegpt.py`

**Interfaces:**
- Produces: `apply_wanda_prefix_obs_stage(layer, plan, sparsity, previous_sparsity=0.0) -> None`.
- Consumes: `ElasticSparseGPTPlan.hessian_inverse`, fixed `plan.permutation`, and Task 1 helper.

- [ ] **Step 1: Write the failing reference test**

```python
def test_prefix_obs_stage_matches_direct_group_solve(self):
    from lib.elastic_sparsegpt import ElasticSparseGPTPlan, apply_wanda_prefix_obs_stage
    layer = nn.Linear(3, 2, bias=False)
    layer.weight.data.copy_(torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
    h_inv = torch.tensor([[2., .2, .1], [.2, 3., .4], [.1, .4, 4.]])
    plan = ElasticSparseGPTPlan(
        permutation=torch.tensor([1, 0, 2]),
        hessian_factor=None,
        dead_columns=torch.zeros(3, dtype=torch.bool),
        hessian_inverse=h_inv,
    )
    ordered_before = layer.weight.detach().clone()[:, plan.permutation]
    ordered_h_inv = h_inv[plan.permutation][:, plan.permutation]
    expected = ordered_before.clone()
    expected[:, 1:] -= ordered_before[:, :1] @ torch.linalg.solve(
        ordered_h_inv[:1, :1], ordered_h_inv[:1, 1:]
    )
    expected[:, :1] = 0
    apply_wanda_prefix_obs_stage(layer, plan, 1 / 3)
    self.assertTrue(torch.allclose(layer.weight[:, plan.permutation], expected))
```

- [ ] **Step 2: Run the test**

```powershell
python -m unittest -v tests.test_elastic_sparsegpt.ElasticSparseGPTTest.test_prefix_obs_stage_matches_direct_group_solve
```

Expected: FAIL because the stage function does not exist.

- [ ] **Step 3: Implement the minimal stage**

```python
def apply_wanda_prefix_obs_stage(layer, plan, sparsity, previous_sparsity=0.0):
    weight = _weight_matrix(layer)
    start, end = prefix_stage_indices(weight.shape[1], previous_sparsity, sparsity)
    if plan.hessian_inverse is None:
        raise ValueError("Wanda prefix OBS stage requires a Hessian inverse")
    permutation = plan.permutation.to(weight.device)
    inverse_permutation = invert_permutation(permutation)
    ordered_weight = weight[:, permutation]
    ordered_h_inv = plan.hessian_inverse.to(weight.device)[permutation][:, permutation]
    removed = ordered_weight[:, start:end].clone()
    if end < weight.shape[1]:
        coupling = torch.linalg.solve(ordered_h_inv[start:end, start:end], ordered_h_inv[start:end, end:])
        ordered_weight[:, end:] -= removed @ coupling
    ordered_weight[:, start:end] = 0
    weight = ordered_weight[:, inverse_permutation]
    layer.weight.data = weight.reshape_as(layer.weight).to(layer.weight.dtype)
```

- [ ] **Step 4: Add and run nested-zero regression test**

```python
def test_prefix_obs_second_stage_keeps_first_prefix_zero(self):
    # Apply 0.0->0.25 and 0.25->0.5 to a four-column Linear layer.
    # Assert the first two ordered columns are zero after the second call.
```

```powershell
python -m unittest -v tests.test_elastic_sparsegpt
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add lib/elastic_sparsegpt.py tests/test_elastic_sparsegpt.py
git commit -m "feat: compensate Wanda column prefixes"
```

### Task 3: Add isolated CLI wiring and retained plan statistics

**Files:**
- Modify: `main.py`
- Modify: `lib/prune.py`
- Test: `tests/test_elastic_sparsegpt.py`

**Interfaces:**
- Consumes: Task 2 `apply_wanda_prefix_obs_stage`.
- Produces: `--prune_method elastic_wanda_prefix_obs`.

- [ ] **Step 1: Write failing parser and ordering tests**

```python
def test_parser_accepts_wanda_prefix_obs(self):
    from main import build_parser
    self.assertEqual(
        build_parser().parse_args(["--prune_method", "elastic_wanda_prefix_obs"]).prune_method,
        "elastic_wanda_prefix_obs",
    )

def test_prefix_obs_uses_wanda_ordering(self):
    from lib.prune import elastic_ordering_for_method
    self.assertEqual(elastic_ordering_for_method("elastic_wanda_prefix_obs"), "wanda")
```

- [ ] **Step 2: Run the tests**

```powershell
python -m unittest -v tests.test_elastic_sparsegpt.ElasticSparseGPTTest.test_parser_accepts_wanda_prefix_obs tests.test_elastic_sparsegpt.ElasticSparseGPTTest.test_prefix_obs_uses_wanda_ordering
```

Expected: FAIL because the method is unknown.

- [ ] **Step 3: Implement wiring**

Add the method to `main.py` parser choices and the `ELASTIC_ORDERING_BY_METHOD` map. Add a dedicated predicate, e.g. `is_prefix_obs_wanda_elastic_method`. In preparation, retain `hessian_inverse` for this method without turning on dynamic reordering. In the model-level stage runner, call Task 2 only when this predicate is true; preserve the existing static and dynamic branches untouched.

- [ ] **Step 4: Run full elastic tests**

```powershell
python -m unittest -v tests.test_elastic_sparsegpt
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add main.py lib/prune.py tests/test_elastic_sparsegpt.py
git commit -m "feat: expose Wanda prefix OBS pruning"
```

### Task 4: Verify disk-plan persistence, syntax, and compatibility

**Files:**
- Test: `tests/test_elastic_plan_store.py`
- Test: `tests/test_reporting.py`

**Interfaces:**
- Consumes: existing `ElasticSparseGPTPlan.cpu()` and plan-store serialization.

- [ ] **Step 1: Write a disk roundtrip test**

```python
def test_disk_plan_store_preserves_hessian_inverse_for_prefix_obs(self):
    # Store a plan containing a 2x2 hessian_inverse, reload it,
    # and assert exact equality and CPU placement.
```

- [ ] **Step 2: Run plan-store tests**

```powershell
python -m unittest -v tests.test_elastic_plan_store
```

Expected: PASS if the existing serializer preserves `hessian_inverse`; otherwise implement the missing field in `lib/elastic_plan_store.py` and re-run.

- [ ] **Step 3: Run complete targeted verification**

```powershell
python -m unittest -v tests.test_data tests.test_elastic_plan_store tests.test_elastic_sparsegpt tests.test_reporting
python -m py_compile main.py lib/prune.py lib/elastic_sparsegpt.py lib/elastic_plan_store.py
git diff --check
```

Expected: all commands exit with code 0.

- [ ] **Step 4: Commit**

```powershell
git add lib/elastic_plan_store.py tests/test_elastic_plan_store.py tests/test_elastic_sparsegpt.py tests/test_reporting.py
git commit -m "test: persist Wanda prefix OBS plans"
```

### Task 5: Run the local-data server experiment

**Files:**
- No source changes.

- [ ] **Step 1: Verify local prerequisites on the server**

```bash
cd ~/projects/wanda
test -f /data/ModelZoo/LLM/llama2/Llama-2-7b-hf/config.json
grep -n "load_from_disk" lib/data.py
```

Expected: model config exists and the loader is local.

- [ ] **Step 2: Run on GPU 0**

```bash
cd ~/projects/wanda
conda activate wanda
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE=/data/huggingface_cache/datasets
CUDA_VISIBLE_DEVICES=0 python main.py \
  --model /data/ModelZoo/LLM/llama2/Llama-2-7b-hf \
  --prune_method elastic_wanda_prefix_obs \
  --sparsity_type unstructured \
  --elastic_sparsities 0.2 0.3 0.4 0.5 0.6 0.7 0.8 \
  --elastic_cache_mode disk \
  --elastic_cache_dir ~/elastic_cache/llama2-7b-prefix-obs \
  --elastic_profile_preparation \
  --nsamples 128 --seed 0 \
  --save ~/projects/wanda/results/llama2-7b-wanda-prefix-obs-128
```

- [ ] **Step 3: Check checkpoints**

```bash
find ~/projects/wanda/results/llama2-7b-wanda-prefix-obs-128 -name 'log_elastic_wanda_prefix_obs.txt' -print -exec cat {} \;
```

Expected: seven logs with monotonically increasing target and actual sparsity.

## Self-Review

- Task 1 covers stage boundaries; Task 2 covers compensation and nesting; Task 3 isolates CLI routing; Task 4 covers persistence and regression checks; Task 5 provides a server run consistent with the local model/C4 constraints.
- The implementation uses one exact function name, `apply_wanda_prefix_obs_stage`, from Task 2 through Task 3.
- No task changes existing methods or asserts an inference-speed benefit.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-23-wanda-prefix-obs-compensation.md`. Two execution options:

1. **Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

