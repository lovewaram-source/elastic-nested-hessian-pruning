# Single-Pass Elastic SparseGPT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace eight repeated Elastic SparseGPT calibrations with one dense-reference preparation pass whose cached plans drive a continuous, strictly nested 10%-to-80% pruning path.

**Architecture:** Separate Hessian collection/factorization from stage application. A reusable per-linear-layer plan stores one fixed permutation, one permuted inverse-Hessian Cholesky factor, and dead-column metadata on CPU or disk; a single model then advances through the requested stages while loading only the current layer plan.

**Tech Stack:** Python 3.10+, PyTorch, Transformers, `unittest`, existing Wanda/SparseGPT layerwise pruning utilities.

**Spec:** `docs/superpowers/specs/2026-08-21-single-pass-elastic-sparsegpt-design.md`

## Global Constraints

- All elastic stages share statistics from one dense-reference calibration sweep.
- Each linear layer computes Hessian, Cholesky factor, and global permutation exactly once.
- Earlier reordered intervals remain byte-for-byte unchanged during later stages; compensation flows only rightward.
- Plans stay `float32`; GPU memory contains only the current layer plan.
- Memory mode is default; disk mode requires `--elastic_cache_dir` and uses a unique run directory.
- Disk mode never overwrites a user file and retains its run directory after an unsuccessful close.
- WikiText evaluation time is excluded from pruning timing.
- Ordinary `sparsegpt`, `wanda`, and magnitude paths remain unchanged.
- Preserve the already-tested Bottom-80% column-cost changes currently present in `README.md`, `lib/elastic_sparsegpt.py`, and `tests/test_elastic_sparsegpt.py`.

---

### Task 1: Reusable Elastic Plan and Cached Stage Application

**Files:**
- Modify: `lib/elastic_sparsegpt.py`
- Modify: `tests/test_elastic_sparsegpt.py`

**Interfaces:**
- Produces: `ElasticSparseGPTPlan(permutation, hessian_factor, dead_columns)` with CPU storage and shape validation.
- Produces: `ElasticSparseGPT.prepare_plan(percdamp=0.01) -> ElasticSparseGPTPlan`.
- Produces: `apply_elastic_sparsegpt_stage(layer, plan, sparsity, previous_sparsity=0.0, blocksize=128) -> None`.
- Keeps: `ElasticSparseGPT.fasterprune(...)` as a compatibility wrapper for single-target tests and callers.

- [ ] **Step 1: Write failing tests for one prepared plan serving two stages**

Add tests that derive expectations from a literal toy layer:

```python
def test_prepared_plan_can_apply_two_nested_stages_without_a_new_hessian(self):
    from torch import nn
    from lib.elastic_sparsegpt import ElasticSparseGPT, apply_elastic_sparsegpt_stage

    layer = nn.Linear(8, 2, bias=False)
    layer.weight.data.copy_(torch.arange(1, 17, dtype=torch.float32).reshape(2, 8))
    collector = ElasticSparseGPT(layer)
    collector.H = torch.eye(8)
    plan = collector.prepare_plan(percdamp=0.01)

    apply_elastic_sparsegpt_stage(layer, plan, 0.10, 0.0, blocksize=2)
    mask_10 = layer.weight == 0
    apply_elastic_sparsegpt_stage(layer, plan, 0.20, 0.10, blocksize=2)
    mask_20 = layer.weight == 0

    assert int(mask_10.sum()) == 2
    assert int(mask_20.sum()) == 3
    assert torch.all(mask_10 <= mask_20)
```

Add a separate freeze test in reordered coordinates:

```python
def test_later_stage_does_not_change_the_frozen_reordered_prefix(self):
    # Prepare the same 8-column toy plan, apply 10%, and clone reordered weights.
    # Apply 20% with the same plan. The first prefix_column_count(8, 0.10)
    # columns must equal the clone exactly.
    assert torch.equal(after_20[:, :frozen_end], after_10[:, :frozen_end])
```

Name the mutation caught: recomputing a plan, revisiting the old interval, or propagating compensation leftward must fail at least one test.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
& 'E:\Users\13657\anaconda3\envs\rose\python.exe' -m unittest tests.test_elastic_sparsegpt.ElasticCompensationTests -v
```

Expected: errors because `prepare_plan` and `apply_elastic_sparsegpt_stage` do not exist.

- [ ] **Step 3: Implement the plan object and preparation**

Add a frozen dataclass and validation:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ElasticSparseGPTPlan:
    permutation: torch.Tensor
    hessian_factor: torch.Tensor
    dead_columns: torch.Tensor

    def validate_for(self, columns: int) -> None:
        if self.permutation.shape != (columns,):
            raise ValueError("plan permutation does not match layer columns")
        if self.hessian_factor.shape != (columns, columns):
            raise ValueError("plan Hessian factor does not match layer columns")
        if self.dead_columns.shape != (columns,):
            raise ValueError("plan dead-column mask does not match layer columns")

    def cpu(self):
        return ElasticSparseGPTPlan(
            self.permutation.detach().cpu(),
            self.hessian_factor.detach().float().cpu(),
            self.dead_columns.detach().cpu(),
        )
```

Move the existing damping, Cholesky inverse, Bottom-80% permutation, permuted-factor calculation from `fasterprune` into `prepare_plan`. Set `self.H = None` after consuming it and return a CPU plan.

- [ ] **Step 4: Implement cached stage application without allocating a Hessian**

Create the module-level `apply_elastic_sparsegpt_stage` function. It must:

1. Inspect the target weight shape without constructing `SparseGPT`, avoiding a new `columns x columns` zero Hessian.
2. Validate the plan.
3. Move only the plan tensors to the layer device.
4. Apply dead-column zeros, fixed permutation, `iter_stage_segments`, exact stage budget, block mask, and the existing rightward compensation formula.
5. Undo the permutation and write the layer weight.
6. Delete device temporaries before returning.

Refactor `fasterprune` to call `prepare_plan`, then `apply_elastic_sparsegpt_stage`, and return `plan.permutation` for compatibility.

- [ ] **Step 5: Run focused and full tests**

Run:

```powershell
& 'E:\Users\13657\anaconda3\envs\rose\python.exe' -m unittest tests.test_elastic_sparsegpt -v
& 'E:\Users\13657\anaconda3\envs\rose\python.exe' -m unittest discover -s tests -v
```

Expected: all tests pass, including the pre-existing Bottom-80% ranking regression.

- [ ] **Step 6: Commit the prerequisite and plan-application behavior**

```bash
git add README.md lib/elastic_sparsegpt.py tests/test_elastic_sparsegpt.py
git commit -m "feat: reuse elastic sparsegpt layer plans"
```

---

### Task 2: Bounded Memory and Disk Plan Stores

**Files:**
- Create: `lib/elastic_plan_store.py`
- Create: `tests/test_elastic_plan_store.py`

**Interfaces:**
- Consumes: `ElasticSparseGPTPlan` from Task 1.
- Produces: `MemoryElasticPlanStore.put_layer(index, plans)`, `get_layer(index)`, and `close(success)`.
- Produces: `DiskElasticPlanStore(base_dir).put_layer(index, plans)`, `get_layer(index)`, `run_dir`, and `close(success)`.
- Produces: `make_elastic_plan_store(mode: str, cache_dir: str | None)` returning the selected backend and rejecting disk mode without a directory.
- Layer plan mappings have type `dict[str, ElasticSparseGPTPlan]`.

- [ ] **Step 1: Write failing real-tensor round-trip tests**

Use a literal two-column plan and `tempfile.TemporaryDirectory`:

```python
def make_plan():
    return ElasticSparseGPTPlan(
        permutation=torch.tensor([1, 0]),
        hessian_factor=torch.tensor([[1.0, 0.25], [0.0, 1.0]]),
        dead_columns=torch.tensor([False, True]),
    )

def test_memory_store_round_trips_a_layer_plan():
    store = MemoryElasticPlanStore()
    store.put_layer(3, {"self_attn.q_proj": make_plan()})
    loaded = store.get_layer(3)["self_attn.q_proj"]
    assert torch.equal(loaded.permutation, torch.tensor([1, 0]))

def test_disk_store_uses_unique_directory_and_cleans_successful_run():
    with tempfile.TemporaryDirectory() as base:
        store = DiskElasticPlanStore(base)
        run_dir = store.run_dir
        store.put_layer(3, {"self_attn.q_proj": make_plan()})
        assert os.path.exists(os.path.join(run_dir, "layer_003.pt"))
        assert torch.equal(
            store.get_layer(3)["self_attn.q_proj"].dead_columns,
            torch.tensor([False, True]),
        )
        store.close(success=True)
        assert not os.path.exists(run_dir)

def test_disk_store_retains_failed_run():
    with tempfile.TemporaryDirectory() as base:
        store = DiskElasticPlanStore(base)
        run_dir = store.run_dir
        store.close(success=False)
        assert os.path.isdir(run_dir)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
& 'E:\Users\13657\anaconda3\envs\rose\python.exe' -m unittest tests.test_elastic_plan_store -v
```

Expected: import failure because `lib.elastic_plan_store` does not exist.

- [ ] **Step 3: Implement serialization with unique paths**

Use `tempfile.mkdtemp(prefix="elastic-plan-", dir=base_dir)` for disk runs. Serialize plain tensor dictionaries rather than pickled live pruner objects:

```python
def _serialize(plans):
    return {
        name: {
            "permutation": plan.permutation.cpu(),
            "hessian_factor": plan.hessian_factor.float().cpu(),
            "dead_columns": plan.dead_columns.cpu(),
        }
        for name, plan in plans.items()
    }
```

Use `torch.load(path, map_location="cpu", weights_only=True)` and reconstruct `ElasticSparseGPTPlan` instances. `put_layer` must reject duplicate layer indices. `get_layer` must raise a clear `KeyError` for missing layers.

`MemoryElasticPlanStore.close` clears its mapping. `DiskElasticPlanStore.close(success=True)` removes only its exact unique `run_dir`; `success=False` prints and retains that directory.

Implement `make_elastic_plan_store` with exactly two accepted modes, `memory` and `disk`; raise `ValueError("elastic_cache_dir is required for disk mode")` before constructing a disk store when the directory is missing.

- [ ] **Step 4: Run store and full tests**

```powershell
& 'E:\Users\13657\anaconda3\envs\rose\python.exe' -m unittest tests.test_elastic_plan_store -v
& 'E:\Users\13657\anaconda3\envs\rose\python.exe' -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add lib/elastic_plan_store.py tests/test_elastic_plan_store.py
git commit -m "feat: cache elastic sparsegpt plans"
```

---

### Task 3: One Dense Preparation Sweep and Stage-Only Application

**Files:**
- Modify: `lib/prune.py`
- Modify: `tests/test_elastic_sparsegpt.py`

**Interfaces:**
- Consumes: a plan store from Task 2.
- Produces: `prepare_elastic_sparsegpt(args, model, tokenizer, dev, plan_store) -> None`.
- Produces: `apply_elastic_sparsegpt_stage(model, dev, plan_store, sparsity, previous_sparsity=0.0) -> None`.
- Produces: `run_prepared_elastic_stages(stages, prepare, apply, on_stage, clock=time.perf_counter)`; `on_stage` receives preparation, incremental, and cumulative seconds.
- Keeps: `prune_elastic_sparsegpt(...)` as a single-target compatibility wrapper that prepares once and applies once.

- [ ] **Step 1: Write failing orchestration and no-mutation tests**

Add a small orchestration helper test around the desired public boundary. Use a tiny fake model/layer only for model plumbing; use real `ElasticSparseGPTPlan` tensors and a real memory store. The preparation test must clone all toy weights before preparation and assert exact equality afterward. The application test must install a prepared plan, set `collector.H = None`, apply a stage, and assert that pruning succeeds, proving stage application cannot depend on a newly collected Hessian.

Add a call-count regression at the orchestration seam:

```python
def test_multi_stage_orchestration_prepares_once():
    calls = {"prepare": 0, "apply": []}

    def prepare():
        calls["prepare"] += 1
        return object()

    def apply(store, previous, current):
        calls["apply"].append((previous, current))

    ticks = iter([0.0, 5.0, 5.0, 7.0, 7.0, 10.0, 10.0, 14.0])
    timings = []
    run_prepared_elastic_stages(
        [0.1, 0.2, 0.3],
        prepare,
        apply,
        lambda store, previous, current, prep, stage, cumulative: timings.append(
            (prep, stage, cumulative)
        ),
        clock=lambda: next(ticks),
    )
    assert calls == {
        "prepare": 1,
        "apply": [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3)],
    }
    assert timings == [(5.0, 2.0, 7.0), (5.0, 3.0, 10.0), (5.0, 4.0, 14.0)]
```

This callback seam is production orchestration: `main.py` will supply real preparation, application, and per-stage result handling in Task 4.

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
& 'E:\Users\13657\anaconda3\envs\rose\python.exe' -m unittest tests.test_elastic_sparsegpt -v
```

Expected: imports fail because the preparation/application/orchestration functions do not exist.

- [ ] **Step 3: Split the current routine into preparation and application**

`prepare_elastic_sparsegpt` reuses the current catcher and hook logic, with these required changes:

1. Capture the first-layer calibration inputs once.
2. For each dense transformer layer, attach real `ElasticSparseGPT` collectors and execute the 128 sample forwards.
3. Call `prepare_plan` for each collector and write the CPU plans with `plan_store.put_layer(layer_index, plans)`.
4. Do not modify any weight.
5. Advance `inps, outs = outs, inps` using the dense outputs.
6. Restore `model.config.use_cache` in `finally` and free collector/Hessian tensors per layer.

`apply_elastic_sparsegpt_stage` loops model layers, loads exactly one layer mapping from the store, calls Task 1's module-level stage function for each named linear module, then deletes the mapping and empties CUDA cache. It performs no dataset loading, hooks, forward calibration, Hessian creation, or Cholesky factorization.

- [ ] **Step 4: Add the explicit one-prepare stage runner**

Implement:

```python
def run_prepared_elastic_stages(
    stages, prepare, apply, on_stage, clock=time.perf_counter
):
    preparation_start = clock()
    prepared = prepare()
    preparation_seconds = clock() - preparation_start
    cumulative_seconds = preparation_seconds
    previous = 0.0
    for current in validate_elastic_sparsegpt_stages(stages):
        stage_start = clock()
        apply(prepared, previous, current)
        stage_seconds = clock() - stage_start
        cumulative_seconds += stage_seconds
        on_stage(
            prepared,
            previous,
            current,
            preparation_seconds,
            stage_seconds,
            cumulative_seconds,
        )
        previous = current
    return prepared
```

Preparation is outside the loop by construction. Do not add fallback recalibration.

The compatibility `prune_elastic_sparsegpt` creates a temporary memory store, prepares once, applies one target, and closes the store in `finally`.

- [ ] **Step 5: Run focused and full tests**

```powershell
& 'E:\Users\13657\anaconda3\envs\rose\python.exe' -m unittest tests.test_elastic_sparsegpt -v
& 'E:\Users\13657\anaconda3\envs\rose\python.exe' -m unittest discover -s tests -v
```

Expected: all tests pass and the orchestration record contains one preparation call.

- [ ] **Step 6: Commit**

```bash
git add lib/prune.py tests/test_elastic_sparsegpt.py
git commit -m "refactor: prepare elastic sparsegpt once"
```

---

### Task 4: CLI Cache Selection, Cumulative Timing, and Result Logs

**Files:**
- Modify: `main.py`
- Modify: `lib/reporting.py`
- Modify: `tests/test_elastic_sparsegpt.py`
- Modify: `tests/test_reporting.py`

**Interfaces:**
- Produces CLI options `--elastic_cache_mode {memory,disk}` and `--elastic_cache_dir PATH`.
- Produces `format_elastic_result_log(method, sparsity, perplexity, preparation_seconds, stage_seconds, cumulative_seconds)`.
- Consumes preparation/application/store APIs from Tasks 2 and 3.

- [ ] **Step 1: Write failing CLI and reporting tests**

Add parser assertions:

```python
args = build_parser().parse_args([
    "--prune_method", "elastic_sparsegpt",
    "--elastic_cache_mode", "disk",
    "--elastic_cache_dir", "/tmp/elastic-cache",
])
assert args.elastic_cache_mode == "disk"
assert args.elastic_cache_dir == "/tmp/elastic-cache"
```

Add validation that disk mode without a directory raises `ValueError("elastic_cache_dir is required for disk mode")` before preparation.

Add a literal reporting expectation:

```python
assert format_elastic_result_log(
    "elastic_sparsegpt", 0.2, 5.8, 200.0, 18.0, 236.0
) == (
    "method\tactual_sparsity\tppl_test\tpreparation_time_seconds\t"
    "stage_pruning_time_seconds\tcumulative_pruning_time_seconds\n"
    "elastic_sparsegpt\t0.2000\t5.8000\t200.00\t18.00\t236.00\n"
)
```

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
& 'E:\Users\13657\anaconda3\envs\rose\python.exe' -m unittest tests.test_elastic_sparsegpt.CliExposureTests tests.test_reporting -v
```

Expected: unrecognized CLI arguments and missing formatter.

- [ ] **Step 3: Implement CLI/store validation and formatter**

Add parser arguments with default `memory`. Add `validate_elastic_cache_args(args)` in `lib/prune.py`; call it before model calibration in the elastic multi-stage branch. Keep `format_result_log` unchanged for baseline compatibility and add the dedicated elastic formatter.

- [ ] **Step 4: Replace the repeated-calibration loop in `main.py`**

Implement this lifecycle using Task 3's runner so preparation structurally remains outside the stage loop:

```python
plan_store = make_elastic_plan_store(args.elastic_cache_mode, args.elastic_cache_dir)
success = False
try:
    def prepare():
        prepare_elastic_sparsegpt(args, model, tokenizer, device, plan_store)
        return plan_store

    def apply(store, previous, current):
        apply_elastic_sparsegpt_stage(model, device, store, current, previous)

    def on_stage(store, previous, current, prep_seconds, stage_seconds, cumulative_seconds):
        actual_sparsity = check_sparsity(model)
        ppl_test = eval_ppl(args, model, tokenizer, device)
        stage_tag = str(current).replace(".", "")
        stage_save = os.path.join(args.save, f"s{stage_tag}")
        os.makedirs(stage_save, exist_ok=True)
        with open(os.path.join(stage_save, "log_elastic_sparsegpt.txt"), "w") as output:
            output.write(format_elastic_result_log(
                args.prune_method,
                actual_sparsity,
                ppl_test,
                prep_seconds,
                stage_seconds,
                cumulative_seconds,
            ))
        if args.save_model:
            stage_model_path = os.path.join(args.save_model, f"s{stage_tag}")
            model.save_pretrained(stage_model_path)
            tokenizer.save_pretrained(stage_model_path)

    run_prepared_elastic_stages(elastic_stages, prepare, apply, on_stage)
    success = True
finally:
    plan_store.close(success=success)
```

Do not include `eval_ppl` or model serialization in `cumulative_seconds`. Remove the multi-stage `permutations` mapping because every plan already owns its fixed permutation.

- [ ] **Step 5: Run focused and full tests**

```powershell
& 'E:\Users\13657\anaconda3\envs\rose\python.exe' -m unittest tests.test_elastic_sparsegpt.CliExposureTests tests.test_reporting -v
& 'E:\Users\13657\anaconda3\envs\rose\python.exe' -m unittest discover -s tests -v
```

Expected: all tests pass; baseline result formatting remains byte-for-byte unchanged.

- [ ] **Step 6: Commit**

```bash
git add main.py lib/prune.py lib/reporting.py tests/test_elastic_sparsegpt.py tests/test_reporting.py
git commit -m "feat: run elastic sparsegpt as one prepared path"
```

---

### Task 5: Documentation and End-to-End Verification

**Files:**
- Modify: `README.md`
- Verify: `main.py`
- Verify: `lib/elastic_sparsegpt.py`
- Verify: `lib/elastic_plan_store.py`
- Verify: `lib/prune.py`
- Verify: `lib/reporting.py`

**Interfaces:**
- Documents memory and disk invocations and timing semantics.
- Produces no new runtime interface beyond Tasks 1-4.

- [ ] **Step 1: Update experiment documentation**

Document that stages share dense-reference Hessians and one permutation. Include both commands:

```bash
# Fastest when host RAM is sufficient
python main.py \
  --model <MODEL_PATH> \
  --prune_method elastic_sparsegpt \
  --sparsity_type unstructured \
  --elastic_sparsities 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 \
  --elastic_cache_mode memory \
  --nsamples 128 --seed 0 \
  --save results/elastic-single-pass

# Bounded host memory, with sufficient disk space
python main.py \
  --model <MODEL_PATH> \
  --prune_method elastic_sparsegpt \
  --sparsity_type unstructured \
  --elastic_sparsities 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 \
  --elastic_cache_mode disk \
  --elastic_cache_dir <CACHE_BASE_DIR> \
  --nsamples 128 --seed 0 \
  --save results/elastic-single-pass
```

State that eight PPL evaluations remain separate and that `--save_model` writes eight full checkpoints.

- [ ] **Step 2: Run complete automated verification**

```powershell
& 'E:\Users\13657\anaconda3\envs\rose\python.exe' -m unittest discover -s tests -v
& 'E:\Users\13657\anaconda3\envs\rose\python.exe' -m compileall -q main.py lib tests
git diff --check
```

Expected: zero test failures, compile exit code 0, and no whitespace errors.

- [ ] **Step 3: Run a CPU toy smoke test**

Create `tests/smoke_elastic_single_pass.py` with this executable body, run it, then remove the smoke file before committing:

```python
import torch
from torch import nn
from lib.elastic_sparsegpt import ElasticSparseGPT, apply_elastic_sparsegpt_stage

layer = nn.Linear(8, 2, bias=False)
layer.weight.data.copy_(torch.arange(1, 17, dtype=torch.float32).reshape(2, 8))
collector = ElasticSparseGPT(layer)
collector.H = torch.eye(8)
plan = collector.prepare_plan()
masks = []
previous = 0.0
for current in (0.1, 0.2, 0.3):
    apply_elastic_sparsegpt_stage(layer, plan, current, previous, blocksize=2)
    masks.append((layer.weight == 0).clone())
    previous = current
print([int(mask.sum()) for mask in masks])
assert torch.all(masks[0] <= masks[1])
assert torch.all(masks[1] <= masks[2])
```

Run:

```powershell
& 'E:\Users\13657\anaconda3\envs\rose\python.exe' tests\smoke_elastic_single_pass.py
```

Expected: zero counts increase at every stage and both subset assertions pass.

- [ ] **Step 4: Review the final diff against every spec constraint**

Confirm from code and tests:

- preparation is outside the stage loop;
- application allocates no Hessian and performs no Cholesky;
- permutation is stored in the plan and never recomputed;
- disk paths are unique and cleanup targets only the created run directory;
- previous prefixes are frozen;
- baseline methods and result logs are unchanged;
- PPL evaluation is outside pruning timers.

- [ ] **Step 5: Commit documentation and any verification-only corrections**

```bash
git add README.md
git commit -m "docs: explain single-pass elastic sparsegpt"
```

- [ ] **Step 6: Prepare a pure server patch without including local experiment data**

Use the feature commit range, not the server's dirty `lib/data.py` or result logs:

```powershell
cmd.exe /d /c "git -C E:\Users\13657\Desktop\wanda diff --binary c34b1d8..HEAD > E:\Users\13657\Desktop\wanda\elastic-single-pass.patch"
```

Verify its first line is `diff --git` before transfer. On the server, use `git apply --check` before `git apply`; never reset or overwrite the server's local C4 dataset configuration.
