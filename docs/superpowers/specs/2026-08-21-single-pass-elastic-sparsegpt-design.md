# Single-Pass Elastic SparseGPT Design

## Goal

Generate strictly nested 10% through 80% sparse checkpoints from one calibration pass, one Hessian factorization per linear layer, and one fixed global column permutation per linear layer. Each stage processes only its newly opened reordered-column interval; earlier intervals remain frozen and compensation flows only to later columns.

## Current Problem

The current `--elastic_sparsities` orchestration invokes the complete calibration and pruning routine once per requested sparsity. Although masks are nested and permutations are reused, Hessians and Cholesky factors are recomputed eight times. The reported stage time therefore measures eight separate pruning preparations rather than one elastic pruning path.

## Chosen Approximation

All elastic stages share statistics collected from one dense-reference calibration sweep. This deliberately differs from independently running SparseGPT at every sparsity, where each target may collect activations from its own already-pruned upstream layers. The shared-statistics approximation is the mechanism that makes one-pass elastic generation possible and must be stated in experiment documentation.

## Architecture

### 1. One dense-reference preparation sweep

Process transformer layers in their original dense state. For every prunable linear module:

1. Accumulate its Hessian from the same 128 C4 calibration samples.
2. Apply the configured damping.
3. Compute the inverse-Hessian Cholesky factor once.
4. Rank all input columns once using the predicted loss of pruning the lowest-scored 80% of weights in each column.
5. Permute the factor into that fixed column order.
6. Store a reusable plan containing the permutation, permuted factor, and dead-column information.

Dense outputs from the current transformer layer become the calibration inputs to the next layer. No model weights are pruned during this preparation sweep.

### 2. Reusable plan storage

Support two storage backends behind the same interface:

- `memory`: store plans as CPU tensors. This is fastest and is the default.
- `disk`: store one plan file per transformer layer beneath a unique temporary run directory created under `--elastic_cache_dir`. Load only the current layer's plan onto CPU/GPU during a stage, then release it.

The disk backend must never overwrite an existing user file. Its run directory is uniquely created. On successful completion it is removed; after an interrupted run it is retained and its location is printed for manual recovery or removal.

Plans remain `float32` to preserve SparseGPT numerical behavior. GPU memory contains only the current model, activation buffers, and the current layer plan.

### 3. One progressive stage path

After preparation, advance a single model through the requested strictly increasing stages:

- 10% processes reordered columns `[0%, 12.5%)`.
- 20% freezes that interval and processes `[12.5%, 25%)`.
- Each later 10-point increase processes the next 12.5% interval.
- 80% finishes at 100% of reordered columns.

Within every newly opened interval, split work into compensation segments of at most 128 columns and prune an exact aggregate budget equivalent to 80% of that interval. Reuse the cached permutation and Cholesky factor. Existing zero weights and all weights in earlier intervals remain unchanged. Error compensation updates only columns to the right, including columns that have not yet been opened.

Each stage operates on the model produced by the previous stage, so masks satisfy:

`M10 subset M20 subset M30 subset ... subset M80`.

### 4. Results and optional model checkpoints

After completing each global stage:

1. Verify actual sparsity.
2. Evaluate WikiText perplexity using the current complete model.
3. Write the stage result under `s01` through `s08`.
4. If `--save_model` is supplied, save the current complete model before advancing to the next stage.

Only one model copy is required in memory. Saving eight models consumes eight checkpoint copies on disk and remains explicitly optional.

### 5. Timing semantics

Measure preparation once. For every stage report:

- `preparation_time_seconds`: the shared calibration, factorization, and ordering cost;
- `stage_pruning_time_seconds`: only the newly opened interval's pruning work;
- `cumulative_pruning_time_seconds`: preparation plus all stage pruning work needed to reach this checkpoint.

WikiText evaluation time is excluded. The 80% cumulative pruning time is the primary measurement for the claim that one elastic path produces all eight sparsities.

## Interfaces

Add these CLI options:

- `--elastic_cache_mode {memory,disk}`, default `memory`;
- `--elastic_cache_dir PATH`, required only for `disk` mode.

Refactor the elastic implementation into two responsibilities:

- prepare a reusable per-linear-layer plan from one Hessian;
- apply one new sparsity interval using an existing plan.

The ordinary `sparsegpt`, `wanda`, and single-target pruning paths remain unchanged.

## Failure Handling

- Reject disk mode without `--elastic_cache_dir` before model calibration.
- Reject invalid or non-increasing stage lists before preparation.
- Reject a plan whose tensor shapes do not match the target linear layer.
- Print the retained temporary cache path if a disk-backed run exits abnormally.
- Do not silently fall back to recomputing Hessians, because that would invalidate the single-pass timing claim.

## Tests

1. A plan-preparation test proves the permutation uses the Bottom-80% predicted loss and records the expected factor shape.
2. A cached-stage test applies 10%, then 20%, without supplying a new Hessian; it proves exact sparsity and mask nesting.
3. A freeze test proves the second stage does not alter weights in the first stage's reordered interval.
4. A compensation test proves the second stage can update future, unopened columns while leaving earlier columns unchanged.
5. A plan-store round-trip test covers memory and disk backends using real tensors and a temporary directory.
6. An orchestration test proves preparation occurs once for a multi-stage run.
7. Existing elastic, CLI, reporting, and compilation tests continue to pass.

## Expected Trade-offs

- Compared with the current implementation, calibration and factorization fall from eight repetitions to one.
- The union of all newly processed column intervals equals one complete traversal of the reordered layer.
- Eight PPL evaluations are still required.
- Memory mode requires substantial host RAM; disk mode trades repeated plan reads for bounded host memory.
- Accuracy may differ from independently calibrated SparseGPT because all stages share dense-reference statistics.
