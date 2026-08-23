# Wanda Prefix OBS Compensation Design

## Goal

Add a new elastic method that retains the existing one-time whole-column Wanda ordering, deletes an increasing prefix at 20%, 30%, ..., 80%, and uses an OBS/SlimGPT-style second-order compensation update after each new prefix interval.

## Scope

This is a column-prefix masking experiment, not architecture shrinking. It does not remove complete LLaMA heads or FFN neurons, change tensor shapes, or claim inference acceleration. Existing methods `elastic_sparsegpt`, `elastic_wanda_sparsegpt`, and `elastic_wanda_dynamic_sparsegpt` remain unchanged. The new method is named `elastic_wanda_prefix_obs`.

## Fixed Wanda coordinate system

For each Linear matrix `W` of shape `(rows, columns)`, use the project’s existing full-column Wanda score:

```
score[j] = sum_i(abs(W[i, j]) * activation_norm[j])
P = argsort(score, ascending=True, stable=True)
```

The ordered computation coordinates are `W_ordered = W[:, P]` and `Hinv_ordered = Hinv[P][:, P]`. The permutation is calculated once from dense calibration statistics; it is never recomputed at later checkpoints.

Before assigning the result to the model, restore original coordinates with the inverse permutation. The ordering is only an internal pruning coordinate system, so inputs still map to their original Linear columns.

## Nested stage rule

At a stage `s`, permanently delete the ordered prefix ending at `round(s * columns)`.

```
G = [round(previous_sparsity * columns) : round(sparsity * columns)]
R = [round(sparsity * columns) : columns]
```

Thus 30% only adds the next 10% after the already-zero 20% prefix. All checkpoints are exactly nested. Do not use the old seven coarse slices or a 128-column compute-block subdivision in this new method.

## Compensation rule

Copy `W_ordered[:, G]` before zeroing it. Apply the group OBS update only to the surviving suffix:

```
coupling = torch.linalg.solve(Hinv_ordered[G, G], Hinv_ordered[G, R])
W_ordered[:, R] -= W_ordered[:, G] @ coupling
W_ordered[:, G] = 0
```

Use `torch.linalg.solve`, not an explicit inverse. Keep the same dense-pass inverse Hessian throughout the full stage trajectory; do not recalibrate, refactor a full Hessian, or dynamically reorder at each checkpoint.

## Plan and output behavior

The plan must retain `hessian_inverse` plus Wanda activation norms and the one-time permutation. It is cached by the existing memory/disk plan store. Each stage changes model weights, so if `--save_model` is set, save each checkpoint immediately. A final 80% snapshot cannot reconstruct earlier compensated weights.

## Correctness requirements

1. CLI accepts `--prune_method elastic_wanda_prefix_obs`.
2. The method uses `ordering="wanda"`.
3. The first stage deletes only its ordered prefix; later stages delete only their newly opened prefix interval.
4. Previously deleted columns stay zero.
5. Compensation writes only to the currently surviving suffix.
6. After inverse permutation, deleted columns are zero in original model coordinates.
7. Existing result log fields remain compatible.
8. No local-C4 loader or local-model path behavior changes.

## Experiment acceptance

Use server-local `/data/ModelZoo/LLM/llama2/Llama-2-7b-hf`, local C4, `nsamples=128`, seed 0, GPU 0. Compare PPL at 20%, 30%, 50%, and 80% with original SparseGPT under the same calibration and evaluation setup. Record preparation and per-stage time, but do not present this column-prefix method as a true structural speedup.

