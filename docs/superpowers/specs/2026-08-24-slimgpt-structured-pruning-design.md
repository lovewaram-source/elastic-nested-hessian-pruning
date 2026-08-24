# SlimGPT Structured Pruning Design

## Goal

Add a paper-derived SlimGPT baseline for LLaMA-2-7B that physically removes attention heads and FFN neurons, evaluates WikiText-2 perplexity, and remains independent of the existing Wanda, SparseGPT, and elastic methods.

## Scope

The initial baseline targets Hugging Face LLaMA models with standard multi-head attention (`num_key_value_heads == num_heads`). It prunes both attention heads and FFN intermediate channels. Grouped-query attention is rejected explicitly because pruning a query head without a matching key/value group requires a different dependency rule.

## Algorithm

For every decoder layer, collect calibration inputs for `self_attn.o_proj` and `mlp.down_proj` and construct a damped inverse Hessian. Use the OBS group loss

`trace(W[:, G] @ inverse(Hinv[G, G]) @ W[:, G].T)`

to score a removable column group `G`. Removing a group applies

`W <- W - W[:, G] @ inverse(Hinv[G, G]) @ Hinv[G, :]`

before deleting the selected columns. The surviving inverse Hessian is updated with the Schur complement.

Attention candidates are complete contiguous heads. At each greedy iteration, score all remaining heads using their full `head_dim` block, remove the lowest-loss head, and update the output projection and inverse Hessian. Then physically keep the corresponding rows in `q_proj`, `k_proj`, and `v_proj`, and the corresponding columns in `o_proj`.

FFN candidates are individual intermediate channels. Recompute single-channel OBS scores after each batch. Start with a batch of at most 1024 channels and halve the batch size as the remaining target shrinks, down to 8; the final batch is exact. Then physically keep the corresponding rows in `gate_proj` and `up_proj` and columns in `down_proj`.

## Layer Ratios and Global Target

The default profile is logarithmically increasing with depth. Let `shape_i = log(i + 1) / log(n)` and use it to define the relative pruning pressure across layers. A scalar multiplier is found by binary search so the predicted number of physically removed parameters, including the discrete head and FFN channel counts, matches the requested reduction of the complete model as closely as possible. Clamp each layer ratio so at least one attention head and one FFN channel survive. A `uniform` profile is exposed as an ablation.

Pruning attention heads changes the output widths of `q_proj`, `k_proj`, and `v_proj` and the input width of `o_proj`. The residual-stream hidden size, projection input width, and `o_proj` output width remain unchanged.

## CLI and Results

Expose `--prune_method slimgpt`, `--slimgpt_ratio_profile {log_increase,uniform}`, and `--slimgpt_ffn_min_group_size`. The regular `--sparsity_ratio` is the target structured ratio. The result log records model, method, target ratio, actual parameter reduction, WikiText-2 PPL, pruning time, and remaining parameter count.

Server experiments use the local model `/data/ModelZoo/LLM/llama2/Llama-2-7b-hf`, local C4 loader, 128 calibration samples, GPU 0, and temporary output under `/tmp`. Only normalized rows are retained in `results/ALL_RESULTS.tsv` and `results/ALL_RESULTS.txt`.

## Verification

Unit tests cover OBS group compensation, layer-ratio normalization, attention dependency slicing, FFN dependency slicing, unsupported GQA rejection, CLI exposure, physical parameter reduction, and a complete dummy LLaMA block forward pass after pruning. The existing test suite and Python compilation must remain clean before upload.
