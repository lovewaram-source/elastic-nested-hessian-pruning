"""Independent single-GPU sorting ablation with fixed dense-reference Hessians.

No token fusion, RIA, dynamic reordering or claim of historical PPL reproduction.
All runs include 20/30/40/50% checkpoints; unstructured has the auxiliary 10% stage.
"""
import argparse
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_metadata import validate_run_arguments, runtime_record

RATES = (.2, .3, .4, .5)
GROUP_ORDERINGS = ("wanda_group_stage_desc", "wanda_group_pruned_part_stage_desc",
                   "wanda_group_stage_quartile_interleave", "wanda_group_stage_pnp_late_interleave")
COLUMN_ORDERINGS = ("wanda", "sparsegpt", "wanda_pruned_part", "sparsegpt_pruned_part")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--mode", choices=("unstructured", "2of4"), required=True)
    p.add_argument("--ordering", choices=("original",) + GROUP_ORDERINGS + COLUMN_ORDERINGS, required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--nsamples", type=int, default=64)
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--percdamp", type=float, default=.01)
    p.add_argument("--unstructured-alpha", type=float, default=1.)
    p.add_argument("--structured-alpha", type=float, default=.25)
    p.add_argument("--eval-samples", type=int, default=None)
    p.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    args = p.parse_args(argv)
    args.rates, args.device_map = RATES, "cpu"
    try:
        validate_run_arguments(args)
    except ValueError as error:
        p.error(str(error))
    if args.nsamples < 1 or args.seqlen < 2:
        p.error("nsamples must be positive and seqlen >= 2")
    if not args.gpu.strip() or "," in args.gpu or args.gpu.strip() == "-1":
        p.error("select exactly one GPU")
    args.gpu = args.gpu.strip()
    if args.mode == "2of4" and args.ordering in COLUMN_ORDERINGS:
        p.error("2of4 sorting uses macroblock-local group modes, not arbitrary column permutations")
    if args.mode == "unstructured" and args.ordering in GROUP_ORDERINGS:
        p.error("group orderings require 2of4 mode")
    return args


def apply_sorted_stage(layer, plans, mode, previous, current, alpha, cumulative):
    import torch
    from lib.elastic_sparsegpt import apply_elastic_2of4_stage
    from run_rate_aligned_nested import _find_layers, _apply_unstructured_stage, _assert_stage_mask

    for name, module in _find_layers(layer).items():
        plan = plans[name]
        if mode == "unstructured":
            mask = _apply_unstructured_stage(module, plan, previous, current, .5, alpha)
        else:
            mask = apply_elastic_2of4_stage(module, plan, current, previous_sparsity=previous,
                                           compensation_alpha=alpha, return_mask=True)
        permutation = plan.permutation.to(mask.device)
        # Kernel returns original-coordinate masks. Validate budgets/grouping in sorted coordinates.
        _assert_stage_mask(mode, mask[:, permutation], previous, current, .5)
        if (cumulative[name] & mask).any():
            raise AssertionError("new stage reuses a previously opened coordinate")
        cumulative[name].logical_or_(mask)
        if (cumulative[name] & module.weight.ne(0)).any():
            raise AssertionError("nested zero was resurrected")
    return sum(int(mask.sum()) for mask in cumulative.values())


def reconstruct_sorted(args, model, initial, attention, positions, cache, device):
    import torch
    from run_rate_aligned_nested import _collect_layer_plans, _find_layers
    from experiment_common import synchronize

    counts = {r: 0 for r in RATES}
    total = 0
    paths = initial
    alpha = args.structured_alpha if args.mode == "2of4" else args.unstructured_alpha
    for index, layer in enumerate(model.model.layers):
        layer.to(device)
        plans = None
        try:
            with torch.no_grad():
                dense_outputs, plans = _collect_layer_plans(
                    layer, paths, attention, positions, device, args.percdamp,
                    ordering=args.ordering, predicted_global_sparsity=.5,
                )
                # Persist actual permutations so regrouped 2:4 is never mistaken for original physical groups.
                torch.save({name: plan.permutation.detach().cpu() for name, plan in plans.items()},
                           args.output / f"permutations_layer_{index:03d}.pt")
                modules = _find_layers(layer)
                total += sum(m.weight.numel() for m in modules.values())
                cumulative = {name: torch.zeros_like(m.weight, dtype=torch.bool) for name, m in modules.items()}
                previous = 0.
                stages = (.1,) + RATES if args.mode == "unstructured" else RATES
                for stage in stages:
                    zeros = apply_sorted_stage(layer, plans, args.mode, previous, stage, alpha, cumulative)
                    if stage in RATES:
                        counts[stage] += zeros
                        if stage != .5:
                            cache.save("target", index, stage, layer)
                    previous = stage
                paths = dense_outputs  # Keep the calibration path dense, independent of sparse targets.
        finally:
            layer.to("cpu")
            del plans
        synchronize(device)
        print(f"SORTING_LAYER_COMPLETE layer={index} ordering={args.ordering}", flush=True)
    return counts, total


def run(args):
    import torch
    from transformers import AutoTokenizer
    from run_checkpoint_token_isolation import CheckpointCache
    from checkpoint_token_isolation_core import load_block_weights
    from run_rate_aligned_nested import _load_model, _load_eval_loader, _load_calibration_inputs, _weight_stats
    from experiment_common import _write_json, _append_tsv, _eval_ppl, synchronize

    for path in (args.output, args.cache):
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(f"must use a new or empty directory: {path}")
    if args.output.resolve() == args.cache.resolve():
        raise ValueError("output and cache must be different directories")
    args.output.mkdir(parents=True, exist_ok=True)
    cache = CheckpointCache(args.cache)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("the full-model sorting runner requires a CUDA GPU")
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    evaluation = _load_eval_loader(args, tokenizer)
    model = _load_model(args, tokenizer)
    model.model.embed_tokens.to(device)
    try:
        initial, attention, positions = _load_calibration_inputs(args, model, tokenizer, device)
    finally:
        model.model.embed_tokens.to("cpu")
    protocol = dict(vars(args), output=str(args.output), cache=str(args.cache),
                    hessian_policy="fixed_dense_reference; no token fusion",
                    ordering_timing="static once per layer before pruning",
                    checkpoint_sharing="one numerical trajectory; frozen processed prefixes",
                    group_constraint="2of4 checked in permuted coordinates; preserve saved permutations",
                    runtime=runtime_record(args, model, evaluation))
    _write_json(args.output / "protocol.json", protocol)
    counts, total = reconstruct_sorted(args, model, initial, attention, positions, cache, device)
    del initial
    synchronize(device)
    preparation_seconds = time.perf_counter() - start
    _write_json(args.output / "preparation_timings.json", dict(preparation_seconds=preparation_seconds))
    model.to(device)
    rows = []
    for rate in reversed(RATES):
        consumed = []
        if rate != .5:
            for index, layer in enumerate(model.model.layers):
                path = cache.path("target", index, rate)
                load_block_weights(layer, path)
                consumed.append(path)
        tolerance = 5e-4 if args.mode == "2of4" else 1e-4
        if abs(counts[rate] / total - rate) > tolerance:
            raise AssertionError("sorting target does not match sparsity budget")
        eval_start = time.perf_counter()
        ppl = float(_eval_ppl(model, evaluation, device))
        zeros, measured = _weight_stats(model)
        if measured != total:
            raise AssertionError("sorting weight accounting mismatch")
        row = dict(mode=args.mode, ordering=args.ordering, sparsity=rate,
                   status="completed", nested_mask_violations=0,
                   alpha=args.structured_alpha if args.mode == "2of4" else args.unstructured_alpha,
                   ppl_wikitext2=ppl, actual_zero_sparsity=zeros / total,
                   explicit_mask_sparsity=counts[rate] / total,
                   evaluation_scope=protocol["runtime"]["evaluation"]["scope"],
                   evaluation_windows=protocol["runtime"]["evaluation"]["actual_windows"],
                   preparation_seconds=preparation_seconds, ppl_eval_seconds=time.perf_counter() - eval_start)
        _append_tsv(args.output / "comparison.tsv", row)
        rows.append(row)
        _write_json(args.output / "results.json", rows)
        for path in consumed:
            path.unlink()  # Only exact snapshots successfully evaluated and written above.
        print(row, flush=True)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.setdefault(key, "1")
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run(args)


if __name__ == "__main__":
    main()
