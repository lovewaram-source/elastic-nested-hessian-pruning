"""Token isolation selected from actual fixed sparse checkpoint paths.

Prepare ordinary nested reference checkpoints once. At each block boundary,
rank the configured real checkpoint representations, select one shared position
mask, collect actual checkpoint linear inputs, and reconstruct a new nested
weight trajectory. Evaluate only the requested isolation ratio.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from checkpoint_token_isolation_core import (
    RATES, collect_checkpoint_hessians, load_block_weights, select_block_private_tokens,
    select_replacement_error_tokens,
    select_all_mean_tokens,
)
from run_rate_aligned_nested import (
    _apply_unstructured_stage, _assert_stage_mask, _base_row, _find_layers,
    _layer_device, _load_calibration_inputs, _load_eval_loader, _load_model, _weight_stats,
)
from experiment_common import _append_tsv, _eval_ppl, _write_json
from experiment_common import input_groups, synchronize
from token_isolation_core import prepare_device_plan
from run_metadata import validate_run_arguments, runtime_record, evaluation_record


def configure_rates(rates):
    """Configure this run's nested rate set while preserving the 20% anchor."""
    global RATES
    rates = tuple(float(rate) for rate in rates)
    if len(rates) < 2 or rates[0] != .2 or any(a >= b for a, b in zip(rates, rates[1:])):
        raise ValueError("rates must be strictly increasing and start with the 20% anchor")
    import checkpoint_token_isolation_core as checkpoint_core
    import token_isolation_core as token_core
    RATES = rates
    checkpoint_core.RATES = rates
    token_core.RATES = rates


class CheckpointCache:
    def __init__(self, root, memory_reference=False):
        self.root = Path(root)
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"cache must be new or empty: {root}")
        self.reference = self.root / "reference"
        self.target = self.root / "target"
        self.reference.mkdir(parents=True)
        self.target.mkdir()
        self.memory_reference = memory_reference
        self.reference_states = {}

    def path(self, kind, layer, stage):
        tag = "dense" if stage is None else f"s{round(stage * 100):02d}"
        return getattr(self, kind) / f"layer_{layer:03d}_{tag}.pt"

    @torch.no_grad()
    def save(self, kind, layer_index, stage, layer):
        if kind == "reference" and self.memory_reference:
            key = (layer_index, stage)
            if key in self.reference_states:
                raise FileExistsError(f"duplicate in-memory reference: {key}")
            self.reference_states[key] = {
                name: module.weight.detach().cpu().clone()
                for name, module in _find_layers(layer).items()
            }
            return
        path = self.path(kind, layer_index, stage)
        if path.exists():
            raise FileExistsError(f"refusing checkpoint overwrite: {path}")
        state = {name: module.weight.detach().cpu()
                 for name, module in _find_layers(layer).items()}
        temporary = path.with_suffix(".pt.tmp")
        torch.save(state, temporary)
        temporary.replace(path)

    def source(self, layer_index, stage):
        if self.memory_reference:
            return self.reference_states[layer_index, stage]
        return self.path("reference", layer_index, stage)

    def release_reference(self, layer_index):
        for key in list(self.reference_states):
            if key[0] == layer_index:
                del self.reference_states[key]


@torch.no_grad()
def collect_dense_reference(layer, inputs, attention_mask, position_ids, device, percdamp,
                            calibration_samples=None):
    from rate_aligned_src.elastic_sparsegpt import ElasticSparseGPT

    modules = _find_layers(layer)
    groups = input_groups(modules)
    pruners = {group: ElasticSparseGPT(modules[group]) for group in groups}
    seen = {}
    calibration_samples = inputs.shape[0] if calibration_samples is None else calibration_samples
    if not 0 < calibration_samples <= inputs.shape[0]:
        raise ValueError("invalid reference calibration sample count")

    def hook_for(group, name):
        def hook(_, values, output):
            value = values[0].detach()
            identity = (value.data_ptr(), tuple(value.shape), value.stride())
            if name == group:
                seen[group] = identity
                if sample < calibration_samples:
                    pruners[group].add_batch(value, output.detach())
            elif seen[group] != identity:
                raise AssertionError("dense reference projection inputs differ")
        return hook

    handles = [modules[name].register_forward_hook(hook_for(group, name))
               for group, members in groups.items() for name in members]
    kwargs = {}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask.to(device)
    if position_ids is not None:
        kwargs["position_ids"] = position_ids.to(device)
    outputs = torch.empty_like(inputs, device="cpu")
    try:
        for sample in range(inputs.shape[0]):
            seen.clear()
            value = layer(inputs[sample:sample + 1].to(device), **kwargs)
            hidden = value[0] if isinstance(value, (tuple, list)) else value
            outputs[sample].copy_(hidden[0].detach().cpu())
    finally:
        for handle in handles:
            handle.remove()
    plans = {group: prepare_device_plan(pruners[group].H, percdamp) for group in groups}
    return outputs, plans


@torch.no_grad()
def apply_stage(layer, plans, mode, previous, current, alpha, cumulative, module_plans=False):
    from rate_aligned_src.elastic_sparsegpt import apply_elastic_2of4_stage

    modules = _find_layers(layer)
    groups = input_groups(modules)
    group_of = {name: group for group, names in groups.items() for name in names}
    for name, module in modules.items():
        plan = plans[name if module_plans else group_of[name]]
        if mode == "unstructured":
            mask = _apply_unstructured_stage(module, plan, previous, current, max(RATES), alpha)
        else:
            mask = apply_elastic_2of4_stage(
                module, plan, current, previous_sparsity=previous,
                prune_n=2, prune_m=4, block_fractions=(.4, .2, .2, .2),
                compensate=True, compensation_alpha=alpha, return_mask=True,
            )
        _assert_stage_mask(mode, mask, previous, current, max(RATES))
        cumulative[name].logical_or_(mask)
        if (cumulative[name] & module.weight.ne(0)).any():
            raise AssertionError(f"nested zeros resurrected: {name}/{current}")
    return sum(int(mask.sum()) for mask in cumulative.values())


@torch.no_grad()
def prepare_reference_layer(args, layer, index, paths, attention_mask, position_ids, cache, device):
    """Keep the dense calibration stream independent of sparse/target streams."""
    alpha = args.unstructured_alpha if args.mode == "unstructured" else args.structured_alpha
    stages = (.1,) + RATES if args.mode == "unstructured" else RATES
    modules = _find_layers(layer)
    cache.save("reference", index, None, layer)
    outputs, plans = collect_dense_reference(
        layer, paths, attention_mask, position_ids, device, args.percdamp,
        calibration_samples=getattr(args, "reference_calibration_samples", None),
    )
    cumulative = {name: torch.zeros_like(module.weight, dtype=torch.bool)
                  for name, module in modules.items()}
    previous = 0.
    total = sum(module.weight.numel() for module in modules.values())
    manifest = []
    for stage in stages:
        zeros = apply_stage(layer, plans, args.mode, previous, stage, alpha, cumulative)
        if stage in RATES:
            cache.save("reference", index, stage, layer)
            location = (f"memory:layer_{index:03d}_s{round(stage * 100):02d}"
                        if cache.memory_reference else str(cache.path("reference", index, stage)))
            manifest.append(dict(layer=index, rate=stage, explicit_zeros=zeros,
                                 total_weights=total, file=location))
        previous = stage
    return outputs, manifest


@torch.no_grad()
def prepare_references(args, model, initial, attention_mask, position_ids, cache):
    """Generate common-dense-H nested checkpoints, preserving the dense stream."""
    paths = initial
    manifest = []
    started = time.perf_counter()
    for index, layer in enumerate(model.model.layers):
        tick = time.perf_counter()
        device = _layer_device(model, index, next(layer.parameters()).device)
        outputs, entries = prepare_reference_layer(
            args, layer, index, paths, attention_mask, position_ids, cache, device,
        )
        manifest.extend(entries)
        paths = outputs
        synchronize(device)
        print(f"REFERENCE_PREPARED layer={index} seconds={time.perf_counter() - tick:.2f}", flush=True)
    elapsed = time.perf_counter() - started
    _write_json(args.output / "reference_checkpoints.json", manifest)
    return elapsed


@torch.no_grad()
def reconstruct(args, model, initial, attention_mask, position_ids, cache,
                stream_reference=False, compute_device=None):
    if stream_reference and not cache.memory_reference:
        raise ValueError("streaming requires bounded in-memory reference storage")
    fraction = getattr(args, "private_fraction", .1)
    fusion = getattr(args, "shared_fusion", "anchor20")
    selection_rule = getattr(args, "selection_rule", "cosine")
    module_selection = selection_rule.startswith("module_")
    if module_selection and (args.mode != "2of4" or fusion != "anchor20" or fraction != .1):
        raise ValueError("module selectors require 2of4, anchor20 and 10% isolation")
    threshold = getattr(args, "share_error_threshold", .05)
    if fusion == "mean_hessian" and selection_rule != "all_mean":
        raise ValueError("mean_hessian requires all_mean selection")
    if selection_rule in ("relative_error", "all_mean") and fusion != "isolated_mean_anchor20" and not (selection_rule == "all_mean" and fusion == "mean_hessian"):
        raise ValueError("relative-error routing requires anchor20 on safe positions and mean on the rest")
    paths = {rate: initial for rate in RATES}
    alpha = args.unstructured_alpha if args.mode == "unstructured" else args.structured_alpha
    stages = (.1,) + RATES if args.mode == "unstructured" else RATES
    counts = {rate: 0 for rate in RATES}
    totals = 0
    timings = {name: 0. for name in ("selection_seconds", "checkpoint_forward_hessian_seconds",
                                    "factorization_seconds", "pruning_seconds", "snapshot_seconds")}
    evaluation_rates = set(getattr(args, "eval_rates", RATES))
    started = time.perf_counter()
    dense_paths = initial
    manifest = []
    reference_seconds = 0.
    error_layers = []
    for index, layer in enumerate(model.model.layers):
        layer_start = time.perf_counter()
        device = (compute_device if compute_device is not None
                  else _layer_device(model, index, next(layer.parameters()).device))
        if compute_device is not None:
            layer.to(device)
        if stream_reference:
            tick = time.perf_counter()
            dense_paths, entries = prepare_reference_layer(
                args, layer, index, dense_paths, attention_mask, position_ids, cache, device,
            )
            manifest.extend(entries)
            synchronize(device)
            reference_seconds += time.perf_counter() - tick
        synchronize(device)
        tick = time.perf_counter()
        if module_selection:
            from module_token_isolation import collect_module_isolation
            hessians, outputs, trace, masks, selection = collect_module_isolation(
                layer, paths, {r:cache.source(index,r) for r in RATES},cache.source(index,None),
                attention_mask,position_ids,device,selection_rule,args.seed+index,
                implementation=getattr(args,"module_hessian_implementation","direct"))
            torch.save(dict(masks=masks,statistics=selection),args.output/f"module_positions_layer_{index:03d}.pt")
            synchronize(device)
            timings["checkpoint_forward_hessian_seconds"] += time.perf_counter()-tick
        elif selection_rule == "all_mean":
            mask, selection = select_all_mean_tokens(paths)
        elif selection_rule == "relative_error":
            mask, selection, errors, valid = select_replacement_error_tokens(paths, device, threshold)
            error_layers.append((errors, valid))
            torch.save(dict(errors=errors, valid=valid), args.output / f"replacement_error_layer_{index:03d}.pt")
        else:
            mask, selection = select_block_private_tokens(paths, device, fraction)
        if not module_selection:
            synchronize(device)
            timings["selection_seconds"] += time.perf_counter() - tick
            # Legacy selectors share one mask across all projections.
            torch.save({"mask": mask, "statistics": selection,
                        "source": "embedding" if index == 0 else f"actual checkpoint block {index - 1} outputs"},
                       args.output / f"private_positions_layer_{index:03d}.pt")
            tick = time.perf_counter()
            hessians, outputs, trace = collect_checkpoint_hessians(
                layer, paths, mask,
                {rate: cache.source(index, rate) for rate in RATES},
                attention_mask, position_ids, device,
                shared_fusion=fusion,
            )
            synchronize(device)
            timings["checkpoint_forward_hessian_seconds"] += time.perf_counter() - tick
        if any(count != args.nsamples for count in trace["forward_counts"].values()):
            raise AssertionError("a sparse reference checkpoint did not execute all samples")
        modules = _find_layers(layer)
        plans = {}
        # Values keep the original tensors alive while identity keys are used.
        # This prevents id reuse and reuses genuinely identical H20 plans only.
        shared_plan_cache = {}
        hessian_uses = Counter(id(h) for values in hessians.values() for h in values.values())
        tick = time.perf_counter()
        for group in list(hessians):
            branch_hessians = hessians.pop(group)
            if all(branch_hessians[r] is branch_hessians[.2] for r in RATES):
                # Corrected isolated-mean fusion produces one identical H.
                common_plan = prepare_device_plan(branch_hessians[.2], args.percdamp)
                plans[group] = {rate: common_plan for rate in RATES}
                branch_hessians.clear()
                del common_plan
            else:
                plans[group] = {}
                for rate in RATES:
                    hessian = branch_hessians.pop(rate)
                    if module_selection and hessian_uses[id(hessian)] > 1:
                        key = id(hessian)
                        if key not in shared_plan_cache:
                            shared_plan_cache[key] = (hessian,prepare_device_plan(hessian,args.percdamp))
                        plans[group][rate] = shared_plan_cache[key][1]
                    else:
                        plans[group][rate] = prepare_device_plan(hessian,args.percdamp)
                    del hessian
        shared_plan_cache.clear()
        synchronize(device)
        timings["factorization_seconds"] += time.perf_counter() - tick
        # Reconstruction starts from original dense weights, not from an
        # already-compensated reference checkpoint (which would double OBS).
        tick = time.perf_counter()
        load_block_weights(layer, cache.source(index, None))
        cache.release_reference(index)
        cumulative = {name: torch.zeros_like(module.weight, dtype=torch.bool)
                      for name, module in modules.items()}
        previous = 0.
        for stage in stages:
            zeros = apply_stage(layer,
                                {group: branches[max(stage, .2)] for group, branches in plans.items()},
                                args.mode, previous, stage, alpha, cumulative, module_plans=module_selection)
            if stage in RATES:
                counts[stage] += zeros
                if stage in evaluation_rates and (stage != max(RATES) or getattr(args, "save_final_target", False)):
                    synchronize(device)
                    timings["pruning_seconds"] += time.perf_counter() - tick
                    snapshot_tick = time.perf_counter()
                    cache.save("target", index, stage, layer)
                    timings["snapshot_seconds"] += time.perf_counter() - snapshot_tick
                    tick = time.perf_counter()
            previous = stage
        synchronize(device)
        timings["pruning_seconds"] += time.perf_counter() - tick
        totals += sum(module.weight.numel() for module in modules.values())
        paths = outputs
        del plans, cumulative
        if compute_device is not None:
            layer.to("cpu")
        report = dict(layer=index, mask=selection, reference_trace=trace,
                      elapsed_seconds=time.perf_counter() - layer_start)
        with (args.output / "execution_trace.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report) + "\n")
        print("TRUE_CHECKPOINT_LAYER_COMPLETE " + json.dumps(report), flush=True)
    # The final block outputs have also been truly computed for every configured
    # checkpoints. Record their alignment even though there is no next H.
    if module_selection:
        final_stats = dict(rule=selection_rule,final_outputs_computed=True,
                           selection="module/rate masks recorded at actual H inputs; no unused output ranking")
    elif selection_rule == "all_mean":
        _, final_stats = select_all_mean_tokens(paths)
    elif selection_rule == "relative_error":
        from replacement_error_stats import summarize_layers
        _, final_stats, _, _ = select_replacement_error_tokens(paths, device, threshold)
        # Only the 32 H input boundaries count. Final unused block output is excluded.
        _write_json(args.output / "replacement_error_summary.json", summarize_layers(error_layers, threshold))
    else:
        _, final_stats = select_block_private_tokens(paths, device, fraction)
    _write_json(args.output / "final_block_output_similarity.json", final_stats)
    del paths, dense_paths
    timings["reconstruction_wall_seconds"] = time.perf_counter() - started
    timings["reference_preparation_seconds"] = reference_seconds
    if stream_reference:
        _write_json(args.output / "reference_checkpoints.json", manifest)
    return counts, totals, timings


@torch.no_grad()
def run(args):
    from transformers import AutoTokenizer

    validate_run_arguments(args)
    configure_rates(args.rates)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("output must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    cache = CheckpointCache(args.cache, memory_reference=True)
    protocol = dict(vars(args), output=str(args.output), cache=str(args.cache))
    fraction = args.private_fraction
    fusion = args.shared_fusion
    relative = args.selection_rule == "relative_error"
    pure_mean = args.selection_rule == "all_mean"
    if pure_mean:
        fraction = 1.
    prefix = "true_checkpoint_block_cosine" + (f"_{fusion}" if fusion != "anchor20" else "")
    policy = f"{prefix}_private_{round(fraction * 100):03d}"
    if relative:
        policy = f"true_checkpoint_relative_error_le_{args.share_error_threshold:g}_{fusion}"
    elif pure_mean:
        policy = f"true_checkpoint_all_mean_{fusion}"
    if args.selection_rule.startswith("module_"):
        policy = f"true_checkpoint_{args.selection_rule}_private_010_anchor20"
    protocol.update(
        rates=RATES, private_fraction=None if relative else fraction, anchor=.2, shared_fusion=fusion,
        share_error_threshold=None if pure_mean else args.share_error_threshold,
        reference="fixed ordinary nested common-dense-H checkpoints, each with its own true weights",
        selection=("all positions; no scoring or threshold; average all configured actual input Grams"
                   if fusion == "mean_hessian" else f"every position averages {len(RATES)} actual reference inputs; no scoring or threshold"
                   if pure_mean else f"share20 only if max_r ||x_r-x20||/||x_r|| <= {args.share_error_threshold}; average otherwise"
                   if relative else f"one global top{fraction:.0%} mask per block boundary from worst cosine distance to actual 20% path"),
        boundary_alignment="block l outputs select positions for block l+1 H; first block uses embedding positions",
        hessian=("all positions: equal mean of configured actual linear-input Grams; one common FP32 buffer"
                 if fusion == "mean_hessian" else f"all positions: FP32 arithmetic mean of {len(RATES)} actual linear inputs, then Gram; common H"
                 if pure_mean else f"safe positions: anchor20; all other positions: arithmetic mean of {len(RATES)} actual linear inputs; common H"
                 if relative else f"isolated{fraction:.0%} arithmetic mean of {len(RATES)} actual linear inputs; "
                 f"remaining{1 - fraction:.0%} 20% anchor input; identical H reused for all rates"
                 if fusion == "isolated_mean_anchor20" else
                 f"same position mask for all projections; shared{1 - fraction:.0%} "
                 + (f"arithmetic mean of {len(RATES)} actual linear inputs" if fusion == "mean_token" else "20% anchor input")
                 + f" plus own{fraction:.0%} actual checkpoint linear input"),
        propagation=f"{len(RATES)} actual checkpoint forward paths, no replacement in true forward propagation",
        reconstruction="original dense weights; stagewise configured rate Hessians; frozen numerical prefixes",
        calibration="C4 64 x 4096 default; original sample normalization and damping",
        evaluation=("full available WikiText2 windows" if args.eval_samples is None
                    else "truncated WikiText2 debug evaluation"),
        evaluation_rates=sorted(getattr(args, "eval_rates", RATES)),
        execution="single visible GPU; CPU-resident model; one active block; layer-streamed in-memory references",
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (Path(__file__), Path(__file__).with_name("checkpoint_token_isolation_core.py"),
                                    Path(__file__).with_name("token_isolation_core.py"),
                                    Path(__file__).with_name("replacement_error_stats.py"),
                                    Path(__file__).with_name("experiment_common.py"),
                                    ROOT / "lib/elastic_sparsegpt.py", ROOT / "lib/sparsegpt.py")},
    )
    if args.selection_rule.startswith("module_"):
        protocol.update(
            selection=("per module/rate top10% ||(W_dense-W_sparse_r)(x_r-x20)||^2"
                       if args.selection_rule == "module_residual" else
                       "per module/rate random10%, seed+layer, one deterministic generator"
                       if args.selection_rule == "module_random" else
                       "original block-boundary worst cosine top10%; identical mask for every module/rate"),
            hessian="per module/rate mixed actual inputs: own10%, anchor20 remaining90%; H20 full actual anchor inputs",
            execution="single GPU, CPU layer streaming; residual selector has one extra scoring forward per reference per layer",
            source_sha256={**protocol["source_sha256"],**{
                str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__).with_name("module_token_isolation.py"),
                             Path(__file__).with_name("token_residual_diagnostics.py"))}})
    _write_json(args.output / "protocol.json", protocol)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    evaluation = _load_eval_loader(args, tokenizer)
    tick = time.perf_counter()
    model = _load_model(args, tokenizer)
    load_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    device = torch.device("cuda:0")
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(device)
    model.model.embed_tokens.to(device)
    try:
        initial, attention_mask, position_ids = _load_calibration_inputs(
            args, model, tokenizer, device,
        )
    finally:
        model.model.embed_tokens.to("cpu")
    protocol["runtime"] = runtime_record(args, model, evaluation)
    _write_json(args.output / "protocol.json", protocol)
    capture_seconds = time.perf_counter() - tick
    counts, totals, timings = reconstruct(
        args, model, initial, attention_mask, position_ids, cache,
        stream_reference=True, compute_device=device,
    )
    reference_seconds = timings["reference_preparation_seconds"]
    del initial
    # Streaming reconstruction wall time already includes reference generation.
    preparation_seconds = load_seconds + capture_seconds + timings["reconstruction_wall_seconds"]
    alpha = args.unstructured_alpha if args.mode == "unstructured" else args.structured_alpha
    coverage = (json.loads((args.output / "replacement_error_summary.json").read_text())
                if relative else None)
    tick = time.perf_counter()
    # Store preparation accounting before moving the full model, so OOM cannot erase it.
    _write_json(args.output / "preparation_timings.json", dict(timings,
                preparation_wall_before_evaluation_seconds=preparation_seconds,
                model_load_seconds=load_seconds, data_seconds=capture_seconds))
    model.to(device)
    synchronize(device)
    reload_seconds = time.perf_counter() - tick
    rows = []
    for rate in sorted(args.eval_rates, reverse=True):
        consumed_paths = []
        if rate == max(RATES) and args.save_final_target:
            consumed_paths = [cache.path("target", index, rate) for index in range(len(model.model.layers))]
        if rate != max(RATES):
            tick = time.perf_counter()
            for index, layer in enumerate(model.model.layers):
                path = cache.path("target", index, rate)
                load_block_weights(layer, path)
                consumed_paths.append(path)
            synchronize(torch.device("cuda:0"))
            reload_seconds += time.perf_counter() - tick
        if abs(counts[rate] / totals - rate) > (5e-4 if args.mode == "2of4" else 1e-4):
            raise AssertionError("wrong final sparsity budget")
        tick = time.perf_counter()
        ppl = float(_eval_ppl(model, evaluation, torch.device("cuda:0")))
        evaluation_seconds = time.perf_counter() - tick
        zeros, measured = _weight_stats(model)
        if measured != totals:
            raise AssertionError("weight counting mismatch")
        row = _base_row(args.mode, policy, rate,
                        alpha, ppl, load_seconds,
                        timings["checkpoint_forward_hessian_seconds"] + timings["factorization_seconds"],
                        timings["pruning_seconds"], 0., evaluation_seconds,
                        counts[rate], totals, zeros, 0)
        row.update(private_fraction=None if relative else fraction, shared_fusion=fusion,
                   evaluation_scope=protocol["runtime"]["evaluation"]["scope"],
                   evaluation_windows=protocol["runtime"]["evaluation"]["actual_windows"],
                   selection_rule=args.selection_rule,
                   share_error_threshold=args.share_error_threshold if relative else None,
                   reference_preparation_seconds=f"{reference_seconds:.2f}",
                   selection_seconds=f"{timings['selection_seconds']:.2f}",
                   snapshot_seconds=f"{timings['snapshot_seconds']:.2f}",
                   data_seconds=f"{capture_seconds:.2f}",
                   checkpoint_load_seconds=f"{reload_seconds:.2f}",
                   preparation_wall_before_evaluation_seconds=f"{preparation_seconds:.2f}",
                   cumulative_total_excluding_ppl_seconds=f"{preparation_seconds + reload_seconds:.2f}")
        if coverage is not None:
            without_embedding = coverage["excluding_embedding_input"]
            row.update(anchor_share_fraction_all_inputs=coverage["all_layer_inputs"]["anchor_share_fraction"],
                       anchor_share_fraction_excluding_embedding=(without_embedding["anchor_share_fraction"]
                                                                 if without_embedding is not None else None))
        elif pure_mean:
            row.update(anchor_share_fraction_all_inputs=0., anchor_share_fraction_excluding_embedding=0.,
                       averaged_fraction_all_inputs=1.,
                       averaging_space="hessian" if fusion == "mean_hessian" else "activation")
        _append_tsv(args.output / "comparison.tsv", row)
        rows.append(row)
        # Keep this rate's snapshots if loading/evaluation/result writing fails.
        for path in consumed_paths:
            path.unlink()  # exact experiment-owned successfully evaluated snapshot
        print("RESULT " + json.dumps(row), flush=True)
    _write_json(args.output / "timings.json", dict(timings, reference_preparation_seconds=reference_seconds,
                                                   model_load_seconds=load_seconds, data_seconds=capture_seconds,
                                                   checkpoint_reload_seconds=reload_seconds,
                                                   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device),
                                                   total_wall_excluding_ppl_seconds=preparation_seconds + reload_seconds))
    _write_json(args.output / "results.json", rows)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("TRUE_CHECKPOINT_EXPERIMENT_COMPLETED", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/ModelZoo/LLM/llama2/Llama-2-7b-hf")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--mode", choices=("unstructured", "2of4"), required=True)
    parser.add_argument("--gpu", default="0", help="one physical GPU ID; multi-GPU lists are rejected")
    parser.add_argument("--device-map", default="cpu", choices=("cpu",),
                        help="keep inactive layers on CPU; use only the selected GPU")
    parser.add_argument("--nsamples", type=int, default=64)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--percdamp", type=float, default=.01)
    parser.add_argument("--private-fraction", type=float, choices=(0., .1, .5, 1.), default=.1)
    parser.add_argument("--rates", default="0.2,0.3,0.4,0.5",
                        help="comma-separated nested rates starting at 0.2; e.g. 0.2,0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--eval-rates", default=None,
                        help="comma-separated rates to evaluate; omitted means every configured rate")
    parser.add_argument("--selection-rule", choices=("cosine", "relative_error", "all_mean", "module_residual", "module_random", "module_cosine"), default="cosine")
    parser.add_argument("--module-hessian-implementation",choices=("direct","delta","cached_direct"),default="direct",
                        help="delta reuses H20 and only accumulates selected-position Gram differences")
    parser.add_argument("--share-error-threshold", type=float, default=.05)
    parser.add_argument("--save-final-target", action="store_true", help="retain the final target until successful PPL")
    parser.add_argument("--shared-fusion", choices=("anchor20", "mean_token", "isolated_mean_anchor20", "mean_hessian"), default="anchor20")
    parser.add_argument("--unstructured-alpha", type=float, default=1.)
    parser.add_argument("--structured-alpha", type=float, default=.25)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--attn-implementation", default="sdpa", choices=("sdpa", "eager"))
    args = parser.parse_args(argv)
    try:
        args.rates = tuple(float(value) for value in args.rates.split(","))
    except ValueError:
        parser.error("rates must be comma-separated numbers")
    if len(args.rates) < 2 or args.rates[0] != .2 or any(a >= b for a, b in zip(args.rates, args.rates[1:])):
        parser.error("rates must be strictly increasing and start with 0.2")
    if any(not math.isfinite(rate) or not 0.2 <= rate < 1.0 for rate in args.rates):
        parser.error("rates must be finite and in [0.2, 1.0)")
    try:
        validate_run_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    if args.eval_rates is None:
        args.eval_rates = args.rates
    else:
        try:
            args.eval_rates = tuple(float(value) for value in args.eval_rates.split(","))
        except ValueError:
            parser.error("eval-rates must be comma-separated numbers")
        if (not args.eval_rates or len(set(args.eval_rates)) != len(args.eval_rates)
                or any(rate not in args.rates for rate in args.eval_rates)):
            parser.error("eval-rates must be a nonempty subset of configured rates")
    if args.nsamples < 1 or args.seqlen < 2:
        parser.error("invalid calibration shape")
    if not math.isfinite(args.share_error_threshold) or args.share_error_threshold < 0:
        parser.error("share error threshold must be finite and nonnegative")
    if args.selection_rule.startswith("module_") and (args.mode != "2of4" or args.shared_fusion != "anchor20" or args.private_fraction != .1):
        parser.error("module selectors require 2of4, anchor20, private-fraction .1")
    if args.shared_fusion == "mean_hessian" and args.selection_rule != "all_mean":
        parser.error("mean_hessian requires --selection-rule all_mean")
    if args.selection_rule in ("relative_error", "all_mean") and args.shared_fusion != "isolated_mean_anchor20" and not (args.selection_rule == "all_mean" and args.shared_fusion == "mean_hessian"):
        parser.error("relative_error requires --shared-fusion isolated_mean_anchor20")
    if not args.gpu.strip() or "," in args.gpu or args.gpu.strip() == "-1":
        parser.error("--gpu must select exactly one GPU")
    args.gpu = args.gpu.strip()
    return args


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.setdefault(key, "1")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
