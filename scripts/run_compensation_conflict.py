"""Single-GPU local cross-rate compensation-conflict diagnosis (no PPL).

References are ordinary common-dense-H nested checkpoints, each with its own
real forward path. A selected module's target trajectory starts from dense W
and uses H20/H30/H40/H50 stagewise, exactly as the own-H checkpoint runner.
The observer never writes a candidate back. Only reference activations, not
the diagnosed target trajectory, are propagated to the next Transformer block.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def requested_groups(modules, requested):
    from experiment_common import input_groups
    if not set(requested).issubset(modules):
        raise ValueError("requested projection absent from model")
    return {g: tuple(n for n in names if n in requested)
            for g, names in input_groups(modules).items() if set(names).intersection(requested)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/data/ModelZoo/LLM/llama2/Llama-2-7b-hf")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--gpu", default="1")
    p.add_argument("--layers", type=int, nargs="+", default=[0, 16, 31])
    p.add_argument("--modules", nargs="+", default=["self_attn.o_proj", "mlp.up_proj",
                                                  "mlp.gate_proj", "mlp.down_proj"])
    p.add_argument("--nsamples", type=int, default=64)
    p.add_argument("--heldout-samples", type=int, default=8)
    p.add_argument("--seqlen", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--percdamp", type=float, default=.01)
    p.add_argument("--probe-tokens", type=int, default=128)
    p.add_argument("--probe-rows", type=int, default=64)
    p.add_argument("--window-groups", type=int, default=3)
    p.add_argument("--stop-after-layer", type=int)
    p.add_argument("--attn-implementation", default="sdpa", choices=("sdpa", "eager"))
    args = p.parse_args(argv)
    if not args.gpu.isdigit() or min(args.nsamples, args.heldout_samples, args.probe_tokens,
                                   args.probe_rows, args.window_groups) < 1 or args.seqlen < 2:
        p.error("one GPU and positive diagnostic dimensions required")
    if len(set(args.layers)) != len(args.layers) or min(args.layers) < 0:
        p.error("diagnostic layers must be unique and nonnegative")
    if len(set(args.modules)) != len(args.modules):
        p.error("diagnostic modules must be unique")
    if args.stop_after_layer is not None and (args.stop_after_layer < 0 or
                                             max(args.layers) > args.stop_after_layer):
        p.error("stop-after-layer must cover all requested diagnostic layers")
    args.mode, args.structured_alpha, args.unstructured_alpha = "2of4", .25, 1.
    args.device_map = "cpu"
    args.reference_calibration_samples = args.nsamples
    return args


def collect_layer(layer, paths, sources, attention, positions, device, args):
    """All train tokens form H; disjoint sequences form held-out probes only."""
    import torch
    from checkpoint_token_isolation_core import load_block_weights
    from run_rate_aligned_nested import _find_layers
    from rate_aligned_src.elastic_sparsegpt import ElasticSparseGPT
    from compensation_conflict import RATES

    modules = _find_layers(layer)
    groups = requested_groups(modules, args.modules)
    train = args.nsamples
    total, length = paths[.2].shape[:2]
    if total != train+args.heldout_samples:
        raise ValueError("input calibration/heldout split mismatch")
    ids = {}
    for split, count, offset in (("train", train, 0), ("heldout", total-train, train*length)):
        flat = torch.linspace(0, count*length-1, min(args.probe_tokens, count*length)).long().unique()
        ids[split] = flat + offset
    outputs, hessians, probes = {}, {}, {g: {s: {} for s in ids} for g in groups}
    current = {}
    pruners = {}

    def hook_for(group):
        def hook(_, values, output):
            value = values[0].detach()
            sample, rate = current["sample"], current["rate"]
            if sample < train:
                pruners[group].add_batch(value, output.detach())
            flat = value.reshape(-1, value.shape[-1])
            for split, selected in ids.items():
                which = ((selected >= sample*length) & (selected < (sample+1)*length)).nonzero().flatten()
                if which.numel():
                    local = selected[which]-sample*length
                    probes[group][split][rate][which] = flat.index_select(0, local.to(device)).cpu()
        return hook

    kwargs = {key: value.to(device) for key, value in (("attention_mask", attention),
              ("position_ids", positions)) if value is not None}
    handles = [modules[g].register_forward_hook(hook_for(g)) for g in groups]
    try:
        for rate in RATES:
            load_block_weights(layer, sources[rate])
            branch = paths.pop(rate)
            outputs[rate] = torch.empty_like(branch, device="cpu")
            current["rate"] = rate
            pruners = {g: ElasticSparseGPT(modules[g]) for g in groups}
            for group in groups:
                for split in ids:
                    probes[group][split][rate] = torch.empty(len(ids[split]), modules[group].weight.shape[1],
                                                             dtype=branch.dtype, device="cpu")
            for sample in range(total):
                current["sample"] = sample
                result = layer(branch[sample:sample+1].to(device), **kwargs)
                hidden = result[0] if isinstance(result, (tuple, list)) else result
                outputs[rate][sample].copy_(hidden[0].cpu())
            for group in groups:
                hessians.setdefault(group, {})[rate] = pruners[group].H
                for split in ids:
                    probes[group][split][rate] = probes[group][split][rate].t().contiguous()
            pruners.clear()
            del branch
    finally:
        for handle in handles:
            handle.remove()
    return outputs, hessians, probes, {s: selected.tolist() for s, selected in ids.items()}


def load_inputs(args, model, tokenizer, device):
    """Keep original train loader; held-out windows come from C4 validation."""
    from rate_aligned_src.data import get_loaders
    from experiment_common import _capture_first_layer_inputs
    loader, validation = get_loaders("c4", nsamples=args.nsamples, seed=args.seed,
                                    seqlen=args.seqlen, tokenizer=tokenizer)
    def digest(value):
        return hashlib.sha256(value.cpu().contiguous().numpy().tobytes()).hexdigest()
    train_hashes = [digest(item[0]) for item in loader]
    seen = set(train_hashes)
    heldout, windows, heldout_hashes = [], [], []
    for window in range(validation.input_ids.numel()//args.seqlen):
        value = validation.input_ids[:, window*args.seqlen:(window+1)*args.seqlen]
        signature = digest(value)
        if signature in seen:
            continue
        seen.add(signature)
        heldout.append((value, value))
        windows.append(window)
        heldout_hashes.append(signature)
        if len(heldout) == args.heldout_samples:
            break
    if len(heldout) != args.heldout_samples:
        raise ValueError("not enough distinct held-out validation windows")
    captured = _capture_first_layer_inputs(model, list(loader)+heldout,
        args.nsamples+args.heldout_samples, args.seqlen, device)
    return (*captured, dict(train_split="C4 train", heldout_split="C4 validation",
        train_token_sha256=train_hashes, heldout_token_sha256=heldout_hashes,
        heldout_validation_windows=windows))


def run(args):
    import torch
    from transformers import AutoTokenizer
    from checkpoint_token_isolation_core import load_block_weights
    from compensation_conflict import ConflictObserver, RATES, same_bits
    from run_checkpoint_token_isolation import CheckpointCache, prepare_reference_layer
    from run_rate_aligned_nested import _load_model, _find_layers
    from experiment_common import _write_json
    from token_isolation_core import prepare_device_plan
    from token_residual_diagnostics import forward_diagnostic_layer
    from rate_aligned_src.elastic_sparsegpt import apply_elastic_2of4_stage

    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("refusing nonempty diagnostic output")
    args.output.mkdir(parents=True, exist_ok=True)
    args._output_owned = True
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = _load_model(args, tokenizer)
    if max(args.layers) >= len(model.model.layers):
        raise ValueError("diagnostic layer outside model")
    protocol = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    protocol.update(task="local diagnostic; no PPL or new pruning method",
        trajectory="dense module W -> stagewise own-reference H20/H30/H40/H50; alpha=.25",
        references="fixed ordinary common-dense-H nested checkpoints; four independent forward streams",
        propagation="reference outputs, not diagnosed target outputs",
        objective="||(W_current-W_dense) X_r||^2 on sampled rows/tokens; NOT dense-input teacher outputs",
        heldout="C4 validation windows, exact token duplicates excluded; never enter reference or branch Hessians",
        donor_contract="same state, first mask, support and alpha; only donor H changes",
        window_contract="future masks recomputed per donor; effective FP32 shadows may differ in rounding from lazy large-block kernel",
        inactive_rates="recorded as counterfactual controls; exclude from shared-user conflict summaries",
        sampling="deterministic evenly spaced output rows and token positions; all training tokens form H",
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), ROOT/"scripts/compensation_conflict.py", ROOT/"lib/elastic_sparsegpt.py",
                      ROOT/"scripts/run_checkpoint_token_isolation.py", ROOT/"scripts/experiment_common.py"]})
    _write_json(args.output/"protocol.json", protocol)
    started = time.perf_counter()
    model.model.embed_tokens.to(device)
    try:
        initial, attention, positions, data_manifest = load_inputs(args, model, tokenizer, device)
    finally:
        model.model.embed_tokens.to("cpu")
    _write_json(args.output/"data_manifest.json", data_manifest)
    cache = CheckpointCache(args.cache, memory_reference=True)
    paths, dense_paths = {r: initial for r in RATES}, initial
    trace_path = args.output/"events.jsonl"
    event_count = 0
    def emit(record):
        nonlocal event_count
        event_count += 1
        with trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False)+"\n")
    for index, layer in enumerate(model.model.layers):
        tick = time.perf_counter()
        layer.to(device)
        dense_paths, _ = prepare_reference_layer(args, layer, index, dense_paths, attention, positions, cache, device)
        sources = {r: cache.source(index, r) for r in RATES}
        identity_checks = []
        if index in args.layers:
            paths, hessians, probes, positions_used = collect_layer(layer, paths, sources, attention, positions, device, args)
            _write_json(args.output/f"probe_positions_layer_{index:03d}.json", positions_used)
            load_block_weights(layer, cache.source(index, None))
            modules = _find_layers(layer)
            for group, raw_h in list(hessians.items()):
                plans = {r: prepare_device_plan(raw_h.pop(r), args.percdamp) for r in RATES}
                del hessians[group], raw_h
                group_modules = requested_groups(modules, args.modules)[group]
                for name in group_modules:
                    module = modules[name]
                    control = copy.deepcopy(module)
                    def emit_module(record, module_name=name):
                        emit(dict(layer=index, module=module_name, **record))
                    observer = ConflictObserver(plans, probes[group], module.weight.detach(), emit_module,
                                                rows=args.probe_rows, window_groups=args.window_groups)
                    previous = 0.
                    for stage in RATES:
                        observer.stage = stage
                        expected = apply_elastic_2of4_stage(control, plans[stage], stage,
                            previous_sparsity=previous, compensation_alpha=.25, return_mask=True)
                        mask = apply_elastic_2of4_stage(module, plans[stage], stage,
                            previous_sparsity=previous, compensation_alpha=.25, return_mask=True,
                            group_observer=observer)
                        if not torch.equal(mask, expected) or not same_bits(module.weight, control.weight):
                            raise AssertionError(f"observer changed main trajectory: {index}/{name}/{stage}")
                        identity_checks.append(dict(module=name, stage=stage, bitwise_weights=True, identical_masks=True))
                        previous = stage
                    print(f"MODULE_COMPLETE layer={index} module={name} events={observer.event_count} "
                          f"diagnostic_seconds={observer.wall_seconds:.2f}", flush=True)
                    del observer, control
                del plans, probes[group]
            del hessians, probes
        else:
            paths, _, _, _ = forward_diagnostic_layer(layer, paths, sources, cache.source(index, None),
                                                       attention, positions, device, diagnose=False)
        cache.release_reference(index)
        del sources
        layer.to("cpu")
        gc.collect()
        torch.cuda.synchronize()
        trace = dict(layer=index, events=event_count, seconds=time.perf_counter()-tick,
                     identity_checks=identity_checks, peak_gpu_bytes=torch.cuda.max_memory_allocated())
        with (args.output/"execution.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(trace)+"\n")
        _write_json(args.output/"state.json", dict(status="running", **trace))
        print("LAYER_COMPLETE "+json.dumps(trace), flush=True)
        if index == args.stop_after_layer:
            break
    from summarize_compensation_conflict import summarize
    summarize(args.output)
    _write_json(args.output/"completion.json", dict(status="completed", events=event_count,
        diagnostic_layers=args.layers, last_propagated_layer=index, ppl_evaluated=False,
        wall_seconds=time.perf_counter()-started))
    _write_json(args.output/"state.json", dict(status="completed", events=event_count))


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.setdefault(key, "1")
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    try:
        with torch.no_grad():
            run(args)
    except Exception as error:
        if getattr(args, "_output_owned", False) and not (args.output/"completion.json").exists():
            from experiment_common import _write_json
            _write_json(args.output/"failure.json", dict(status="failed", error=repr(error)))
        raise
