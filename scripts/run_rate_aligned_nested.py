"""Compare common-Hessian nested pruning with fully rate-aligned replay.

Each aligned branch starts from the dense checkpoint, uses its own activation
Hessian in every layer, replays inherited masks from lower rates, and chooses
only the newly opened slice with that branch's Hessian. The common-Hessian
baseline reuses the dense-branch plans across all nested checkpoints.
"""

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_common import (  # noqa: E402
    _append_tsv,
    _capture_first_layer_inputs,
    _eval_ppl,
    _layer_device,
    _layer_forward_cpu,
    _write_json,
)


UNSTRUCTURED_STAGES = (0.1, 0.2, 0.3, 0.4, 0.5)
STRUCTURED_STAGES = (0.2, 0.3, 0.4, 0.5)
STRUCTURED_BLOCK_FRACTIONS = (0.4, 0.2, 0.2, 0.2)


def _stage_tag(stage):
    return f"s{round(float(stage) * 100):03d}"


def _find_layers(module):
    import torch.nn as nn
    import transformers

    layer_types = (nn.Linear, nn.Conv2d, transformers.Conv1D)
    found = {}

    def visit(parent, prefix=""):
        for name, child in parent.named_children():
            key = f"{prefix}.{name}" if prefix else name
            if isinstance(child, layer_types):
                found[key] = child
            else:
                visit(child, key)

    visit(module)
    return found


class _PlanStore:
    """Small CPU/disk store keeping a single dense-reference plan per layer."""

    def __init__(self, parent):
        import torch

        self.torch = torch
        Path(parent).mkdir(parents=True, exist_ok=True)
        self.path = Path(tempfile.mkdtemp(prefix="rate-aligned-plans-", dir=parent))
        self.layers = set()

    def put_layer(self, layer_index, plans):
        if layer_index in self.layers:
            raise ValueError(f"plan layer {layer_index} already exists")
        payload = {
            name: {
                key: None if getattr(plan, key) is None else getattr(plan, key).detach().cpu()
                for key in (
                    "permutation", "hessian_factor", "dead_columns",
                    "activation_norms", "hessian_inverse", "processed_columns",
                )
            }
            for name, plan in plans.items()
        }
        self.torch.save(payload, self.path / f"layer_{layer_index:03d}.pt")
        self.layers.add(layer_index)

    def get_layer(self, layer_index):
        from rate_aligned_src.elastic_sparsegpt import ElasticSparseGPTPlan

        if layer_index not in self.layers:
            raise KeyError(f"missing plan layer {layer_index}")
        payload = self.torch.load(
            self.path / f"layer_{layer_index:03d}.pt",
            map_location="cpu", weights_only=True,
        )
        return {
            name: ElasticSparseGPTPlan(**values)
            for name, values in payload.items()
        }

    def close(self, success):
        if success:
            shutil.rmtree(self.path)
        else:
            print(f"incomplete Hessian plans retained at {self.path}", flush=True)


def _mask_file(root, mode, stage, layer_index, name):
    module_name = name.replace(".", "_").replace("/", "_")
    return Path(root) / "mask_schedules" / mode / _stage_tag(stage) / (
        f"layer_{layer_index:02d}_{module_name}.npz"
    )


def _save_mask(path, mask):
    import numpy as np
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = mask.detach().to(device="cpu", dtype=torch.bool).contiguous()
    packed = np.packbits(value.numpy().reshape(-1))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, packed=packed, shape=np.asarray(value.shape, dtype=np.int64),
                 numel=np.asarray(value.numel(), dtype=np.int64))
    os.replace(temporary, path)


def _load_mask(path, expected_shape):
    import numpy as np
    import torch

    with np.load(path, allow_pickle=False) as data:
        shape = tuple(int(value) for value in data["shape"])
        numel = int(data["numel"])
        if shape != tuple(expected_shape) or math.prod(shape) != numel:
            raise ValueError(f"mask archive shape mismatch at {path}")
        flat = np.unpackbits(data["packed"])[:numel]
        return torch.from_numpy(flat.reshape(shape).astype(np.bool_))


def _mask_matrix(module, mask, device):
    import torch
    import torch.nn as nn
    import transformers

    value = mask.to(device=device, dtype=torch.bool)
    if isinstance(module, nn.Conv2d):
        return value.flatten(1)
    if isinstance(module, transformers.Conv1D):
        return value.t()
    return value


def _matrix_mask_to_weight(module, mask):
    import torch.nn as nn
    import transformers

    if isinstance(module, nn.Conv2d):
        return mask.reshape(module.weight.shape)
    if isinstance(module, transformers.Conv1D):
        return mask.t()
    return mask.reshape(module.weight.shape)


def _apply_unstructured_stage(module, plan, previous, current, maximum, alpha,
                              forced_mask=None):
    """Apply one elastic slice with an optional inherited, fixed mask."""
    import torch
    import torch.nn as nn
    import transformers
    from rate_aligned_src.elastic_sparsegpt import (
        _weight_matrix,
        build_block_mask,
        invert_permutation,
    )

    if not 0.0 <= alpha <= 1.0 or not 0.0 <= previous < current <= maximum:
        raise ValueError("invalid unstructured stage or compensation alpha")
    weight = _weight_matrix(module)
    rows, columns = weight.shape
    start = round(columns * previous / maximum)
    end = round(columns * current / maximum)
    budget = round(current * rows * columns) - round(previous * rows * columns)
    if not 0 <= start < end <= columns or budget > rows * (end - start):
        raise ValueError("invalid elastic unstructured slice or mask budget")

    device = weight.device
    permutation = plan.permutation.to(device)
    inverse = invert_permutation(permutation)
    factor = plan.hessian_factor.to(device=device, dtype=torch.float32)
    dead = plan.dead_columns.to(device=device, dtype=torch.bool)
    weight[:, dead] = 0
    ordered = weight[:, permutation]
    ordered_dead = dead[permutation]
    ordered[:, ordered_dead] = 0

    block_weight = ordered[:, start:end].clone()
    block_factor = factor[start:end, start:end]
    if forced_mask is None:
        score = block_weight.square() / torch.diag(block_factor).reshape(1, -1).square()
        block_mask = build_block_mask(score, budget=budget)
    else:
        fixed = _mask_matrix(module, forced_mask, device)[:, permutation]
        block_mask = fixed[:, start:end].clone()
        if int(block_mask.sum().item()) != budget:
            raise AssertionError(
                f"inherited slice has {int(block_mask.sum().item())} masks; expected {budget}"
            )

    quantized = torch.zeros_like(block_weight)
    errors = torch.zeros_like(block_weight)
    for column in range(end - start):
        current_weight = block_weight[:, column]
        diagonal = block_factor[column, column]
        quantized[:, column] = current_weight.masked_fill(block_mask[:, column], 0)
        error = (current_weight - quantized[:, column]) / diagonal
        error = alpha * error
        errors[:, column] = error
        if alpha:
            block_weight[:, column:] -= error.unsqueeze(1) @ block_factor[
                column, column:end - start
            ].unsqueeze(0)

    ordered[:, start:end] = quantized
    if alpha and end < columns:
        ordered[:, end:] -= errors @ factor[start:end, end:]
    ordered[:, ordered_dead] = 0
    result = ordered[:, inverse]
    if isinstance(module, nn.Conv2d):
        result = result.reshape(module.weight.shape)
    elif isinstance(module, transformers.Conv1D):
        result = result.t()
    module.weight.data.copy_(result.to(module.weight.dtype))
    return _matrix_mask_to_weight(
        module,
        _scatter_slice_mask(block_mask, rows, columns, start, end, permutation, inverse),
    )


def _scatter_slice_mask(block_mask, rows, columns, start, end, permutation, inverse):
    import torch

    ordered_mask = torch.zeros((rows, columns), dtype=torch.bool, device=block_mask.device)
    ordered_mask[:, start:end] = block_mask
    return ordered_mask[:, inverse]


def _collect_layer_plans(layer, inputs, attention_mask, position_ids,
                         layer_device, percdamp):
    import torch
    from rate_aligned_src.elastic_sparsegpt import ElasticSparseGPT

    modules = _find_layers(layer)
    pruners = {name: ElasticSparseGPT(module) for name, module in modules.items()}

    def add_batch(name):
        def hook(_, module_inputs, module_output):
            pruners[name].add_batch(module_inputs[0].detach(), module_output.detach())
        return hook

    handles = [modules[name].register_forward_hook(add_batch(name)) for name in modules]
    try:
        outputs = _layer_forward_cpu(
            layer, inputs, attention_mask, position_ids, layer_device
        )
    finally:
        for handle in handles:
            handle.remove()

    plans = {}
    for name, pruner in pruners.items():
        plans[name] = pruner.prepare_plan(percdamp=percdamp, ordering="original")
        pruner.free()
    del pruners
    if torch.cuda.is_available() and layer_device.type == "cuda":
        torch.cuda.empty_cache()
    return outputs, plans


def _load_model(args, tokenizer):
    import torch
    from transformers import AutoModelForCausalLM

    return_value = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )
    return_value.seqlen = args.seqlen
    return_value.eval()
    return_value.config.use_cache = False
    return return_value


def _load_calibration_inputs(args, model, tokenizer, device):
    import numpy as np
    import torch
    from rate_aligned_src.data import get_loaders

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    loader, _ = get_loaders(
        "c4", nsamples=args.nsamples, seed=args.seed,
        seqlen=model.seqlen, tokenizer=tokenizer,
    )
    return _capture_first_layer_inputs(
        model, loader, args.nsamples, model.seqlen, device
    )


def _load_eval_loader(args, tokenizer):
    from types import SimpleNamespace
    from rate_aligned_src.data import get_loaders

    _, loader = get_loaders(
        "wikitext2", seed=0, seqlen=args.seqlen, tokenizer=tokenizer
    )
    if args.eval_samples is not None:
        loader = SimpleNamespace(
            input_ids=loader.input_ids[:, : args.eval_samples * args.seqlen]
        )
    return loader


def _weight_stats(model):
    zeros = total = 0
    for layer in model.model.layers:
        for module in _find_layers(layer).values():
            weight = module.weight.data
            zeros += int((weight == 0).sum().item())
            total += weight.numel()
    return zeros, total


def _append_result(output, row):
    _append_tsv(output / "comparison.tsv", row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def _base_row(mode, policy, sparsity, alpha, ppl, model_load_seconds,
              hessian_seconds, pruning_seconds, forward_seconds,
              eval_seconds, explicit_zeros, total_weights, actual_zeros,
              nested_violations):
    return {
        "pruning_type": mode,
        "hessian_policy": policy,
        "sparsity": f"{sparsity:.2f}",
        "alpha": f"{alpha:.2f}",
        "ppl_wikitext2": f"{ppl:.8f}",
        "model_load_seconds": f"{model_load_seconds:.2f}",
        "hessian_seconds": f"{hessian_seconds:.2f}",
        "pruning_seconds": f"{pruning_seconds:.2f}",
        "branch_forward_seconds": f"{forward_seconds:.2f}",
        "ppl_eval_seconds": f"{eval_seconds:.2f}",
        "explicit_mask_sparsity": f"{explicit_zeros / max(total_weights, 1):.8f}",
        "actual_zero_sparsity": f"{actual_zeros / max(total_weights, 1):.8f}",
        "nested_mask_violations": int(nested_violations),
        "status": "completed",
    }


def _prepare_dense_plans(args, model, tokenizer, device, store):
    import torch

    inputs, attention_mask, position_ids = _load_calibration_inputs(
        args, model, tokenizer, device
    )
    hessian_seconds = 0.0
    for layer_index, layer in enumerate(model.model.layers):
        layer_device = _layer_device(model, layer_index, device)
        start = time.perf_counter()
        outputs, plans = _collect_layer_plans(
            layer, inputs, attention_mask, position_ids,
            layer_device, args.percdamp,
        )
        hessian_seconds += time.perf_counter() - start
        store.put_layer(layer_index, plans)
        inputs = outputs
        del outputs, plans
    return hessian_seconds


def _apply_common_stage(args, mode, model, store, previous, current, device):
    import torch
    from rate_aligned_src.elastic_sparsegpt import (
        ELASTIC_2OF4_BLOCK_FRACTIONS,
        apply_elastic_2of4_stage,
    )

    stage_zeros = total = 0
    for layer_index, layer in enumerate(model.model.layers):
        layer_device = _layer_device(model, layer_index, device)
        modules = _find_layers(layer)
        plans = store.get_layer(layer_index)
        for name, module in modules.items():
            plan = plans[name]
            if mode == "unstructured":
                mask = _apply_unstructured_stage(
                    module, plan, previous, current, 0.5, args.unstructured_alpha
                )
            else:
                mask = apply_elastic_2of4_stage(
                    module, plan, current, previous_sparsity=previous,
                    prune_n=2, prune_m=4,
                    block_fractions=ELASTIC_2OF4_BLOCK_FRACTIONS,
                    compensate=True, compensation_alpha=args.structured_alpha,
                    return_mask=True,
                )
            stage_zeros += int(mask.sum().item())
            total += module.weight.numel()
        del plans
        if layer_device.type == "cuda":
            torch.cuda.empty_cache()
    return stage_zeros, total


def _run_common_baseline(args, output, mode, tokenizer, eval_loader, device):
    import torch

    stages = UNSTRUCTURED_STAGES if mode == "unstructured" else STRUCTURED_STAGES
    alpha = args.unstructured_alpha if mode == "unstructured" else args.structured_alpha
    model_load_start = time.perf_counter()
    model = _load_model(args, tokenizer)
    model_load_seconds = time.perf_counter() - model_load_start
    cache_parent = output / ".work" / f"common_{mode}"
    cache_parent.mkdir(parents=True, exist_ok=True)
    store = _PlanStore(cache_parent)
    success = False
    try:
        hessian_seconds = _prepare_dense_plans(
            args, model, tokenizer, device, store
        )
        previous = 0.0
        cumulative_explicit = 0
        total_weights = 0
        for current in stages:
            start = time.perf_counter()
            stage_zeros, stage_weights = _apply_common_stage(
                args, mode, model, store, previous, current, device
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            pruning_seconds = time.perf_counter() - start
            cumulative_explicit += stage_zeros
            total_weights = stage_weights
            eval_start = time.perf_counter()
            ppl = float(_eval_ppl(model, eval_loader, device))
            eval_seconds = time.perf_counter() - eval_start
            actual_zeros, actual_weights = _weight_stats(model)
            _append_result(
                output,
                _base_row(
                    mode, "common_dense_H", current, alpha, ppl,
                    model_load_seconds, hessian_seconds, pruning_seconds,
                    0.0, eval_seconds, cumulative_explicit,
                    total_weights, actual_zeros,
                    0,
                ),
            )
            if actual_weights != total_weights:
                raise AssertionError("weight accounting differs between model scans")
            previous = current
        success = True
    finally:
        store.close(success=success)
        if success:
            shutil.rmtree(cache_parent, ignore_errors=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _assert_stage_mask(mode, mask, previous, current, maximum):
    import torch
    from rate_aligned_src.elastic_sparsegpt import iter_2of4_stage_segments

    if mode == "unstructured":
        _, columns = mask.shape
        start = round(columns * previous / maximum)
        end = round(columns * current / maximum)
        expected = round(current * mask.shape[0] * columns) - round(
            previous * mask.shape[0] * columns
        )
        if int(mask.sum().item()) != expected:
            raise AssertionError("unstructured stage does not meet its exact mask budget")
        if int(mask[:, :start].sum().item()) or int(mask[:, end:].sum().item()):
            raise AssertionError("unstructured stage mask escaped its opened slice")
        return

    rows, columns = mask.shape
    segments = iter_2of4_stage_segments(
        columns, previous, current,
        block_fractions=STRUCTURED_BLOCK_FRACTIONS,
    )
    if len(segments) != 1:
        raise AssertionError("expected one 2:4 macroblock per checkpoint")
    start, end = segments[0]
    opened = mask[:, start:end]
    if int(mask[:, :start].sum().item()) or int(mask[:, end:].sum().item()):
        raise AssertionError("2:4 stage mask escaped its opened macroblock")
    counts = opened.reshape(rows, -1, 4).sum(dim=2)
    if not torch.all(counts == 2):
        raise AssertionError("2:4 mask is not exactly 2:4 within the opened macroblock")


def _run_aligned_rate(args, output, mode, rate, tokenizer, eval_loader, device):
    import torch
    from rate_aligned_src.elastic_sparsegpt import (
        ELASTIC_2OF4_BLOCK_FRACTIONS,
        apply_elastic_2of4_stage,
    )

    stages = UNSTRUCTURED_STAGES if mode == "unstructured" else STRUCTURED_STAGES
    active_stages = [stage for stage in stages if stage <= rate + 1e-8]
    alpha = args.unstructured_alpha if mode == "unstructured" else args.structured_alpha

    load_start = time.perf_counter()
    model = _load_model(args, tokenizer)
    model_load_seconds = time.perf_counter() - load_start
    inputs, attention_mask, position_ids = _load_calibration_inputs(
        args, model, tokenizer, device
    )

    hessian_seconds = pruning_seconds = forward_seconds = 0.0
    explicit_zeros = total_weights = nested_violations = 0
    for layer_index, layer in enumerate(model.model.layers):
        layer_device = _layer_device(model, layer_index, device)
        start = time.perf_counter()
        outputs, plans = _collect_layer_plans(
            layer, inputs, attention_mask, position_ids,
            layer_device, args.percdamp,
        )
        hessian_seconds += time.perf_counter() - start
        modules = _find_layers(layer)
        prune_start = time.perf_counter()
        for name, module in modules.items():
            plan = plans[name]
            total_weights += module.weight.numel()
            previous_stage = 0.0
            for stage in active_stages:
                inherited = None
                if stage < rate - 1e-8:
                    path = _mask_file(output, mode, stage, layer_index, name)
                    if not path.is_file():
                        raise FileNotFoundError(
                            f"missing lower-rate mask needed for nesting: {path}"
                        )
                    inherited = _load_mask(path, tuple(module.weight.shape))
                if mode == "unstructured":
                    stage_mask = _apply_unstructured_stage(
                        module, plan, previous_stage, stage, 0.5, alpha,
                        forced_mask=inherited,
                    )
                else:
                    stage_mask = apply_elastic_2of4_stage(
                        module, plan, stage,
                        previous_sparsity=previous_stage,
                        prune_n=2, prune_m=4,
                        block_fractions=ELASTIC_2OF4_BLOCK_FRACTIONS,
                        compensate=True, compensation_alpha=alpha,
                        return_mask=True,
                        precomputed_mask=inherited,
                    )
                _assert_stage_mask(mode, stage_mask, previous_stage, stage, 0.5)
                if inherited is not None and not torch.equal(
                    stage_mask.detach().cpu(), inherited
                ):
                    nested_violations += int(
                        (inherited & ~stage_mask.detach().cpu()).sum().item()
                    )
                    raise AssertionError(
                        f"rate {rate:.0%} failed to replay nested stage {stage:.0%} "
                        f"in layer {layer_index} module {name}"
                    )
                explicit_zeros += int(stage_mask.sum().item())
                if stage >= rate - 1e-8:
                    _save_mask(
                        _mask_file(output, mode, stage, layer_index, name),
                        stage_mask,
                    )
                previous_stage = stage
                del inherited, stage_mask
        if torch.cuda.is_available() and layer_device.type == "cuda":
            torch.cuda.synchronize(layer_device)
        pruning_seconds += time.perf_counter() - prune_start
        del plans

        forward_start = time.perf_counter()
        inputs = _layer_forward_cpu(
            layer, inputs, attention_mask, position_ids, layer_device
        )
        if torch.cuda.is_available() and layer_device.type == "cuda":
            torch.cuda.synchronize(layer_device)
        forward_seconds += time.perf_counter() - forward_start
        del outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    expected_sparsity = rate
    explicit_sparsity = explicit_zeros / max(total_weights, 1)
    # 2:4 macroblocks have small discrete budget offsets after row-wise
    # four-column-group rounding; the common-H reference exhibits the same.
    budget_tolerance = 5e-4 if mode == "2of4" else 1e-4
    if abs(explicit_sparsity - expected_sparsity) > budget_tolerance:
        raise AssertionError(
            f"explicit mask sparsity {explicit_sparsity:.6f} does not match "
            f"target {expected_sparsity:.6f}"
        )
    eval_start = time.perf_counter()
    ppl = float(_eval_ppl(model, eval_loader, device))
    eval_seconds = time.perf_counter() - eval_start
    actual_zeros, actual_weights = _weight_stats(model)
    if actual_weights != total_weights:
        raise AssertionError("weight accounting differs between model scans")
    row = _base_row(
        mode, "fully_rate_aligned_H", rate, alpha, ppl,
        model_load_seconds, hessian_seconds, pruning_seconds,
        forward_seconds, eval_seconds, explicit_zeros,
        total_weights, actual_zeros, nested_violations,
    )
    _append_result(output, row)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def _run_aligned(args, output, mode, tokenizer, eval_loader, device, rates=None):
    stages = UNSTRUCTURED_STAGES if mode == "unstructured" else STRUCTURED_STAGES
    rows = []
    for rate in stages if rates is None else rates:
        row = _run_aligned_rate(
            args, output, mode, rate, tokenizer, eval_loader, device,
        )
        rows.append(row)
    return rows


def run(args):
    import torch
    import numpy as np
    from transformers import AutoTokenizer

    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    eval_loader = _load_eval_loader(args, tokenizer)
    device = torch.device("cuda:0")
    protocol = {
        "model": args.model,
        "calibration": {"dataset": "C4", "nsamples": args.nsamples,
                        "seqlen": args.seqlen, "seed": args.seed},
        "evaluation": "WikiText2 test PPL; evaluation time is reported separately",
        "evaluation_windows": args.eval_samples,
        "visible_gpus": args.gpu,
        "selected_modes": ("unstructured", "2of4") if args.mode == "all" else [args.mode],
        "selected_rates": args.rates,
        "common_baseline_run": not args.skip_common,
        "device_map": args.device_map,
        "attention_implementation": args.attn_implementation,
        "percdamp": args.percdamp,
        "column_ordering": "original; no sorting",
        "unstructured": {
            "stages": UNSTRUCTURED_STAGES,
            "alpha": args.unstructured_alpha,
            "slices": "maximum sparsity 50%; each 10-point checkpoint opens the next 20% input columns",
        },
        "2:4": {
            "stages": STRUCTURED_STAGES,
            "alpha": args.structured_alpha,
            "block_fractions": STRUCTURED_BLOCK_FRACTIONS,
            "mask_timing": "per 4-column group, default elastic SparseGPT",
            "note": "20% is the first valid global 2:4 checkpoint",
        },
        "comparison": {
            "common_dense_H": "Hessian plans calibrated on dense-path activations and shared across all checkpoints",
            "fully_rate_aligned_H": "each target rate starts from dense weights, uses its own layerwise branch activations/Hessians for all opened slices, and inherits all lower-rate slice masks",
            "weights": "numerical values are independently recomputed for each aligned rate; explicit masks remain nested",
        },
    }
    source_files = [
        ROOT / "scripts" / "run_rate_aligned_nested.py",
        ROOT / "scripts" / "experiment_common.py",
        ROOT / "lib" / "elastic_sparsegpt.py",
        ROOT / "lib" / "sparsegpt.py",
        ROOT / "lib" / "data.py",
    ]
    protocol["source_sha256"] = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_files
    }
    _write_json(args.output / "protocol.json", protocol)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    modes = ("unstructured", "2of4") if args.mode == "all" else (args.mode,)
    for mode in modes:
        if not args.skip_common:
            print(f"START common-H baseline: {mode}", flush=True)
            _run_common_baseline(
                args, args.output, mode, tokenizer, eval_loader, device
            )
        print(f"START fully rate-aligned branches: {mode}", flush=True)
        _run_aligned(
            args, args.output, mode, tokenizer, eval_loader, device, args.rates
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/data/ModelZoo/LLM/llama2/Llama-2-7b-hf")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("all", "unstructured", "2of4"), default="all")
    parser.add_argument("--rates", help="Comma-separated target rates, e.g. 0.30")
    parser.add_argument("--skip-common", action="store_true",
                        help="skip the common-H baseline when it has already been measured")
    parser.add_argument("--resume", action="store_true",
                        help="continue in an existing output directory and append rate results")
    parser.add_argument("--gpu", default="0,1")
    parser.add_argument("--device-map", choices=("auto", "balanced"), default="balanced")
    parser.add_argument("--nsamples", type=int, default=64)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--unstructured-alpha", type=float, default=1.0)
    parser.add_argument("--structured-alpha", type=float, default=0.25)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    args = parser.parse_args()
    if args.rates:
        try:
            args.rates = tuple(float(value) for value in args.rates.split(","))
        except ValueError:
            parser.error("rates must be comma-separated numbers")
        if len(set(args.rates)) != len(args.rates):
            parser.error("rates must not contain duplicates")
        allowed_rates = {
            "unstructured": UNSTRUCTURED_STAGES,
            "2of4": STRUCTURED_STAGES,
        }
        selected_modes = ("unstructured", "2of4") if args.mode == "all" else (args.mode,)
        if any(rate not in allowed_rates[mode] for mode in selected_modes for rate in args.rates):
            parser.error("a requested rate is not valid for the selected mode")
    else:
        args.rates = None
    if args.skip_common and args.rates is None:
        parser.error("--skip-common requires --rates so the aligned branches are explicit")
    for value in (args.unstructured_alpha, args.structured_alpha):
        if not 0.0 <= value <= 1.0:
            parser.error("compensation alpha must be in [0, 1]")
    if args.nsamples < 1 or args.seqlen < 2:
        parser.error("nsamples and seqlen must be positive")
    if args.eval_samples is not None and args.eval_samples < 1:
        parser.error("eval-samples must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.setdefault(key, "1")
    run(args)


if __name__ == "__main__":
    main()
