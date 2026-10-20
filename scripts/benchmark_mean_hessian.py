"""One real 7B block: compare mean-input Gram against mean of input Grams.

This diagnostic does not produce publication PPL. The production experiment
must run separately, so benchmarking overhead is excluded from its timings.
"""
import gc
import json
import os
import time

import torch

from checkpoint_token_isolation_core import RATES, collect_checkpoint_hessians, select_all_mean_tokens
from run_checkpoint_token_isolation import (
    CheckpointCache, _load_calibration_inputs, _load_model, parse_args,
    prepare_reference_layer, synchronize,
)


@torch.no_grad()
def main():
    from transformers import AutoTokenizer
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.setdefault(key, "1")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("benchmark output must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = _load_model(args, tokenizer)
    model.model.embed_tokens.to(device)
    initial, attention, positions = _load_calibration_inputs(args, model, tokenizer, device)
    model.model.embed_tokens.to("cpu")
    cache = CheckpointCache(args.cache, memory_reference=True)
    layer = model.model.layers[0].to(device)
    prepare_reference_layer(args, layer, 0, initial, attention, positions, cache, device)
    mask, _ = select_all_mean_tokens({r: initial for r in RATES})
    rows = []
    # Reverse the order on repetition to expose first-run/order effects.
    for repeat, order in enumerate((("isolated_mean_anchor20", "mean_hessian"),
                                    ("mean_hessian", "isolated_mean_anchor20"))):
        for fusion in order:
            gc.collect()
            torch.cuda.empty_cache()
            synchronize(device)
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            hessians, outputs, trace = collect_checkpoint_hessians(
                layer, {r: initial for r in RATES}, mask,
                {r: cache.source(0, r) for r in RATES}, attention, positions, device,
                shared_fusion=fusion)
            synchronize(device)
            row = dict(repeat=repeat, fusion=fusion, seconds=time.perf_counter()-started,
                       baseline_allocated_bytes=baseline,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       peak_extra_bytes=torch.cuda.max_memory_allocated(device)-baseline,
                       forward_counts=trace["forward_counts"], nsamples=args.nsamples,
                       seqlen=args.seqlen, block=0)
            rows.append(row)
            print("BENCHMARK " + json.dumps(row), flush=True)
            del hessians, outputs, trace
    (args.output/"benchmark.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("BENCHMARK_COMPLETED", flush=True)


if __name__ == "__main__":
    main()
