"""Real first-block cost and numerical checks, separate from production PPL."""
import gc
import json
import os
from pathlib import Path
import time

import torch

from checkpoint_token_isolation_core import RATES,load_block_weights
from module_token_isolation import collect_masked_module_hessians,select_module_masks
from token_residual_diagnostics import block_cosine_scores
from run_checkpoint_token_isolation import (CheckpointCache,_load_model,_load_calibration_inputs,
    parse_args,prepare_reference_layer,prepare_device_plan,apply_stage,_find_layers,synchronize)


@torch.no_grad()
def main():
    from transformers import AutoTokenizer
    args=parse_args()
    if args.selection_rule not in ("module_random","module_cosine"):
        raise ValueError("benchmark supports random and cosine masks")
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    for key in ("HF_HUB_OFFLINE","HF_DATASETS_OFFLINE","TRANSFORMERS_OFFLINE"):
        os.environ.setdefault(key,"1")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(args.seed)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("benchmark output must be new or empty")
    args.output.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda:0")
    tokenizer=AutoTokenizer.from_pretrained(args.model,use_fast=False)
    model=_load_model(args,tokenizer)
    model.model.embed_tokens.to(device)
    try:
        initial,attention,positions=_load_calibration_inputs(args,model,tokenizer,device)
    finally:
        model.model.embed_tokens.to("cpu")
    layer=model.model.layers[0].to(device)
    cache=CheckpointCache(args.cache,memory_reference=True)
    dense_outputs,_=prepare_reference_layer(args,layer,0,initial,attention,positions,cache,device)
    del dense_outputs
    cosine=block_cosine_scores({r:initial for r in RATES},device) if args.selection_rule=="module_cosine" else None
    masks,_=select_module_masks(tuple(_find_layers(layer)),initial.shape[:2],args.selection_rule,
                                cosine=cosine,seed=args.seed)
    baseline_h,baseline_targets={},{}
    rows=[]
    for implementation in ("direct","cached_direct"):
        gc.collect()
        torch.cuda.empty_cache()
        synchronize(device)
        baseline_allocated=torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        started=time.perf_counter()
        hs,_,trace=collect_masked_module_hessians(layer,{r:initial for r in RATES},
            {r:cache.source(0,r) for r in RATES},masks,attention,positions,device,
            capture_outputs=False,implementation=implementation)
        synchronize(device)
        row=dict(implementation=implementation,collector_seconds=time.perf_counter()-started,
                 selection_rule=args.selection_rule,
                 baseline_allocated_bytes=baseline_allocated,
                 collector_peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        if implementation=="direct":
            baseline_h={n:{r:h.cpu() for r,h in branches.items()} for n,branches in hs.items()}
        else:
            errors=[]
            for n,branches in hs.items():
                for r,h in branches.items():
                    old=baseline_h[n][r].to(device)
                    errors.append(dict(module=n,rate=r,
                        relative_frobenius=float((h-old).norm()/old.norm().clamp_min(1e-20)),
                        max_abs=float((h-old).abs().max())))
            row["hessian_errors"]=errors
            del baseline_h,old
        plans={}
        by_identity={}
        for n,branches in hs.items():
            plans[n]={}
            for r,h in branches.items():
                if r==.2 and id(h) in by_identity:
                    plan=by_identity[id(h)]
                else:
                    plan=prepare_device_plan(h,args.percdamp)
                    if r==.2:
                        by_identity[id(h)]=plan
                plans[n][r]=plan
        del hs,h,branches,plan,by_identity
        load_block_weights(layer,cache.source(0,None))
        cumulative={n:torch.zeros_like(m.weight,dtype=torch.bool) for n,m in _find_layers(layer).items()}
        previous=0.
        target_errors=[]
        for stage in RATES:
            apply_stage(layer,{n:p[stage] for n,p in plans.items()},"2of4",previous,stage,.25,cumulative,module_plans=True)
            for n,m in _find_layers(layer).items():
                if implementation=="direct":
                    baseline_targets[stage,n]=m.weight.detach().cpu().clone()
                else:
                    old=baseline_targets.pop((stage,n)).to(device)
                    target_errors.append(dict(module=n,rate=stage,
                        mask_disagreements=int((m.weight.eq(0)!=old.eq(0)).sum()),
                        weights=m.weight.numel(),max_abs=float((m.weight-old).abs().max()),
                        relative_frobenius=float((m.weight.float()-old.float()).norm()/old.float().norm().clamp_min(1e-20))))
            previous=stage
        if implementation!="direct":
            row["target_errors"]=target_errors
        rows.append(row)
        del plans,cumulative
        (args.output/"benchmark.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
        print("BENCHMARK "+json.dumps(row),flush=True)
    print("MODULE_TOKEN_OPTIMIZATION_BENCHMARK_COMPLETED",flush=True)


if __name__=="__main__":
    main()
