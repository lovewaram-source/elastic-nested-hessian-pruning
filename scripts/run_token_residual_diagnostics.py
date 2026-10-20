"""Real-reference Token selector diagnosis; no new pruning method or PPL."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))

import torch

from checkpoint_token_isolation_core import RATES
from run_checkpoint_token_isolation import (
    CheckpointCache, _load_calibration_inputs, _load_model, prepare_reference_layer,
)
from experiment_common import _write_json
from experiment_common import synchronize
from token_residual_diagnostics import compare_selectors, forward_diagnostic_layer, top_ids


def write_report(output, records):
    lines = ["# Token selector diagnostic (not PPL)", "",
             "All selectors choose the same top 10% budget. Random coverage uses 20 fixed-seed draws.",
             "Residual coverage is evaluated using the residual score itself, so its advantage is tautological, not efficacy evidence.",
             "Zero-total-error cases have undefined coverage (NA); first-layer Q/K/V are expected to be degenerate.", "",
             "At layer 0 the block input is the same embedding for all branches: boundary cosine has no signal for ANY module; floating-point tie ordering is not meaningful selection.", "",
             "|Layer|Module|Rate|Cosine informative|Spearman|Top overlap|Cosine coverage|Residual coverage|Random coverage|",
             "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "NA" if value is None else f"{value:.6f}"
    for row in records:
        lines.append("|"+"|".join([str(row["layer"]),row["module"],str(row["rate"]),
                                    str(row["cosine_boundary_informative"])] +
                      [fmt(row[k]) for k in ("spearman","top_overlap_fraction","coverage_cosine",
                                             "coverage_residual","coverage_random_mean")])+"|")
    (output/"summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


@torch.no_grad()
def run(args):
    from transformers import AutoTokenizer
    from rate_aligned_src import elastic_sparsegpt as pruning_kernel
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("diagnostic output must be new or empty")
    args.output.mkdir(parents=True,exist_ok=True)
    device = torch.device("cuda:0")
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model,use_fast=False)
    model = _load_model(args,tokenizer)
    if not args.layers or any(i<0 or i>=len(model.model.layers) for i in args.layers):
        raise ValueError("invalid diagnostic layer indices")
    _write_json(args.output/"protocol.json",dict(
        model=args.model, mode=args.mode, structured_alpha=args.structured_alpha,
        nsamples=args.nsamples,seqlen=args.seqlen,seed=args.seed,layers=args.layers,
        selected_fraction=.1,random_repeats=20,gpu=args.gpu,
        reference="fixed ordinary common-dense-H nested references; actual four independent forward paths",
        residual="||(W_dense-W_sparse_r)(x_r-x20)||_2^2 at each actual linear input",
        cosine="original block-boundary max over r=30/40/50 cosine distance to x20; same mask across modules",
        aggregation="none across modules or branches; each module/rate reported separately",
        task="diagnostic only: no target reconstruction or PPL; no change to OBS",
        normalization="raw squared residual; no denominator or epsilon changes to residual ranking",
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__),Path(__file__).with_name("token_residual_diagnostics.py"),
                                 Path(__file__).with_name("checkpoint_token_isolation_core.py"),
                                 Path(__file__).with_name("run_checkpoint_token_isolation.py"),
                                 Path(__file__).with_name("experiment_common.py"),
                                 Path(pruning_kernel.__file__).resolve()]}))
    model.model.embed_tokens.to(device)
    try:
        initial,attention,positions = _load_calibration_inputs(args,model,tokenizer,device)
    finally:
        model.model.embed_tokens.to("cpu")
    paths={r:initial for r in RATES}
    dense_paths=initial
    cache=CheckpointCache(args.cache,memory_reference=True)
    records,execution=[],[]
    for index,layer in enumerate(model.model.layers):
        tick=time.perf_counter()
        layer.to(device)
        dense_paths,_=prepare_reference_layer(args,layer,index,dense_paths,attention,positions,cache,device)
        synchronize(device)
        preparation_seconds=time.perf_counter()-tick
        torch.cuda.reset_peak_memory_stats(device)
        paths,scores,cosine,trace=forward_diagnostic_layer(
            layer,paths,{r:cache.source(index,r) for r in RATES},cache.source(index,None),
            attention,positions,device,diagnose=index in args.layers,chunk_size=args.chunk_size)
        trace.update(layer=index,reference_preparation_seconds=preparation_seconds,
                     collector_peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(device))
        if index in args.layers:
            stats_started=time.perf_counter()
            for rate,modules in scores.items():
                for name,score in modules.items():
                    summary=compare_selectors(score,cosine,seed=args.seed+index)
                    summary.update(layer=index,module=name,rate=rate,
                                   cosine_boundary_informative=index!=0 and not summary["cosine_score_constant"])
                    records.append(summary)
            torch.save(dict(residual_scores=scores,boundary_cosine=cosine,
                            cosine_top_ids=top_ids(cosine),
                            residual_top_ids={r:{n:top_ids(s) for n,s in groups.items()}
                                              for r,groups in scores.items()}),
                       args.output/f"scores_layer_{index:03d}.pt")
            _write_json(args.output/"diagnostics.json",records)
            write_report(args.output,records)
            trace["cpu_ranking_and_score_save_seconds"]=time.perf_counter()-stats_started
        execution.append(trace)
        _write_json(args.output/"execution.json",execution)
        cache.release_reference(index)
        del scores,cosine
        layer.to("cpu")
        gc.collect()
        print("LAYER_COMPLETE "+json.dumps(trace),flush=True)
    _write_json(args.output/"completion.json",dict(status="completed",layer_count=len(execution),
                diagnostic_layers=args.layers,record_count=len(records),
                total_wall_seconds=time.perf_counter()-started,
                ppl_evaluated=False))
    print("TOKEN_RESIDUAL_DIAGNOSTIC_COMPLETED",flush=True)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",default="/data/ModelZoo/LLM/llama2/Llama-2-7b-hf")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--gpu",default="1")
    parser.add_argument("--layers",type=int,nargs="+",default=[0,16,31])
    parser.add_argument("--nsamples",type=int,default=64)
    parser.add_argument("--seqlen",type=int,default=4096)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--percdamp",type=float,default=.01)
    parser.add_argument("--chunk-size",type=int,default=512)
    parser.add_argument("--attn-implementation",default="sdpa",choices=("sdpa","eager"))
    args=parser.parse_args(argv)
    if not args.gpu.strip() or "," in args.gpu or args.gpu.strip()=="-1":
        parser.error("exactly one GPU is required")
    if args.nsamples<1 or args.seqlen<2 or args.chunk_size<1 or len(set(args.layers))!=len(args.layers):
        parser.error("invalid calibration shape, chunk size or duplicate layer")
    args.mode="2of4"
    args.structured_alpha=.25
    args.unstructured_alpha=1.
    args.device_map="cpu"
    return args


if __name__=="__main__":
    args=parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    for key in ("HF_HUB_OFFLINE","HF_DATASETS_OFFLINE","TRANSFORMERS_OFFLINE"):
        os.environ.setdefault(key,"1")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(args.seed)
    run(args)
