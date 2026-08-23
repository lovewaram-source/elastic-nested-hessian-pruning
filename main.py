import argparse
import os 
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from lib.prune import (
    apply_elastic_sparsegpt_stage,
    check_sparsity,
    is_dynamic_wanda_elastic_method,
    is_prefix_obs_wanda_elastic_method,
    is_elastic_sparsegpt_method,
    find_layers,
    prepare_elastic_sparsegpt,
    prune_ablate,
    prune_elastic_sparsegpt,
    prune_magnitude,
    prune_sparsegpt,
    prune_wanda,
    run_prepared_elastic_stages,
    validate_elastic_sparsegpt_stages,
)
from lib.elastic_plan_store import make_elastic_plan_store
from lib.eval import eval_ppl, eval_zero_shot
from lib.reporting import format_elastic_result_log, format_result_log

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True, 
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings 
    return model

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument(
        '--sparsegpt_blocksize',
        type=int,
        default=128,
        help='Column block size used only by the original SparseGPT pruner.',
    )
    parser.add_argument(
        '--elastic_sparsities',
        type=float,
        nargs='+',
        default=None,
        help='Increasing elastic_sparsegpt checkpoints chosen from 0.2 0.3 ... 0.8.',
    )
    parser.add_argument(
        '--elastic_cache_mode',
        choices=['memory', 'disk'],
        default='memory',
        help='Store shared elastic SparseGPT plans in CPU memory or on disk.',
    )
    parser.add_argument(
        '--elastic_cache_dir',
        type=str,
        default=None,
        help='Base directory for temporary disk-backed elastic plans.',
    )
    parser.add_argument(
        '--elastic_profile_preparation',
        action='store_true',
        help='Print a synchronized breakdown of elastic preparation time.',
    )
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", "elastic_sparsegpt", "elastic_wanda_sparsegpt", "elastic_wanda_dynamic_sparsegpt", "elastic_wanda_prefix_obs",
                        "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", "ablate_wanda_iter", "search"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str )
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')

    parser.add_argument("--eval_zero_shot", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if is_elastic_sparsegpt_method(args.prune_method) and args.elastic_sparsities:
        elastic_stages = validate_elastic_sparsegpt_stages(args.elastic_sparsities)
        if prune_n != 0 or prune_m != 0:
            raise ValueError("elastic SparseGPT methods support unstructured sparsity only")

        multiple_stages = len(elastic_stages) > 1
        plan_store = make_elastic_plan_store(
            args.elastic_cache_mode,
            args.elastic_cache_dir,
        )
        success = False
        preparation_breakdown = None

        def prepare():
            nonlocal preparation_breakdown
            preparation_breakdown = prepare_elastic_sparsegpt(
                args,
                model,
                tokenizer,
                device,
                plan_store,
            )
            return plan_store

        def apply(store, previous_sparsity, stage_sparsity):
            source = "dense (0.0)" if previous_sparsity == 0.0 else f"{previous_sparsity:.1f}"
            print(f"Pruning stage: {source} -> {stage_sparsity:.1f}")
            apply_elastic_sparsegpt_stage(
                model,
                device,
                store,
                stage_sparsity,
                previous_sparsity=previous_sparsity,
                dynamic_wanda=is_dynamic_wanda_elastic_method(args.prune_method),
                wanda_prefix_obs=is_prefix_obs_wanda_elastic_method(
                    args.prune_method
                ),
            )

        def on_stage(
            store,
            previous_sparsity,
            stage_sparsity,
            preparation_time_seconds,
            stage_pruning_time_seconds,
            cumulative_pruning_time_seconds,
        ):
            actual_sparsity = check_sparsity(model)
            print(f"sparsity sanity check {actual_sparsity:.4f}")
            ppl_test = eval_ppl(args, model, tokenizer, device)
            print(f"wikitext perplexity {ppl_test}")
            print(
                f"elastic preparation {preparation_time_seconds:.2f}s, "
                f"stage pruning {stage_pruning_time_seconds:.2f}s, "
                f"cumulative pruning {cumulative_pruning_time_seconds:.2f}s"
            )

            stage_tag = str(stage_sparsity).replace(".", "")
            stage_save = (
                os.path.join(args.save, f"s{stage_tag}")
                if multiple_stages
                else args.save
            )
            os.makedirs(stage_save, exist_ok=True)
            save_filepath = os.path.join(stage_save, f"log_{args.prune_method}.txt")
            with open(save_filepath, "w") as f:
                f.write(
                    format_elastic_result_log(
                        model_name=model_name,
                        method=args.prune_method,
                        sparsity_ratio=actual_sparsity,
                        perplexity=ppl_test,
                        preparation_time_seconds=preparation_time_seconds,
                        stage_pruning_time_seconds=stage_pruning_time_seconds,
                        cumulative_pruning_time_seconds=cumulative_pruning_time_seconds,
                        preparation_breakdown=preparation_breakdown,
                        previous_sparsity=previous_sparsity,
                        target_sparsity=stage_sparsity,
                    )
                )

            if args.save_model:
                stage_model_path = (
                    os.path.join(args.save_model, f"s{stage_tag}")
                    if multiple_stages
                    else args.save_model
                )
                model.save_pretrained(stage_model_path)
                tokenizer.save_pretrained(stage_model_path)

        try:
            run_prepared_elastic_stages(
                elastic_stages,
                prepare,
                apply,
                on_stage,
                synchronize=torch.cuda.synchronize,
            )
            success = True
        finally:
            plan_store.close(success=success)
        return

    pruning_time_seconds = 0.0
    if args.sparsity_ratio != 0:
        print("pruning starts")
        pruning_start = time.perf_counter()
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif is_elastic_sparsegpt_method(args.prune_method):
            prune_elastic_sparsegpt(
                args,
                model,
                tokenizer,
                device,
                prune_n=prune_n,
                prune_m=prune_m,
            )
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        pruning_time_seconds = time.perf_counter() - pruning_start
        print(f"pruning time {pruning_time_seconds:.2f} seconds")

    ################################################################
    print("*"*30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    ppl_test = eval_ppl(args, model, tokenizer, device)
    print(f"wikitext perplexity {ppl_test}")

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    with open(save_filepath, "w") as f:
        f.write(
            format_result_log(
                model_name=model_name,
                method=args.prune_method,
                sparsity_ratio=sparsity_ratio,
                perplexity=ppl_test,
                pruning_time_seconds=pruning_time_seconds,
            )
        )

    if args.eval_zero_shot:
        accelerate=False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

    if args.save_model:
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()
