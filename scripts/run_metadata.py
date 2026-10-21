"""Early argument checks and automatically captured experiment provenance."""
import hashlib
from importlib import metadata
import math
from pathlib import Path
import platform
import sys


def validate_run_arguments(args):
    for name in ("unstructured_alpha", "structured_alpha"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    if not math.isfinite(args.percdamp) or args.percdamp <= 0:
        raise ValueError("percdamp must be finite and positive")
    if args.eval_samples is not None and args.eval_samples < 1:
        raise ValueError("eval-samples must be positive")
    if any(not math.isclose(r * 100, round(r * 100), abs_tol=1e-8, rel_tol=0)
           for r in args.rates):
        raise ValueError("rates must be integer percentages to avoid checkpoint name collisions")
    if args.mode == "2of4":
        allowed = (.2, .3, .4, .5)
        if tuple(args.rates) != allowed[:len(args.rates)]:
            raise ValueError("2of4 rates must be a consecutive prefix of 0.2,0.3,0.4,0.5")


def evaluation_record(args, evaluation):
    windows = evaluation.input_ids.numel() // args.seqlen
    return dict(scope="full_available_windows" if args.eval_samples is None else "truncated_debug",
                requested_windows=args.eval_samples, actual_windows=windows,
                sequence_length=args.seqlen, evaluated_next_token_targets=windows * (args.seqlen - 1),
                dropped_tail_tokens=evaluation.input_ids.numel() % args.seqlen)


def runtime_record(args, model, evaluation):
    import torch
    from rate_aligned_src.data import DATASET_RECORDS
    from export_github import ROOT, select_files

    versions = {}
    for package in ("torch", "transformers", "datasets", "accelerate", "numpy", "sentencepiece", "protobuf"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    config = model.config.to_dict()
    runtime_sources = select_files(ROOT, seeds=["scripts/run_checkpoint_token_isolation.py",
                                               "scripts/run_sorting_nested.py"])
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
               for name in runtime_sources if name.endswith(".py")}
    model_path = Path(args.model)
    config_file = model_path / "config.json"
    return dict(python=sys.version, platform=platform.platform(), packages=versions,
                cuda_version=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                datasets=dict(DATASET_RECORDS), evaluation=evaluation_record(args, evaluation),
                model=dict(path=str(model_path.resolve()), config=config,
                           config_sha256=hashlib.sha256(config_file.read_bytes()).hexdigest()
                           if config_file.is_file() else None,
                           revision=getattr(model.config, "_commit_hash", None),
                           weights_identity="local weights not content-hashed; preserve model artifact separately"),
                public_python_source_sha256=sources)
