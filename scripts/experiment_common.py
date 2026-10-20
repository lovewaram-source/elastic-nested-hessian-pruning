"""Shared I/O, calibration, evaluation and input-group helpers for Hessian experiments."""
import csv
import json
import math
import os
from pathlib import Path


class _CaughtFirstLayer(Exception):
    pass


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


def _append_tsv(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row), delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _eval_ppl(model, eval_loader, input_device):
    """Evaluate the same WikiText2 next-token loss while supporting sharded models."""
    import torch
    import torch.nn.functional as F

    token_ids = eval_loader.input_ids
    samples = token_ids.numel() // model.seqlen
    if samples < 1:
        raise ValueError("evaluation data contains no complete sequence window")
    total_nll = 0.0
    with torch.no_grad():
        for index in range(samples):
            inputs = token_ids[:, index * model.seqlen : (index + 1) * model.seqlen].to(input_device)
            outputs = model(inputs, use_cache=False)
            logits = outputs.logits
            labels = inputs[:, 1:].to(logits.device)
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                labels.contiguous().view(-1),
            )
            total_nll += float(loss.float().item()) * model.seqlen
            del inputs, outputs, logits, labels, loss
    return math.exp(total_nll / (samples * model.seqlen))


def _layer_device(model, layer_index, fallback):
    import torch

    mapped = getattr(model, "hf_device_map", {}).get(
        f"model.layers.{layer_index}"
    )
    if mapped is None:
        return fallback
    if isinstance(mapped, int):
        return torch.device(f"cuda:{mapped}")
    if isinstance(mapped, str) and mapped.isdigit():
        return torch.device(f"cuda:{mapped}")
    return torch.device(mapped)


def _capture_first_layer_inputs(model, dataloader, nsamples, seqlen, device):
    import torch
    import torch.nn as nn

    layers = model.model.layers
    inputs = torch.zeros(
        (nsamples, seqlen, model.config.hidden_size),
        dtype=next(model.parameters()).dtype,
        device="cpu",
    )
    cache = {"index": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, hidden_states, **kwargs):
            inputs[cache["index"]].copy_(hidden_states[0].detach().cpu())
            cache["index"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            raise _CaughtFirstLayer

    original = layers[0]
    layers[0] = Catcher(original)
    try:
        for batch in dataloader:
            try:
                model(batch[0].to(device))
            except _CaughtFirstLayer:
                pass
            if cache["index"] >= nsamples:
                break
    finally:
        layers[0] = original
    if cache["index"] != nsamples:
        raise RuntimeError(
            f"captured {cache['index']} calibration inputs; expected {nsamples}"
        )
    if inputs.shape[1] != seqlen:
        raise AssertionError("captured calibration sequence length changed")
    return inputs, cache["attention_mask"], cache["position_ids"]


def _layer_forward_cpu(layer, inputs, attention_mask, position_ids, device):
    import torch

    outputs = torch.empty_like(inputs, device="cpu")
    mask = None if attention_mask is None else attention_mask.to(device)
    positions = None if position_ids is None else position_ids.to(device)
    for index in range(inputs.shape[0]):
        kwargs = {}
        if mask is not None:
            kwargs["attention_mask"] = mask
        if positions is not None:
            kwargs["position_ids"] = positions
        result = layer(inputs[index : index + 1].to(device), **kwargs)
        hidden = result[0] if isinstance(result, (tuple, list)) else result
        outputs[index].copy_(hidden[0].detach().cpu())
    return outputs


def input_groups(modules):
    groups = {
        "self_attn.q_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
        "self_attn.o_proj": ("self_attn.o_proj",),
        "mlp.gate_proj": ("mlp.gate_proj", "mlp.up_proj"),
        "mlp.down_proj": ("mlp.down_proj",),
    }
    if {name for members in groups.values() for name in members} != set(modules):
        raise ValueError("this runner requires the seven standard Llama linear projections")
    return groups


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_gate(path,expected_rule=None):
    rows=json.loads(Path(path).read_text())
    if len(rows)!=2 or [r["implementation"] for r in rows]!=["direct","cached_direct"]:
        return False,"incomplete benchmark"
    fast=rows[1]
    if not fast.get("hessian_errors") or not fast.get("target_errors"):
        return False,"missing numerical checks"
    if expected_rule is not None:
        if any(r.get("selection_rule")!=expected_rule for r in rows):
            return False,"wrong selector benchmark"
        names=("self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj",
               "mlp.gate_proj","mlp.up_proj","mlp.down_proj")
        expected={(n,r) for n in names for r in (.2,.3,.4,.5)}
        for field in ("hessian_errors","target_errors"):
            values=fast[field]
            if len(values)!=28 or {(r.get("module"),r.get("rate")) for r in values}!=expected:
                return False,"incomplete module/rate numerical coverage"
    if any(r["max_abs"]!=0 for r in fast["hessian_errors"]):
        return False,"Hessian not bitwise equal"
    if any(r["mask_disagreements"]!=0 or r["max_abs"]!=0 for r in fast["target_errors"]):
        return False,"target masks or values not bitwise equal"
    return True,"bitwise Hessian and target checks passed"
