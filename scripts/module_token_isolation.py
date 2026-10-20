"""Module/rate-specific top-10% isolation, with true fixed reference paths."""
import hashlib
import math
import time
from collections.abc import Mapping

import torch

from checkpoint_token_isolation_core import RATES, load_block_weights
from run_rate_aligned_nested import _find_layers
from experiment_common import input_groups, synchronize
from token_residual_diagnostics import block_cosine_scores, forward_diagnostic_layer, top_ids


def select_module_masks(names, shape, rule, scores=None, cosine=None, seed=0):
    total = math.prod(shape)
    k = math.ceil(total*.1)
    generator = torch.Generator().manual_seed(seed)
    masks, evidence = {}, {}
    for rate in RATES[1:]:
        masks[rate], evidence[str(rate)] = {}, {}
        for name in names:
            if rule == "module_residual":
                ids = top_ids(scores[rate][name])
            elif rule == "module_random":
                ids = torch.randperm(total, generator=generator)[:k]
            elif rule == "module_cosine":
                ids = top_ids(cosine)
            else:
                raise ValueError("unknown module isolation rule")
            mask = torch.zeros(total, dtype=torch.bool)
            mask[ids] = True
            masks[rate][name] = mask.reshape(shape)
            evidence[str(rate)][name] = dict(private_tokens=k,total_tokens=total,
                position_sha256=hashlib.sha256(ids.sort().values.numpy().tobytes()).hexdigest())
    return masks, dict(rule=rule, private_fraction=.1,seed=seed,per_rate_module=evidence,
                       anchor20="all actual own inputs; independent of routing mask")


@torch.no_grad()
def collect_masked_module_hessians(layer, paths, sources, masks, attention, positions,
                                   device, capture_outputs=True, implementation="direct"):
    if implementation not in ("direct","delta","cached_direct"):
        raise ValueError("unknown Hessian accumulation implementation")
    modules = _find_layers(layer)
    groups = input_groups(modules)
    group_of = {n:g for g,members in groups.items() for n in members}
    count,length = paths[.2].shape[:2]
    for rate in RATES[1:]:
        for name in modules:
            mask = masks[rate][name]
            if mask.shape != (count,length) or mask.dtype != torch.bool or int(mask.sum()) != math.ceil(count*length*.1):
                raise ValueError("every module/rate must isolate exactly ceil(10% of positions)")
    anchors, anchor_h, branch_h, outputs = {}, {}, {}, {}
    for group in groups:
        width = modules[group].weight.shape[1]
        anchor_h[group] = torch.zeros(width,width,device=device,dtype=torch.float32)
    indices = {r:{n:[row.nonzero().flatten() for row in masks[r][n]] for n in modules}
               for r in RATES[1:]}
    gpu_indices = {r:{n:[ids.to(device) for ids in rows] for n,rows in names.items()}
                   for r,names in indices.items()}
    owners = {r:{n:n for n in modules} for r in RATES[1:]}
    if implementation == "cached_direct":
        for rate in RATES[1:]:
            for group,members in groups.items():
                for j,name in enumerate(members):
                    for earlier in members[:j]:
                        if torch.equal(masks[rate][name],masks[rate][earlier]):
                            owners[rate][name] = owners[rate][earlier]
                            break
    current = dict(rate=None,sample=None,seen={},anchor_union={},full_anchor={})
    kwargs = {}
    if attention is not None:
        kwargs["attention_mask"] = attention.to(device)
    if positions is not None:
        kwargs["position_ids"] = positions.to(device)

    def hook_for(name):
        def hook(_,inputs,output):
            value = inputs[0].detach()
            group,rate,sample = group_of[name],current["rate"],current["sample"]
            identity = (value.data_ptr(),tuple(value.shape),value.stride())
            if name == group:
                current["seen"][group] = identity
            elif current["seen"].get(group) != identity:
                raise AssertionError("shared-input groups no longer aligned")
            if rate == .2:
                if name != group:
                    return
                if group not in anchors:
                    anchors[group] = torch.empty(count,length,value.shape[-1],dtype=value.dtype)
                anchors[group][sample].copy_(value[0].cpu())
                x = value[0].float().t()
                anchor_h[group].addmm_(x,x.t(),alpha=2/count)
            else:
                if owners[rate][name] != name:
                    return
                ids = gpu_indices[rate][name][sample]
                if implementation in ("direct","cached_direct"):
                    if implementation == "cached_direct":
                        if group not in current["full_anchor"]:
                            current["full_anchor"][group] = anchors[group][sample].to(device,dtype=torch.float32,copy=True)
                        mixed = current["full_anchor"][group].clone()
                    else:
                        mixed = anchors[group][sample].to(device,dtype=torch.float32,copy=True)
                    mixed.index_copy_(0,ids,value[0].index_select(0,ids).float())
                    x = mixed.t()
                    branch_h[name][rate].addmm_(x,x.t(),alpha=2/count)
                else:
                    # Upload the union of selected anchor rows once per input
                    # group/sample. No full-sequence transfer or mixed Gram.
                    if group not in current["anchor_union"]:
                        union = torch.stack([masks[rate][n][sample] for n in groups[group]]).any(0).nonzero().flatten()
                        selected = anchors[group][sample].index_select(0,union).to(device).float()
                        current["anchor_union"][group] = (union,selected)
                    union,selected = current["anchor_union"][group]
                    local_ids = torch.searchsorted(union,indices[rate][name][sample]).to(device)
                    anchor = selected.index_select(0,local_ids).t()
                    own = value[0].index_select(0,ids).float().t()
                    branch_h[name][rate].addmm_(own,own.t(),alpha=2/count)
                    branch_h[name][rate].addmm_(anchor,anchor.t(),alpha=-2/count)
        return hook

    handles = [m.register_forward_hook(hook_for(n)) for n,m in modules.items()]
    try:
        for rate in RATES:
            load_block_weights(layer,sources[rate])
            current["rate"] = rate
            if rate != .2:
                for name,module in modules.items():
                    branch_h.setdefault(name,{})
                    if owners[rate][name] == name:
                        branch_h[name][rate] = torch.zeros(
                            module.weight.shape[1],module.weight.shape[1],device=device,dtype=torch.float32)
                    else:
                        branch_h[name][rate] = branch_h[owners[rate][name]][rate]
            branch = paths.pop(rate)
            if capture_outputs:
                outputs[rate] = torch.empty_like(branch,device="cpu")
            for sample in range(count):
                current["sample"] = sample
                current["seen"].clear()
                current["anchor_union"].clear()
                current["full_anchor"].clear()
                result = layer(branch[sample:sample+1].to(device),**kwargs)
                if capture_outputs:
                    hidden = result[0] if isinstance(result,(tuple,list)) else result
                    outputs[rate][sample].copy_(hidden[0].detach().cpu())
            del branch
    finally:
        for handle in handles:
            handle.remove()
    if implementation == "delta":
        for name in modules:
            for rate in RATES[1:]:
                branch_h[name][rate].add_(anchor_h[group_of[name]])
    hessians = {n:{.2:anchor_h[group_of[n]],**branch_h[n]} for n in modules}
    return hessians,outputs,dict(forward_counts={str(r):count for r in RATES},
        selection_scope="per module and per rate",input_reuse_checked=True,
        fusion="10% actual own input, 90% actual anchor20 input",
        propagation="no replacement of real reference outputs",hessian_implementation=implementation,
        accumulation=("H20 + sum_selected(own Gram - anchor Gram), FP32; mathematically equivalent, reordered sums"
                      if implementation == "delta" else "full mixed input Gram; original summation order"),
        shared_H_owners={str(r):names for r,names in owners.items()})


@torch.no_grad()
def collect_module_isolation(layer,paths,sources,dense_state,attention,positions,device,rule,seed,
                             implementation="direct"):
    started = time.perf_counter()
    names = tuple(_find_layers(layer))
    shape = paths[.2].shape[:2]
    first_outputs = None
    scoring_trace = None
    if rule == "module_residual":
        if not isinstance(dense_state,Mapping):
            dense_state = torch.load(dense_state,map_location="cpu",weights_only=True)
        first_outputs,scores,_,scoring_trace = forward_diagnostic_layer(
            layer,dict(paths),sources,dense_state,attention,positions,device,diagnose=True)
        masks,selection = select_module_masks(names,shape,rule,scores=scores,seed=seed)
        del scores
    else:
        cosine = block_cosine_scores(paths,device) if rule == "module_cosine" else None
        masks,selection = select_module_masks(names,shape,rule,cosine=cosine,seed=seed)
    synchronize(device)
    selection_seconds = time.perf_counter()-started
    hessians,outputs,trace = collect_masked_module_hessians(
        layer,paths,sources,masks,attention,positions,device,capture_outputs=first_outputs is None,
        implementation=implementation)
    if first_outputs is not None:
        outputs = first_outputs
    trace.update(rule=rule,scoring_pass=scoring_trace,
                 selection_including_scoring_seconds=selection_seconds,
                 hessian_passes=1,scoring_passes=1 if scoring_trace else 0)
    return hessians,outputs,trace,masks,selection
