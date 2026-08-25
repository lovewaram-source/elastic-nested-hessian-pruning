"""CPU-memory and disk stores for reusable Elastic SparseGPT plans."""

import os
import shutil
import tempfile

import torch

from .elastic_sparsegpt import ElasticSparseGPTPlan


def _serialize(plans):
    serialized = {}
    for name, plan in plans.items():
        tensors = {
            "permutation": plan.permutation.detach().cpu(),
            "dead_columns": plan.dead_columns.detach().cpu(),
        }
        if plan.hessian_factor is not None:
            tensors["hessian_factor"] = plan.hessian_factor.detach().float().cpu()
        if plan.activation_norms is not None:
            tensors["activation_norms"] = plan.activation_norms.detach().float().cpu()
        if plan.hessian_inverse is not None:
            tensors["hessian_inverse"] = plan.hessian_inverse.detach().float().cpu()
        if plan.processed_columns is not None:
            tensors["processed_columns"] = plan.processed_columns.detach().cpu()
        serialized[name] = tensors
    return serialized


def _deserialize(payload):
    return {
        name: ElasticSparseGPTPlan(
            permutation=tensors["permutation"],
            hessian_factor=tensors.get("hessian_factor"),
            dead_columns=tensors["dead_columns"],
            activation_norms=tensors.get("activation_norms"),
            hessian_inverse=tensors.get("hessian_inverse"),
            processed_columns=tensors.get("processed_columns"),
        )
        for name, tensors in payload.items()
    }


class MemoryElasticPlanStore:
    def __init__(self):
        self._layers = {}

    def put_layer(self, layer_index, plans):
        if layer_index in self._layers:
            raise ValueError(f"layer {layer_index} already exists")
        self._layers[layer_index] = _deserialize(_serialize(plans))

    def get_layer(self, layer_index):
        if layer_index not in self._layers:
            raise KeyError(f"missing elastic plan layer {layer_index}")
        return self._layers[layer_index]

    def replace_layer(self, layer_index, plans):
        if layer_index not in self._layers:
            raise KeyError(f"missing elastic plan layer {layer_index}")
        self._layers[layer_index] = _deserialize(_serialize(plans))

    def close(self, success):
        self._layers.clear()


class DiskElasticPlanStore:
    def __init__(self, base_dir=None, keep_on_success=False):
        if base_dir is None:
            base_dir = tempfile.gettempdir()
        os.makedirs(base_dir, exist_ok=True)
        self.run_dir = tempfile.mkdtemp(
            prefix="elastic-plan-",
            dir=base_dir,
        )
        self._layer_indices = set()
        self.keep_on_success = keep_on_success

    def _layer_path(self, layer_index):
        return os.path.join(self.run_dir, f"layer_{layer_index:03d}.pt")

    def put_layer(self, layer_index, plans):
        if layer_index in self._layer_indices:
            raise ValueError(f"layer {layer_index} already exists")
        torch.save(_serialize(plans), self._layer_path(layer_index))
        self._layer_indices.add(layer_index)

    def get_layer(self, layer_index):
        if layer_index not in self._layer_indices:
            raise KeyError(f"missing elastic plan layer {layer_index}")
        payload = torch.load(
            self._layer_path(layer_index),
            map_location="cpu",
            weights_only=True,
        )
        return _deserialize(payload)

    def replace_layer(self, layer_index, plans):
        if layer_index not in self._layer_indices:
            raise KeyError(f"missing elastic plan layer {layer_index}")
        torch.save(_serialize(plans), self._layer_path(layer_index))

    def close(self, success):
        if success and not self.keep_on_success:
            shutil.rmtree(self.run_dir)
        else:
            print(f"retaining elastic plan cache at {self.run_dir}")


def make_elastic_plan_store(mode, cache_dir=None, keep_on_success=False):
    if mode == "memory":
        return MemoryElasticPlanStore()
    if mode == "disk":
        return DiskElasticPlanStore(
            cache_dir,
            keep_on_success=keep_on_success,
        )
    raise ValueError(f"unsupported elastic cache mode: {mode}")
