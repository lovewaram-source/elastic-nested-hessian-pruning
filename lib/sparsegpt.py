import hashlib
import math
import time

import torch
import torch.nn as nn
import transformers
from .nm import build_nm_prune_mask

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

## SparseGPT: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
class SparseGPT:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self,
        sparsity,
        prune_n=0,
        prune_m=0,
        blocksize=128,
        percdamp=.01,
        compensate=True,
        compensation_alpha=1.0,
        mask_before_compensation=False,
        precomputed_mask=None,
        return_diagnostics=False,
        return_mask=False,
    ):
        if compensation_alpha < 0.0 or compensation_alpha > 1.0:
            raise ValueError("compensation_alpha must be in [0, 1]")
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        # Keep the diagnostic snapshot on CPU to avoid doubling GPU weight memory.
        W_before = W.detach().cpu() if return_diagnostics else None

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        frozen_mask = None
        if precomputed_mask is not None:
            frozen_mask = precomputed_mask.to(device=self.dev, dtype=torch.bool)
            expected_shape = (self.rows, self.columns)
            if tuple(frozen_mask.shape) != expected_shape:
                raise ValueError(
                    f"precomputed mask shape {tuple(frozen_mask.shape)} "
                    f"does not match {expected_shape}"
                )
        elif prune_n != 0 and mask_before_compensation:
            frozen_mask = torch.zeros(
                (self.rows, self.columns), dtype=torch.bool, device=self.dev
            )
            hdiag = torch.diag(Hinv)
            for start in range(0, self.columns, prune_m):
                end = min(start + prune_m, self.columns)
                if end - start != prune_m:
                    raise ValueError("2:4 mask requires columns divisible by prune_m")
                scores = W[:, start:end] ** 2 / (
                    hdiag[start:end].reshape((1, -1))
                ) ** 2
                frozen_mask[:, start:end] = build_nm_prune_mask(
                    scores, prune_n, prune_m
                )

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0:
                # A precomputed mask is authoritative for strict coefficient
                # ablations. Previously this branch ignored frozen_mask for
                # unstructured pruning and silently selected a new mask.
                if frozen_mask is not None:
                    mask1 = frozen_mask[:, i1:i2]
                elif mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            elif frozen_mask is not None:
                mask1 = frozen_mask[:, i1:i2]
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and frozen_mask is None and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1[:, i:(i + prune_m)] = build_nm_prune_mask(tmp, prune_n, prune_m)

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                if compensate and compensation_alpha != 0.0:
                    err1 = (w - q) / d
                    err1 = compensation_alpha * err1
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            if compensate and compensation_alpha != 0.0:
                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()

        diagnostics = {}
        if return_diagnostics or return_mask:
            mask = W == 0
            W_cpu = W.detach().cpu()
            mask_cpu = mask.detach().cpu()
            if return_mask:
                schedule_mask = (
                    frozen_mask if frozen_mask is not None else mask
                ).detach().cpu()
            if return_diagnostics:
                sparse_only = W_before.masked_fill(mask_cpu, 0)
                compensation_delta = W_cpu - sparse_only
                mask_hash = hashlib.sha256(
                    mask_cpu.contiguous().numpy().tobytes()
                ).hexdigest()
                eps = torch.finfo(W.dtype).eps
                diagnostics = {
                    "mask_hash": mask_hash,
                    "mask_zeros": int(mask.sum().item()),
                    "weight_numel": int(W.numel()),
                    "weight_norm": float(W_before.norm().item()),
                    "pruned_delta_norm": float((sparse_only - W_before).norm().item()),
                    "compensation_delta_norm": float(compensation_delta.norm().item()),
                    "total_delta_norm": float((W_cpu - W_before).norm().item()),
                    "compensation_relative_norm": float(
                        compensation_delta.norm().item() / (W_before.norm().item() + float(eps))
                    ),
                }
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if return_mask:
            diagnostics["_mask"] = schedule_mask
        if return_diagnostics or return_mask:
            return diagnostics

    def free(self):
        self.H = None
        torch.cuda.empty_cache()







