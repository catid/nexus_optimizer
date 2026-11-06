#!/usr/bin/env python3
"""Matched ~100M decoder benchmark for AdamW, Muon, and Nexus.

Run one rank with ``python optimizer_experiments/benchmark_decoder_100m.py
--optimizer adamw`` or use ``torchrun`` for DDP.  The model, deterministic
tokens, precision, and timing window are shared across optimizer choices.
Use ``--train-data`` for the repository's real WikiText-103 next-token stream;
without it the benchmark uses deterministic random tokens for a fast smoke run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from nexus_optimizer import NexusOptimizer
except ImportError:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    from nexus_optimizer import NexusOptimizer


@dataclass(frozen=True)
class Distributed:
    rank: int
    world_size: int
    local_rank: int
    enabled: bool


def init_distributed(backend: str) -> Distributed:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world > 1
    if enabled:
        dist.init_process_group(backend=backend)
    return Distributed(rank, world, local, enabled)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class DecoderBlock(nn.Module):
    def __init__(self, width: int, ffn_width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.norm1 = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.norm2 = RMSNorm(width)
        self.up = nn.Linear(width, 2 * ffn_width, bias=False)
        self.down = nn.Linear(ffn_width, width, bias=False)
        self.heads = heads
        self.head_dim = width // heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, self.head_dim).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).contiguous().view(b, t, d))
        gate, value = self.up(self.norm2(x)).chunk(2, dim=-1)
        return x + self.down(F.silu(gate) * value)


class DecoderLM(nn.Module):
    """Tied-embedding decoder; defaults to 100.6M parameters."""

    def __init__(self, vocab=34_816, width=768, layers=10, ffn_width=3_200, heads=12, dtype=torch.bfloat16):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab, width, dtype=dtype)
        self.blocks = nn.ModuleList([DecoderBlock(width, ffn_width, heads) for _ in range(layers)])
        self.norm = RMSNorm(width)
        self.apply(self._init_weights)
        self.to(dtype=dtype)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_embeddings(ids)
        for block in self.blocks:
            x = block(x)
        return F.linear(self.norm(x), self.tok_embeddings.weight)


def matrix_partition(model: nn.Module):
    muon, adam = [], []
    for name, p in model.named_parameters():
        if p.ndim == 2 and "tok_embeddings" not in name:
            muon.append(p)
        else:
            adam.append(p)
    return muon, adam


@torch.no_grad()
def orthogonalize(grad: torch.Tensor, steps: int) -> torch.Tensor:
    transposed = grad.shape[0] < grad.shape[1]
    x = grad.float().t() if transposed else grad.float()
    x = x / x.norm().clamp_min(1e-7)
    for _ in range(steps):
        gram = x @ x.t()
        x = 3.4445 * x - 4.7750 * (gram @ x) + 2.0315 * (gram @ (gram @ x))
    return x.t() if transposed else x


class FastMuon(torch.optim.Optimizer):
    """Official fused Muon matrices plus AdamW for non-matrix parameters.

    PyTorch 2.14+ provides a multi-tensor Muon implementation.  Use it when
    available so the baseline reflects the fast production path; retain the
    compact reference implementation below for older torch builds.
    """

    def __init__(self, muon_params, adam_params, *, lr=3e-4, muon_lr=None, weight_decay=0.1, ns_steps=2, momentum=0.9, nesterov=False):
        self._official = hasattr(torch.optim, "Muon")
        self._adam_params = list(adam_params)
        self._muon_params = list(muon_params)
        if self._official:
            self.muon = torch.optim.Muon(
                self._muon_params,
                lr=float(muon_lr or lr),
                weight_decay=float(weight_decay),
                momentum=float(momentum),
                nesterov=bool(nesterov),
                ns_steps=int(ns_steps),
            )
            self.adam = torch.optim.AdamW(
                self._adam_params,
                lr=float(lr),
                betas=(0.9, 0.95),
                weight_decay=float(weight_decay),
            )
            return
        self.ns_steps = int(ns_steps)
        defaults = dict(lr=float(lr), betas=(float(momentum), 0.95), weight_decay=float(weight_decay))
        super().__init__([
            {"params": list(muon_params), "kind": "muon", "lr": float(muon_lr or lr)},
            {"params": list(adam_params), "kind": "adam", "lr": float(lr)},
        ], defaults)

    @torch.no_grad()
    def step(self, closure=None):
        if self._official:
            loss = self.muon.step()
            self.adam.step()
            return loss
        loss = closure() if closure is not None else None
        beta1, beta2 = self.defaults["betas"]
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                lr, decay = group["lr"], group["weight_decay"]
                g = p.grad.detach()
                state = self.state[p]
                if group["kind"] == "muon":
                    momentum = state.setdefault("momentum", torch.zeros_like(p))
                    momentum.mul_(beta1).add_(g, alpha=1 - beta1)
                    if decay:
                        p.mul_(1 - lr * decay)
                    scale = math.sqrt(max(1.0, p.shape[0] / max(1, p.shape[1])))
                    p.add_(orthogonalize(momentum, self.ns_steps).to(p.dtype), alpha=-lr * scale)
                else:
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    state["step"] += 1
                    if decay:
                        p.mul_(1 - lr * decay)
                    g = g.float()
                    m, v = state["exp_avg"], state["exp_avg_sq"]
                    m.mul_(beta1).add_(g, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    bias1, bias2 = 1 - beta1**state["step"], 1 - beta2**state["step"]
                    p.addcdiv_(m.to(p.dtype), v.sqrt().add_(1e-8).to(p.dtype), value=-lr * math.sqrt(bias2) / bias1)
        return loss

    def zero_grad(self, set_to_none: bool = True):
        if self._official:
            self.muon.zero_grad(set_to_none=set_to_none)
            self.adam.zero_grad(set_to_none=set_to_none)
            return
        return super().zero_grad(set_to_none=set_to_none)


def reduce_grads(model: nn.Module, world: int) -> None:
    if world == 1:
        return
    for p in model.parameters():
        if p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world)


def autocast_for(device, dtype):
    return torch.autocast("cuda", dtype=dtype) if device.type == "cuda" and dtype != torch.float32 else nullcontext()


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--optimizer", choices=("adamw", "muon", "nexus-adamw"), required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--vocab", type=int, default=34_816); p.add_argument("--width", type=int, default=768)
    p.add_argument("--layers", type=int, default=10); p.add_argument("--ffn-width", type=int, default=3_200)
    p.add_argument("--heads", type=int, default=12); p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seq", type=int, default=512); p.add_argument("--accumulation", type=int, default=1)
    p.add_argument(
        "--train-data",
        default="",
        help="directory containing tokenized WikiText .npy shards; empty uses deterministic random tokens",
    )
    p.add_argument(
        "--eval-data",
        default="",
        help="raw uint16 token shard for post-timing validation (for example validation-00000-of-00001.npy)",
    )
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--steps", type=int, default=20); p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--ddp-sync-every-microbatch",
        action="store_true",
        help="force DDP all-reduce after every accumulation microbatch (diagnostic; slower)",
    )
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--muon-lr", type=float, default=None); p.add_argument("--ns-steps", type=int, default=2)
    p.add_argument("--muon-momentum", type=float, default=0.9)
    p.add_argument("--muon-nesterov", action="store_true")
    p.add_argument("--nexus-inner-lr", type=float, default=0.05); p.add_argument("--seed", type=int, default=1234)
    p.add_argument(
        "--nexus-assume-synchronized-inner",
        action="store_true",
        help="skip Nexus outer pseudo-gradient all-reduce when all ranks are asserted identical",
    )
    p.add_argument(
        "--nexus-persistent-grads",
        action="store_true",
        help="attach inner gradients to reusable flat buffers before backward",
    )
    p.add_argument("--output", default=""); p.add_argument("--no-ddp", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.no_ddp:
        d = Distributed(0, 1, 0, False)
    else:
        d = init_distributed(args.backend)
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device(f"cuda:{d.local_rank}" if args.device == "cuda" else args.device)
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed)
    model = DecoderLM(args.vocab, args.width, args.layers, args.ffn_width, args.heads, dtype=dtype).to(device)
    params = sum(p.numel() for p in model.parameters())
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    elif args.optimizer == "muon":
        muon, adam = matrix_partition(model)
        optimizer = FastMuon(muon, adam, lr=args.lr, muon_lr=args.muon_lr, weight_decay=args.weight_decay, ns_steps=args.ns_steps, momentum=args.muon_momentum, nesterov=args.muon_nesterov)
    else:
        optimizer = NexusOptimizer(
            model,
            torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay),
            args.nexus_inner_lr,
            sync_gradients=d.enabled,
            process_group=dist.group.WORLD if d.enabled else None,
            assume_synchronized_inner=args.nexus_assume_synchronized_inner,
            persistent_inner_grads=args.nexus_persistent_grads,
        )
    nexus = optimizer if isinstance(optimizer, NexusOptimizer) else None
    if d.enabled and nexus is None:
        # ``static_graph`` and ``no_sync`` are not compatible in the PyTorch
        # reducer on all supported releases: ending a no-sync accumulation
        # cycle can trip the reducer's autograd-hook assertion.  Retain the
        # static-graph fast path when there is no accumulation, and use the
        # regular reducer for no-sync accumulation.
        ddp_kwargs = {
            "gradient_as_bucket_view": True,
            "static_graph": args.accumulation == 1 or args.ddp_sync_every_microbatch,
        }
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [d.local_rank]
        train_model = DDP(model, **ddp_kwargs)
    else:
        train_model = nexus.inner_model if nexus is not None else model
    gen = torch.Generator(device=device).manual_seed(args.seed + 10000 * d.rank)
    train_tokens = None
    data_pos = 0
    if args.train_data:
        # Stage real contiguous WikiText tokens once on each GPU.  This keeps
        # host I/O out of the optimizer timing while preserving a real LLM
        # next-token task.
        import glob
        import numpy as np

        shards = sorted(glob.glob(os.path.join(args.train_data, "*.npy")))
        if not shards:
            raise FileNotFoundError(f"no .npy token shards found under {args.train_data!r}")
        # These repository shards intentionally use a raw uint16 payload with
        # an ``.npy`` suffix (the sidecar metadata records the dtype), so they
        # are memory-mapped rather than passed through ``numpy.load``.
        arrays = [np.memmap(path, mode="r", dtype=np.uint16) for path in shards]
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("train token shards must be one-dimensional")
        tokens = np.concatenate([np.asarray(array) for array in arrays]).astype(np.int64, copy=False)
        required = args.batch * (args.seq + 1)
        if tokens.size < required + 1:
            raise ValueError("train token stream is shorter than one batch")
        train_tokens = torch.as_tensor(tokens, device=device, dtype=torch.long)
        data_pos = (d.rank * required) % (train_tokens.numel() - 1)

    def step():
        # Draw a fresh deterministic microbatch.  Reusing one batch would turn
        # this into a memorization test and make the loss number meaningless.
        def next_batch():
            nonlocal data_pos
            if train_tokens is None:
                ids = torch.randint(args.vocab, (args.batch, args.seq + 1), device=device, generator=gen)
                return ids[:, :-1], ids[:, 1:]
            count = args.batch * (args.seq + 1)
            end = data_pos + count
            if end <= train_tokens.numel():
                flat = train_tokens[data_pos:end]
            else:
                flat = torch.cat((train_tokens[data_pos:], train_tokens[: end - train_tokens.numel()]), dim=0)
            data_pos = (end - 1) % (train_tokens.numel() - 1)
            ids = flat.view(args.batch, args.seq + 1)
            return ids[:, :-1], ids[:, 1:]

        if nexus is not None:
            nexus.zero_grad(set_to_none=not args.nexus_persistent_grads)
            for _ in range(args.accumulation):
                x, y = next_batch()
                with autocast_for(device, dtype):
                    # NSGD normalizes each microbatch gradient; do not
                    # average or accumulate it before the inner step.
                    loss = F.cross_entropy(train_model(x).float().flatten(0, 1), y.flatten())
                loss.backward()
                # NexusOptimizer performs the bucketed inner all-reduce and
                # constructs/reduces the outer pseudo-gradient itself.
                nexus.inner_step()
            nexus.outer_step()
            # Nexus does not divide the microbatch loss: NSGD consumes each
            # microbatch independently, so this is already the CE value.
            return float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        for microbatch in range(args.accumulation):
            x, y = next_batch()
            # Accumulation can retain local gradients on all but the final
            # microbatch.  The final DDP backward reduces the accumulated
            # bucket once, which is mathematically identical to reducing every
            # microbatch separately after dividing the loss by accumulation.
            # Nexus intentionally does not use this path: its inner trajectory
            # requires a globally averaged gradient after each microbatch.
            sync_context = nullcontext()
            if (
                d.enabled
                and isinstance(train_model, DDP)
                and not args.ddp_sync_every_microbatch
                and microbatch < args.accumulation - 1
            ):
                sync_context = train_model.no_sync()
            with sync_context:
                with autocast_for(device, dtype):
                    loss = F.cross_entropy(train_model(x).float().flatten(0, 1), y.flatten()) / args.accumulation
                loss.backward()
        optimizer.step(); return float(loss.detach()) * args.accumulation

    for _ in range(args.warmup): step()
    if device.type == "cuda": torch.cuda.synchronize(device)
    if d.enabled: dist.barrier()
    times, last_loss = [], 0.0
    for _ in range(args.steps):
        if device.type == "cuda":
            start, end = torch.cuda.Event(True), torch.cuda.Event(True); start.record(); last_loss = step(); end.record(); end.synchronize(); elapsed = start.elapsed_time(end)
        else:
            started = time.perf_counter(); last_loss = step(); elapsed = (time.perf_counter() - started) * 1000
        if d.enabled: dist.barrier()
        times.append(float(elapsed))
    mean_ms = sum(times) / len(times)
    if d.enabled:
        t = torch.tensor(mean_ms, device=device, dtype=torch.float64); dist.all_reduce(t, op=dist.ReduceOp.MAX); mean_ms = float(t.item())
        t = torch.tensor(last_loss, device=device, dtype=torch.float64); dist.all_reduce(t); last_loss = float(t.item() / d.world_size)
    eval_loss = None
    if args.eval_data:
        import numpy as np

        eval_tokens = torch.as_tensor(
            # ``memmap`` is read-only; copy before handing it to torch to avoid
            # the non-writable NumPy warning and make the ownership explicit.
            np.asarray(np.memmap(args.eval_data, mode="r", dtype=np.uint16)).copy(),
            device=device,
            dtype=torch.long,
        )
        eval_model = model
        was_training = eval_model.training
        eval_model.eval()
        total = torch.zeros((), device=device, dtype=torch.float64)
        count = 0
        with torch.no_grad():
            for batch_index in range(max(1, args.eval_batches)):
                start = (batch_index * args.batch * (args.seq + 1)) % max(1, eval_tokens.numel() - args.seq - 1)
                flat = eval_tokens[start : start + args.batch * (args.seq + 1)]
                if flat.numel() < args.batch * (args.seq + 1):
                    flat = torch.cat((flat, eval_tokens[: args.batch * (args.seq + 1) - flat.numel()]))
                ids = flat.view(args.batch, args.seq + 1)
                with autocast_for(device, dtype):
                    value = F.cross_entropy(eval_model(ids[:, :-1]).float().flatten(0, 1), ids[:, 1:].flatten())
                total += value.double()
                count += 1
        eval_model.train(was_training)
        if d.enabled:
            dist.all_reduce(total, op=dist.ReduceOp.SUM)
            count *= d.world_size
        eval_loss = float(total.item() / count)
    result = {
        "schema": "decoder-100m-v1",
        "optimizer": args.optimizer,
        "parameters": params,
        "parameters_millions": params / 1e6,
        "dtype": args.dtype,
        "data_source": args.train_data or "synthetic-random",
        "eval_data": args.eval_data or None,
        "eval_batches": args.eval_batches if args.eval_data else 0,
        "batch_per_rank": args.batch,
        "sequence_length": args.seq,
        "gradient_accumulation": args.accumulation,
        "world_size": d.world_size,
        "global_tokens_per_step": d.world_size * args.batch * args.seq * args.accumulation,
        "mean_step_ms": mean_ms,
        "tokens_per_second_global": d.world_size * args.batch * args.seq * args.accumulation / (mean_ms / 1000),
        "tokens_per_second_per_rank": args.batch * args.seq * args.accumulation / (mean_ms / 1000),
        "warmup_steps": args.warmup,
        "timed_steps": args.steps,
        "seed": args.seed,
        "hyperparameters": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "muon_lr": args.muon_lr,
            "muon_momentum": args.muon_momentum,
            "muon_nesterov": args.muon_nesterov,
            "ns_steps": args.ns_steps,
            "nexus_inner_lr": args.nexus_inner_lr,
            "nexus_assume_synchronized_inner": args.nexus_assume_synchronized_inner,
            "nexus_persistent_grads": args.nexus_persistent_grads,
        },
        "last_loss": last_loss,
        "eval_loss": eval_loss,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
        "step_times_ms": times,
        "host": os.uname().nodename,
    }
    if d.rank == 0:
        print(json.dumps(result, sort_keys=True), flush=True)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f: json.dump(result, f, indent=2, sort_keys=True); f.write("\n")
    if d.enabled: dist.destroy_process_group()


if __name__ == "__main__":
    main()
