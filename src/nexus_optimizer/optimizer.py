"""Nexus optimizer from arXiv:2604.09258v1.

The paper's practical algorithm keeps two copies of the model:

* ``model`` is updated by an arbitrary outer optimizer (AdamW, SGD, Muon, ...).
* ``inner_model`` is updated after every microbatch by memory-free normalized
  SGD (NSGD).

At an accumulation boundary the displacement is converted to a gradient for
the outer optimizer.  The paper writes ``g_hat = inner_model - model`` and
``outer.step(grad=-g_hat)``.  Consequently the gradient handed to a normal
PyTorch optimizer is exactly ``model - inner_model``.  Keeping this sign
explicit is important: passing the displacement itself would turn the update
into ascent.

The implementation uses per-parameter normalization, as specified in the
engineering adaptation (Algorithm 3, Sec. 3.3), rather than normalizing one
flattened vector over the whole model.  It does not add momentum or weight
decay to the inner loop; those belong to the outer optimizer.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Tuple

import torch
from torch import nn


class _GradientBucket:
    """Reusable flat reduction buffer for a fixed parameter bucket.

    Autograd gives the inner model one gradient tensor per parameter.  Building
    a fresh ``torch.cat`` for every Nexus microbatch makes a large allocation
    and synchronization point on the critical path.  This bucket keeps one
    flat buffer and shaped views alive, so each microbatch only copies the
    newly produced gradients into storage that is then all-reduced in place.
    """

    def __init__(self, params: Iterable[nn.Parameter], flat: Optional[torch.Tensor] = None) -> None:
        self.params = tuple(params)
        if not self.params:
            raise ValueError("gradient buckets cannot be empty")
        device = self.params[0].device
        dtype = self.params[0].dtype
        if any(p.device != device or p.dtype != dtype for p in self.params):
            raise ValueError("gradient bucket parameters must share device and dtype")
        numel = sum(p.numel() for p in self.params)
        if flat is None:
            flat = torch.empty(numel, device=device, dtype=dtype)
        elif flat.numel() != numel or flat.device != device or flat.dtype != dtype:
            raise ValueError("reusable gradient storage does not match bucket")
        self.flat = flat
        self.views = []
        offset = 0
        for p in self.params:
            size = p.numel()
            self.views.append(self.flat[offset : offset + size].view_as(p))
            offset += size


def _make_gradient_buckets(
    params: Iterable[nn.Parameter],
    storage: Optional[Iterable[torch.Tensor]] = None,
) -> Tuple[_GradientBucket, ...]:
    """Create deterministic, device/dtype-homogeneous reusable grad buckets."""

    grouped: Dict[Tuple[torch.device, torch.dtype], list[nn.Parameter]] = {}
    for param in params:
        grouped.setdefault((param.device, param.dtype), []).append(param)
    storage_values = None if storage is None else tuple(storage)
    if storage_values is not None and len(storage_values) != len(grouped):
        raise ValueError("reusable gradient storage bucket count does not match parameters")
    return tuple(
        _GradientBucket(entries, None if storage_values is None else storage_values[index])
        for index, entries in enumerate(grouped.values())
    )


def _trainable_pairs(model: nn.Module, inner_model: nn.Module) -> Iterator[Tuple[nn.Parameter, nn.Parameter]]:
    """Yield corresponding trainable parameters and validate model structure."""

    model_params = OrderedDict((name, p) for name, p in model.named_parameters())
    inner_params = OrderedDict((name, p) for name, p in inner_model.named_parameters())
    if tuple(model_params) != tuple(inner_params):
        raise ValueError("model and inner_model must have identical parameter names")
    for name, p in model_params.items():
        inner_p = inner_params[name]
        if p.shape != inner_p.shape or p.dtype != inner_p.dtype:
            raise ValueError(f"parameter mismatch for {name}: {tuple(p.shape)}/{p.dtype} vs "
                             f"{tuple(inner_p.shape)}/{inner_p.dtype}")
        if p.requires_grad != inner_p.requires_grad:
            raise ValueError(f"requires_grad mismatch for {name}")
        if p.requires_grad:
            yield p, inner_p


@torch.no_grad()
def _copy_module_state(dst: nn.Module, src: nn.Module) -> None:
    """Copy parameters and buffers without reallocating the inner module."""

    src_state = src.state_dict()
    dst_state = dst.state_dict()
    if tuple(src_state) != tuple(dst_state):
        raise ValueError("model and inner_model must have identical state_dict keys")
    for name, value in src_state.items():
        target = dst_state[name]
        if target.shape != value.shape or target.dtype != value.dtype:
            raise ValueError(f"state mismatch for {name}")
        target.copy_(value)


@torch.no_grad()
def _copy_state_pairs(
    pairs: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> None:
    """Copy a validated list of destination/source tensors in foreach batches."""

    _copy_state_buckets(_make_state_buckets(pairs))


def _make_state_buckets(
    pairs: Iterable[Tuple[torch.Tensor, torch.Tensor]],
) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor], ...], ...]:
    """Build immutable device/dtype buckets for repeated state copies."""

    buckets: Dict[Tuple[torch.device, torch.dtype], list[Tuple[torch.Tensor, torch.Tensor]]] = {}
    for dst, src in pairs:
        if dst.shape != src.shape or dst.dtype != src.dtype:
            raise ValueError("model and inner_model state tensors must match")
        buckets.setdefault((dst.device, dst.dtype), []).append((dst, src))
    return tuple(tuple(entries) for entries in buckets.values())


@torch.no_grad()
def _copy_state_buckets(
    buckets: Iterable[Iterable[Tuple[torch.Tensor, torch.Tensor]]],
) -> None:
    """Copy prevalidated state buckets with one foreach launch per bucket."""

    for entries in buckets:
        entries = tuple(entries)
        if not entries:
            continue
        torch._foreach_copy_([dst for dst, _ in entries], [src for _, src in entries])


@torch.no_grad()
def _average_gradients(
    params: Iterable[nn.Parameter],
    *,
    world_size: int,
    all_reduce: Any,
    process_group: Optional[Any] = None,
) -> None:
    """Average gradients with one collective per device/dtype bucket.

    The cloned inner model is outside DDP, and the outer model's forward is
    also bypassed by Algorithm 3.  Both paths therefore need an explicit
    reduction.  Concatenating compatible gradients avoids launching one NCCL
    collective for every parameter tensor, which is a significant latency cost
    for a transformer with thousands of tensors.
    """

    buckets: Dict[
        Tuple[torch.device, torch.dtype], list[Tuple[nn.Parameter, Optional[torch.Tensor]]]
    ] = {}
    for param in params:
        grad = param.grad
        # Include unused parameters with a zero placeholder.  Every rank must
        # execute the same collectives in the same order; skipping a local None
        # gradient can deadlock when another rank used that parameter.
        key = (param.device, param.dtype) if grad is None else (grad.device, grad.dtype)
        buckets.setdefault(key, []).append((param, grad))
    for entries in buckets.values():
        if len(entries) == 1 and entries[0][1] is not None:
            flat = entries[0][1]
            kwargs = {"op": torch.distributed.ReduceOp.SUM}
            if process_group is not None:
                kwargs["group"] = process_group
            all_reduce(flat, **kwargs)
            flat.div_(world_size)
            continue
        # The zero placeholder is local to this rank.  If another rank has a
        # real gradient, reduction fills it in and we install that result into
        # this rank's ``param.grad`` below.
        flat = torch.cat(
            [
                (grad.detach() if grad is not None else torch.zeros_like(param))
                .reshape(-1)
                for param, grad in entries
            ],
            dim=0,
        )
        kwargs = {"op": torch.distributed.ReduceOp.SUM}
        if process_group is not None:
            kwargs["group"] = process_group
        all_reduce(flat, **kwargs)
        flat.div_(world_size)
        offset = 0
        for param, _grad in entries:
            size = param.numel()
            # Keep the reduced flat storage alive through the parameter grad
            # views instead of copying it back into every autograd tensor.
            # ``zero_grad(set_to_none=True)`` releases these views after the
            # inner/outer update.  This preserves unused-parameter handling
            # while removing hundreds of device-to-device copy kernels.
            param.grad = flat[offset : offset + size].view_as(param)
            offset += size


@torch.no_grad()
def _average_gradient_buckets(
    buckets: Iterable[_GradientBucket],
    *,
    world_size: int,
    all_reduce: Any,
    process_group: Optional[Any] = None,
) -> bool:
    """Reduce gradients through reusable flat buffers.

    Returns ``False`` when a gradient has a device or dtype different from its
    parameter, allowing the caller to use the general path above.  Missing
    gradients are represented by zeros in the reusable buffer so all ranks
    still participate in the same collectives.
    """

    kwargs = {"op": torch.distributed.ReduceOp.SUM}
    if process_group is not None:
        kwargs["group"] = process_group
    # Validate all buckets before reducing any of them.  Otherwise a late
    # dtype/device mismatch could make the fallback path reduce an earlier
    # bucket twice.
    buckets = tuple(buckets)
    for bucket in buckets:
        for param in bucket.params:
            grad = param.grad
            if grad is not None and (grad.is_sparse or grad.device != param.device or grad.dtype != param.dtype):
                return False
    for bucket in buckets:
        active_views = []
        active_grads = []
        has_missing = False
        # In persistent mode autograd writes directly into the reusable flat
        # views.  Avoid copying each gradient back onto the same storage.
        already_packed = True
        for param, view in zip(bucket.params, bucket.views):
            grad = param.grad
            if grad is None:
                has_missing = True
                already_packed = False
                continue
            if grad is not view:
                already_packed = False
            active_views.append(view)
            active_grads.append(grad)
        if has_missing:
            bucket.flat.zero_()
        if active_views and not already_packed:
            # One multi-tensor copy replaces one cat allocation and the many
            # individual D2D copies needed to refill a persistent buffer.
            torch._foreach_copy_(active_views, active_grads)
            # The autograd-owned gradient tensors are no longer needed after
            # the copy.  Drop their parameter references before the collective
            # so the reusable flat buffer does not raise the peak footprint by
            # another full model-sized gradient allocation.
            for param in bucket.params:
                param.grad = None
            del active_grads
        all_reduce(bucket.flat, **kwargs)
        bucket.flat.div_(world_size)
        for param, view in zip(bucket.params, bucket.views):
            param.grad = view
    return True


class NormalizedSGD:
    """Memory-free per-parameter normalized SGD used by Nexus.

    Each trainable tensor receives ``-lr * grad / max(||grad||_2, eps)``.  The
    norm is accumulated in FP32 to keep BF16/FP16 inner trajectories stable.
    No momentum, weight decay, or optimizer state is used by design.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float,
        eps: float = 1e-12,
        *,
        foreach: bool = True,
    ):
        if lr < 0:
            raise ValueError("lr must be non-negative")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.params = list(params)
        self.lr = float(lr)
        self.eps = float(eps)
        self.foreach = bool(foreach)

    @torch.no_grad()
    def step(self) -> None:
        active = [p for p in self.params if p.grad is not None]
        if self.foreach and active:
            # Foreach launches one norm/update batch per device and dtype,
            # avoiding a Python/CUDA launch for every transformer parameter.
            # The gradients are disposable.  Keep the normalized update in a
            # single multi-tensor addcdiv kernel instead of launching separate
            # divide, multiply, and add passes over every parameter.
            buckets: Dict[Tuple[torch.device, torch.dtype], list[nn.Parameter]] = {}
            for p in active:
                if p.grad.is_sparse:
                    raise RuntimeError("Nexus NormalizedSGD does not support sparse gradients")
                buckets.setdefault((p.device, p.dtype), []).append(p)
            for params in buckets.values():
                grads = [p.grad for p in params]
                # Foreach norm preserves the input dtype.  Cast only the
                # reductions to FP32 so BF16/FP16 norms match the reference
                # ``vector_norm(grad.float())`` without storing FP32 grads.
                norms = torch._foreach_norm([g.float() for g in grads], 2.0)
                # Avoid allocating one scalar scale tensor per parameter:
                # addcdiv fuses the division and parameter update while
                # preserving the FP32 norm calculation.
                torch._foreach_clamp_min_(norms, self.eps)
                torch._foreach_addcdiv_(params, grads, norms, value=-self.lr)
            return
        for p in self.params:
            grad = p.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError("Nexus NormalizedSGD does not support sparse gradients")
            # Compute the norm in FP32, but keep the update in parameter dtype.
            norm = torch.linalg.vector_norm(grad.detach().float())
            scale = self.lr / norm.clamp_min(self.eps)
            p.add_(grad, alpha=0.0) if self.lr == 0 else p.add_(grad * scale.to(grad.dtype), alpha=-1.0)

    @torch.no_grad()
    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.zero_()


class NexusOptimizer:
    """Algorithm 3 Nexus wrapper around a standard outer optimizer.

    Parameters
    ----------
    model:
        The model whose parameters are updated by ``outer_optimizer``.
    outer_optimizer:
        A normal PyTorch optimizer built from ``model.parameters()``.  Its
        moments and weight decay are retained exactly; Nexus only supplies its
        gradients at accumulation boundaries.
    inner_lr:
        NSGD learning rate used for every microbatch.
    assume_synchronized_inner:
        Opt-in distributed fast path. When every rank starts with identical
        model and outer-optimizer state and inner gradients are globally
        averaged after each microbatch, the outer pseudo-gradient is already
        identical on every rank and its second all-reduce can be omitted.
        Leave this false for general distributed use.
    inner_model:
        Optional pre-created clone.  Supplying one is useful with wrappers
        such as FSDP, where ``copy.deepcopy`` may be unsupported.  It must have
        the same parameter and buffer structure as ``model``.
    """

    def __init__(
        self,
        model: nn.Module,
        outer_optimizer: torch.optim.Optimizer,
        inner_lr: float,
        *,
        inner_model: Optional[nn.Module] = None,
        norm_eps: float = 1e-12,
        sync_gradients: bool = True,
        process_group: Optional[Any] = None,
        persistent_inner_grads: bool = False,
        assume_synchronized_inner: bool = False,
    ) -> None:
        if inner_lr < 0:
            raise ValueError("inner_lr must be non-negative")
        self.model = model
        self._outer_module = getattr(model, "module", model)
        if not isinstance(self._outer_module, nn.Module):
            raise TypeError("model must be a torch.nn.Module or DDP wrapper")
        self.outer = outer_optimizer
        self.inner_model = copy.deepcopy(self._outer_module) if inner_model is None else inner_model
        # Preserve frozen parameters.  Calling ``module.requires_grad_(True)``
        # here would silently unfreeze embeddings or tied auxiliary weights and
        # make the inner backward both slower and semantically different.
        model_named = dict(self._outer_module.named_parameters())
        for name, inner_p in self.inner_model.named_parameters():
            if name in model_named:
                inner_p.requires_grad_(model_named[name].requires_grad)
        self.inner_model.train(self._outer_module.training)
        # Cache parameter/buffer correspondence once.  Rebuilding two full
        # state_dicts at every outer boundary is measurable for a transformer;
        # the cached pairs also allow the synchronization copy to use one
        # foreach launch per device/dtype bucket.
        outer_state = OrderedDict(self._outer_module.named_parameters())
        outer_state.update((name, value) for name, value in self._outer_module.named_buffers())
        inner_state = OrderedDict(self.inner_model.named_parameters())
        inner_state.update((name, value) for name, value in self.inner_model.named_buffers())
        if tuple(outer_state) != tuple(inner_state):
            raise ValueError("model and inner_model must have identical parameter/buffer names")
        self._state_pairs = []
        for name, source in outer_state.items():
            target = inner_state[name]
            if target.shape != source.shape or target.dtype != source.dtype:
                raise ValueError(f"state mismatch for {name}")
            self._state_pairs.append((target, source))
        self._state_buckets = _make_state_buckets(self._state_pairs)
        # Validate correspondence before constructing the inner optimizer.
        self._pairs = list(_trainable_pairs(self._outer_module, self.inner_model))
        # Keep one flat reduction buffer per device/dtype bucket.  The buffer
        # avoids allocating a new 100M-element ``torch.cat`` result for every
        # synchronized microbatch while preserving the same one-collective
        # reduction topology and exact Algorithm 3 trajectory.
        self._gradient_buckets = _make_gradient_buckets(inner_p for _, inner_p in self._pairs)
        # Reuse storage for the outer model-minus-inner pseudo-gradient too.
        # This avoids a second full-model ``torch.cat`` allocation at each
        # accumulation boundary while preserving its exact values.
        self._outer_gradient_buckets = _make_gradient_buckets(
            (p for p, _ in self._pairs),
            storage=(bucket.flat for bucket in self._gradient_buckets),
        )
        self._inner_for_outer = {id(p): inner_p for p, inner_p in self._pairs}
        model_ids = {id(p) for group in outer_optimizer.param_groups for p in group["params"]}
        missing = [name for name, p in self._outer_module.named_parameters() if p.requires_grad and id(p) not in model_ids]
        if missing:
            raise ValueError("outer optimizer is missing model parameters: " + ", ".join(missing[:8]))
        self.inner = NormalizedSGD((q for _, q in self._pairs), inner_lr, norm_eps)
        # The cloned inner module is intentionally not wrapped in DDP.  When
        # training the outer model with DDP, explicitly average each inner
        # micro-batch gradient before its NSGD update so every rank follows the
        # same trajectory.  This is required for the displacement to represent
        # the globally averaged mini-batch gradient rather than rank-local
        # noise.  The flag also makes single-process and custom distributed
        # setups easy to control.
        self.sync_gradients = bool(sync_gradients)
        self.process_group = process_group
        self.persistent_inner_grads = bool(persistent_inner_grads)
        # If every inner gradient is globally averaged and all ranks start
        # with identical model/optimizer state, the model-inner pseudo-gradient
        # is identical on every rank and its second all-reduce is redundant.
        # This remains opt-in because custom collectives, stochastic rank-local
        # state, or divergent optimizer state can violate that invariant.
        self.assume_synchronized_inner = bool(assume_synchronized_inner)
        self._inner_gradients_ready = False
        self.microbatches = 0

    @property
    def inner_lr(self) -> float:
        return self.inner.lr

    def train(self, mode: bool = True) -> "NexusOptimizer":
        self._outer_module.train(mode)
        if self.model is not self._outer_module:
            self.model.train(mode)
        self.inner_model.train(mode)
        return self

    def zero_inner_grad(self, set_to_none: bool = True) -> None:
        self.inner.zero_grad(set_to_none=set_to_none)
        self._inner_gradients_ready = False
        if self.persistent_inner_grads and not set_to_none:
            self.prepare_inner_gradients()

    @torch.no_grad()
    def prepare_inner_gradients(self) -> None:
        """Attach reusable flat views as inner gradients before backward.

        This is an opt-in allocation optimization.  Autograd accumulates
        directly into the bucket views, so the distributed inner reduction can
        all-reduce those same buffers without a per-microbatch ``cat`` or
        foreach copy.  The default mode retains ordinary autograd-owned
        gradients and the public ``zero_grad`` behavior.
        """

        if not self.persistent_inner_grads:
            return
        for bucket in self._gradient_buckets:
            bucket.flat.zero_()
            for param, view in zip(bucket.params, bucket.views):
                param.grad = view
        self._inner_gradients_ready = True

    def inner_step(self) -> None:
        """Apply one NSGD update after a microbatch backward pass."""

        if self.persistent_inner_grads and not self._inner_gradients_ready:
            # Manual callers may invoke ``loss.backward(); inner_step()``
            # without the convenience ``microbatch_step`` wrapper.  Preserve
            # an already-computed first gradient; later microbatches are
            # prepared by the previous call before their backward pass.
            if not any(param.grad is not None for param in self.inner.params):
                self.prepare_inner_gradients()
        self._sync_inner_gradients()
        self.inner.step()
        if self.persistent_inner_grads:
            # The gradient objects are already the persistent bucket views;
            # clear the flat storage in one operation instead of walking every
            # parameter to set ``grad=None`` and then reattaching it.
            self._inner_gradients_ready = False
            self.prepare_inner_gradients()
        else:
            self.inner.zero_grad(set_to_none=True)
        self.microbatches += 1

    @torch.no_grad()
    def _sync_inner_gradients(self) -> None:
        """Average cloned-inner gradients across initialized DDP ranks."""

        if not self.sync_gradients:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        world = (
            torch.distributed.get_world_size()
            if self.process_group is None
            else torch.distributed.get_world_size(group=self.process_group)
        )
        if not _average_gradient_buckets(
            self._gradient_buckets,
            world_size=world,
            all_reduce=torch.distributed.all_reduce,
            process_group=self.process_group,
        ):
            _average_gradients(
                (inner_p for _, inner_p in self._pairs),
                world_size=world,
                all_reduce=torch.distributed.all_reduce,
                process_group=self.process_group,
            )

    def microbatch_step(self, loss: torch.Tensor) -> None:
        """Backpropagate ``loss`` through the inner model and advance NSGD.

        The caller must compute the loss using ``inner_model`` (for example,
        ``loss = nexus.inner_model(batch).loss``).  This explicit API prevents
        accidentally updating the main model during the inner loop.
        """

        if loss.ndim != 0:
            raise ValueError("loss must be a scalar")
        if self.persistent_inner_grads and not self._inner_gradients_ready:
            self.prepare_inner_gradients()
        loss.backward()
        self.inner_step()

    @torch.no_grad()
    def _set_outer_pseudograd(self) -> None:
        if self.microbatches <= 0:
            raise RuntimeError("Nexus outer_step requires at least one inner microbatch")
        # Algorithm 3: g_hat = inner - model; outer.step(grad=-g_hat).
        # Therefore p.grad is model - inner.  Allocate an ordinary tensor so
        # outer optimizers are free to mutate it during their update.
        for bucket in self._outer_gradient_buckets:
            for p, view in zip(bucket.params, bucket.views):
                view.copy_(p).sub_(self._inner_for_outer[id(p)])
        if (
            self.sync_gradients
            and not self.assume_synchronized_inner
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            world = (
                torch.distributed.get_world_size()
                if self.process_group is None
                else torch.distributed.get_world_size(group=self.process_group)
            )
            kwargs = {"op": torch.distributed.ReduceOp.SUM}
            if self.process_group is not None:
                kwargs["group"] = self.process_group
            for bucket in self._outer_gradient_buckets:
                torch.distributed.all_reduce(bucket.flat, **kwargs)
                bucket.flat.div_(world)
        for bucket in self._outer_gradient_buckets:
            for p, view in zip(bucket.params, bucket.views):
                p.grad = view

    def outer_step(self, closure=None) -> Any:
        """Apply the outer optimizer to the Nexus pseudo-gradient.

        After the update the inner model is synchronized in-place with the new
        main model, exactly as ``inner_model <- model.clone()`` in Algorithm 3.
        The return value is whatever the wrapped optimizer returns.
        """

        if closure is not None:
            raise ValueError(
                "Nexus outer_step does not accept an optimizer closure: the closure's backward "
                "would replace the Nexus pseudo-gradient"
            )
        self._set_outer_pseudograd()
        with torch.no_grad():
            result = self.outer.step()
        self.outer.zero_grad(set_to_none=True)
        _copy_state_buckets(self._state_buckets)
        self.microbatches = 0
        self.zero_inner_grad(set_to_none=True)
        return result

    def step(self, closure=None) -> Any:
        """Alias for :meth:`outer_step` for optimizer-like call sites."""

        return self.outer_step(closure=closure)

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Clear both outer and inner gradients."""

        self.outer.zero_grad(set_to_none=set_to_none)
        self.zero_inner_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def sync_inner(self) -> None:
        """Discard a partial inner trajectory and resynchronize it to ``model``."""

        _copy_state_buckets(self._state_buckets)
        self.microbatches = 0
        self.zero_inner_grad(set_to_none=True)

    def state_dict(self) -> Dict[str, Any]:
        """Return outer optimizer state plus the partial inner trajectory."""

        return {
            "outer": self.outer.state_dict(),
            "inner": self.inner_model.state_dict(),
            "microbatches": self.microbatches,
            "inner_lr": self.inner_lr,
            "norm_eps": self.inner.eps,
            "foreach": self.inner.foreach,
            "assume_synchronized_inner": self.assume_synchronized_inner,
            "persistent_inner_grads": self.persistent_inner_grads,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore a Nexus checkpoint, including an in-progress accumulation."""

        if "outer" not in state_dict or "inner" not in state_dict:
            raise ValueError("Nexus checkpoint must contain 'outer' and 'inner' entries")
        self.outer.load_state_dict(state_dict["outer"])
        self.inner_model.load_state_dict(state_dict["inner"])
        # ``inner_lr`` and ``norm_eps`` are part of the serialized optimizer
        # state, rather than constructor-only metadata.  Restoring them is
        # necessary for an interrupted accumulation to continue with exactly
        # the same NSGD trajectory after the caller recreates the wrapper.
        if "inner_lr" in state_dict:
            inner_lr = float(state_dict["inner_lr"])
            if inner_lr < 0:
                raise ValueError("checkpoint contains a negative inner_lr")
            self.inner.lr = inner_lr
        if "norm_eps" in state_dict:
            norm_eps = float(state_dict["norm_eps"])
            if norm_eps <= 0:
                raise ValueError("checkpoint contains a non-positive norm_eps")
            self.inner.eps = norm_eps
        if "foreach" in state_dict:
            self.inner.foreach = bool(state_dict["foreach"])
        if "assume_synchronized_inner" in state_dict:
            self.assume_synchronized_inner = bool(state_dict["assume_synchronized_inner"])
        if "persistent_inner_grads" in state_dict:
            self.persistent_inner_grads = bool(state_dict["persistent_inner_grads"])
        self.microbatches = int(state_dict.get("microbatches", 0))
        if self.microbatches < 0:
            raise ValueError("checkpoint contains a negative microbatch count")
        self.zero_grad(set_to_none=True)
