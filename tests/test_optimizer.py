import copy
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from nexus_optimizer import NexusOptimizer, NormalizedSGD  # noqa: E402
from nexus_optimizer.optimizer import (  # noqa: E402
    _average_gradient_buckets,
    _average_gradients,
    _make_gradient_buckets,
)


def test_normalized_sgd_is_per_parameter_and_memory_free():
    a = nn.Parameter(torch.tensor([3.0, 4.0]))
    b = nn.Parameter(torch.tensor([2.0, 0.0]))
    a.grad = torch.tensor([3.0, 4.0])
    b.grad = torch.tensor([2.0, 0.0])
    NormalizedSGD([a, b], lr=0.5).step()
    # Both tensors have norm-one updates; a global flattened normalization
    # would produce a different result.
    assert torch.allclose(a, torch.tensor([2.7, 3.6]))
    assert torch.allclose(b, torch.tensor([1.5, 0.0]))


def test_algorithm3_pseudogradient_sign_and_sync():
    torch.manual_seed(0)
    model = nn.Linear(3, 1, bias=False)
    outer = torch.optim.SGD(model.parameters(), lr=0.25)
    nexus = NexusOptimizer(model, outer, inner_lr=0.1)
    start = model.weight.detach().clone()

    x = torch.tensor([[1.0, 2.0, -1.0]])
    # Build a loss on the inner model and run exactly one inner microbatch.
    inner_loss = (nexus.inner_model(x).sum() - 2.0) ** 2
    inner_loss.backward()
    grad = nexus.inner_model.weight.grad.detach().clone()
    normed = grad / grad.norm()
    nexus.inner_step()
    expected_inner = start - 0.1 * normed
    assert torch.allclose(nexus.inner_model.weight, expected_inner)

    # Algorithm 3 says outer receives model - inner, not inner - model.
    expected_grad = start - expected_inner
    nexus.outer_step()
    assert torch.allclose(model.weight, start - 0.25 * expected_grad)
    assert torch.allclose(model.weight, nexus.inner_model.weight)
    assert nexus.microbatches == 0


def test_outer_pseudogradient_reuses_inner_bucket_storage():
    model = nn.Sequential(nn.Linear(3, 2, bias=False), nn.Linear(2, 1, bias=False))
    nexus = NexusOptimizer(model, torch.optim.SGD(model.parameters(), lr=0.25), inner_lr=0.1)
    assert len(nexus._gradient_buckets) == len(nexus._outer_gradient_buckets)
    for inner_bucket, outer_bucket in zip(nexus._gradient_buckets, nexus._outer_gradient_buckets):
        assert inner_bucket.flat.data_ptr() == outer_bucket.flat.data_ptr()


def test_inner_gradient_sync_hook_averages_when_distributed(monkeypatch):
    model = nn.Sequential(nn.Linear(2, 1, bias=False), nn.Linear(1, 1, bias=False))
    nexus = NexusOptimizer(model, torch.optim.SGD(model.parameters(), lr=0.1), inner_lr=0.01)
    for parameter in nexus.inner_model.parameters():
        parameter.grad = torch.ones_like(parameter)
    calls = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def fake_all_reduce(tensor, op=None):
        calls.append((tensor.clone(), op))
        tensor.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    nexus._sync_inner_gradients()
    assert len(calls) == 1  # compatible tensors are packed into one collective
    for parameter in nexus.inner_model.parameters():
        assert torch.allclose(parameter.grad, torch.ones_like(parameter))


def test_average_gradients_collects_unused_parameters(monkeypatch):
    p_used = nn.Parameter(torch.zeros(2))
    p_unused = nn.Parameter(torch.zeros(3))
    p_used.grad = torch.ones(2)
    calls = []

    monkeypatch.setattr(torch.distributed, "ReduceOp", type("ReduceOp", (), {"SUM": 0}))

    def fake_all_reduce(tensor, op=None):
        calls.append(tensor.numel())
        # Simulate another rank contributing one to every element.
        tensor.add_(1)

    _average_gradients([p_used, p_unused], world_size=2, all_reduce=fake_all_reduce)
    assert calls == [5]
    assert torch.allclose(p_used.grad, torch.full((2,), 1.0))
    assert torch.allclose(p_unused.grad, torch.full((3,), 0.5))


def test_reusable_gradient_bucket_matches_average(monkeypatch):
    p_used = nn.Parameter(torch.zeros(2))
    p_unused = nn.Parameter(torch.zeros(3))
    p_used.grad = torch.ones(2)
    buckets = _make_gradient_buckets([p_used, p_unused])
    calls = []

    monkeypatch.setattr(torch.distributed, "ReduceOp", type("ReduceOp", (), {"SUM": 0}))

    def fake_all_reduce(tensor, op=None):
        calls.append(tensor.numel())
        # Simulate another rank contributing one to every element.
        tensor.add_(1)

    assert _average_gradient_buckets(
        buckets,
        world_size=2,
        all_reduce=fake_all_reduce,
    )
    assert calls == [5]
    assert torch.allclose(p_used.grad, torch.full((2,), 1.0))
    assert torch.allclose(p_unused.grad, torch.full((3,), 0.5))


def test_persistent_inner_grad_views_match_regular_trajectory():
    torch.manual_seed(42)
    reference = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 2))
    regular_model = copy.deepcopy(reference)
    persistent_model = copy.deepcopy(reference)
    regular = NexusOptimizer(
        regular_model,
        torch.optim.SGD(regular_model.parameters(), lr=0.07),
        inner_lr=0.03,
        sync_gradients=False,
    )
    persistent = NexusOptimizer(
        persistent_model,
        torch.optim.SGD(persistent_model.parameters(), lr=0.07),
        inner_lr=0.03,
        sync_gradients=False,
        persistent_inner_grads=True,
    )
    batches = [torch.randn(3, 4), torch.randn(3, 4)]
    for optimizer, set_to_none in ((regular, True), (persistent, False)):
        optimizer.zero_grad(set_to_none=set_to_none)
        for batch in batches:
            # Exercise the same explicit backward/inner_step API as the DDP
            # benchmark, including the first persistent-view attachment.
            optimizer.inner_model(batch).sum().backward()
            optimizer.inner_step()
        optimizer.outer_step()
    for lhs, rhs in zip(regular_model.parameters(), persistent_model.parameters()):
        assert torch.equal(lhs, rhs)
    for lhs, rhs in zip(regular.inner_model.parameters(), persistent.inner_model.parameters()):
        assert torch.equal(lhs, rhs)


def test_assume_synchronized_inner_skips_only_outer_reduce(monkeypatch):
    model = nn.Linear(2, 2, bias=False)
    nexus = NexusOptimizer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.1),
        inner_lr=0.01,
        assume_synchronized_inner=True,
    )
    calls = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def fake_all_reduce(tensor, op=None):
        calls.append(tensor.numel())
        tensor.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    nexus.inner_model(torch.ones(1, 2)).sum().backward()
    nexus.inner_step()
    assert len(calls) == 1  # the required inner-gradient reduction
    nexus.outer_step()
    assert len(calls) == 1  # the opt-in flag omits only the redundant outer reduction


def test_checkpoint_restores_partial_inner_trajectory():
    torch.manual_seed(1)
    model = nn.Linear(2, 2)
    nexus = NexusOptimizer(model, torch.optim.SGD(model.parameters(), lr=0.2), inner_lr=0.03)
    loss = nexus.inner_model(torch.ones(1, 2)).sum()
    loss.backward()
    nexus.inner_step()
    state = nexus.state_dict()

    model2 = copy.deepcopy(model)
    nexus2 = NexusOptimizer(model2, torch.optim.SGD(model2.parameters(), lr=0.2), inner_lr=0.03)
    nexus2.load_state_dict(state)
    assert nexus2.microbatches == 1
    assert nexus2.inner_lr == nexus.inner_lr
    assert nexus2.inner.eps == nexus.inner.eps
    for p, q in zip(nexus.inner_model.parameters(), nexus2.inner_model.parameters()):
        assert torch.equal(p, q)


def test_checkpoint_restores_inner_hyperparameters():
    model = nn.Linear(2, 2)
    nexus = NexusOptimizer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.2),
        inner_lr=0.03,
        norm_eps=7e-7,
    )
    state = nexus.state_dict()
    model2 = copy.deepcopy(model)
    restored = NexusOptimizer(
        model2,
        torch.optim.SGD(model2.parameters(), lr=0.2),
        inner_lr=0.9,
        norm_eps=1e-3,
    )
    restored.load_state_dict(state)
    assert restored.inner_lr == 0.03
    assert restored.inner.eps == 7e-7


def test_checkpoint_restores_foreach_setting():
    model = nn.Linear(2, 2)
    nexus = NexusOptimizer(model, torch.optim.SGD(model.parameters(), lr=0.2), inner_lr=0.03)
    nexus.inner.foreach = False
    state = nexus.state_dict()
    model2 = copy.deepcopy(model)
    restored = NexusOptimizer(model2, torch.optim.SGD(model2.parameters(), lr=0.2), inner_lr=0.03)
    restored.load_state_dict(state)
    assert restored.inner.foreach is False
