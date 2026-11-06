# Nexus Optimizer

A production-ready PyTorch implementation of the Nexus optimizer from
[arXiv:2604.09258](https://arxiv.org/abs/2604.09258). Nexus keeps a cloned
inner model and applies per-parameter normalized SGD after every microbatch;
at an accumulation boundary, the displacement becomes the gradient for an
ordinary outer optimizer such as AdamW.

The implementation is designed for BF16 DDP training. It keeps the paper's
model dimensions and update equations intact, batches gradient reductions into
flat device/dtype buckets, and reuses one flat buffer for the inner and outer
collectives. The package has no dependency on a particular model stack.

## Install

```bash
pip install -e .
# for tests
pip install -e '.[dev]'
```

## Use in a training loop

```python
import torch
import torch.distributed as dist
from nexus_optimizer import NexusOptimizer

# model is already placed on the local CUDA device.
outer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.03
)
nexus = NexusOptimizer(
    model,
    outer,
    inner_lr=0.10,
    sync_gradients=dist.is_initialized(),
    process_group=dist.group.WORLD if dist.is_initialized() else None,
)

for batch_group in loader:              # one outer step / accumulation window
    nexus.zero_grad(set_to_none=True)
    for inputs, targets in batch_group:  # each item is one microbatch
        loss = loss_fn(nexus.inner_model(inputs), targets)
        nexus.microbatch_step(loss)
    nexus.outer_step()
```

`microbatch_step` performs backward on `inner_model`, globally averages the
inner gradient when DDP is initialized, applies normalized SGD, and clears the
inner gradients. `outer_step` constructs the exact Algorithm 3 pseudo-gradient
`model - inner_model`, calls the wrapped optimizer, and synchronizes the clone.
The outer optimizer remains a normal PyTorch optimizer, so its moments and
weight decay are unchanged.

For an explicit benchmark and a real WikiText-103 training stream:

```bash
python examples/benchmark_decoder_100m.py \
  --optimizer nexus-adamw --ffn-width 2150 --device cuda \
  --batch 2 --seq 512 --accumulation 4 \
  --train-data /path/to/wikitext103_tokenized/train \
  --eval-data /path/to/validation-00000-of-00001.npy \
  --warmup 10 --steps 50 --eval-batches 32
```

The same program can be launched across two hosts with one process per host:

```bash
torchrun --nnodes=2 --nproc-per-node=1 --node-rank=$RANK \
  --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT \
  examples/benchmark_decoder_100m.py \
  --optimizer nexus-adamw --ffn-width 2150 --device cuda \
  --batch 2 --seq 512 --accumulation 4 --warmup 10 --steps 50
```

The training example uses the full 99,883,776-parameter decoder when
`--ffn-width 2150` is selected. `--train-data` is optional; without it the
example uses deterministic synthetic tokens for implementation smoke tests.

## Distributed fast paths

The default path performs every reduction required by Algorithm 3. The
persistent flat gradient buckets are enabled by the implementation and reuse
one BF16 buffer across the inner gradient and outer pseudo-gradient phases.
They add about 190 MiB at this benchmark size and improve throughput by about
1.8% on the measured GB300 pair.

`--nexus-assume-synchronized-inner` is an opt-in DDP fast path. It omits the
outer pseudo-gradient all-reduce only when every rank starts with identical
model and outer-optimizer state, every inner gradient is globally averaged, and
rank-local computation and collectives are deterministic. Under those
conditions the pseudo-gradient is already identical on every rank. Keep the
flag off for general distributed use.

`persistent_inner_grads=True` is an experimental API that attaches autograd
views to the reusable buffers. It preserves the trajectory but did not improve
end-to-end throughput in the measured 100M DDP case, so it is not the default.

## Measured results

All records below use the same 99.88M BF16 decoder, WikiText-103, one GB300
on each of `gemini1` and `gemini2`, batch 2 per rank, sequence length 512,
four accumulation microbatches, 10 warmup steps, and 50 timed steps.

| Implementation | Step | Global tok/s | Validation CE | Peak VRAM |
|---|---:|---:|---:|---:|
| Previous Nexus baseline | 57.332 ms | 142,886 | 6.325 | 2,645 MiB |
| Persistent shared bucket | 56.306 ms | 145,492 | 6.330 | 2,835 MiB |
| Shared bucket + synchronized-inner flag | 51.173 ms | **160,084** | 6.330 | 2,835 MiB |

The synchronized-inner result is an opt-in result with strict preconditions;
the default full-reduction implementation is the shared-bucket row. A second
50-step repeat of the opt-in path measured 51.781 ms / 158,204 tok/s. Raw JSON
records and the optimization notes are in [`benchmarks/`](benchmarks/) and
[`docs/research.md`](docs/research.md).

## Verify

```bash
pytest -q
python -m py_compile src/nexus_optimizer/*.py examples/*.py
```

The current test suite covers the Algorithm 3 sign, per-parameter normalized
updates, unused gradients, bucket packing and storage reuse, checkpoint
restoration, persistent-view parity, and the synchronized-inner collective
invariant.

## Scope and limitations

Nexus requires an inner update after every microbatch. Accumulating local inner
grads with DDP `no_sync()` changes the trajectory and is not used. The
synchronized-inner fast path assumes synchronized replicas; the safe default
retains the outer all-reduce. A one-block custom CUDA NSGD prototype was
profiled as an idea but was not included because its reduction order and
end-to-end performance were not validated against the FP32-norm reference.
