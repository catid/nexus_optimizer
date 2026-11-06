# How the Nexus implementation was sped up

This note records the optimization work that produced the code in this
repository. The target was the Nexus Algorithm 3 trajectory from
[arXiv:2604.09258](https://arxiv.org/abs/2604.09258), with the model dimensions,
per-microbatch inner update, and numerical meaning of the update held fixed.
The measurements use a 99,883,776-parameter BF16 causal decoder trained on
WikiText-103 with one GB300 on each of `gemini1` and `gemini2`, batch 2 per rank,
sequence length 512, four microbatches per outer step, 10 warmup steps, and 50
timed steps.

## Algorithmic invariants

Let `theta` be the outer model and `phi` be its clone. For every microbatch,
Nexus computes the loss using `phi`, averages the resulting gradient across DDP
ranks, and applies the per-parameter update

```
phi <- phi - inner_lr * grad(phi) / max(||grad(phi)||_2, eps).
```

At the accumulation boundary, Algorithm 3 defines `g_hat = phi - theta` and
passes `-g_hat` to the outer optimizer. A conventional PyTorch optimizer
therefore receives exactly `theta - phi`. After the outer step, the clone is
copied from the updated outer model. These ordering and sign rules are tested
explicitly and are not changed by the optimizations below.

The norm is computed in FP32 even when parameters and gradients are BF16. The
normalization is per parameter tensor, rather than one norm over a flattened
model vector. The inner path has no momentum or weight decay.

## Where the time went

The first DDP implementation concatenated every parameter gradient into a new
full-model tensor after every inner microbatch, reduced it, and rebuilt views.
It also created another set of temporaries for the outer pseudo-gradient. The
Nexus trajectory needs a global inner reduction after each microbatch, so the
ordinary DDP `no_sync()` accumulation optimization cannot be applied.

The production implementation now builds device/dtype-homogeneous flat buckets
once. The bucket owns shaped views and is reused for both phases. Each inner
microbatch copies gradients into the flat buffer, performs the same one
all-reduce, and applies the same normalized update. At the outer boundary the
same storage holds `theta - phi` for the outer reduction. This removes repeated
full-model allocations and preserves the collective count and order.

## Results

| Implementation | Step | Global tok/s | Validation CE | Peak VRAM |
|---|---:|---:|---:|---:|
| Previous Nexus baseline | 57.332 ms | 142,886 | 6.325 | 2,645 MiB |
| Persistent shared bucket | **56.306 ms** | **145,492** | 6.330 | 2,835 MiB |
| Shared bucket + synchronized-inner flag | **51.173 ms** | **160,084** | 6.330 | 2,835 MiB |

The shared bucket is approximately 1.8% faster than the old baseline and costs
about 190 MiB of peak allocation for this model. The opt-in synchronized-inner
row is about 10% faster than the full-reduction shared-bucket row. A second
50-step opt-in run measured 51.781 ms / 158,204 tok/s, showing normal timing
variation.

## Why the synchronized-inner fast path is safe only under a precondition

Once each inner gradient is globally averaged, all ranks take the same inner
step if they started from identical clone parameters and use deterministic
rank-local computation. The clone parameters therefore remain identical. The
outer model and outer optimizer state also remain identical when every rank
applies the same deterministic outer update. Consequently `theta - phi` is
already identical on every rank, making the outer all-reduce redundant.

The implementation exposes this as `assume_synchronized_inner=False` by
default and as `--nexus-assume-synchronized-inner` in the benchmark. The
default still performs the outer reduction. The opt-in is appropriate for the
controlled DDP setup used here; custom collectives, rank-local stochastic state,
unequal initialization, or divergent optimizer state require the safe default.
A two-rank BF16 invariant check over 20 outer steps observed zero cross-rank
parameter difference with and without the outer reduction.

## Experiments that were not promoted

* Persistent autograd gradient views preserve the trajectory, but the matched
  100M DDP run was about 57.5 ms versus 56.3 ms for shared buckets. They remain
  available as an experimental option for further profiling, not as the
  production default.
* A foreach-fused outer pseudo-gradient fill was slower (about 56.9 ms) than
  the shared loop (about 56.3 ms) and was rejected.
* A one-block-per-tensor custom CUDA NSGD kernel was sketched and compiled in a
  development container, but its reduction order was not proven equivalent to
  the FP32 reference and its end-to-end performance was not established. It is
  intentionally absent from this repository.
* Native BF16 norm reductions were faster in isolation but changed the norm
  enough to violate the FP32-norm stability contract. The production path keeps
  FP32 norm accumulation.

The raw records are in `benchmarks/`. The test suite has 11 tests and includes
CPU/Gloo-compatible collective mocks, exact sign checks, bucket storage reuse,
checkpoint restoration, persistent-view trajectory parity, and the
synchronized-inner reduction invariant.
