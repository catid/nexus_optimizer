# API notes

## `NexusOptimizer`

```python
NexusOptimizer(
    model,
    outer_optimizer,
    inner_lr,
    *,
    inner_model=None,
    norm_eps=1e-12,
    sync_gradients=True,
    process_group=None,
    persistent_inner_grads=False,
    assume_synchronized_inner=False,
)
```

`model` and `outer_optimizer` must refer to the same trainable parameters. The
optional `inner_model` can be supplied when cloning through a wrapper such as
FSDP is not supported. `norm_eps` guards zero or very small gradients.

For DDP, pass the process group explicitly when using a non-default group. The
inner clone is intentionally not wrapped in DDP: Nexus must finish its global
inner reduction before applying each normalized update.

`state_dict()` includes the outer optimizer state, clone state, partial
microbatch count, inner hyperparameters, and opt-in mode flags. This allows a
checkpoint taken inside an accumulation window to resume the same trajectory.

## Calling conventions

The convenience API is:

```python
nexus.zero_grad()
for x, y in microbatches:
    nexus.microbatch_step(loss_fn(nexus.inner_model(x), y))
nexus.outer_step()
```

The lower-level equivalent is `loss.backward(); nexus.inner_step()` for each
microbatch, followed by `nexus.outer_step()`.
