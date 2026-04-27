# cpu-mamba

CPU-only Mamba S6 block with **state_dict parity** to `mamba_ssm.Mamba`. Drop-in replacement for serving Mamba models on machines without CUDA.

`mamba_ssm` 2.x has no working CPU path — the package's `causal_conv1d` dependency unconditionally calls a CUDA kernel even with `use_fast_path=False`. This package solves that: a single-file PyTorch module + JIT-compiled C++ scan kernel, no `mamba_ssm` or `causal_conv1d` install required at runtime.

## Why

You trained a Mamba model on GPU. You want to serve it on CPU. Without retrain.

```python
import torch, mamba_ssm  # GPU env: training
gpu_block = mamba_ssm.Mamba(d_model=384, d_state=24, d_conv=4, expand=2).cuda()
# ... train ...
torch.save(gpu_block.state_dict(), "trained.pt")
```

```python
import torch
from cpu_mamba import CPUMamba  # CPU env: inference
block = CPUMamba(d_model=384, d_state=24, d_conv=4, expand=2)
block.load_state_dict(torch.load("trained.pt"))  # works; identical key layout
y = block(x)  # x: (B, T, d_model) on CPU
```

## Numerical parity

`CPUMamba` produces outputs identical to `mamba_ssm.Mamba(use_fast_path=False)` to fp32 noise floor:

| Test | Max abs diff | Mean abs diff |
|---|---|---|
| 384-dim block, B=2, T=180 vs `mamba_ssm.Mamba` (CUDA) | 9.7e-8 | 1.5e-8 |
| C++ scan vs Python time-loop, B=151, T=180 | 3.8e-6 | 1.4e-7 |

(The fp32 floor for these tensor sizes is ~1e-6, so `CPUMamba` matches `mamba_ssm` exactly within precision.)

## Performance

Two backends, both produce identical outputs:

| Backend | Speedup vs Python time-loop |
|---|---|
| **C++ kernel** (default; OpenMP across (B, d_inner), sequential across T) | **~11×** |
| **Python time-loop** (fallback; pure PyTorch) | 1× (reference) |

Bench at v60-class shape (B=151, T=180, d_inner=1536, d_state=24) on 8-thread x86_64:
- Python time-loop: ~1535 ms
- C++ kernel: ~133 ms

The C++ kernel JIT-compiles on first import (~15 s) and caches in `~/.cache/torch_extensions/`. Subsequent runs are instant.

## Install

```bash
pip install cpu-mamba           # core
pip install cpu-mamba[fast]     # + ninja for faster JIT compile
```

`gcc` and `ninja` must be on PATH. Pure-PyTorch fallback runs without them — set `CPU_MAMBA_FORCE_TIME_LOOP=1` to bypass the C++ kernel.

## State_dict parity (the load-bearing claim)

`CPUMamba`'s `state_dict` keys and shapes match `mamba_ssm.Mamba` 2.3.x exactly:

```
A_log              (d_inner, d_state)
D                  (d_inner,)
in_proj.weight     (2 * d_inner, d_model)
conv1d.weight      (d_inner, 1, d_conv)
conv1d.bias        (d_inner,)
x_proj.weight      (dt_rank + 2 * d_state, d_inner)
dt_proj.weight     (d_inner, dt_rank)
dt_proj.bias       (d_inner,)
out_proj.weight    (d_model, d_inner)
```

So `cpu_block.load_state_dict(mamba_ssm_block.state_dict())` is byte-equivalent to passing through.

For models that wrap `Mamba` inside a larger architecture, `cpu_mamba.surgery` walks the module tree and swaps every `mamba_ssm.Mamba` instance with a `CPUMamba` of matching dimensions, transferring weights:

```python
from cpu_mamba.surgery import swap_mamba_to_cpu

# Build the full architecture under a stubbed mamba_ssm
# (so the original code imports cleanly without mamba_ssm installed):
import sys
sys.modules["mamba_ssm"] = ...  # see examples/

model = MyModelWithMambaBlocks(...)
model.load_state_dict(torch.load("checkpoint.pt"))
swap_mamba_to_cpu(model)  # in-place; preserves weights
y = model(x)  # CPU
```

## Forcing the fallback

For parity testing or numerical debugging:

```bash
CPU_MAMBA_FORCE_TIME_LOOP=1 python my_script.py
```

The Python time-loop is byte-identical to the C++ kernel up to fp32 reordering. Useful when you want to confirm a numerical issue isn't from the C++ side.

## Limitations

- **No backward pass.** Inference only. For training on CPU, you want the autograd-capable mainline implementation, not this.
- **fp32 only.** bf16/fp16 paths in `mamba_ssm.Mamba` use the CUDA kernel; this package casts to fp32 for the scan, then casts back to the input dtype.
- **No KV cache / inference_params.** The full T-step scan runs every call. Adding stateful inference is straightforward but isn't implemented.
- **Fixed d_state ≤ 256.** The C++ kernel uses a stack-allocated buffer for the hidden state. Larger states would need heap allocation; trivial to extend.

## Roadmap

- **0.2.0:** Associative parallel scan (Heinsen-style) for log(T) reduction across timesteps. Targets ~5× additional speedup on machines with many cores.
- **0.3.0:** bf16 scan path for half-precision serving.
- **0.4.0:** Stateful KV-cache-equivalent for autoregressive generation.

## License

MIT.
