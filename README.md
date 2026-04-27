# cpu-mamba

CPU-only Mamba S6 block with **state_dict parity** to `mamba_ssm.Mamba`. Drop-in replacement for serving Mamba models on machines without CUDA.

`mamba_ssm` 2.x has no working CPU path — the package's `causal_conv1d` dependency unconditionally calls a CUDA kernel even with `use_fast_path=False`. This package solves that: a single torch dependency + JIT-compiled C++ scan kernel, no `mamba_ssm` or `causal_conv1d` install required at runtime.

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

## What's in the box

| Version | Feature |
|---|---|
| **0.1.0** | C++ scan kernel + Python time-loop fallback, state_dict parity |
| **0.2.0** | Associative parallel scan via recursive doubling (log(T) reductions) |
| **0.3.0** | bf16 forward path — 2× memory savings on weights + activations |
| **0.4.0** | Stateful (autoregressive) inference + int8 / int4 / nf4 weight-only quant |

## Numerical parity

`CPUMamba` produces outputs identical to `mamba_ssm.Mamba(use_fast_path=False)` to fp32 noise floor:

| Test | Max abs diff |
|---|---|
| 384-dim block, B=2, T=180 vs `mamba_ssm.Mamba` (CUDA) | 9.7e-8 |
| C++ kernel vs Python time-loop, B=151, T=180 | 3.8e-6 |
| Associative scan vs time-loop (B=2, T=32) | 3.0e-8 |
| forward_step prefill+step vs full forward(T=16) | 9.0e-8 |

(The fp32 floor for these tensor sizes is ~1e-6, so all backends match within precision.)

## Performance

Three independent scan backends, all produce identical outputs:

| Backend | Speedup vs Python time-loop | Memory | Use when |
|---|---|---|---|
| **C++ kernel** (default) | **~11×** | O(B × d_inner × d_state) | Default. Best for production. |
| **Associative scan** | varies (better on many cores) | O(B × T × d_inner × d_state) | Many cores, short T |
| **Python time-loop** (fallback) | 1× (reference) | O(B × d_inner × d_state) | Parity reference / debugging |

Bench at large-batch panel shape (B=151, T=180, d_inner=1536, d_state=24) on 8-thread x86_64:
- Python time-loop: ~1535 ms
- C++ kernel: ~133 ms

The C++ kernel JIT-compiles on first import (~15 s) and caches in `~/.cache/torch_extensions/`. Subsequent runs are instant.

## bf16 (v0.3.0)

```python
block = CPUMamba(d_model=384, d_state=24).to(torch.bfloat16)
y = block(x_bf16)  # bf16 in, bf16 out — half the weight memory
```

Internally the selective scan upcasts to fp32 for stability and downcasts back. Linear, Conv1d, and SiLU run natively in bf16 on x86_64. Typical mean relative error vs fp32: ~1.8% (matches bf16's 7-bit mantissa).

## Stateful inference (v0.4.0)

For autoregressive generation, prefill the prompt once and step token-by-token:

```python
block = CPUMamba(d_model=384, d_state=24).eval()

# Prefill the prompt
y_prompt, state = block.forward_with_final_state(prompt)  # prompt: (B, T_prompt, D)

# Generate one token at a time
for _ in range(max_new_tokens):
    y, state = block.forward_step(next_token, state)  # next_token: (B, D)
    # ... sample, append, repeat
```

The state captures both the selective-scan hidden `h` and the causal conv ring buffer. `forward_step` outputs match the corresponding slice of a full forward to fp32 floor.

## Quantization (v0.4.0)

Weight-only quantization with three schemes:

```python
from cpu_mamba import CPUMamba, QuantConfig, quantize_block

block = CPUMamba(d_model=384, d_state=24)
block.load_state_dict(torch.load("trained.pt"))

quantize_block(block, QuantConfig(scheme="int4"))   # in-place
y = block(x)                                         # 5.6× smaller weights
```

| Scheme | Bits | Compression vs fp32 | Mean abs roundtrip err |
|---|---|---|---|
| **int8** | 8 (signed) | 3.4× (29.6% of fp32) | 0.006 |
| **int4** | 4 (signed, packed) | 5.6× (17.7%) | 0.110 |
| **nf4** | 4 (NormalFloat LUT) | 5.6× (17.7%) | 0.082 |

Activations stay in fp32 (or whatever input dtype). Quantized weights dequantize on the fly during forward — no fused matmul kernels yet, so the speedup is purely from cache pressure / memory bandwidth.

Quantization replaces only `in_proj`, `x_proj`, `dt_proj`, `out_proj` (the four Linear layers). `conv1d`, `A_log`, `D`, biases keep their original dtype.

**Caveat:** per-channel scaling underutilizes NF4's near-zero LUT density when weight rows have outliers. Blockwise NF4 (group_size=64, à la QLoRA) is planned for v0.5.

## Install

```bash
pip install cpu-mamba           # core
pip install cpu-mamba[fast]     # + ninja for faster JIT compile
```

`gcc` and `ninja` must be on PATH for the C++ scan kernel. Pure-PyTorch fallback runs without them — set `CPU_MAMBA_FORCE_TIME_LOOP=1` to bypass the C++ kernel.

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

So `cpu_block.load_state_dict(mamba_ssm_block.state_dict())` is byte-equivalent.

For models that wrap `Mamba` inside a larger architecture, `cpu_mamba.surgery` walks the module tree and swaps every `mamba_ssm.Mamba` instance with a `CPUMamba` of matching dimensions, transferring weights:

```python
from cpu_mamba.surgery import install_mamba_ssm_stub, swap_mamba_to_cpu

# Option 1: stub mamba_ssm before importing the model
install_mamba_ssm_stub()
import my_model_module  # `from mamba_ssm import Mamba` resolves to CPUMamba

# Option 2: build the model with mamba_ssm, then swap after loading weights
model = MyModelWithMambaBlocks(...)
model.load_state_dict(torch.load("ckpt.pt"))
swap_mamba_to_cpu(model)  # in-place; preserves trained weights
```

## API summary

```python
from cpu_mamba import (
    CPUMamba,                       # the block
    assert_state_dict_compatible,   # explain mismatches when loading
    QuantConfig,
    QuantizedLinear,
    quantize_block,                 # in-place int8/int4/nf4
    memory_footprint,               # report buffer + param sizes
)
from cpu_mamba.surgery import install_mamba_ssm_stub, swap_mamba_to_cpu
from cpu_mamba.scan_assoc import (
    selective_scan_assoc,           # full-T parallel scan
    selective_scan_assoc_chunked,   # bounded-memory chunked variant
)
```

`CPUMamba` block instance attributes that you can flip:
- `block._scan_backend = "auto"` (default) | `"assoc"` (use parallel scan)
- `block._assoc_chunk_T = 32` (chunk size for chunked associative scan; 0 = full)
- `block._force_time_loop = True` (bypass C++ kernel; parity reference)

## Forcing the fallback

For parity testing or numerical debugging:

```bash
CPU_MAMBA_FORCE_TIME_LOOP=1 python my_script.py
```

The Python time-loop is byte-identical to the C++ kernel up to fp32 reordering.

## Limitations

- **No backward pass.** Inference only. For training on CPU, use the autograd-capable mainline implementations.
- **Quantization activations stay fp32.** Fused dequant+matmul kernels are not yet implemented; weight-only quant gives memory savings but only modest latency improvements.
- **No KV cache for attention.** Mamba is attention-free, so there is no K/V — but stateful generation is supported via `forward_step` (v0.4.0).
- **Fixed d_state ≤ 256** in the C++ kernel (stack-allocated). Easy to extend.

## Roadmap

- **0.5.0:** Blockwise NF4 (group_size=64) for tighter accuracy; fused dequant+matmul C++ kernel for int4/nf4 forward speedups.
- **0.6.0:** SIMD-vectorized parallel scan kernel (AVX-512 / NEON paths).
- **0.7.0:** ONNX export + ggml-style file format for headless inference.

## License

MIT.
