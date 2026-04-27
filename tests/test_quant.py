"""Smoke test for v0.4.0 weight-only quantization (int8/int4/nf4)."""
from __future__ import annotations

import pytest
import torch

from cpu_mamba import CPUMamba
from cpu_mamba.quant import (
    QuantConfig, QuantizedLinear, quantize_block, memory_footprint,
    quantize_int8, dequantize_int8,
    quantize_int4, dequantize_int4,
    quantize_nf4, dequantize_nf4,
)


def _build(seed=0, d_model=128, d_state=16):
    torch.manual_seed(seed)
    return CPUMamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2).eval()


# ============================================================================
# Quant primitives
# ============================================================================

def test_int8_roundtrip_close():
    """int8 quant/dequant returns weight close to original (absolute error scale of ~1/127)."""
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    w_q, scale = quantize_int8(w)
    w_r = dequantize_int8(w_q, scale)
    assert w_q.dtype == torch.int8
    abs_err = (w - w_r).abs()
    # Each weight quantizes within scale/127 = max_abs/127 of the true value.
    # Per-channel max_abs ≈ 3-4σ for N(0,1), so abs_err ≤ ~3/127 ≈ 0.024 per element.
    print(f"\nint8 roundtrip: abs_err max={abs_err.max().item():.4f} mean={abs_err.mean().item():.4f}")
    assert abs_err.max().item() < 0.05


def test_int4_roundtrip_close():
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    packed, scale, in_dim = quantize_int4(w)
    w_r = dequantize_int4(packed, scale, in_dim)
    assert packed.dtype == torch.uint8
    assert packed.shape[-1] * 2 >= in_dim
    abs_err = (w - w_r).abs()
    # int4 has only 15 levels in [-1,1] normalized → expect ~max_abs/14 ≈ 0.21 per element max
    print(f"\nint4 roundtrip: abs_err max={abs_err.max().item():.4f} mean={abs_err.mean().item():.4f}")
    assert abs_err.max().item() < 0.5


def test_nf4_roundtrip_better_than_int4():
    """NF4 should beat int4 on normal-distributed weights."""
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    _, scale_i4, _ = quantize_int4(w)
    _, scale_nf4, _ = quantize_nf4(w)
    packed_i4, _, in_dim = quantize_int4(w)
    packed_nf4, _, in_dim2 = quantize_nf4(w)
    w_r_i4 = dequantize_int4(packed_i4, scale_i4, in_dim)
    w_r_nf4 = dequantize_nf4(packed_nf4, scale_nf4, in_dim2)
    err_i4 = (w - w_r_i4).abs().mean().item()
    err_nf4 = (w - w_r_nf4).abs().mean().item()
    print(f"\nint4 mean abs err: {err_i4:.4f}")
    print(f"nf4  mean abs err: {err_nf4:.4f}")
    assert err_nf4 < err_i4, "NF4 should be more accurate than int4 on normal weights"


def test_int4_handles_odd_in_dim():
    """Packing must handle odd in_features by padding."""
    torch.manual_seed(0)
    w = torch.randn(8, 33)  # odd in_dim
    packed, scale, in_dim = quantize_int4(w)
    assert in_dim == 33
    w_r = dequantize_int4(packed, scale, in_dim)
    assert w_r.shape == w.shape


# ============================================================================
# QuantizedLinear
# ============================================================================

def test_quantized_linear_forward():
    torch.manual_seed(0)
    lin = torch.nn.Linear(128, 256)
    qlin_int8 = QuantizedLinear(lin.weight, lin.bias, "int8")
    qlin_int4 = QuantizedLinear(lin.weight, lin.bias, "int4")
    qlin_nf4 = QuantizedLinear(lin.weight, lin.bias, "nf4")

    x = torch.randn(4, 128)
    y_fp = lin(x)
    y_i8 = qlin_int8(x)
    y_i4 = qlin_int4(x)
    y_nf4 = qlin_nf4(x)

    rel = lambda a, b: ((a - b).abs() / (a.abs().clamp(min=1e-3))).mean().item()
    print(f"\nLinear forward rel err vs fp32:")
    print(f"  int8: {rel(y_fp, y_i8):.4f}")
    print(f"  int4: {rel(y_fp, y_i4):.4f}")
    print(f"  nf4 : {rel(y_fp, y_nf4):.4f}")
    assert rel(y_fp, y_i8) < 0.05
    assert rel(y_fp, y_i4) < 0.50
    assert rel(y_fp, y_nf4) < 0.50


# ============================================================================
# Block-level
# ============================================================================

@pytest.mark.parametrize("scheme", ["int8", "int4", "nf4"])
def test_quantize_block_runs(scheme):
    """Quantizing a CPUMamba block produces a working forward."""
    torch.manual_seed(0)
    block = _build()
    n = quantize_block(block, QuantConfig(scheme=scheme))
    assert n == 4, f"expected 4 Linear replaces, got {n}"

    x = torch.randn(2, 32, 128)
    with torch.no_grad():
        y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("scheme,max_rel_err", [
    ("int8", 0.10),
    ("int4", 0.50),
    ("nf4", 0.60),  # per-channel scaling underutilizes NF4 LUT — see docstring
])
def test_quantized_block_accuracy_vs_fp32(scheme, max_rel_err):
    """Quantized block output should be close to fp32 reference."""
    torch.manual_seed(0)
    block_fp = _build()
    block_q = _build()
    block_q.load_state_dict(block_fp.state_dict())
    quantize_block(block_q, QuantConfig(scheme=scheme))

    x = torch.randn(2, 32, 128)
    with torch.no_grad():
        y_fp = block_fp(x)
        y_q = block_q(x)

    rel = ((y_fp - y_q).abs() / (y_fp.abs().clamp(min=1e-3))).mean().item()
    print(f"\n{scheme}: mean rel err vs fp32 = {rel:.4f}")
    assert rel < max_rel_err


def test_memory_footprint():
    """Quantized blocks must take less memory than fp32."""
    block_fp = _build()
    block_int8 = _build()
    block_int8.load_state_dict(block_fp.state_dict())
    block_int4 = _build()
    block_int4.load_state_dict(block_fp.state_dict())
    block_nf4 = _build()
    block_nf4.load_state_dict(block_fp.state_dict())

    quantize_block(block_int8, QuantConfig(scheme="int8"))
    quantize_block(block_int4, QuantConfig(scheme="int4"))
    quantize_block(block_nf4, QuantConfig(scheme="nf4"))

    mem_fp = memory_footprint(block_fp)["total_bytes"]
    mem_i8 = memory_footprint(block_int8)["total_bytes"]
    mem_i4 = memory_footprint(block_int4)["total_bytes"]
    mem_nf4 = memory_footprint(block_nf4)["total_bytes"]
    print(f"\nfp32:  {mem_fp:>10,} bytes")
    print(f"int8:  {mem_i8:>10,} bytes  ({mem_i8/mem_fp:.2%} of fp32)")
    print(f"int4:  {mem_i4:>10,} bytes  ({mem_i4/mem_fp:.2%} of fp32)")
    print(f"nf4:   {mem_nf4:>10,} bytes  ({mem_nf4/mem_fp:.2%} of fp32)")
    # Linear layers dominate: int8 ~25% of fp, int4/nf4 ~15-20% (conv1d + biases stay fp32)
    assert mem_i8 < mem_fp
    assert mem_i4 < mem_i8
    assert mem_nf4 <= mem_i4 * 1.05  # Same packing; allow tiny overhead from LUT cache
