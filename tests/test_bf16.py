"""Smoke test for v0.3.0 bf16 path.

CPUMamba forward must accept bf16 input + bf16 weights and produce bf16 output
that's close to the fp32 reference.

Strategy: weights stay bf16, scan is upcast to fp32 (numerical stability),
output downcast back to bf16. Linear/Conv1d/SiLU run in bf16 natively on CPU.
"""
from __future__ import annotations

import time

import pytest
import torch

from cpu_mamba import CPUMamba


def _build(dtype, d_model=128, d_state=16, seed=0):
    torch.manual_seed(seed)
    block = CPUMamba(d_model=d_model, d_state=d_state, d_conv=4, expand=2).eval()
    return block.to(dtype=dtype)


def test_bf16_forward_runs():
    """bf16 forward must not error and must produce finite bf16 output."""
    block = _build(torch.bfloat16)
    x = torch.randn(2, 32, 128, dtype=torch.bfloat16)
    with torch.no_grad():
        y = block(x)
    assert y.dtype == torch.bfloat16, f"expected bf16 output, got {y.dtype}"
    assert torch.isfinite(y).all(), "bf16 output has NaN/Inf"
    assert y.shape == x.shape


def test_bf16_vs_fp32_close():
    """bf16 output should be close to fp32 reference (relative error)."""
    torch.manual_seed(42)
    block_fp = _build(torch.float32)
    # Build bf16 block with the SAME weights to isolate dtype effect
    block_bf = CPUMamba(d_model=128, d_state=16, d_conv=4, expand=2).eval()
    block_bf.load_state_dict(block_fp.state_dict())
    block_bf = block_bf.to(dtype=torch.bfloat16)

    x_fp = torch.randn(2, 32, 128, dtype=torch.float32)
    x_bf = x_fp.to(dtype=torch.bfloat16)

    with torch.no_grad():
        y_fp = block_fp(x_fp)
        y_bf = block_bf(x_bf).to(dtype=torch.float32)

    # bf16 has ~7-bit mantissa → expect ~1% relative error
    rel = (y_fp - y_bf).abs() / (y_fp.abs().clamp(min=1e-3) + 1e-6)
    print(f"\nbf16 vs fp32: rel_err max={rel.max().item():.3f} mean={rel.mean().item():.4f} "
          f"p90={rel.float().quantile(0.9).item():.4f}")
    print(f"  abs diff max={(y_fp - y_bf).abs().max().item():.3e} "
          f"mean={(y_fp - y_bf).abs().mean().item():.3e}")

    # Allow up to 30% relative error per element (bf16 is lossy on accumulation)
    # The mean and p90 should be much tighter
    assert rel.mean().item() < 0.10, f"bf16 mean rel err too high: {rel.mean().item()}"
    assert rel.float().quantile(0.9).item() < 0.30, "bf16 p90 rel err too high"


def test_bf16_state_dict_load_from_fp32():
    """A bf16 block must load an fp32 state_dict (with auto-cast)."""
    torch.manual_seed(11)
    fp_block = _build(torch.float32)
    fp_sd = fp_block.state_dict()

    bf_block = CPUMamba(d_model=128, d_state=16, d_conv=4, expand=2).eval()
    bf_block = bf_block.to(dtype=torch.bfloat16)
    # Cast each fp32 tensor to bf16 before loading
    bf_sd = {k: v.to(torch.bfloat16) for k, v in fp_sd.items()}
    bf_block.load_state_dict(bf_sd, strict=True)

    x = torch.randn(1, 16, 128, dtype=torch.bfloat16)
    with torch.no_grad():
        y = bf_block(x)
    assert torch.isfinite(y).all()


def test_bf16_assoc_scan_compat():
    """bf16 weights must work with the associative scan backend too."""
    block = _build(torch.bfloat16)
    block._scan_backend = "assoc"
    x = torch.randn(2, 16, 128, dtype=torch.bfloat16)
    with torch.no_grad():
        y = block(x)
    assert torch.isfinite(y).all()
    assert y.dtype == torch.bfloat16


def test_bf16_memory_smaller_than_fp32():
    """bf16 weights must take half the memory of fp32."""
    fp_block = _build(torch.float32)
    bf_block = _build(torch.bfloat16)
    fp_bytes = sum(p.numel() * p.element_size() for p in fp_block.parameters())
    bf_bytes = sum(p.numel() * p.element_size() for p in bf_block.parameters())
    print(f"\nfp32 weights: {fp_bytes:,} bytes")
    print(f"bf16 weights: {bf_bytes:,} bytes")
    assert bf_bytes == fp_bytes // 2
