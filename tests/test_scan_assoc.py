"""Smoke test for v0.2.0 associative scan: parity with time-loop + C++ kernel."""
from __future__ import annotations

import os
import time

import pytest
import torch

from cpu_mamba import CPUMamba


def _all_three_outputs(block, x):
    """Run forward with each of the three scan backends; return outputs."""
    # Time-loop (reference)
    block._force_time_loop = True
    block._scan_backend = "auto"
    with torch.no_grad():
        y_loop = block(x).clone()

    # C++ kernel
    block._force_time_loop = False
    block._scan_backend = "auto"
    with torch.no_grad():
        y_cpp = block(x).clone()

    # Associative scan (full)
    block._scan_backend = "assoc"
    block._assoc_chunk_T = 0
    with torch.no_grad():
        y_assoc = block(x).clone()

    return y_loop, y_cpp, y_assoc


def test_assoc_parity_small():
    """Small shape: assoc scan must match time-loop + C++ kernel to fp32 floor."""
    torch.manual_seed(0)
    block = CPUMamba(d_model=128, d_state=16, d_conv=4, expand=2).eval()
    x = torch.randn(2, 32, 128)

    y_loop, y_cpp, y_assoc = _all_three_outputs(block, x)

    diff_assoc_loop = (y_assoc - y_loop).abs()
    diff_assoc_cpp = (y_assoc - y_cpp).abs()
    diff_cpp_loop = (y_cpp - y_loop).abs()

    print(f"\nshape: {tuple(x.shape)}")
    print(f"assoc vs loop: max={diff_assoc_loop.max().item():.3e} mean={diff_assoc_loop.mean().item():.3e}")
    print(f"assoc vs cpp:  max={diff_assoc_cpp.max().item():.3e} mean={diff_assoc_cpp.mean().item():.3e}")
    print(f"cpp vs loop:   max={diff_cpp_loop.max().item():.3e} mean={diff_cpp_loop.mean().item():.3e}")

    assert diff_assoc_loop.max().item() < 1e-3, f"assoc vs loop diff too large"
    assert diff_assoc_cpp.max().item() < 1e-3, f"assoc vs cpp diff too large"


def test_assoc_chunked_parity():
    """Chunked assoc scan must match unchunked output (and time-loop)."""
    torch.manual_seed(7)
    block = CPUMamba(d_model=128, d_state=16, d_conv=4, expand=2).eval()
    x = torch.randn(2, 64, 128)

    block._force_time_loop = False
    block._scan_backend = "assoc"
    block._assoc_chunk_T = 0
    with torch.no_grad():
        y_full = block(x).clone()

    for chunk_T in [4, 8, 16, 32, 50]:  # 50 doesn't evenly divide 64 — must still match
        block._assoc_chunk_T = chunk_T
        with torch.no_grad():
            y_chunked = block(x).clone()
        diff = (y_chunked - y_full).abs()
        assert diff.max().item() < 1e-3, \
            f"chunk_T={chunk_T}: max_diff={diff.max().item():.3e}"
        print(f"chunk_T={chunk_T}: max_diff={diff.max().item():.3e}")


@pytest.mark.skipif(os.environ.get("SKIP_LARGE") == "1", reason="memory-heavy test")
def test_assoc_parity_medium():
    """Medium shape (closer to real model). Memory: ~67 MB per intermediate."""
    torch.manual_seed(0)
    block = CPUMamba(d_model=256, d_state=16, d_conv=4, expand=2).eval()
    x = torch.randn(4, 64, 256)

    y_loop, y_cpp, y_assoc = _all_three_outputs(block, x)
    diff_assoc = (y_assoc - y_loop).abs()
    print(f"\nmedium: assoc vs loop max={diff_assoc.max().item():.3e}")
    assert diff_assoc.max().item() < 5e-3
