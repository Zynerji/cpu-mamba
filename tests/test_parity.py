"""Parity + sanity tests for CPUMamba.

These tests run without `mamba_ssm` (the package CPUMamba replaces). Tests
that compare against a real CUDA `mamba_ssm.Mamba` are gated on availability.
"""
from __future__ import annotations

import os
import pytest
import torch

from cpu_mamba import CPUMamba, assert_state_dict_compatible
from cpu_mamba.surgery import install_mamba_ssm_stub, swap_mamba_to_cpu


def _build(d_model=384, d_state=24, d_conv=4, expand=2, seed=0):
    torch.manual_seed(seed)
    return CPUMamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand).eval()


def test_forward_shape():
    block = _build()
    x = torch.randn(2, 180, 384)
    with torch.no_grad():
        y = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_state_dict_keys():
    """The state_dict layout must match mamba_ssm.Mamba exactly."""
    block = _build(d_model=384, d_state=24, d_conv=4, expand=2)
    sd = block.state_dict()
    expected = {
        "A_log", "D",
        "in_proj.weight",
        "conv1d.weight", "conv1d.bias",
        "x_proj.weight",
        "dt_proj.weight", "dt_proj.bias",
        "out_proj.weight",
    }
    assert set(sd.keys()) == expected
    assert sd["A_log"].shape == (768, 24)        # d_inner = 2 * d_model = 768
    assert sd["D"].shape == (768,)
    assert sd["in_proj.weight"].shape == (1536, 384)
    assert sd["conv1d.weight"].shape == (768, 1, 4)
    assert sd["x_proj.weight"].shape == (24 + 48, 768)  # dt_rank=24 + 2*d_state=48
    assert sd["dt_proj.weight"].shape == (768, 24)
    assert sd["out_proj.weight"].shape == (384, 768)


def test_cpp_vs_python_parity():
    """C++ kernel and Python time-loop must produce identical outputs."""
    block = _build()
    x = torch.randn(4, 64, 384)

    # C++ kernel (default)
    with torch.no_grad():
        y_cpp = block(x).clone()

    # Python time-loop fallback
    block._force_time_loop = True
    with torch.no_grad():
        y_loop = block(x).clone()
    block._force_time_loop = False

    diff = (y_cpp - y_loop).abs()
    assert diff.max().item() < 1e-4, f"max diff {diff.max().item():.3e} too large"


def test_state_dict_compatibility_helper():
    """`assert_state_dict_compatible` raises on mismatch, passes on match."""
    block = _build()
    sd = block.state_dict()
    # Identity check passes
    assert_state_dict_compatible(block, sd)

    # Wrong shape → raises
    bad = dict(sd)
    bad["A_log"] = torch.zeros(100, 24)
    with pytest.raises(ValueError, match="shape mismatch"):
        assert_state_dict_compatible(block, bad)

    # Extra key → raises
    bad = dict(sd)
    bad["extra_key"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected keys"):
        assert_state_dict_compatible(block, bad)


def test_install_mamba_ssm_stub():
    """After install, `from mamba_ssm import Mamba` returns CPUMamba."""
    import sys
    sys.modules.pop("mamba_ssm", None)
    install_mamba_ssm_stub()
    import mamba_ssm
    assert mamba_ssm.Mamba is CPUMamba

    # Build via stub: must produce a CPUMamba instance
    block = mamba_ssm.Mamba(d_model=384, d_state=24, d_conv=4, expand=2)
    assert isinstance(block, CPUMamba)


def test_swap_no_mamba_ssm():
    """swap_mamba_to_cpu is a no-op when mamba_ssm isn't installed (and our stub
    is the same as CPUMamba)."""
    import sys
    sys.modules.pop("mamba_ssm", None)
    install_mamba_ssm_stub()  # stub.Mamba IS CPUMamba — swap should detect and skip

    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = CPUMamba(384, 24, 4, 2)

    n = swap_mamba_to_cpu(Wrap())
    assert n == 0


@pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("SKIP_CUDA_TESTS") == "1",
    reason="requires CUDA + mamba_ssm installed",
)
def test_parity_against_mamba_ssm_cuda():
    """If real mamba_ssm is installed, CPUMamba outputs must match its CUDA path."""
    try:
        import mamba_ssm
        if mamba_ssm.Mamba is CPUMamba:
            pytest.skip("mamba_ssm is a stub — can't compare against CUDA")
    except ImportError:
        pytest.skip("mamba_ssm not installed")

    torch.manual_seed(7)
    src = mamba_ssm.Mamba(d_model=384, d_state=24, d_conv=4, expand=2).cuda().eval()
    cpu = CPUMamba(d_model=384, d_state=24, d_conv=4, expand=2).cpu().eval()
    cpu.load_state_dict(src.state_dict(), strict=True)

    x = torch.randn(2, 64, 384)
    with torch.no_grad():
        y_cuda = src(x.cuda()).cpu()
        y_cpu = cpu(x)

    diff = (y_cuda - y_cpu).abs()
    # fp32 noise floor ~1e-6 for these shapes
    assert diff.max().item() < 1e-4, f"CUDA vs CPU max diff {diff.max().item():.3e}"
