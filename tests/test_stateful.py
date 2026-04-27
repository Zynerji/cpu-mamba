"""Smoke test for v0.4.0 stateful (autoregressive) inference.

Test invariant: forward(x[:, :T]) must produce the same outputs as
prefill(x[:, :T_prefill]) followed by T - T_prefill calls to forward_step.

This is the property that makes Mamba serve as a streaming model: state
captured at end of prefill should be a perfect substitute for re-running
the full sequence.
"""
from __future__ import annotations

import pytest
import torch

from cpu_mamba import CPUMamba


def test_forward_step_matches_full_forward():
    """Single-step generation matches the corresponding slice of full forward."""
    torch.manual_seed(0)
    block = CPUMamba(d_model=64, d_state=8, d_conv=4, expand=2).eval()
    block._force_time_loop = True  # parity reference

    B, T, D = 2, 16, 64
    x = torch.randn(B, T, D)

    # Full reference
    with torch.no_grad():
        y_full = block(x)

    # Prefill first T_pref tokens, then step the rest one at a time
    T_pref = 8
    with torch.no_grad():
        y_pref, state = block.forward_with_final_state(x[:, :T_pref])
    assert y_pref.shape == (B, T_pref, D)

    # Step through remaining tokens
    y_steps = []
    for t in range(T_pref, T):
        with torch.no_grad():
            y_step, state = block.forward_step(x[:, t], state)
        y_steps.append(y_step.unsqueeze(1))
    y_step_cat = torch.cat(y_steps, dim=1)                                # (B, T-T_pref, D)

    # Combined: prefill output + stepped output
    y_combined = torch.cat([y_pref, y_step_cat], dim=1)
    diff = (y_combined - y_full).abs()
    print(f"\nfull vs prefill+step: max={diff.max().item():.3e} mean={diff.mean().item():.3e}")
    assert diff.max().item() < 1e-4, f"max diff {diff.max().item():.3e} too large"


def test_pure_step_from_init_state():
    """Stepping from zeroed state matches full forward when prefill is empty."""
    torch.manual_seed(7)
    block = CPUMamba(d_model=64, d_state=8, d_conv=4, expand=2).eval()
    block._force_time_loop = True

    B, T, D = 2, 12, 64
    x = torch.randn(B, T, D)

    # Full reference
    with torch.no_grad():
        y_full = block(x)

    # All steps from init_state
    state = block.init_state(B)
    y_steps = []
    for t in range(T):
        with torch.no_grad():
            y_step, state = block.forward_step(x[:, t], state)
        y_steps.append(y_step.unsqueeze(1))
    y_seq = torch.cat(y_steps, dim=1)

    diff = (y_seq - y_full).abs()
    print(f"\npure-step vs full: max={diff.max().item():.3e} mean={diff.mean().item():.3e}")
    assert diff.max().item() < 1e-4


def test_init_state_shapes():
    """init_state produces the correct shapes."""
    block = CPUMamba(d_model=64, d_state=8, d_conv=4, expand=2)
    state = block.init_state(batch_size=4)
    assert state["h"].shape == (4, 128, 8)         # (B, d_inner=2*d_model, d_state)
    assert state["conv_state"].shape == (4, 128, 3)  # (B, d_inner, d_conv-1)


def test_step_accepts_2d_or_3d_input():
    """forward_step accepts (B, D) or (B, 1, D) input."""
    block = CPUMamba(d_model=64, d_state=8, d_conv=4, expand=2).eval()
    state = block.init_state(2)
    x_2d = torch.randn(2, 64)
    with torch.no_grad():
        y_2d, state = block.forward_step(x_2d, state)

    state2 = block.init_state(2)
    x_3d = x_2d.unsqueeze(1)
    with torch.no_grad():
        y_3d, state2 = block.forward_step(x_3d, state2)

    assert torch.allclose(y_2d, y_3d, atol=1e-6)
