"""Associative parallel scan for the Mamba selective scan recurrence.

The recurrence
    h_t = a_t * h_{t-1} + b_t,    h_0 = 0
with associative composition operator
    (a_l, b_l) ⊕ (a_r, b_r) = (a_r * a_l, a_r * b_l + b_r)
admits a parallel prefix scan in log(T) sequential passes (recursive doubling).

This implementation works directly in (a, b) space — no log/exp transforms,
so it is numerically stable for any positive a_t (which is the Mamba case,
since a_t = exp(dt * A) with A < 0 means a_t ∈ (0, 1]).

**Memory: O(B * T * d_inner * d_state)** for the (a, b) prefix tensors plus
log2(T) workspace tensors during reduction. Use the chunked variant for
long sequences or memory-pressured machines.
"""
from __future__ import annotations

import math

import torch


def _associative_prefix_scan(
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Recursive-doubling prefix scan over time dim (dim 1).

    Args:
        a: (B, T, ...) — multiplicative recurrence coefficients
        b: (B, T, ...) — additive recurrence terms

    Returns:
        h: (B, T, ...) where h[t] = a[t]·h[t-1] + b[t], h[-1] = 0.
    """
    T = a.shape[1]
    a_pref = a
    b_pref = b
    stride = 1
    while stride < T:
        a_shift = a_pref[:, :-stride]
        b_shift = b_pref[:, :-stride]
        a_now = a_pref[:, stride:]
        b_now = b_pref[:, stride:]

        a_combined = a_now * a_shift
        b_combined = a_now * b_shift + b_now

        # Out-of-place update — required for parallel scan correctness
        a_pref = torch.cat([a_pref[:, :stride], a_combined], dim=1)
        b_pref = torch.cat([b_pref[:, :stride], b_combined], dim=1)
        stride *= 2

    return b_pref


def selective_scan_assoc(
    x: torch.Tensor,    # (B, T, d_inner)
    dt: torch.Tensor,   # (B, T, d_inner) — already softplus'd
    A: torch.Tensor,    # (d_inner, d_state) — negative
    Bp: torch.Tensor,   # (B, T, d_state)
    Cp: torch.Tensor,   # (B, T, d_state)
    D: torch.Tensor,    # (d_inner,)
) -> torch.Tensor:
    """Recursive-doubling associative scan (numerically stable).

    Returns y (B, T, d_inner).
    """
    # a_t = exp(dt_t · A): bounded in (0, 1] since dt > 0 and A < 0.
    a = (dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)).exp()  # (B, T, d_inner, d_state)
    # b_t = dt_t · B_t · x_t (broadcast across d_state).
    b = (dt.unsqueeze(-1) * Bp.unsqueeze(2)) * x.unsqueeze(-1)  # (B, T, d_inner, d_state)

    h = _associative_prefix_scan(a, b)                          # (B, T, d_inner, d_state)

    y = (h * Cp.unsqueeze(2)).sum(-1) + D.view(1, 1, -1) * x    # (B, T, d_inner)
    return y


def selective_scan_assoc_chunked(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    Bp: torch.Tensor,
    Cp: torch.Tensor,
    D: torch.Tensor,
    chunk_T: int = 32,
) -> torch.Tensor:
    """Chunked associative scan: parallel within each chunk, recurrent across chunks.

    Trades full-T parallelism for ~chunk_T memory cost. State at the end of
    each chunk feeds the next chunk's initial state. Result is bit-identical
    (modulo float reordering) to the full-T scan and the time-loop.

    Memory peak per chunk: O(B * chunk_T * d_inner * d_state).
    """
    B, T, d_inner = x.shape
    d_state = A.shape[1]
    if chunk_T >= T:
        return selective_scan_assoc(x, dt, A, Bp, Cp, D)

    y_chunks = []
    h_carry = torch.zeros(B, d_inner, d_state, dtype=torch.float32, device=x.device)
    Da_view = D.view(1, 1, -1)

    for s in range(0, T, chunk_T):
        e = min(s + chunk_T, T)
        x_c = x[:, s:e]
        dt_c = dt[:, s:e]
        Bp_c = Bp[:, s:e]
        Cp_c = Cp[:, s:e]

        a_c = (dt_c.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)).exp()  # (B, t, d_inner, d_state)
        b_c = (dt_c.unsqueeze(-1) * Bp_c.unsqueeze(2)) * x_c.unsqueeze(-1)

        # Run scan within chunk with h_carry as initial state.
        # Trick: prepend a "step 0" with a=1, b=h_carry; run scan; drop step 0.
        a_init = torch.ones_like(a_c[:, :1])
        b_init = h_carry.unsqueeze(1)
        a_full = torch.cat([a_init, a_c], dim=1)
        b_full = torch.cat([b_init, b_c], dim=1)

        h_full = _associative_prefix_scan(a_full, b_full)
        h_c = h_full[:, 1:]  # drop the carry step

        y_c = (h_c * Cp_c.unsqueeze(2)).sum(-1) + Da_view * x_c
        y_chunks.append(y_c)

        h_carry = h_c[:, -1]

    return torch.cat(y_chunks, dim=1)
