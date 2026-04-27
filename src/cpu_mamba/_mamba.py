"""CPU-compatible Mamba block that loads `mamba_ssm.Mamba` state_dict.

Pure PyTorch fallback path + JIT-compiled C++ selective_scan kernel for speed.
No causal_conv1d, no selective_scan_cuda, no mamba_ssm imports required.

State_dict parity: identical key names + shapes to mamba_ssm.Mamba 2.3.1, so
`cpu_module.load_state_dict(mamba_ssm_module.state_dict())` works directly.

Set CPU_MAMBA_FORCE_TIME_LOOP=1 to bypass the C++ kernel (parity testing).
"""
from __future__ import annotations

import math
import os
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F

# Lazy-loaded C++ scan kernel. Falls back to time-loop if compile fails.
_SCAN_OP = None
_SCAN_BUILD_TRIED = False


def _try_load_cpp_scan():
    global _SCAN_OP, _SCAN_BUILD_TRIED
    if _SCAN_BUILD_TRIED:
        return _SCAN_OP
    _SCAN_BUILD_TRIED = True
    if os.environ.get("CPU_MAMBA_FORCE_TIME_LOOP") == "1":
        return None
    try:
        from torch.utils.cpp_extension import load
        cpp_path = pathlib.Path(__file__).parent / "selective_scan_cpu.cpp"
        if not cpp_path.exists():
            return None
        _SCAN_OP = load(
            name="selective_scan_cpu",
            sources=[str(cpp_path)],
            extra_cflags=["-O3", "-fopenmp", "-march=native", "-ffast-math"],
            extra_ldflags=["-fopenmp"],
            verbose=False,
        )
    except Exception:
        _SCAN_OP = None
    return _SCAN_OP


class CPUMamba(nn.Module):
    """Pure-PyTorch Mamba S6 block. State_dict-compatible with mamba_ssm.Mamba."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dt_rank: str | int = "auto",
                 dt_min: float = 0.001, dt_max: float = 0.1, dt_init: str = "random",
                 dt_scale: float = 1.0, dt_init_floor: float = 1e-4,
                 conv_bias: bool = True, bias: bool = False, **_unused):
        # Absorb extra kwargs from mamba_ssm.Mamba 2.3.x (use_fast_path, layer_idx, device, dtype, ...)
        # so the trained module imports as a drop-in replacement.
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else int(dt_rank)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=d_conv, groups=self.d_inner, padding=d_conv - 1,
            bias=conv_bias,
        )
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A: (d_inner, d_state). Stored as log of -A (ensures A < 0 after -exp).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    # ---- v0.4.0: stateful (autoregressive) inference -------------------------
    def init_state(self, batch_size: int, device=None, dtype=None) -> dict:
        """Allocate zero state for `forward_step`.

        Returns a dict with:
            'h':          (B, d_inner, d_state) — selective-scan hidden state
            'conv_state': (B, d_inner, d_conv-1) — past inputs to causal conv
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return {
            "h": torch.zeros(batch_size, self.d_inner, self.d_state, device=device, dtype=torch.float32),
            "conv_state": torch.zeros(batch_size, self.d_inner, self.d_conv - 1, device=device, dtype=dtype),
        }

    def forward_step(self, x_step: torch.Tensor, state: dict) -> tuple:
        """Run one autoregressive step.

        Args:
            x_step: (B, D) or (B, 1, D) — single-token input
            state:  dict from init_state() or a prior forward_step call

        Returns:
            (y_step, new_state) where y_step is (B, D) matching x_step layout (last dim).

        State is updated in-place AND returned (caller may keep either reference).
        """
        if x_step.dim() == 3:
            assert x_step.shape[1] == 1, "forward_step expects T=1"
            x_step = x_step.squeeze(1)
        elif x_step.dim() != 2:
            raise ValueError(f"x_step must be (B, D) or (B, 1, D), got {x_step.shape}")
        B, D = x_step.shape
        assert D == self.d_model

        xz = self.in_proj(x_step)                                # (B, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)                               # each (B, d_inner)

        # Causal conv: append new x to conv_state buffer of size d_conv-1, then apply conv.
        # conv_state: (B, d_inner, d_conv-1)
        conv_input = torch.cat(
            [state["conv_state"], x.unsqueeze(-1)], dim=-1
        )                                                         # (B, d_inner, d_conv)
        # Apply depthwise causal conv: per-channel weighted sum.
        # conv1d.weight is (d_inner, 1, d_conv); squeeze to (d_inner, d_conv).
        w = self.conv1d.weight.squeeze(1)                         # (d_inner, d_conv)
        x_conv = (conv_input * w.unsqueeze(0)).sum(dim=-1)        # (B, d_inner)
        if self.conv1d.bias is not None:
            x_conv = x_conv + self.conv1d.bias
        # Update state: shift the buffer left (drop oldest, keep current as last)
        state["conv_state"] = conv_input[..., 1:]
        x = self.act(x_conv)

        # x_proj → split into dt, B, C
        x_dbl = self.x_proj(x)                                    # (B, dt_rank + 2*d_state)
        dt, Bp, Cp = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)                                     # (B, d_inner)
        dt = F.softplus(dt)

        # Selective scan, single step. fp32 throughout for stability.
        A = -torch.exp(self.A_log.float())                        # (d_inner, d_state)
        x_f = x.float()
        dt_f = dt.float()
        Bp_f = Bp.float()
        Cp_f = Cp.float()
        D_f = self.D.float()

        dA = torch.exp(dt_f.unsqueeze(-1) * A.unsqueeze(0))       # (B, d_inner, d_state)
        dB_x = (dt_f.unsqueeze(-1) * Bp_f.unsqueeze(1)) * x_f.unsqueeze(-1)  # (B, d_inner, d_state)
        h_new = dA * state["h"] + dB_x                            # (B, d_inner, d_state)
        state["h"] = h_new
        y = (h_new * Cp_f.unsqueeze(1)).sum(dim=-1) + D_f * x_f   # (B, d_inner)

        y = y.to(x.dtype)
        y = y * self.act(z)
        return self.out_proj(y), state

    def forward_with_final_state(self, hidden_states: torch.Tensor) -> tuple:
        """Run a full forward pass and ALSO return the final state.

        Useful for prefilling: process the prompt, then continue with forward_step.
        """
        B, T, D = hidden_states.shape
        # Run the main forward (output ignored — we re-derive state below for correctness)
        y_full = self.forward(hidden_states)

        # Re-derive final state. The cleanest correct approach is to run the time-loop
        # path and capture h, plus the last (d_conv-1) inputs to the conv1d.
        xz = self.in_proj(hidden_states)
        x_pre, _ = xz.chunk(2, dim=-1)                            # (B, T, d_inner)
        # conv_state at end of prefill: last (d_conv-1) post-in_proj inputs
        x_t = x_pre.transpose(1, 2)                                # (B, d_inner, T)
        conv_state = x_t[..., -(self.d_conv - 1):].clone()         # (B, d_inner, d_conv-1)

        # Run the conv to get the gated x for the scan
        x_conv = self.conv1d(x_t)[..., :T]
        x_conv = self.act(x_conv.transpose(1, 2))                  # (B, T, d_inner)
        x_dbl = self.x_proj(x_conv)
        dt, Bp, Cp = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())

        # Compute final h via the time-loop (cheap relative to full forward)
        h = torch.zeros(B, self.d_inner, self.d_state, dtype=torch.float32, device=hidden_states.device)
        x_f = x_conv.float(); dt_f = dt.float(); Bp_f = Bp.float()
        for t in range(T):
            dt_t = dt_f[:, t]
            B_t = Bp_f[:, t].unsqueeze(1)
            x_t_f = x_f[:, t].unsqueeze(-1)
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            dB_x = (dt_t.unsqueeze(-1) * B_t) * x_t_f
            h = dA * h + dB_x

        state = {"h": h, "conv_state": conv_state}
        return y_full, state

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: (B, T, D). Returns (B, T, D)."""
        B, T, D = hidden_states.shape

        xz = self.in_proj(hidden_states)  # (B, T, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)         # each (B, T, d_inner)

        # Causal conv: (B, T, d_inner) -> (B, d_inner, T) -> conv -> trim padding
        x = x.transpose(1, 2)
        x = self.conv1d(x)[..., :T]
        x = x.transpose(1, 2)
        x = self.act(x)

        x_dbl = self.x_proj(x)  # (B, T, dt_rank + 2*d_state)
        dt, Bp, Cp = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)   # (B, T, d_inner)
        dt = F.softplus(dt)     # discretization step

        A = -torch.exp(self.A_log.float())   # (d_inner, d_state)

        # Cast to fp32 for numerical stability of the scan.
        x_f = x.float().contiguous()
        dt_f = dt.float().contiguous()
        Bp_f = Bp.float().contiguous()
        Cp_f = Cp.float().contiguous()
        D_f = self.D.float().contiguous()

        backend = getattr(self, "_scan_backend", "auto")
        force_loop = bool(getattr(self, "_force_time_loop", False))
        scan_op = None if force_loop else _try_load_cpp_scan()

        if backend == "assoc" and not force_loop:
            from cpu_mamba.scan_assoc import selective_scan_assoc, selective_scan_assoc_chunked
            chunk_T = int(getattr(self, "_assoc_chunk_T", 0) or 0)
            if chunk_T > 0:
                y = selective_scan_assoc_chunked(x_f, dt_f, A.contiguous(), Bp_f, Cp_f, D_f, chunk_T=chunk_T)
            else:
                y = selective_scan_assoc(x_f, dt_f, A.contiguous(), Bp_f, Cp_f, D_f)
        elif scan_op is not None and not force_loop:
            # Fast path: C++ kernel runs the full T-step recurrence + skip + sum-to-y
            y = scan_op.selective_scan_cpu(x_f, dt_f, A.contiguous(), Bp_f, Cp_f, D_f)
        else:
            # Time-loop fallback (parity reference). y_t = C_t · h_t + D · x_t
            h = torch.zeros(B, self.d_inner, self.d_state,
                            device=x.device, dtype=torch.float32)
            ys = []
            for t in range(T):
                dt_t = dt_f[:, t]                                 # (B, d_inner)
                B_t = Bp_f[:, t].unsqueeze(1)                     # (B, 1, d_state)
                C_t = Cp_f[:, t].unsqueeze(1)                     # (B, 1, d_state)
                x_t = x_f[:, t].unsqueeze(-1)                     # (B, d_inner, 1)
                dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
                dB_x = (dt_t.unsqueeze(-1) * B_t) * x_t
                h = dA * h + dB_x
                y_t = (h * C_t).sum(dim=-1)                       # (B, d_inner)
                ys.append(y_t)
            y = torch.stack(ys, dim=1)                            # (B, T, d_inner)
            y = y + D_f * x_f                                     # skip connection

        y = y.to(x.dtype)
        y = y * self.act(z)                                       # gating
        return self.out_proj(y)


def assert_state_dict_compatible(cpu_mamba: CPUMamba, sd: dict):
    """Check that `sd` (from mamba_ssm.Mamba) is loadable into `cpu_mamba`."""
    own = dict(cpu_mamba.state_dict())
    extra = set(sd.keys()) - set(own.keys())
    missing = set(own.keys()) - set(sd.keys())
    if extra:
        raise ValueError(f"unexpected keys in source state_dict: {sorted(extra)}")
    if missing:
        raise ValueError(f"missing keys in source state_dict: {sorted(missing)}")
    for k in own:
        if own[k].shape != sd[k].shape:
            raise ValueError(f"shape mismatch on {k}: own {own[k].shape} vs src {sd[k].shape}")


