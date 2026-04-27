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

        scan_op = _try_load_cpp_scan()
        if scan_op is not None and not getattr(self, "_force_time_loop", False):
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


