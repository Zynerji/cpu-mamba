"""Weight-only quantization for CPUMamba.

Three schemes (each with per-channel symmetric scale):
  * int8:  signed 8-bit, 4× weight compression vs fp32
  * int4:  signed 4-bit packed (2 weights/byte), 8× compression
  * nf4:   4-bit NormalFloat code (QLoRA-style LUT), 8× compression with
           better accuracy than int4 on normal-distributed weights

All schemes are weight-only — activations stay in fp32 (or whatever dtype
the input is). Quantized weights dequantize on the fly during forward.
This keeps the API simple and the math correct, at the cost of some
forward-time overhead (one extra matmul-equivalent dequant per layer).

Use:
    from cpu_mamba import CPUMamba
    from cpu_mamba.quant import quantize_block, QuantConfig

    block = CPUMamba(d_model=384, d_state=24)
    block.load_state_dict(trained_state_dict)

    quantize_block(block, QuantConfig(scheme="int4"))   # in-place
    y = block(x)  # 8× smaller weights, ~5-10% latency overhead
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# NF4 LUT (QLoRA / Dettmers et al. 2023). 16 values, quantiles of N(0,1) mapped to [-1, 1].
_NF4_LUT_VALUES = (
    -1.0, -0.6961928, -0.5250730, -0.39491748,
    -0.28444138, -0.18477343, -0.09105602, 0.0,
    0.07958029, 0.16093020, 0.24611230, 0.33791524,
    0.44070983, 0.56261897, 0.72295684, 1.0,
)


def _nf4_lut(device, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(_NF4_LUT_VALUES, device=device, dtype=dtype)


@dataclass
class QuantConfig:
    scheme: str = "int8"  # "int8" | "int4" | "nf4"

    def __post_init__(self):
        if self.scheme not in ("int8", "int4", "nf4"):
            raise ValueError(f"unknown scheme {self.scheme!r}")


# ============================================================================
# Quantization primitives
# ============================================================================

def quantize_int8(weight: torch.Tensor) -> tuple:
    """Per-channel symmetric int8 quant. weight: (out, in)."""
    w = weight.float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0     # (out, 1)
    w_q = (w / scale).round().clamp(-127, 127).to(torch.int8)             # (out, in)
    return w_q, scale.squeeze(-1)                                          # scale: (out,)


def dequantize_int8(w_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return w_q.float() * scale.unsqueeze(-1)


def quantize_int4(weight: torch.Tensor) -> tuple:
    """Per-channel symmetric int4 quant, packed 2 weights/byte."""
    w = weight.float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 7.0
    w_q = (w / scale).round().clamp(-7, 7).to(torch.int8)                  # (out, in)

    # Convert to unsigned [0, 14] then pack
    in_dim = w_q.shape[-1]
    pad = in_dim % 2
    if pad:
        w_q = F.pad(w_q, (0, 1))                                            # right-pad to even
    w_u = (w_q + 7).to(torch.uint8)                                         # [0, 14]
    high = w_u[..., 0::2]
    low = w_u[..., 1::2]
    packed = (high << 4) | low                                              # (out, ceil(in/2))
    return packed, scale.squeeze(-1), in_dim


def dequantize_int4(packed: torch.Tensor, scale: torch.Tensor, in_dim: int) -> torch.Tensor:
    """Unpack int4 nibbles, recenter to signed [-7, 7], multiply by scale."""
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    n_out, n_packed = packed.shape
    n_unpacked = n_packed * 2
    w_u = torch.empty((n_out, n_unpacked), dtype=torch.uint8, device=packed.device)
    w_u[..., 0::2] = high
    w_u[..., 1::2] = low
    w_signed = w_u.to(torch.int8) - 7                                       # [-7, 7]
    w_q = w_signed[..., :in_dim]                                            # drop padding
    return w_q.float() * scale.unsqueeze(-1)


def quantize_nf4(weight: torch.Tensor) -> tuple:
    """Per-channel NF4 quant (QLoRA-style). Packed 2 codes/byte."""
    w = weight.float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)               # (out, 1)
    w_norm = (w / scale).clamp(-1.0, 1.0)                                    # in [-1, 1]
    lut = _nf4_lut(w.device, w.dtype)                                        # (16,)
    # Find nearest LUT entry for each weight (vectorized over (out, in))
    distances = (w_norm.unsqueeze(-1) - lut).abs()                           # (out, in, 16)
    idx = distances.argmin(dim=-1).to(torch.uint8)                           # (out, in), values 0..15

    in_dim = idx.shape[-1]
    pad = in_dim % 2
    if pad:
        idx = F.pad(idx, (0, 1))
    high = idx[..., 0::2]
    low = idx[..., 1::2]
    packed = (high << 4) | low
    return packed, scale.squeeze(-1), in_dim


def dequantize_nf4(packed: torch.Tensor, scale: torch.Tensor, in_dim: int) -> torch.Tensor:
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    n_out, n_packed = packed.shape
    n_unpacked = n_packed * 2
    idx = torch.empty((n_out, n_unpacked), dtype=torch.uint8, device=packed.device)
    idx[..., 0::2] = high
    idx[..., 1::2] = low
    idx = idx[..., :in_dim].to(torch.long)
    lut = _nf4_lut(packed.device, torch.float32)
    return lut[idx] * scale.unsqueeze(-1)


# ============================================================================
# Quantized Linear module
# ============================================================================

class QuantizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear with weight-only quantization.

    Buffers (not parameters — quant weights aren't trained):
        scheme: str
        weight_q: int8 (int8 scheme) or uint8 packed (int4/nf4)
        scale: (out,) float32
        in_features, out_features: int
    """

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor], scheme: str):
        super().__init__()
        self.scheme = scheme
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]

        if scheme == "int8":
            w_q, scale = quantize_int8(weight)
            self.register_buffer("weight_q", w_q)
            self.register_buffer("scale", scale)
            self._packed_in_dim = self.in_features
        elif scheme == "int4":
            packed, scale, in_dim = quantize_int4(weight)
            self.register_buffer("weight_q", packed)
            self.register_buffer("scale", scale)
            self._packed_in_dim = in_dim
        elif scheme == "nf4":
            packed, scale, in_dim = quantize_nf4(weight)
            self.register_buffer("weight_q", packed)
            self.register_buffer("scale", scale)
            self._packed_in_dim = in_dim
        else:
            raise ValueError(f"unknown scheme {scheme!r}")

        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone())
        else:
            self.register_parameter("bias", None)

    def dequantize_weight(self) -> torch.Tensor:
        if self.scheme == "int8":
            return dequantize_int8(self.weight_q, self.scale)
        elif self.scheme == "int4":
            return dequantize_int4(self.weight_q, self.scale, self._packed_in_dim)
        elif self.scheme == "nf4":
            return dequantize_nf4(self.weight_q, self.scale, self._packed_in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize_weight()
        return F.linear(x, w.to(x.dtype), self.bias)

    def extra_repr(self) -> str:
        return (f"in={self.in_features}, out={self.out_features}, "
                f"scheme={self.scheme!r}")


# ============================================================================
# Block-level surgery
# ============================================================================

# Linear submodule names inside CPUMamba (and the standard mamba_ssm.Mamba).
_QUANTIZABLE_LINEARS = ("in_proj", "x_proj", "dt_proj", "out_proj")


def quantize_block(module: nn.Module, config: QuantConfig,
                   target_names: Iterable[str] = _QUANTIZABLE_LINEARS) -> int:
    """Replace nn.Linear children of `module` (named in target_names) with
    QuantizedLinear in-place. Returns the number replaced.

    Recurses through children; safe to call on a model that contains many
    CPUMamba blocks. Conv1d / A_log / D / biases are left at their original dtype.
    """
    n_replaced = 0
    target_set = set(target_names)
    for name, parent in list(module.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and child_name in target_set:
                qlin = QuantizedLinear(
                    weight=child.weight.detach(),
                    bias=child.bias.detach() if child.bias is not None else None,
                    scheme=config.scheme,
                )
                qlin = qlin.to(device=child.weight.device)
                setattr(parent, child_name, qlin)
                n_replaced += 1
    return n_replaced


def memory_footprint(module: nn.Module) -> dict:
    """Report per-component memory footprint of a (possibly quantized) module."""
    fp_bytes = 0
    quant_bytes = 0
    for p in module.parameters():
        fp_bytes += p.numel() * p.element_size()
    for b_name, b in module.named_buffers():
        bytes_ = b.numel() * b.element_size()
        if b_name.endswith(".weight_q") or b_name.endswith(".scale"):
            quant_bytes += bytes_
        else:
            fp_bytes += bytes_
    return {
        "param_bytes": fp_bytes,
        "quant_buffer_bytes": quant_bytes,
        "total_bytes": fp_bytes + quant_bytes,
    }
