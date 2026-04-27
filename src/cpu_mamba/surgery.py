"""Module surgery + import compat helpers.

Two ways to use cpu_mamba in a larger model:

1. **Stub-then-import** (recommended for new code paths):
       from cpu_mamba.surgery import install_mamba_ssm_stub
       install_mamba_ssm_stub()
       # now `from mamba_ssm import Mamba` returns CPUMamba
       import my_model_module

2. **Build-then-swap** (for code already running with `mamba_ssm` imported):
       model = MyModelWithMambaBlocks(...)
       model.load_state_dict(torch.load("ckpt.pt"))
       from cpu_mamba.surgery import swap_mamba_to_cpu
       swap_mamba_to_cpu(model)  # walks module tree, replaces every Mamba
       model.cpu().eval()
"""
from __future__ import annotations

import sys
import types
from typing import Optional

import torch.nn as nn

from cpu_mamba._mamba import CPUMamba


def install_mamba_ssm_stub(force: bool = False) -> None:
    """Install a stub `mamba_ssm` module that aliases `Mamba` to `CPUMamba`.

    After calling this, `from mamba_ssm import Mamba` returns `CPUMamba`. Code
    that imports `mamba_ssm` will work even if the real package isn't installed.

    Args:
        force: replace an already-loaded `mamba_ssm` module.
    """
    if "mamba_ssm" in sys.modules and not force:
        return
    stub = types.ModuleType("mamba_ssm")
    stub.Mamba = CPUMamba
    stub.__version__ = f"cpu-mamba-stub"
    sys.modules["mamba_ssm"] = stub


def swap_mamba_to_cpu(module: nn.Module, verbose: bool = False) -> int:
    """Walk `module` and replace every `mamba_ssm.Mamba` instance with `CPUMamba`.

    State is transferred via `state_dict()`, so the swap preserves trained weights.

    Returns the number of blocks replaced.
    """
    try:
        import mamba_ssm
        mamba_class = mamba_ssm.Mamba
    except Exception:
        # mamba_ssm not importable; nothing to do
        return 0

    if mamba_class is CPUMamba:
        # Stub already installed — nothing to swap
        return 0

    n_replaced = 0
    for name, parent in list(module.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, mamba_class):
                cpu_block = CPUMamba(
                    d_model=child.d_model,
                    d_state=child.d_state,
                    d_conv=child.d_conv,
                    expand=child.expand,
                )
                cpu_block.load_state_dict(child.state_dict(), strict=True)
                setattr(parent, child_name, cpu_block)
                n_replaced += 1
                if verbose:
                    print(f"swapped {name}.{child_name}: mamba_ssm.Mamba → CPUMamba")
    return n_replaced
