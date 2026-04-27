"""cpu-mamba — CPU-only Mamba S6 block, state_dict-compatible with mamba_ssm.

Drop-in replacement for `mamba_ssm.Mamba` on CPU-only environments. Loads
the same state_dict (identical key names + shapes) so a model trained with
`mamba_ssm.Mamba` on GPU can be served on a CPU box with no retrain.

Two scan implementations, identical numerics:
  * Default: JIT-compiled C++ kernel with OpenMP across (batch × d_inner),
    sequential across timesteps. ~10x faster than the Python fallback.
  * Fallback: pure-PyTorch time loop (slow but always works; useful as a
    parity reference when porting/debugging).

Quick start:
    from cpu_mamba import CPUMamba
    block = CPUMamba(d_model=384, d_state=24, d_conv=4, expand=2)
    block.load_state_dict(trained_mamba_block.state_dict())  # from mamba_ssm
    y = block(x)  # x: (B, T, d_model) on CPU

Force the time-loop fallback (for parity testing):
    CPU_MAMBA_FORCE_TIME_LOOP=1 python ...
"""
from cpu_mamba._mamba import CPUMamba, assert_state_dict_compatible

__version__ = "0.2.0"
__all__ = ["CPUMamba", "assert_state_dict_compatible", "__version__"]
