"""Benchmark CPUMamba: C++ kernel vs Python time-loop, with parity check.

Run:
    python bench/bench.py
"""
from __future__ import annotations

import argparse
import time

import torch

from cpu_mamba import CPUMamba


def time_one(block, x, n=2):
    with torch.no_grad():
        _ = block(x)  # warmup (triggers C++ JIT build on first call)
    times = []
    for _ in range(n):
        t0 = time.time()
        with torch.no_grad():
            _ = block(x)
        times.append(time.time() - t0)
    return sum(times) / len(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=768)
    ap.add_argument("--d_state", type=int, default=24)
    ap.add_argument("--d_conv", type=int, default=4)
    ap.add_argument("--expand", type=int, default=2)
    ap.add_argument("--T", type=int, default=180)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    block = CPUMamba(d_model=args.d_model, d_state=args.d_state,
                     d_conv=args.d_conv, expand=args.expand).eval()

    n_params = sum(p.numel() for p in block.parameters())
    print(f"CPUMamba: d_model={args.d_model} d_state={args.d_state} d_conv={args.d_conv} "
          f"expand={args.expand} T={args.T} threads={args.threads}")
    print(f"  params: {n_params:,}")
    print()

    print("=== Parity (C++ kernel vs Python time-loop) ===")
    x = torch.randn(4, args.T, args.d_model)
    block._force_time_loop = False
    with torch.no_grad():
        y_cpp = block(x).clone()
    block._force_time_loop = True
    with torch.no_grad():
        y_loop = block(x).clone()
    block._force_time_loop = False
    diff = (y_cpp - y_loop).abs()
    print(f"  B=4 T={args.T}: max_abs_diff={diff.max().item():.3e}  mean_abs_diff={diff.mean().item():.3e}")
    print()

    print("=== Throughput ===")
    for B in [16, 64, 151]:
        x = torch.randn(B, args.T, args.d_model)
        # C++
        block._force_time_loop = False
        cpp_t = time_one(block, x, n=3)
        # Python time-loop
        block._force_time_loop = True
        loop_t = time_one(block, x, n=2)
        block._force_time_loop = False
        speedup = loop_t / cpp_t
        print(f"  B={B:3d}: C++ {cpp_t*1000:7.1f} ms   Python {loop_t*1000:7.1f} ms   "
              f"speedup {speedup:5.1f}x")


if __name__ == "__main__":
    main()
