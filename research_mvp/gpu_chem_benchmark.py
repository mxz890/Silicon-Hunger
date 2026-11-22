#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark chemical diffusion on CPU vs RTX 5070 Ti.

The GPU path keeps C, edges and sources resident on the GPU.  Only the final
gain vector is copied back.  This isolates the cost of a persistent chemical
field from the CPU/GPU transfer cost.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from large_math_2000 import N, build_graph

ROOT = Path(__file__).resolve().parent
STEPS = 20
DT = 0.08
DECAY = np.array([0.12, 0.06, 0.16, 0.20, 0.10, 0.16], dtype=np.float32)
DIFFUSION = np.array([0.10, 0.06, 0.08, 0.12, 0.09, 0.10], dtype=np.float32)


def source_array():
    sources = np.zeros((6, N), dtype=np.float32)
    sources[0, 0] = 0.22
    sources[1, 1] = 0.12
    sources[2, 0] = 0.18
    sources[3, 1] = 0.14
    sources[4, 1999] = 0.16
    sources[5, 1999] = 0.16
    return sources


def gains_np(C):
    exc = 0.8 * C[5] + 0.6 * C[0] + 0.4 * C[2] + 0.3 * C[3]
    inh = 0.9 * C[4] + 0.4 * C[1]
    gain = 1.0 + 0.8 * exc / (1.0 + exc) - 0.45 * inh / (1.0 + inh)
    return np.clip(gain, 0.4, 1.8)


def cpu_run(eu, ev, ew, steps=STEPS):
    C = np.full((6, N), 0.05, dtype=np.float32)
    sources = source_array()
    for _ in range(steps):
        for k in range(6):
            incoming = np.bincount(ev, weights=ew * C[k, eu], minlength=N)
            degree = np.bincount(ev, weights=ew, minlength=N)
            flux = incoming - degree * C[k]
            C[k] = C[k] + DT * (DIFFUSION[k] * flux - DECAY[k] * C[k] + sources[k])
            C[k] = np.clip(C[k], 0.0, 3.0)
    return C, gains_np(C)


def gpu_run(eu, ev, ew, device="cuda", steps=STEPS, sync_each=False, include_transfer=True):
    if include_transfer:
        eu_t = torch.as_tensor(eu, dtype=torch.long, device=device)
        ev_t = torch.as_tensor(ev, dtype=torch.long, device=device)
        ew_t = torch.as_tensor(ew, dtype=torch.float32, device=device)
    else:
        eu_t, ev_t, ew_t = eu, ev, ew
    C = torch.full((6, N), 0.05, dtype=torch.float32, device=device)
    sources = torch.as_tensor(source_array(), dtype=torch.float32, device=device)
    decay = torch.as_tensor(DECAY, dtype=torch.float32, device=device).view(6, 1)
    diffusion = torch.as_tensor(DIFFUSION, dtype=torch.float32, device=device).view(6, 1)
    degree = torch.bincount(ev_t, weights=ew_t, minlength=N).to(torch.float32)
    with torch.no_grad():
        for _ in range(steps):
            incoming = torch.zeros_like(C)
            incoming.index_add_(1, ev_t, C[:, eu_t] * ew_t)
            flux = incoming - degree.unsqueeze(0) * C
            C = C + DT * (diffusion * flux - decay * C + sources)
            C = torch.clamp(C, 0.0, 3.0)
            if sync_each:
                torch.cuda.synchronize()
        exc = 0.8 * C[5] + 0.6 * C[0] + 0.4 * C[2] + 0.3 * C[3]
        inh = 0.9 * C[4] + 0.4 * C[1]
        gain = 1.0 + 0.8 * exc / (1.0 + exc) - 0.45 * inh / (1.0 + inh)
        gain = torch.clamp(gain, 0.4, 1.8)
        torch.cuda.synchronize()
        return C.detach().cpu().numpy(), gain.detach().cpu().numpy()


def timed(fn, repeats=5):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return out, times


def main() -> None:
    pos, incoming, edges, eu, ev, ew = build_graph()
    print("GRAPH", "nodes", N, "edges", len(edges))
    print("DEVICE", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

    eu_np, ev_np, ew_np = np.asarray(eu), np.asarray(ev), np.asarray(ew)
    cpu_out, cpu_times = timed(lambda: cpu_run(eu_np, ev_np, ew_np), repeats=5)
    print("CPU_MS", [round(t, 3) for t in cpu_times], "median", round(float(np.median(cpu_times)), 3))

    if not torch.cuda.is_available():
        payload = {"error": "CUDA not available"}
        (ROOT / "gpu_chem_result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    # Persistent GPU: edge tensors and C already resident.
    eu_t = torch.as_tensor(eu, dtype=torch.long, device="cuda")
    ev_t = torch.as_tensor(ev, dtype=torch.long, device="cuda")
    ew_t = torch.as_tensor(ew, dtype=torch.float32, device="cuda")
    gpu_out, gpu_times = timed(lambda: gpu_run(eu_t, ev_t, ew_t, include_transfer=False), repeats=5)
    print("GPU_PERSISTENT_MS", [round(t, 3) for t in gpu_times], "median", round(float(np.median(gpu_times)), 3))

    cold_out, cold_times = timed(lambda: gpu_run(eu_np, ev_np, ew_np, include_transfer=True), repeats=5)
    print("GPU_COLD_MS", [round(t, 3) for t in cold_times], "median", round(float(np.median(cold_times)), 3))

    sync_out, sync_times = timed(lambda: gpu_run(eu_t, ev_t, ew_t, include_transfer=False, sync_each=True), repeats=3)
    print("GPU_SYNC_EACH_MS", [round(t, 3) for t in sync_times], "median", round(float(np.median(sync_times)), 3))

    diff = float(np.max(np.abs(cpu_out[0] - gpu_out[0])))
    gain_diff = float(np.max(np.abs(cpu_out[1] - gpu_out[1])))
    payload = {
        "graph": {"nodes": N, "edges": len(edges)},
        "device": torch.cuda.get_device_name(0),
        "steps": STEPS,
        "cpu_ms": [round(float(t), 4) for t in cpu_times],
        "gpu_persistent_ms": [round(float(t), 4) for t in gpu_times],
        "gpu_cold_ms": [round(float(t), 4) for t in cold_times],
        "gpu_sync_each_ms": [round(float(t), 4) for t in sync_times],
        "median": {
            "cpu": round(float(np.median(cpu_times)), 4),
            "gpu_persistent": round(float(np.median(gpu_times)), 4),
            "gpu_cold": round(float(np.median(cold_times)), 4),
            "gpu_sync_each": round(float(np.median(sync_times)), 4),
        },
        "speedup_persistent": round(float(np.median(cpu_times) / np.median(gpu_times)), 3),
        "speedup_cold": round(float(np.median(cpu_times) / np.median(cold_times)), 3),
        "max_concentration_diff": round(diff, 6),
        "max_gain_diff": round(gain_diff, 6),
        "output_files": {"result_json": "research_mvp/gpu_chem_result.json"},
    }
    (ROOT / "gpu_chem_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
