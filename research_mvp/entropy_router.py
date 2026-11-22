#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task B: route a signal through a binary tree using secrets vs PRNG."""

from __future__ import annotations

import json
import math
import random
import secrets
from collections import Counter
from pathlib import Path

LEAVES = [
    "答案是 2。",
    "两个一相加，得到 2。",
    "2。",
    "1 + 1 = 2。",
    "结果就是 2，没有别的。",
    "二。",
    "如果你问 1+1，那就是 2。",
    "2 —— 这是确定的结果。",
]


def route(bits: list[int]) -> tuple[str, int]:
    node = 1
    for bit in bits:
        node = node * 2 + bit
    leaf = node - 8
    return "".join(str(b) for b in bits), leaf


def shannon_entropy(values: list[str]) -> float:
    counts = Counter(values)
    total = len(values)
    return -sum((n / total) * math.log2(n / total) for n in counts.values()) if total else 0.0


def longest_run(values: list[str]) -> int:
    best = 0
    run = 1
    for previous, current in zip(values, values[1:]):
        if previous == current:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return max(best, 1)


def collect(mode: str, n: int = 500) -> dict:
    rng = random.Random(42)
    outputs = []
    paths = []
    for _ in range(n):
        if mode == "fixed":
            bits = [0, 0, 0]
        elif mode == "pseudo":
            bits = [rng.randrange(2) for _ in range(3)]
        elif mode == "secrets":
            bits = [secrets.randbits(1) for _ in range(3)]
        else:
            raise ValueError(mode)
        path, leaf = route(bits)
        paths.append(path)
        outputs.append(LEAVES[leaf])
    return {
        "mode": mode,
        "n": n,
        "unique_outputs": len(set(outputs)),
        "unique_paths": len(set(paths)),
        "entropy_bits": round(shannon_entropy(outputs), 4),
        "longest_run": longest_run(outputs),
        "most_common": Counter(outputs).most_common(3),
        "path_distribution": dict(Counter(paths)),
        "sample": outputs[:12],
    }


def main() -> None:
    results = [collect(mode) for mode in ("fixed", "pseudo", "secrets")]
    out = Path(__file__).resolve().parent
    (out / "entropy_router_result.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for row in results:
        print(
            f"{row['mode']:8s} unique={row['unique_outputs']} "
            f"entropy={row['entropy_bits']:.3f} "
            f"longest_run={row['longest_run']}"
        )
        print("  sample:", " | ".join(row["sample"][:5]))
    print("SAVED", out / "entropy_router_result.json")


if __name__ == "__main__":
    main()
