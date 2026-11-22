#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 1: 200-round development phase; saves a frozen graph for XOR."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from graph_core import DevelopmentalGraph

ROOT = Path(__file__).resolve().parent
EXTREME = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
CONFLICT_A = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
CONFLICT_B = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
CATASTROPHE_STEP = 50
CATASTROPHE_END = 70


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    model = DevelopmentalGraph(seed=args.seed)
    log = []

    for step in range(1, args.cycles + 1):
        if step == CATASTROPHE_STEP:
            env = list(EXTREME)
            model.enter_catastrophe()
            print(f"CATASTROPHE_START step={step} env={env} prune_after={model.prune_after}")
        elif step > CATASTROPHE_STEP:
            env = list(EXTREME)
        else:
            env = [rng.uniform(-1.0, 1.0) for _ in range(7)]

        event = model.step(env, conflict_envs=(CONFLICT_A, CONFLICT_B))
        mutation = None
        if step % 10 == 0:
            mutation = model.topology_mutation()
            print(f"MUTATION step={step} {mutation}")

        if step == CATASTROPHE_END:
            model.exit_catastrophe()
            print(f"CATASTROPHE_END step={step} prune_after={model.prune_after}")

        if step == 1 or step % 20 == 0 or step == args.cycles:
            entry = {"step": step, **event, "mutation": mutation, "summary": model.summary()}
            log.append(entry)
            print(
                f"step={step:03d} nodes={event['nodes']} edges={event['edges']} "
                f"max_degree={model.summary()['max_degree']} hub={model.summary()['hub']} "
                f"hub_degree={model.summary()['hub_degree']} "
                f"capture={event['capture']} split={event['split']} reason={event['split_reason']} "
                f"conflict={event['conflict']}"
            )

    final = model.summary()
    (ROOT / "phase1_graph.json").write_text(
        json.dumps(model.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "phase1_log.json").write_text(
        json.dumps({"log": log, "final": final}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("FINAL", json.dumps(final, ensure_ascii=False))
    print("SAVED", ROOT / "phase1_graph.json")


if __name__ == "__main__":
    main()
