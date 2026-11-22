#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2 extra tasks on the developed graph: 3-bit parity and delayed match."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from graph_core import DevelopmentalGraph
from phase2_learning import XORLearner

ROOT = Path(__file__).resolve().parent


def parity3():
    data = []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                data.append(((float(a), float(b), float(c)), float(a ^ b ^ c)))
    return [0, 1, 3], data


def delayed_match():
    data = [((0.0, 0.0), 1.0), ((0.0, 1.0), 0.0), ((1.0, 0.0), 0.0), ((1.0, 1.0), 1.0)]
    return [0, 1], data


def xor_data():
    return [0, 1], [((0.0, 0.0), 0.0), ((0.0, 1.0), 1.0), ((1.0, 0.0), 1.0), ((1.0, 1.0), 0.0)]


def run_task(base_data: dict, name: str, input_ids, data, epochs: int = 1600):
    model = DevelopmentalGraph.from_dict(base_data)
    model.max_nodes = model.graph.number_of_nodes() + 3
    print(
        f"TASK {name} inputs={input_ids} nodes={model.graph.number_of_nodes()} "
        f"edges={model.graph.number_of_edges()} "
        f"paths={[nx.has_path(model.graph, i, 9) for i in input_ids]}"
    )
    learner = XORLearner(model, input_ids=input_ids, data=data)
    result = learner.train(epochs=epochs, lr=2.0, max_growth=2)
    learner.write_back()
    predictions = []
    for values, target in learner.data:
        act, _ = learner.forward(values)
        predictions.append(
            {"inputs": list(values), "target": target, "output": round(act.get(9, 0.0), 5)}
        )
    return {
        "name": name,
        "input_ids": list(input_ids),
        "topology_before": {
            "nodes": int(base_data.get("nodes") and len(base_data["nodes"])),
            "edges": len(base_data["edges"]),
        },
        **result,
        "predictions": predictions,
    }


def main() -> None:
    graph_path = ROOT / "phase1_graph.json"
    base = json.loads(graph_path.read_text(encoding="utf-8"))
    tasks = []
    for name, builder in [
        ("xor", xor_data),
        ("parity3", parity3),
        ("delayed_match", delayed_match),
    ]:
        input_ids, data = builder()
        tasks.append(run_task(base, name, input_ids, data))
    out = {"tasks": tasks}
    (ROOT / "phase2_tasks_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for task in tasks:
        print(
            "RESULT",
            task["name"],
            "loss", task["final_loss"],
            "acc", task["final_accuracy"],
            "nodes", task["nodes"],
            "growth", len(task["growth"]),
        )
    print("SAVED", ROOT / "phase2_tasks_result.json")


if __name__ == "__main__":
    main()
