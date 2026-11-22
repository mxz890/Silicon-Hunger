#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2: freeze the developed graph, learn XOR, allow local growth on stall."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import networkx as nx

from graph_core import DevelopmentalGraph, INPUTS, OUTPUT, sigmoid

ROOT = Path(__file__).resolve().parent
XOR = [(0.0, 0.0, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)]


class XORLearner:
    def __init__(self, model: DevelopmentalGraph, input_ids=INPUTS, data=None):
        self.model = model
        self.graph = model.graph
        self.inputs = list(input_ids)
        self.data = list(data) if data is not None else [((a, b), y) for a, b, y in XOR]
        self.output = OUTPUT
        # The developmental phase created the topology, not a trained
        # function.  Reset trainable parameters to small values so the
        # dense 40-node graph does not start fully saturated.
        rng = random.Random(42)
        self.biases = {i: 0.0 for i in model.nodes if i not in self.inputs}
        self.weights = {
            (u, v): rng.uniform(-1.2, 1.2) / math.sqrt(max(1, self.graph.in_degree(v)))
            for u, v in self.graph.edges
        }
        self.topo = list(nx.topological_sort(self.graph))

    def forward(self, values) -> tuple[dict[int, float], dict[int, float]]:
        if isinstance(values, dict):
            act = {i: float(values.get(i, 0.0)) for i in self.inputs}
        else:
            act = {i: float(values[k]) for k, i in enumerate(self.inputs)}
        z = {}
        for v in self.topo:
            if v in self.inputs:
                continue
            pre = self.biases.get(v, 0.0)
            for u in self.graph.predecessors(v):
                pre += self.weights.get((u, v), 0.0) * act.get(u, 0.0)
            z[v] = pre
            act[v] = sigmoid(pre)
        return act, z

    def evaluate(self) -> tuple[float, float]:
        loss = 0.0
        correct = 0
        for values, target in self.data:
            act, _ = self.forward(values)
            out = act.get(self.output, 0.0)
            loss += 0.5 * (out - target) ** 2
            correct += int((out >= 0.5) == (target >= 0.5))
        return loss / len(self.data), correct / len(self.data)

    def gradients(self) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
        grad_b = {v: 0.0 for v in self.biases}
        grad_w = {e: 0.0 for e in self.weights}
        for values, target in self.data:
            act, z = self.forward(values)
            out = act.get(self.output, 0.0)
            dz = {self.output: (out - target) * out * (1.0 - out)}
            for v in reversed(self.topo):
                if v in self.inputs:
                    continue
                g = dz.get(v, 0.0)
                if abs(g) < 1e-12:
                    continue
                grad_b[v] += g
                for u in self.graph.predecessors(v):
                    grad_w[(u, v)] = grad_w.get((u, v), 0.0) + g * act.get(u, 0.0)
                    upstream = g * self.weights.get((u, v), 0.0) * act.get(u, 0.0) * (1.0 - act.get(u, 0.0))
                    dz[u] = dz.get(u, 0.0) + upstream
        n = len(self.data)
        return {k: v / n for k, v in grad_b.items()}, {k: v / n for k, v in grad_w.items()}

    def step(self, lr: float) -> None:
        grad_b, grad_w = self.gradients()
        for v in self.biases:
            self.biases[v] -= lr * grad_b.get(v, 0.0)
        for e in self.weights:
            self.weights[e] -= lr * grad_w.get(e, 0.0)

    def write_back(self) -> None:
        for v, b in self.biases.items():
            if v in self.model.nodes:
                self.model.nodes[v].bias = float(b)
        for (u, v), w in self.weights.items():
            if self.model.graph.has_edge(u, v):
                self.model.graph[u][v]["weight"] = float(w)

    def grow(self) -> dict:
        node = self.model.grow_near_output()
        if node is None:
            return {"ok": False}
        self.biases[node.idx] = node.bias
        for u, v in self.model.graph.edges:
            if (u, v) not in self.weights:
                self.weights[(u, v)] = random.uniform(-0.3, 0.3) / math.sqrt(max(1, self.model.graph.in_degree(v)))
        self.topo = list(nx.topological_sort(self.model.graph))
        return {"ok": True, "id": node.idx, "bias": node.bias}

    def train(self, epochs: int = 1600, lr: float = 2.0, max_growth: int = 3) -> dict:
        history = []
        growth = []
        last_growth_epoch = 0
        best = (1e9, 0.0)
        for epoch in range(1, epochs + 1):
            self.step(lr)
            loss, acc = self.evaluate()
            if loss < best[0]:
                best = (loss, acc)
            if epoch == 1 or epoch % 100 == 0:
                history.append({"epoch": epoch, "loss": round(loss, 5), "acc": round(acc, 3)})
                print(f"epoch={epoch:04d} loss={loss:.5f} acc={acc:.2f} nodes={self.graph.number_of_nodes()}")
            if loss < 0.02 and acc >= 1.0:
                break
            if epoch - last_growth_epoch >= 500 and loss > 0.12 and len(growth) < max_growth:
                event = self.grow()
                event["epoch"] = epoch
                event["loss_before"] = round(loss, 5)
                growth.append(event)
                last_growth_epoch = epoch
                print("LOCAL_GROWTH", event)
        loss, acc = self.evaluate()
        return {
            "epochs_run": epoch,
            "final_loss": round(loss, 6),
            "final_accuracy": round(acc, 3),
            "best_loss": round(best[0], 6),
            "best_accuracy": round(best[1], 3),
            "growth": growth,
            "history": history,
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1600)
    parser.add_argument("--lr", type=float, default=2.0)
    parser.add_argument("--output", default="phase2_result.json")
    args = parser.parse_args()
    graph_path = ROOT / "phase1_graph.json"
    if not graph_path.exists():
        raise SystemExit("Run phase1_development.py first to create phase1_graph.json")
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    model = DevelopmentalGraph.from_dict(data)
    model.max_nodes = model.graph.number_of_nodes() + 3
    print("LOADED", graph_path)
    print(
        "TOPOLOGY",
        "nodes", model.graph.number_of_nodes(),
        "edges", model.graph.number_of_edges(),
        "input", list(INPUTS),
        "output", OUTPUT,
    )
    learner = XORLearner(model)
    result = learner.train(epochs=args.epochs, lr=args.lr)
    learner.write_back()
    predictions = []
    for values, y in learner.data:
        act, _ = learner.forward(values)
        predictions.append(
            {"inputs": list(values), "target": y, "output": round(act.get(OUTPUT, 0.0), 5)}
        )
    payload = {
        "inputs": list(learner.inputs),
        "x_or": [{"inputs": list(values), "target": y} for values, y in learner.data],
        "topology_before": {
            "nodes": model.graph.number_of_nodes(),
            "edges": model.graph.number_of_edges(),
        },
        **result,
        "predictions": predictions,
        "final_graph": model.to_dict(),
    }
    out_path = ROOT / args.output
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    grown_path = out_path.with_name(out_path.stem.replace("_result", "_grown_graph") + ".json")
    grown_path.write_text(
        json.dumps(model.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("FINAL", json.dumps({k: payload[k] for k in ("final_loss", "final_accuracy", "growth", "nodes", "edges")}, ensure_ascii=False))
    print("SAVED", out_path)
    print("SAVED", grown_path)


if __name__ == "__main__":
    main()
