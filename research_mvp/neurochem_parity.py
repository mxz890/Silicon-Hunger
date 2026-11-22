#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3-bit parity with six common neurotransmitter fields.

This is a phenomenological test, not a biological brain simulation.
The chemical field is a graph reaction-diffusion layer that modulates
the gain of each node during parity learning.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import networkx as nx

from graph_core import DevelopmentalGraph
from phase2_learning import XORLearner

ROOT = Path(__file__).resolve().parent
INPUTS = [0, 1, 3]
OUTPUT = 9

CHEMICALS = ["DA", "5HT", "NE", "ACh", "GABA", "Glu"]
DECAY = {"DA": 0.12, "5HT": 0.06, "NE": 0.16, "ACh": 0.20, "GABA": 0.10, "Glu": 0.16}
DIFFUSION = {"DA": 0.10, "5HT": 0.06, "NE": 0.08, "ACh": 0.12, "GABA": 0.09, "Glu": 0.10}


def parity_data():
    data = []
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                data.append(((float(a), float(b), float(c)), float(a ^ b ^ c)))
    return data


class ChemField:
    def __init__(self, graph: nx.Graph, seed: int = 7):
        self.graph = graph
        self.rng = random.Random(seed)
        self.c = {name: {i: 0.05 for i in graph.nodes} for name in CHEMICALS}

    def step(self, sources: dict[str, dict[int, float]], dt: float = 0.20):
        for name in CHEMICALS:
            old = self.c[name]
            D = DIFFUSION[name]
            decay = DECAY[name]
            new = {}
            for i in self.graph.nodes:
                flux = 0.0
                for j, data in self.graph[i].items():
                    weight = float(data.get("weight", 1.0))
                    flux += weight * (old[j] - old[i])
                src = sources.get(name, {}).get(i, 0.0)
                value = old[i] + dt * (D * flux + src - decay * old[i])
                new[i] = max(0.0, min(3.0, value))
            self.c[name] = new

    def gains(self) -> dict[int, float]:
        gains = {}
        for i in self.graph.nodes:
            exc = (
                0.8 * self.c["Glu"][i]
                + 0.6 * self.c["DA"][i]
                + 0.4 * self.c["NE"][i]
                + 0.3 * self.c["ACh"][i]
            )
            inh = 0.9 * self.c["GABA"][i] + 0.4 * self.c["5HT"][i]
            gain = 1.0 + 0.8 * exc / (1.0 + exc) - 0.45 * inh / (1.0 + inh)
            gains[i] = max(0.35, min(1.8, gain))
        return gains

    def means(self) -> dict[str, float]:
        return {
            name: round(sum(self.c[name].values()) / max(1, len(self.c[name])), 5)
            for name in CHEMICALS
        }


class NeuroChemParityLearner(XORLearner):
    def __init__(self, model, chem: ChemField | None = None):
        super().__init__(model, input_ids=INPUTS, data=parity_data())
        self.chem = chem
        self.gains = {i: 1.0 for i in model.graph.nodes}
        if self.chem is not None:
            self.gains = self.chem.gains()

    def forward(self, values):
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
            z[v] = self.gains.get(v, 1.0) * pre
            act[v] = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z[v]))))
        return act, z

    def gradients(self):
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
                    upstream = (
                        g
                        * self.weights.get((u, v), 0.0)
                        * act.get(u, 0.0)
                        * (1.0 - act.get(u, 0.0))
                        * self.gains.get(u, 1.0)
                    )
                    dz[u] = dz.get(u, 0.0) + upstream
        n = len(self.data)
        return {k: v / n for k, v in grad_b.items()}, {k: v / n for k, v in grad_w.items()}

    def update_chem(self, loss: float, acc: float):
        if self.chem is None:
            return
        sources = {name: {} for name in CHEMICALS}
        hub = max(
            (i for i in self.model.nodes if self.model.nodes[i].kind == "hidden"),
            key=lambda i: self.model.graph.degree(i),
            default=OUTPUT,
        )
        for i in self.model.graph.nodes:
            sources["DA"][i] = 0.025 + 0.10 * acc
            sources["5HT"][i] = 0.020 + 0.04 * (1.0 - loss)
            sources["NE"][i] = 0.020 + 0.10 * loss
            sources["ACh"][i] = 0.025 + 0.06 * (1.0 - acc)
            sources["GABA"][i] = 0.015 + 0.06 * loss
            sources["Glu"][i] = 0.035
        for i in self.inputs:
            sources["NE"][i] += 0.08 * loss
            sources["ACh"][i] += 0.06 * (1.0 - acc)
        sources["DA"][OUTPUT] += 0.18 * acc
        sources["Glu"][OUTPUT] += 0.08
        sources["GABA"][hub] += 0.12 * loss
        sources["5HT"][hub] += 0.04 * (1.0 - loss)
        self.chem.step(sources)
        self.gains = self.chem.gains()

    def lr_factor(self) -> float:
        if self.chem is None:
            return 1.0
        mean = self.chem.means()
        factor = 1.0 + 0.25 * mean["DA"] + 0.20 * mean["ACh"] - 0.15 * mean["GABA"] + 0.05 * mean["5HT"]
        return max(0.5, min(1.8, factor))

    def train(self, epochs: int = 2400, lr: float = 2.0):
        history = []
        best = (1e9, 0.0)
        for epoch in range(1, epochs + 1):
            loss, acc = self.evaluate()
            if self.chem is not None:
                self.update_chem(loss, acc)
            effective_lr = lr * self.lr_factor()
            self.step(effective_lr)
            loss2, acc2 = self.evaluate()
            if loss2 < best[0]:
                best = (loss2, acc2)
            if epoch == 1 or epoch % 100 == 0:
                entry = {
                    "epoch": epoch,
                    "loss": round(loss2, 6),
                    "acc": round(acc2, 4),
                    "gain_min": round(min(self.gains.values()), 4),
                    "gain_max": round(max(self.gains.values()), 4),
                    "lr_factor": round(self.lr_factor(), 4),
                    "chem_mean": self.chem.means() if self.chem is not None else None,
                }
                history.append(entry)
                print(
                    f"epoch={epoch:04d} loss={loss2:.6f} acc={acc2:.3f} "
                    f"gain=[{min(self.gains.values()):.3f},{max(self.gains.values()):.3f}] "
                    f"chem={entry['chem_mean']}"
                )
            if loss2 < 0.02 and acc2 >= 1.0:
                break
        loss, acc = self.evaluate()
        return {
            "epochs_run": epoch,
            "final_loss": round(loss, 6),
            "final_accuracy": round(acc, 4),
            "best_loss": round(best[0], 6),
            "best_accuracy": round(best[1], 4),
            "history": history,
            "chem_final": self.chem.means() if self.chem is not None else None,
            "chem_per_node": {
                name: {str(i): round(v, 5) for i, v in self.chem.c[name].items()}
                for name in CHEMICALS
            } if self.chem is not None else None,
            "gains_final": {str(i): round(g, 5) for i, g in self.gains.items()},
        }


def expand_for_parity(model: DevelopmentalGraph, count: int = 12) -> DevelopmentalGraph:
    """Add hidden capacity near the three inputs and the existing hub."""
    model.max_nodes = model.graph.number_of_nodes() + count + 2
    anchors = [3, 22, 9, 0, 1]
    for k in range(count):
        anchor_id = anchors[k % len(anchors)]
        if anchor_id not in model.nodes:
            anchor_id = 22 if 22 in model.nodes else 9
        anchor = model.nodes[anchor_id]
        parent = anchor if anchor.kind == "hidden" else None
        node = model.add_hidden(parent, None, "parity_expansion", anchor=anchor)
        if node is not None:
            # Give the third input a direct route into the new node.
            model._add_edge(3, node.idx, model.rng.uniform(0.2, 0.7))
    return model


def graph_paths(model: DevelopmentalGraph) -> dict[str, list[int]]:
    paths = {}
    for i in INPUTS:
        if i in model.graph and nx.has_path(model.graph, i, OUTPUT):
            paths[str(i)] = nx.shortest_path(model.graph, i, OUTPUT)
        else:
            paths[str(i)] = []
    return paths


def run_one(base: dict, use_chem: bool) -> dict:
    model = DevelopmentalGraph.from_dict(base)
    chem = ChemField(model.graph) if use_chem else None
    learner = NeuroChemParityLearner(model, chem=chem)
    print(
        "RUN", "neurochem" if use_chem else "baseline",
        "nodes", model.graph.number_of_nodes(),
        "edges", model.graph.number_of_edges(),
        "inputs", INPUTS,
        "paths", [nx.has_path(model.graph, i, OUTPUT) for i in INPUTS],
    )
    start = time.perf_counter()
    result = learner.train()
    elapsed = time.perf_counter() - start
    result["train_seconds"] = round(elapsed, 3)
    result["ms_per_epoch"] = round(1000.0 * elapsed / max(1, result["epochs_run"]), 3)
    predictions = []
    for values, target in learner.data:
        act, _ = learner.forward(values)
        predictions.append(
            {"inputs": list(values), "target": target, "output": round(act.get(OUTPUT, 0.0), 5)}
        )
    result["mode"] = "neurochem" if use_chem else "baseline"
    result["input_ids"] = INPUTS
    result["nodes"] = model.graph.number_of_nodes()
    result["edges"] = model.graph.number_of_edges()
    result["predictions"] = predictions
    result["graph_paths"] = graph_paths(model)
    return result


def main() -> None:
    base = json.loads((ROOT / "phase1_graph.json").read_text(encoding="utf-8"))
    expanded = expand_for_parity(DevelopmentalGraph.from_dict(base), count=12)
    expanded_data = expanded.to_dict()
    (ROOT / "neurochem_expanded_graph.json").write_text(
        json.dumps(expanded_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "EXPANDED",
        "nodes", expanded.graph.number_of_nodes(),
        "edges", expanded.graph.number_of_edges(),
        "paths", graph_paths(expanded),
    )
    results = [run_one(expanded_data, use_chem=False), run_one(expanded_data, use_chem=True)]
    payload = {
        "task": "3-bit parity",
        "inputs": INPUTS,
        "output": OUTPUT,
        "topology": {
            "nodes": expanded.graph.number_of_nodes(),
            "edges": expanded.graph.number_of_edges(),
            "expanded_by": 12,
        },
        "input_to_output_paths": graph_paths(expanded),
        "output_files": {
            "result_json": "research_mvp/neurochem_parity_result.json",
            "expanded_graph": "research_mvp/neurochem_expanded_graph.json",
            "report": "research_mvp/NEUROCHEM_PARITY_REPORT.md",
            "curves_png": "research_mvp/neurochem_parity_curves.png",
        },
        "chemicals": {
            "DA": "dopamine: reward / gain",
            "5HT": "serotonin: stability / inhibition",
            "NE": "noradrenaline: arousal / error gain",
            "ACh": "acetylcholine: attention / plasticity",
            "GABA": "inhibitory gain",
            "Glu": "glutamatergic excitation",
        },
        "results": results,
    }
    (ROOT / "neurochem_parity_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for result in results:
        print(
            "RESULT", result["mode"],
            "loss", result["final_loss"],
            "acc", result["final_accuracy"],
            "epochs", result["epochs_run"],
        )
    print("SAVED", ROOT / "neurochem_parity_result.json")


if __name__ == "__main__":
    main()
