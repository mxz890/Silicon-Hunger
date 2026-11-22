#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task A: grow and prune a 10-node arithmetic graph without PyTorch."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

OPS = {
    "add": lambda a, b: a + b,
    "add_bias": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "avg": lambda a, b: (a + b) / 2.0,
}


@dataclass
class Node:
    idx: int
    op: str
    bias: float
    noise: float
    score: float = 1.0
    correct: int = 0
    wrong: int = 0
    dormant: bool = False

    def evaluate(self, rng: random.Random, a: float = 1.0, b: float = 1.0) -> tuple[float, bool]:
        if self.op == "add":
            raw = a + b + rng.gauss(0.0, self.noise * 0.2)
        elif self.op == "add_bias":
            raw = a + b + self.bias + rng.gauss(0.0, self.noise)
        else:
            raw = OPS[self.op](a, b) + rng.gauss(0.0, self.noise * 0.05)
        return raw, round(raw) == 2


def build_graph(n: int, rng: random.Random) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    for i in range(n):
        graph.add_edge(i, (i + 1) % n, weight=0.12)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.30:
                graph.add_edge(i, j, weight=round(rng.uniform(0.15, 0.50), 3))
    return graph


def make_nodes(n: int, rng: random.Random) -> list[Node]:
    ops = [rng.choice(list(OPS.keys())) for _ in range(n)]
    # Guarantee a seed coalition, otherwise the graph can have no correct node.
    for i in range(2):
        if ops.count("add") < 2:
            ops[i] = "add"
    nodes = []
    for i, op in enumerate(ops):
        bias = rng.uniform(-0.35, 0.35) if op == "add_bias" else 0.0
        noise = rng.uniform(0.05, 0.25)
        nodes.append(Node(i, op, bias, noise))
    return nodes


def simple_pagerank(graph: nx.Graph, alpha: float = 0.85, iterations: int = 100) -> dict[int, float]:
    nodes = list(graph.nodes)
    count = len(nodes)
    if count == 0:
        return {}
    rank = {i: 1.0 / count for i in nodes}
    for _ in range(iterations):
        nxt = {i: (1.0 - alpha) / count for i in nodes}
        dangling = sum(rank[i] for i in nodes if graph.degree(i) == 0)
        for u, v in graph.edges:
            du = graph.degree(u)
            dv = graph.degree(v)
            if du:
                nxt[v] += alpha * rank[u] / du
            if dv:
                nxt[u] += alpha * rank[v] / dv
        if dangling:
            for i in nodes:
                nxt[i] += alpha * dangling / count
        rank = nxt
    return rank


def metrics(graph: nx.Graph, nodes: list[Node]) -> dict:
    degree = dict(graph.degree())
    betweenness = nx.betweenness_centrality(graph) if graph.number_of_nodes() else {}
    pagerank = simple_pagerank(graph) if graph.number_of_edges() else {i: 0.0 for i in graph.nodes}
    hubs = sorted(
        graph.nodes,
        key=lambda i: (degree.get(i, 0), pagerank.get(i, 0.0), nodes[i].score),
        reverse=True,
    )
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": round(nx.density(graph), 4) if graph.number_of_nodes() > 1 else 0.0,
        "components": nx.number_connected_components(graph),
        "max_degree": max(degree.values()) if degree else 0,
        "degree": degree,
        "betweenness": {i: round(v, 4) for i, v in betweenness.items()},
        "pagerank": {i: round(v, 4) for i, v in pagerank.items()},
        "hub_order": hubs,
        "top_hubs": [
            {
                "node": i,
                "degree": degree.get(i, 0),
                "score": round(nodes[i].score, 3),
                "op": nodes[i].op,
                "correct": nodes[i].correct,
            }
            for i in hubs[:5]
        ],
    }


def run(cycles: int = 100, seed: int = 7) -> dict:
    rng = random.Random(seed)
    n = 10
    graph = build_graph(n, rng)
    nodes = make_nodes(n, rng)
    history = []

    for cycle in range(1, cycles + 1):
        correct_nodes = []
        for node in nodes:
            _, ok = node.evaluate(rng)
            if ok:
                correct_nodes.append(node.idx)
                node.correct += 1
                node.score = min(10.0, node.score * 1.08 + 0.40)
                node.dormant = False
            else:
                node.wrong += 1
                node.score = max(0.01, node.score * 0.90)
                node.dormant = graph.degree(node.idx) == 0

        correct_set = set(correct_nodes)

        # Growth: correct nodes form a coalition.  The two most successful
        # nodes become hubs and also project links into the rest of the graph.
        for u in correct_nodes:
            for v in correct_nodes:
                if u < v:
                    if graph.has_edge(u, v):
                        graph[u][v]["weight"] = min(1.0, graph[u][v]["weight"] + 0.15)
                    else:
                        graph.add_edge(u, v, weight=0.22)

        hubs = sorted(correct_nodes, key=lambda i: nodes[i].score, reverse=True)[:2]
        wrong = [i for i in graph.nodes if i not in set(correct_nodes)]
        for u in hubs:
            targets = rng.sample(wrong, min(3, len(wrong))) if wrong else []
            for v in targets:
                if graph.has_edge(u, v):
                    graph[u][v]["weight"] = min(1.0, graph[u][v]["weight"] + 0.04)
                else:
                    graph.add_edge(u, v, weight=0.08)

        # Pruning: edges not touching a correct node decay and disappear.
        for u, v in list(graph.edges):
            weight = graph[u][v]["weight"]
            if u in correct_set and v in correct_set:
                graph[u][v]["weight"] = min(1.0, weight + 0.02)
            else:
                weight *= 0.88
                if weight < 0.06:
                    graph.remove_edge(u, v)
                else:
                    graph[u][v]["weight"] = weight

        # Mutation: occasionally let a wrong node change its arithmetic rule.
        if cycle % 17 == 0:
            wrong = [n for n in nodes if n.idx not in correct_set]
            if wrong:
                candidate = rng.choice(wrong)
                candidate.op = rng.choice(list(OPS.keys()))
                candidate.bias = rng.uniform(-0.8, 0.8)

        if cycle == 1 or cycle % 10 == 0 or cycle == cycles:
            history.append({"cycle": cycle, **metrics(graph, nodes)})

    result = {
        "seed": seed,
        "cycles": cycles,
        "initial_nodes": n,
        "final": metrics(graph, nodes),
        "history": history,
        "nodes": [
            {
                "id": n.idx,
                "op": n.op,
                "bias": round(n.bias, 4),
                "noise": round(n.noise, 4),
                "score": round(n.score, 3),
                "correct": n.correct,
                "wrong": n.wrong,
                "degree": graph.degree(n.idx),
                "dormant": graph.degree(n.idx) == 0,
            }
            for n in nodes
        ],
    }
    return result, graph, nodes


def plot(graph: nx.Graph, nodes: list[Node], out_png: Path, seed: int = 7) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.35, 1]})
    pos = nx.spring_layout(graph, seed=seed, weight="weight", iterations=200)
    degree = dict(graph.degree())
    max_degree = max(degree.values()) if degree else 1
    colors = ["#2f7d5b" if nodes[i].correct >= nodes[i].wrong else "#b9c4bd" for i in graph.nodes]
    sizes = [160 + 260 * (degree.get(i, 0) / max_degree) for i in graph.nodes]
    widths = [0.4 + 2.4 * graph[u][v]["weight"] for u, v in graph.edges]
    nx.draw_networkx_edges(graph, pos, ax=axes[0], width=widths, edge_color="#9bb8a8", alpha=0.65)
    nx.draw_networkx_nodes(graph, pos, ax=axes[0], node_color=colors, node_size=sizes, edgecolors="#173d2c", linewidths=1.0)
    nx.draw_networkx_labels(graph, pos, ax=axes[0], font_size=9, font_color="white", font_weight="bold")
    axes[0].set_title("Final topology", fontsize=12, color="#173d2c")
    axes[0].axis("off")

    order = sorted(graph.nodes, key=lambda i: degree.get(i, 0), reverse=True)
    axes[1].barh([str(i) for i in order], [degree.get(i, 0) for i in order], color="#2f7d5b")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("degree")
    axes[1].set_title("Hub ranking", fontsize=12, color="#173d2c")
    axes[1].grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent
    result, graph, nodes = run(args.cycles, args.seed)
    (out_dir / "graph_growth_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot(graph, nodes, out_dir / "graph_growth.png", args.seed)
    print(json.dumps(result["final"], ensure_ascii=False, indent=2))
    print("SAVED", out_dir / "graph_growth_result.json")
    print("SAVED", out_dir / "graph_growth.png")


if __name__ == "__main__":
    main()
