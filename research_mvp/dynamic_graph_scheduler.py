#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task A v2: secrets wakes a subgraph; reachability + finite-difference score."""

from __future__ import annotations

import argparse
import json
import math
import random
import secrets
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

TARGET = 2.0
DEATH_AFTER = 8
SOURCE = 0
SINK = 9
HIDDEN = list(range(1, 9))

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
    weight: float
    score: float = 1.0
    correct: int = 0
    wrong: int = 0
    dead_streak: int = 0
    dead: bool = False
    pruned_at: int | None = None
    active: bool = False
    functional: bool = False


def node_value(node: Node, a: float = 1.0, b: float = 1.0) -> float:
    """Pure node computation. No graph mutation, no global state."""
    if node.op == "add":
        return a + b
    if node.op == "add_bias":
        return a + b + node.bias
    return OPS[node.op](a, b)


def make_nodes(rng: random.Random) -> list[Node]:
    ops = [rng.choice(list(OPS.keys())) for _ in HIDDEN]
    for i in range(2):
        if ops.count("add") < 2:
            ops[i] = "add"
    nodes = []
    for idx in range(10):
        if idx == SOURCE:
            nodes.append(Node(idx, "source", 0.0, 0.0))
        elif idx == SINK:
            nodes.append(Node(idx, "sink", 0.0, 0.0))
        else:
            op = ops[idx - 1]
            bias = rng.uniform(-0.35, 0.35) if op == "add_bias" else 0.0
            nodes.append(Node(idx, op, bias, rng.uniform(0.5, 1.5)))
    return nodes


def build_graph(rng: random.Random) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(range(10))
    for i in HIDDEN:
        graph.add_edge(SOURCE, i, weight=round(rng.uniform(0.08, 0.20), 3))
        graph.add_edge(i, SINK, weight=round(rng.uniform(0.08, 0.20), 3))
    for i in HIDDEN:
        for j in HIDDEN:
            if i < j and rng.random() < 0.18:
                graph.add_edge(i, j, weight=round(rng.uniform(0.05, 0.18), 3))
    return graph


def reachability(graph: nx.DiGraph, active: set[int]) -> tuple[set[int], set[int], set[int]]:
    forward = nx.descendants(graph, SOURCE) | {SOURCE}
    backward = nx.ancestors(graph, SINK) | {SINK}
    active_hidden = {i for i in active if i in HIDDEN}
    functional = active_hidden & forward & backward
    return forward, backward, functional


def final_output(graph: nx.DiGraph, nodes: list[Node], active: set[int], weights: dict[int, float] | None = None) -> float:
    _, _, functional = reachability(graph, active)
    pairs = []
    for i in functional:
        w = weights[i] if weights and i in weights else nodes[i].weight
        pairs.append((w, node_value(nodes[i])))
    total = sum(w for w, _ in pairs)
    return sum(w * v for w, v in pairs) / total if total else 0.0


def numerical_sensitivity(graph: nx.DiGraph, nodes: list[Node], active: set[int], idx: int) -> float:
    eps = 1e-3
    base = nodes[idx].weight
    plus = final_output(graph, nodes, active, {idx: base + eps})
    minus = final_output(graph, nodes, active, {idx: base - eps})
    return abs(plus - minus) / (2.0 * eps)


def wake_subgraph(graph: nx.DiGraph, nodes: list[Node]) -> tuple[set[int], int]:
    alive = [i for i in HIDDEN if not nodes[i].dead]
    if not alive:
        return {SOURCE, SINK}, -1
    seed = secrets.choice(alive)
    active = {seed}
    neighbors = [
        n for n in set(graph.predecessors(seed)) | set(graph.successors(seed))
        if n in HIDDEN and not nodes[n].dead
    ]
    for _ in range(min(3, len(neighbors))):
        candidate = secrets.choice(neighbors)
        active.add(candidate)
        neighbors.remove(candidate)
    return active | {SOURCE, SINK}, seed


def add_edge_safe(graph: nx.DiGraph, u: int, v: int, weight: float = 0.10) -> None:
    if u == v:
        return
    if u == SINK or v == SOURCE:
        return
    if graph.has_edge(u, v):
        graph[u][v]["weight"] = min(1.0, graph[u][v]["weight"] + 0.04)
        return
    if nx.has_path(graph, v, u):
        return
    graph.add_edge(u, v, weight=weight)


def metrics(graph: nx.DiGraph, nodes: list[Node], active: set[int], output: float, cycle: int) -> dict:
    forward, backward, functional = reachability(graph, active)
    degree = dict(graph.degree())
    pruned = [i for i in HIDDEN if nodes[i].dead]
    hubs = sorted(
        [i for i in HIDDEN if not nodes[i].dead],
        key=lambda i: (degree.get(i, 0), nodes[i].score),
        reverse=True,
    )
    return {
        "cycle": cycle,
        "active": sorted(active - {SOURCE, SINK}),
        "dead_nodes": pruned,
        "functional": sorted(functional),
        "forward": sorted(forward - {SOURCE, SINK}),
        "backward": sorted(backward - {SOURCE, SINK}),
        "output": round(output, 5),
        "error": round(abs(output - TARGET), 5),
        "edges": graph.number_of_edges(),
        "max_degree": max(degree.values()) if degree else 0,
        "hub_order": hubs[:5],
        "dead_streak": {i: nodes[i].dead_streak for i in HIDDEN},
        "score": {i: round(nodes[i].score, 3) for i in HIDDEN},
    }


def run(cycles: int, init_seed: int) -> tuple[dict, nx.DiGraph, list[Node]]:
    rng = random.Random(init_seed)
    graph = build_graph(rng)
    nodes = make_nodes(rng)
    history = []
    selected_seeds = []

    for cycle in range(1, cycles + 1):
        active, seed = wake_subgraph(graph, nodes)
        selected_seeds.append(seed)
        forward, backward, functional = reachability(graph, active)

        for i in HIDDEN:
            node = nodes[i]
            if i not in active:
                node.active = False
                node.functional = False
                node.dead_streak += 1
                node.score *= 0.88
                if node.dead_streak >= DEATH_AFTER and not node.dead:
                    node.dead = True
                    node.pruned_at = cycle
                    for u, v in list(graph.edges):
                        if u == i or v == i:
                            graph.remove_edge(u, v)
            else:
                node.dead_streak = 0
                value = node_value(node)
                correct = round(value) == 2
                func = i in functional
                node.active = True
                node.functional = func
                sens = numerical_sensitivity(graph, nodes, active, i)
                node.correct += int(correct)
                node.wrong += int(not correct)
                node.score = max(
                    0.0,
                    node.score * 0.84
                    + (1.2 if correct else 0.0)
                    + (1.5 if func else 0.0)
                    + min(2.5, sens * 8.0),
                )

        output = final_output(graph, nodes, active)
        alive = [i for i in HIDDEN if not nodes[i].dead]
        winners = sorted(
            [i for i in alive if i in active and i in functional],
            key=lambda i: nodes[i].score,
            reverse=True,
        )[:3]
        for u in winners:
            add_edge_safe(graph, SOURCE, u, 0.12)
            add_edge_safe(graph, u, SINK, 0.12)
            for v in winners:
                if u < v:
                    add_edge_safe(graph, u, v, 0.08)

        # Structural pruning: dead incident edges are already gone.
        for u, v in list(graph.edges):
            w = graph[u][v]["weight"]
            if u in HIDDEN and v in HIDDEN:
                w *= 0.90
            else:
                w *= 0.985
            if w < 0.04:
                graph.remove_edge(u, v)
            else:
                graph[u][v]["weight"] = w

        if cycle == 1 or cycle % 10 == 0 or cycle == cycles:
            history.append(metrics(graph, nodes, active, output, cycle))

    result = {
        "init_seed": init_seed,
        "cycles": cycles,
        "death_after": DEATH_AFTER,
        "selection": "secrets.choice seed + up to 2 neighbors; source/sink always awake",
        "scheduler": {i: nodes[i].pruned_at for i in HIDDEN if nodes[i].dead},
        "final": metrics(graph, nodes, active, final_output(graph, nodes, active), cycles),
        "history": history,
        "nodes": [
            {
                "id": i,
                "op": nodes[i].op,
                "weight": round(nodes[i].weight, 4),
                "score": round(nodes[i].score, 3),
                "correct": nodes[i].correct,
                "wrong": nodes[i].wrong,
                "dead": nodes[i].dead,
                "pruned_at": nodes[i].pruned_at,
                "active": nodes[i].active,
                "functional": nodes[i].functional,
                "degree": graph.degree(i),
            }
            for i in range(10)
        ],
    }
    return result, graph, nodes


def plot(graph: nx.DiGraph, nodes: list[Node], out_png: Path, cycle: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8))
    components = list(nx.connected_components(graph.to_undirected()))
    main_nodes = max(components, key=len) if components else set(graph.nodes)
    main_graph = graph.subgraph(main_nodes)
    pos = nx.spring_layout(main_graph, seed=4, k=1.15, iterations=300)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    span = max(maxx - minx, maxy - miny, 1e-9)
    pos = {
        n: (
            (p[0] - (minx + maxx) / 2) / span * 1.6,
            (p[1] - (miny + maxy) / 2) / span * 1.6,
        )
        for n, p in pos.items()
    }
    isolated = [n for n in graph.nodes if n not in main_nodes]
    for j, n in enumerate(isolated):
        x = -0.7 + 1.4 * j / max(1, len(isolated) - 1) if len(isolated) > 1 else 0.0
        pos[n] = (x, -1.05)

    degree = dict(graph.degree())
    max_degree = max(degree.values()) if degree else 1
    colors = []
    for i in graph.nodes:
        if i in (SOURCE, SINK):
            colors.append("#173d2c")
        elif nodes[i].dead:
            colors.append("#c05a4a")
        elif nodes[i].active and nodes[i].functional:
            colors.append("#2f7d5b")
        elif nodes[i].active:
            colors.append("#6f9f86")
        else:
            colors.append("#b9c4bd")
    sizes = [260 + 420 * (degree.get(i, 0) / max_degree) for i in graph.nodes]
    edges = list(graph.edges)
    widths = [0.5 + 2.5 * graph[u][v]["weight"] for u, v in edges]
    nx.draw_networkx_edges(
        graph, pos, edgelist=edges, ax=ax, width=widths, edge_color="#8aa99a",
        alpha=0.7, arrows=True, arrowsize=12, connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_nodes(
        graph, pos, ax=ax, node_color=colors, node_size=sizes,
        edgecolors="#173d2c", linewidths=1.2,
    )
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=10, font_color="white", font_weight="bold")
    dead = [i for i in HIDDEN if nodes[i].dead]
    ax.text(
        0.5, -0.02,
        f"cycle {cycle} | green=active/functional | red=dead | gray=unused | dead={dead}",
        transform=ax.transAxes, ha="center", va="top", fontsize=9, color="#5f6b64",
    )
    ax.set_title(
        f"Cycle {cycle} | secrets-woken subgraph + death clock",
        fontsize=12, color="#173d2c",
    )
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.05)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    result, graph, nodes = run(args.cycles, args.seed)
    (out / "dynamic_graph_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot(graph, nodes, out / "dynamic_graph.png", args.cycles)
    print(json.dumps(result["final"], ensure_ascii=False, indent=2))
    print("PRUNED", result["scheduler"])
    print("SAVED", out / "dynamic_graph_result.json")
    print("SAVED", out / "dynamic_graph.png")


if __name__ == "__main__":
    main()
