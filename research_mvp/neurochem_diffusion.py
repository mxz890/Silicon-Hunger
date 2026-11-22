#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal graph reaction-diffusion model for a neurotransmitter field.

This is a phenomenological model, not a biological brain simulation.
Each node stores an extracellular concentration c_i.  A node exchanges
chemical with neighbours, has a source and a reuptake/decay term.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parent


@dataclass
class ChemNode:
    idx: int
    pos: tuple[float, float, float]
    concentration: float = 0.0
    source: float = 0.0
    decay: float = 0.08
    receptor_k: float = 0.5


def neighbourhood_flux(graph: nx.Graph, c: dict[int, float], idx: int) -> float:
    """Net chemical flux into node i from all neighbours.

    flux_i = sum_j w_ij * (c_j - c_i)
    """
    total = 0.0
    for j, data in graph[idx].items():
        weight = float(data.get("weight", 1.0))
        total += weight * (c[j] - c[idx])
    return total


def neighbourhood_mean(graph: nx.Graph, c: dict[int, float], idx: int) -> float:
    """Weighted mean concentration around node i (excluding i)."""
    total = 0.0
    weight_sum = 0.0
    for j, data in graph[idx].items():
        weight = float(data.get("weight", 1.0))
        total += weight * c[j]
        weight_sum += weight
    return total / weight_sum if weight_sum else c[idx]


def receptor_activation(c: float, k: float) -> float:
    """Hill/Langmuir-like receptor occupancy: c / (k + c)."""
    return c / (k + c) if c + k > 0 else 0.0


def diffuse(
    graph: nx.Graph,
    nodes: dict[int, ChemNode],
    dt: float = 0.1,
    D: float = 0.08,
    sources: dict[int, float] | None = None,
) -> dict[int, float]:
    """One explicit Euler step of graph reaction-diffusion.

    dc_i/dt = D * sum_j w_ij (c_j - c_i) + source_i - decay_i c_i
    """
    sources = sources or {}
    c = {i: n.concentration for i, n in nodes.items()}
    nxt = {}
    for i, node in nodes.items():
        flux = neighbourhood_flux(graph, c, i)
        source = sources.get(i, 0.0) + node.source
        nxt[i] = c[i] + dt * (D * flux + source - node.decay * c[i])
        nxt[i] = max(0.0, nxt[i])
    for i, value in nxt.items():
        nodes[i].concentration = value
    return nxt


def build_demo() -> tuple[nx.Graph, dict[int, ChemNode]]:
    # 12 nodes in 3D, connected only when close enough.
    positions = [
        (20, 22, 24), (29, 31, 28), (39, 26, 33), (49, 35, 27),
        (58, 45, 36), (67, 36, 44), (74, 50, 48), (64, 60, 55),
        (50, 55, 62), (38, 64, 55), (28, 52, 48), (18, 58, 39),
    ]
    graph = nx.Graph()
    nodes = {}
    for i, pos in enumerate(positions):
        graph.add_node(i)
        nodes[i] = ChemNode(i, pos, decay=0.09)
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            distance = math.dist(positions[i], positions[j])
            if distance <= 26.0:
                graph.add_edge(i, j, weight=max(0.15, 1.0 - distance / 30.0))
    return graph, nodes


def main() -> None:
    graph, nodes = build_demo()
    print("nodes", graph.number_of_nodes(), "edges", graph.number_of_edges())
    print("degree", dict(graph.degree()))

    # Two pulses: node 1 and node 7.
    pulses = {
        1: {5: 1.0, 6: 0.8, 7: 0.6, 8: 0.4},
        7: {13: 1.0, 14: 0.8, 15: 0.6, 16: 0.4},
    }
    history = []
    for step in range(1, 81):
        sources = {i: pulses[i].get(step, 0.0) for i in pulses}
        c = diffuse(graph, nodes, dt=0.12, D=0.12, sources=sources)
        if step % 16 == 0 or step in (1, 20, 40):
            history.append(
                {
                    "step": step,
                    "top": sorted(
                        ({"node": i, "c": round(v, 4)} for i, v in c.items()),
                        key=lambda x: x["c"], reverse=True,
                    )[:5],
                }
            )

    # A node-local query: what is around node 5, and what receptor occupancy?
    node = 5
    mean = neighbourhood_mean(graph, {i: n.concentration for i, n in nodes.items()}, node)
    flux = neighbourhood_flux(graph, {i: n.concentration for i, n in nodes.items()}, node)
    print("node", node, "c", round(nodes[node].concentration, 5))
    print("neighbour_mean", round(mean, 5), "neighbour_flux", round(flux, 5))
    print("receptor_activation", round(receptor_activation(nodes[node].concentration, nodes[node].receptor_k), 4))

    (ROOT / "neurochem_result.json").write_text(
        json.dumps(
            {
                "nodes": [
                    {
                        "id": i,
                        "pos": n.pos,
                        "c": round(n.concentration, 6),
                        "receptor": round(receptor_activation(n.concentration, n.receptor_k), 5),
                    }
                    for i, n in nodes.items()
                ],
                "history": history,
                "node_query": {
                    "node": node,
                    "neighbour_mean": round(mean, 6),
                    "neighbour_flux": round(flux, 6),
                    "receptor_activation": round(receptor_activation(nodes[node].concentration, nodes[node].receptor_k), 6),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    xs = [n.pos[0] for n in nodes.values()]
    ys = [n.pos[1] for n in nodes.values()]
    zs = [n.pos[2] for n in nodes.values()]
    cs = [n.concentration for n in nodes.values()]
    for u, v in graph.edges:
        ax.plot(
            [nodes[u].pos[0], nodes[v].pos[0]],
            [nodes[u].pos[1], nodes[v].pos[1]],
            [nodes[u].pos[2], nodes[v].pos[2]],
            color="#8aa99a", linewidth=1.0, alpha=0.5,
        )
    scatter = ax.scatter(xs, ys, zs, c=cs, s=[80 + 450 * c for c in cs], cmap="viridis", depthshade=True)
    for i in nodes:
        ax.text(*nodes[i].pos, str(i), fontsize=8)
    fig.colorbar(scatter, ax=ax, shrink=0.6, label="concentration")
    ax.set_title("Graph reaction-diffusion: neurotransmitter field")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(ROOT / "neurochem_3d.png", dpi=180)
    print("SAVED", ROOT / "neurochem_result.json")
    print("SAVED", ROOT / "neurochem_3d.png")


if __name__ == "__main__":
    main()
