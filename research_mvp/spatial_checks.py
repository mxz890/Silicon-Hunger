#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small invariants for the 3D spatial constraint / neighbour rewiring."""

from __future__ import annotations

import json
from pathlib import Path

from graph_core import DevelopmentalGraph


def main() -> None:
    model = DevelopmentalGraph(seed=3)
    max_pos = max(max(n.pos) for n in model.nodes.values())
    min_pos = min(min(n.pos) for n in model.nodes.values())
    edge_distances = [
        model._distance_nodes(u, v) for u, v in model.graph.edges
    ]
    target = next(
        i for i in sorted(model.nodes)
        if model.nodes[i].kind == "hidden" and model.graph.degree(i) >= 4
    )
    before_edges = model.graph.number_of_edges()
    model.prune(target)
    after_edges = model.graph.number_of_edges()
    post_distances = [
        model._distance_nodes(u, v) for u, v in model.graph.edges
    ]
    payload = {
        "seed": 3,
        "pos_min": round(min_pos, 3),
        "pos_max": round(max_pos, 3),
        "spatial_threshold": model.spatial_threshold,
        "edge_count_before": before_edges,
        "edge_count_after": after_edges,
        "max_edge_distance_before": round(max(edge_distances), 3),
        "max_edge_distance_after": round(max(post_distances), 3),
        "pruned_node": target,
        "rewires": model.rewires,
        "blocked_edges": model.blocked_edges,
        "invariants": {
            "all_positions_0_100": 0.0 <= min_pos and max_pos <= 100.0,
            "all_edges_within_threshold_before": max(edge_distances) <= model.spatial_threshold,
            "all_edges_within_threshold_after": max(post_distances) <= model.spatial_threshold,
            "rewire_positive": len(model.rewires) > 0,
        },
    }
    out = Path(__file__).resolve().parent
    (out / "spatial_check.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
