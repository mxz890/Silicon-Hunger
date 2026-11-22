#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared developmental graph with a 3D spatial constraint."""

from __future__ import annotations

import math
import random
import secrets
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

INPUTS = (0, 1)
OUTPUT = 9
ENV_DIM = 7
SPATIAL_THRESHOLD = 30.0


def sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def clamp(v: float) -> float:
    return max(0.0, min(100.0, v))


def dist3(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


@dataclass
class Node:
    idx: int
    kind: str
    bias: float
    receptor: list[float]
    pos: list[float] = field(default_factory=lambda: [50.0, 50.0, 50.0])
    activation: float = 0.0
    last_active: int = 0
    born_at: int = 0
    dead: bool = False
    pruned_at: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.idx,
            "kind": self.kind,
            "bias": round(self.bias, 6),
            "receptor": [round(v, 6) for v in self.receptor],
            "pos": [round(v, 4) for v in self.pos],
            "last_active": self.last_active,
            "born_at": self.born_at,
            "dead": self.dead,
            "pruned_at": self.pruned_at,
        }


@dataclass
class Ghost:
    idx: int
    pruned_at: int
    last_activation: float
    degree: int
    pos: list[float] = field(default_factory=lambda: [50.0, 50.0, 50.0])


class DevelopmentalGraph:
    def __init__(
        self,
        seed: int = 11,
        initial_hidden: int = 7,
        max_nodes: int = 40,
        split_degree: int = 3,
        prune_after: int = 20,
        activation_threshold: float = 0.50,
        capture_threshold: float = 0.45,
        spatial_threshold: float = SPATIAL_THRESHOLD,
        split_mode: str = "conflict",
        conflict_weight_var: float = 0.015,
        conflict_min_sensitivity: float = 1e-4,
        conflict_cooldown: int = 4,
    ):
        self.rng = random.Random(seed)
        self.max_nodes = max_nodes
        self.split_degree = split_degree
        self.prune_after = prune_after
        self.activation_threshold = activation_threshold
        self.capture_threshold = capture_threshold
        self.spatial_threshold = spatial_threshold
        self.split_mode = split_mode
        self.conflict_weight_var = conflict_weight_var
        self.conflict_min_sensitivity = conflict_min_sensitivity
        self.conflict_cooldown = conflict_cooldown
        self.last_split_step = 0
        self.catastrophe = False
        self.normal_prune_after = prune_after
        self.normal_activation_threshold = activation_threshold
        self.mutations: list[dict[str, Any]] = []
        self.center = [self.rng.uniform(30.0, 70.0) for _ in range(3)]
        self.graph = nx.DiGraph()
        self.nodes: dict[int, Node] = {}
        self.ghosts: list[Ghost] = []
        self.next_id = 10
        self.births: list[dict[str, Any]] = []
        self.deaths: list[dict[str, Any]] = []
        self.rewires: list[dict[str, Any]] = []
        self.blocked_edges: list[dict[str, Any]] = []
        self.step_no = 0

        for idx in (0, 1, OUTPUT):
            self.graph.add_node(idx)
            kind = "input" if idx in INPUTS else "output"
            self.nodes[idx] = Node(idx, kind, 0.0, [0.0] * ENV_DIM, self._random_pos())

        hidden_ids = list(range(2, 2 + initial_hidden))
        for idx in hidden_ids:
            self.nodes[idx] = self._new_node(idx, "hidden", None)
            self.graph.add_node(idx)

        # Spatial initial graph: connect only local neighbours.
        for idx in hidden_ids:
            for u in (0, 1):
                self._add_edge(u, idx, self.rng.uniform(0.2, 0.8))
            if self._distance_nodes(idx, OUTPUT) <= self.spatial_threshold:
                self._add_edge(idx, OUTPUT, self.rng.uniform(0.2, 0.8))
        for i in hidden_ids:
            for j in hidden_ids:
                if i < j and self.rng.random() < 0.45:
                    self._add_edge(i, j, self.rng.uniform(0.1, 0.4))

        # Repair obvious dead ends with a local neighbour that can reach output.
        for idx in hidden_ids:
            self._connect_to_output_if_possible(idx)

    def _random_pos(self) -> list[float]:
        spread = 16.0
        return [clamp(self.center[k] + self.rng.uniform(-spread, spread)) for k in range(3)]

    def _new_node(self, idx: int, kind: str, parent: Node | None, anchor: Node | None = None) -> Node:
        if parent is None:
            receptor = [self.rng.uniform(-1.0, 1.0) for _ in range(ENV_DIM)]
            bias = self.rng.uniform(-0.3, 0.3)
            anchor = anchor or self.nodes[self.rng.choice(list(self.nodes))]
            pos = [clamp(anchor.pos[k] + self.rng.uniform(-10.0, 10.0)) for k in range(3)]
        else:
            receptor = [v + self.rng.uniform(-0.15, 0.15) for v in parent.receptor]
            bias = parent.bias + self.rng.uniform(-0.15, 0.15)
            pos = [clamp(parent.pos[k] + self.rng.uniform(-9.0, 9.0)) for k in range(3)]
        return Node(idx, kind, bias, receptor, pos)

    def _distance_nodes(self, u: int, v: int) -> float:
        return dist3(self.nodes[u].pos, self.nodes[v].pos)

    def _add_edge(self, u: int, v: int, weight: float) -> bool:
        if u == v or u not in self.nodes or v not in self.nodes:
            return False
        distance = self._distance_nodes(u, v)
        if distance > self.spatial_threshold:
            if len(self.blocked_edges) < 500:
                self.blocked_edges.append({"u": u, "v": v, "distance": round(distance, 3)})
            return False
        if self.graph.has_edge(u, v):
            self.graph[u][v]["weight"] = max(self.graph[u][v]["weight"], weight)
            return False
        if nx.has_path(self.graph, v, u):
            return False
        self.graph.add_edge(u, v, weight=float(weight))
        return True

    def _connect_to_output_if_possible(self, idx: int) -> bool:
        if idx not in self.nodes or idx == OUTPUT:
            return False
        if nx.has_path(self.graph, idx, OUTPUT):
            return True
        candidates = [
            v for v in self.nodes
            if v != idx and v != OUTPUT and self._distance_nodes(idx, v) <= self.spatial_threshold
            and nx.has_path(self.graph, v, OUTPUT)
        ]
        for v in sorted(candidates, key=lambda v: self._distance_nodes(idx, v)):
            if self._add_edge(idx, v, self.rng.uniform(0.2, 0.8)):
                return True
        return False

    def add_hidden(
        self,
        parent: Node | None,
        env: list[float] | None,
        reason: str,
        anchor: Node | None = None,
    ) -> Node | None:
        if len(self.nodes) >= self.max_nodes:
            return None
        idx = self.next_id
        self.next_id += 1
        node = self._new_node(idx, "hidden", parent, anchor)
        if env:
            node.receptor = [v + self.rng.uniform(-0.1, 0.1) for v in env]
        node.born_at = self.step_no
        self.nodes[idx] = node
        self.graph.add_node(idx)

        for u in (0, 1):
            self._add_edge(u, idx, self.rng.uniform(0.2, 0.8))
        if parent is not None:
            self._add_edge(parent.idx, idx, self.rng.uniform(0.3, 0.9))
        if self._distance_nodes(idx, OUTPUT) <= self.spatial_threshold:
            self._add_edge(idx, OUTPUT, self.rng.uniform(0.3, 0.9))
        else:
            self._connect_to_output_if_possible(idx)

        self.births.append(
            {"id": idx, "at": self.step_no, "parent": parent.idx if parent else None, "reason": reason, "pos": node.pos}
        )
        return node

    def _rewire_after_prune(self, idx: int, incoming: list[int], outgoing: list[int], pos: list[float]) -> None:
        if not self.nodes:
            return
        survivors = sorted(self.nodes, key=lambda v: dist3(self.nodes[v].pos, pos))
        for u in incoming:
            if u not in self.nodes:
                continue
            for k in survivors[:3]:
                if k == u or k not in self.nodes:
                    continue
                if self._add_edge(u, k, self.rng.uniform(0.15, 0.5)):
                    self.rewires.append({"removed": idx, "from": u, "to": k, "at": self.step_no})
                    break
        for w in outgoing:
            if w not in self.nodes:
                continue
            for k in survivors[:3]:
                if k == w or k not in self.nodes:
                    continue
                if self._add_edge(k, w, self.rng.uniform(0.15, 0.5)):
                    self.rewires.append({"removed": idx, "from": k, "to": w, "at": self.step_no})
                    break

    def prune(self, idx: int) -> None:
        node = self.nodes[idx]
        degree = self.graph.degree(idx)
        incoming = list(self.graph.predecessors(idx))
        outgoing = list(self.graph.successors(idx))
        pos = list(node.pos)
        self.deaths.append(
            {"id": idx, "at": self.step_no, "activation": node.activation, "degree": degree, "pos": pos}
        )
        self.ghosts.append(Ghost(idx, self.step_no, node.activation, degree, pos))
        self.graph.remove_node(idx)
        node.dead = True
        node.pruned_at = self.step_no
        self.nodes.pop(idx, None)
        self._rewire_after_prune(idx, incoming, outgoing, pos)

    def forward(self, env: list[float] | None = None, inputs: tuple[float, float] | None = None) -> dict[int, float]:
        if inputs is None:
            if env is None:
                raise ValueError("env or inputs required")
            inputs = (env[0], env[1])
        activations: dict[int, float] = {0: inputs[0], 1: inputs[1]}
        for idx in nx.topological_sort(self.graph):
            if idx in INPUTS:
                continue
            node = self.nodes[idx]
            pre = node.bias + sum(
                self.graph[pred][idx]["weight"] * activations.get(pred, 0.0)
                for pred in self.graph.predecessors(idx)
            )
            if node.kind == "hidden" and env is not None:
                pre += 0.25 * sum(a * b for a, b in zip(node.receptor, env))
            node.activation = sigmoid(pre)
            activations[idx] = node.activation
        return activations

    def incoming_variance(self, idx: int) -> float:
        weights = [self.graph[u][idx]["weight"] for u in self.graph.predecessors(idx)]
        if len(weights) < 2:
            return 0.0
        mean = sum(weights) / len(weights)
        return sum((w - mean) ** 2 for w in weights) / len(weights)

    def _sensitivity(self, idx: int, env: list[float], eps: float = 1e-3) -> float:
        node = self.nodes[idx]
        base = node.bias
        try:
            node.bias = base + eps
            plus = self.forward(env=env).get(OUTPUT, 0.0)
            node.bias = base - eps
            minus = self.forward(env=env).get(OUTPUT, 0.0)
        finally:
            node.bias = base
        return (plus - minus) / (2.0 * eps)

    def conflict_candidates(self, env_a: list[float], env_b: list[float]) -> list[tuple[float, int, float, float, float]]:
        candidates = []
        for idx in self.nodes:
            node = self.nodes[idx]
            if node.kind != "hidden":
                continue
            variance = self.incoming_variance(idx)
            if variance < self.conflict_weight_var:
                continue
            act_a = self.forward(env=env_a).get(idx, 0.0)
            act_b = self.forward(env=env_b).get(idx, 0.0)
            feedback_a = 2.0 * act_a - 1.0
            feedback_b = 2.0 * act_b - 1.0
            # Functional conflict: the node gives the same feedback direction
            # under A and B, although B needs the opposite direction.
            same_direction = feedback_a * feedback_b > 0
            strong = min(abs(feedback_a), abs(feedback_b)) > 0.06
            if same_direction and strong:
                sa = self._sensitivity(idx, env_a)
                sb = self._sensitivity(idx, env_b)
                if min(abs(sa), abs(sb)) < self.conflict_min_sensitivity:
                    continue
                score = min(abs(feedback_a), abs(feedback_b)) * math.sqrt(variance)
                candidates.append((score, idx, sa, sb, variance, act_a, act_b))
        candidates.sort(reverse=True)
        return candidates

    def enter_catastrophe(self, prune_after: int = 3, activation_threshold: float = 0.68) -> None:
        if self.catastrophe:
            return
        self.normal_prune_after = self.prune_after
        self.normal_activation_threshold = self.activation_threshold
        self.prune_after = prune_after
        self.activation_threshold = activation_threshold
        self.catastrophe = True

    def exit_catastrophe(self) -> None:
        if not self.catastrophe:
            return
        self.prune_after = self.normal_prune_after
        self.activation_threshold = self.normal_activation_threshold
        self.catastrophe = False

    def _add_mutation_edge(self, u: int, v: int) -> bool:
        if u == v or u not in self.nodes or v not in self.nodes:
            return False
        if self.graph.has_edge(u, v) or nx.has_path(self.graph, v, u):
            return False
        weight = self.rng.uniform(0.15, 0.45)
        self.graph.add_edge(u, v, weight=weight)
        self.mutations.append(
            {
                "at": self.step_no,
                "type": "long_edge",
                "u": u,
                "v": v,
                "distance": round(self._distance_nodes(u, v), 3),
                "bypass_spatial": True,
            }
        )
        return True

    def topology_mutation(self) -> dict[str, Any]:
        alive = [i for i in self.nodes if self.nodes[i].kind == "hidden"]
        if not alive:
            return {"ok": False}
        node = secrets.choice(alive)
        action = secrets.choice(["cut", "long"])
        if action == "cut":
            incident = list(self.graph.in_edges(node)) + list(self.graph.out_edges(node))
            if not incident:
                return {"ok": False, "node": node, "action": "cut"}
            u, v = max(incident, key=lambda e: self.graph[e[0]][e[1]]["weight"])
            weight = self.graph[u][v]["weight"]
            self.graph.remove_edge(u, v)
            event = {
                "at": self.step_no, "type": "cut", "node": node,
                "u": u, "v": v, "weight": round(weight, 4),
            }
            self.mutations.append(event)
            return {"ok": True, **event}
        far = [
            v for v in self.nodes
            if v != node and self._distance_nodes(node, v) > self.spatial_threshold
        ]
        secrets.SystemRandom().shuffle(far)
        for v in far:
            if self._add_mutation_edge(node, v):
                return {"ok": True, **self.mutations[-1]}
            if self._add_mutation_edge(v, node):
                return {"ok": True, **self.mutations[-1]}
        return {"ok": False, "node": node, "action": "long"}

    def step(self, env: list[float], conflict_envs: tuple[list[float], list[float]] | None = None) -> dict[str, Any]:
        self.step_no += 1
        activations = self.forward(env=env)
        output = activations.get(OUTPUT, 0.0)
        hidden = [i for i in self.nodes if self.nodes[i].kind == "hidden" and not self.nodes[i].dead]
        max_hidden = max((activations.get(i, 0.0) for i in hidden), default=0.0)

        for i in hidden:
            node = self.nodes[i]
            if activations.get(i, 0.0) > self.activation_threshold:
                node.last_active = self.step_no
            elif self.step_no - node.last_active >= self.prune_after:
                self.prune(i)

        capture = max_hidden < self.capture_threshold
        if capture:
            anchor = self.nodes[self.rng.choice(list(self.nodes))]
            self.add_hidden(None, env, "environment_not_captured", anchor=anchor)

        split = None
        conflict_info = None
        if (
            self.split_mode == "conflict"
            and conflict_envs is not None
            and len(self.nodes) < self.max_nodes
            and self.step_no - self.last_split_step >= self.conflict_cooldown
        ):
            candidates = self.conflict_candidates(conflict_envs[0], conflict_envs[1])
            if candidates:
                score, parent, sa, sb, variance, act_a, act_b = candidates[0]
                conflict_info = {
                    "score": round(score, 6),
                    "node": parent,
                    "act_a": round(act_a, 5),
                    "act_b": round(act_b, 5),
                    "feedback_a": round(2.0 * act_a - 1.0, 5),
                    "feedback_b": round(2.0 * act_b - 1.0, 5),
                    "sens_a": round(sa, 6),
                    "sens_b": round(sb, 6),
                    "weight_variance": round(variance, 6),
                }
                split = self.add_hidden(self.nodes[parent], env, "functional_conflict")
                if split is not None:
                    self.last_split_step = self.step_no
        elif self.split_mode == "degree":
            candidates = [
                i for i in self.nodes
                if self.nodes[i].kind == "hidden" and not self.nodes[i].dead
                and self.graph.degree(i) > self.split_degree
                and self.nodes[i].last_active >= self.step_no - 1
            ]
            if candidates and len(self.nodes) < self.max_nodes:
                parent = max(candidates, key=lambda i: self.graph.degree(i))
                split = self.add_hidden(self.nodes[parent], env, "degree_split")

        return {
            "step": self.step_no,
            "output": round(output, 5),
            "max_hidden": round(max_hidden, 5),
            "capture": capture,
            "split": split.idx if split else None,
            "split_reason": "functional_conflict" if conflict_info else ("degree_split" if split else None),
            "conflict": conflict_info,
            "nodes": len(self.nodes),
            "edges": self.graph.number_of_edges(),
        }

    def summary(self) -> dict[str, Any]:
        degree = dict(self.graph.degree())
        hidden = [i for i in self.nodes if self.nodes[i].kind == "hidden" and not self.nodes[i].dead]
        hubs = sorted(hidden, key=lambda i: degree.get(i, 0), reverse=True)
        return {
            "nodes": len(self.nodes),
            "edges": self.graph.number_of_edges(),
            "max_degree": max(degree.values()) if degree else 0,
            "hub": hubs[0] if hubs else None,
            "hub_degree": degree.get(hubs[0], 0) if hubs else 0,
            "hidden": len(hidden),
            "ghosts": len(self.ghosts),
            "births": len(self.births),
            "deaths": len(self.deaths),
            "rewires": len(self.rewires),
            "blocked_edges": len(self.blocked_edges),
            "mutations": len(self.mutations),
            "catastrophe": self.catastrophe,
            "spatial_threshold": self.spatial_threshold,
            "degree": degree,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_nodes": list(INPUTS),
            "output_node": OUTPUT,
            "spatial_threshold": self.spatial_threshold,
            "nodes": [self.nodes[i].as_dict() for i in sorted(self.nodes) if not self.nodes[i].dead],
            "edges": [
                {"u": u, "v": v, "weight": round(self.graph[u][v]["weight"], 6)}
                for u, v in self.graph.edges
            ],
            "ghosts": [g.__dict__ for g in self.ghosts],
            "births": self.births,
            "deaths": self.deaths,
            "rewires": self.rewires,
            "mutations": self.mutations,
            "steps": self.step_no,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DevelopmentalGraph":
        obj = cls(seed=0, initial_hidden=0)
        obj.graph = nx.DiGraph()
        obj.nodes = {}
        obj.ghosts = [Ghost(**g) for g in data.get("ghosts", [])]
        obj.births = list(data.get("births", []))
        obj.deaths = list(data.get("deaths", []))
        obj.rewires = list(data.get("rewires", []))
        obj.mutations = list(data.get("mutations", []))
        obj.step_no = int(data.get("steps", 0))
        obj.spatial_threshold = float(data.get("spatial_threshold", SPATIAL_THRESHOLD))
        obj.next_id = max([n["id"] for n in data["nodes"]] + [9]) + 1
        for item in data["nodes"]:
            obj.nodes[item["id"]] = Node(
                idx=item["id"], kind=item["kind"], bias=item["bias"],
                receptor=item["receptor"], pos=item.get("pos", [50.0, 50.0, 50.0]),
                last_active=item.get("last_active", 0),
                born_at=item.get("born_at", 0),
            )
            obj.graph.add_node(item["id"])
        for edge in data["edges"]:
            obj.graph.add_edge(edge["u"], edge["v"], weight=edge["weight"])
        return obj

    def grow_near_output(self) -> Node | None:
        return self.add_hidden(None, None, "xor_stall_growth", anchor=self.nodes[OUTPUT])
