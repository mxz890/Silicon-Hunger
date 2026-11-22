#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 0: developmental graph with pyvis snapshots every 10 rounds."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import plotly.graph_objects as go

from pyvis.network import Network

from graph_core import DevelopmentalGraph

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "visualizations"
EXTREME = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
CONFLICT_A = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
CONFLICT_B = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
CATASTROPHE_STEP = 50
CATASTROPHE_END = 70


def render_graph(model: DevelopmentalGraph, step: int, out_dir: Path = OUT) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = model.graph
    degree = dict(graph.degree())
    max_degree = max(degree.values()) if degree else 1
    new_ids = {b["id"] for b in model.births if b["at"] > step - 10}

    net = Network(
        height="820px",
        width="100%",
        directed=True,
        notebook=False,
        cdn_resources="in_line",
    )
    net.toggle_physics(True)
    net.barnes_hut(gravity=-9000, central_gravity=0.25, spring_length=180, spring_strength=0.02, damping=0.9)

    for idx in graph.nodes:
        if idx in new_ids:
            color = "#2ecc71"
            group = "new"
        elif idx in (0, 1):
            color = "#f1c40f"
            group = "input"
        elif idx == 9:
            color = "#9b59b6"
            group = "output"
        else:
            color = "#3498db"
            group = "alive"
        size = 12 + 7.0 * degree.get(idx, 0)
        title = f"node {idx} | degree {degree.get(idx, 0)} | {group}"
        net.add_node(str(idx), label=str(idx), color=color, size=size, title=title)

    for u, v, data in graph.edges(data=True):
        weight = float(data.get("weight", 0.1))
        net.add_edge(
            str(u),
            str(v),
            width=1.0 + 5.0 * weight,
            title=f"w={weight:.3f}",
            color="#7f8c8d",
            arrows="to",
        )

    for ghost in model.ghosts:
        net.add_node(
            f"ghost-{ghost.idx}",
            label=f"{ghost.idx}†",
            color="#e74c3c",
            size=14,
            title=f"pruned at cycle {ghost.pruned_at} | degree {ghost.degree}",
        )

    path = out_dir / f"step_{step:03d}.html"
    html = net.generate_html(notebook=False)
    path.write_text(html, encoding="utf-8")
    return path


def render_graph_3d(model: DevelopmentalGraph, step: int, out_dir: Path = OUT) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = model.graph
    degree = dict(graph.degree())
    max_degree = max(degree.values()) if degree else 1
    new_ids = {b["id"] for b in model.births if b["at"] > step - 10}

    nodes = []
    for idx in graph.nodes:
        if idx in new_ids:
            color = "#2ecc71"
            group = "new"
        elif idx in (0, 1):
            color = "#f1c40f"
            group = "input"
        elif idx == 9:
            color = "#9b59b6"
            group = "output"
        else:
            color = "#3498db"
            group = "alive"
        node = model.nodes[idx]
        nodes.append(
            {
                "id": str(idx),
                "name": str(idx),
                "title": f"node {idx} | degree {degree.get(idx, 0)} | {group} | pos {[round(v, 1) for v in node.pos]}",
                "x": node.pos[0],
                "y": node.pos[1],
                "z": node.pos[2],
                "color": color,
                "val": 4 + 2.2 * degree.get(idx, 0),
                "group": group,
            }
        )
    for ghost in model.ghosts:
        nodes.append(
            {
                "id": f"ghost-{ghost.idx}",
                "name": f"{ghost.idx}†",
                "title": f"PRUNED node {ghost.idx} at cycle {ghost.pruned_at} | degree {ghost.degree}",
                "x": ghost.pos[0],
                "y": ghost.pos[1],
                "z": ghost.pos[2],
                "color": "#e74c3c",
                "val": 3,
                "group": "pruned",
            }
        )

    links = [
        {
            "source": str(u),
            "target": str(v),
            "width": 0.4 + 2.5 * float(data.get("weight", 0.1)),
            "color": "#8aa99a",
            "title": f"w={data.get('weight', 0):.3f} d={model._distance_nodes(u, v):.1f}",
        }
        for u, v, data in graph.edges(data=True)
    ]

    data = {"nodes": nodes, "links": links}
    html = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Silicon Hunger 3D - step {step:03d}</title>
<style>
  html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:#0b0f14;font-family:system-ui,sans-serif}}
  #graph{{width:100vw;height:100vh}}
  .hud{{position:fixed;left:18px;top:16px;color:#d8e7de;z-index:10;background:rgba(11,15,20,.72);
       padding:12px 16px;border:1px solid #2f7d5b;border-radius:10px;font-size:13px;line-height:1.7}}
  .dot{{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}}
  .muted{{color:#8aa99a}}
</style>
</head>
<body>
<div class="hud">
  <b>Silicon Hunger / 3D spatial graph</b><br>
  step {step:03d} · nodes {nodes} · edges {edges}<br>
  <span class="dot" style="background:#2ecc71"></span>new
  <span class="dot" style="background:#e74c3c"></span>pruned
  <span class="dot" style="background:#3498db"></span>alive
  <span class="dot" style="background:#f1c40f"></span>input
  <span class="dot" style="background:#9b59b6"></span>output<br>
  <span class="muted">drag rotate · wheel zoom · auto-rotate on</span>
</div>
<div id="graph"></div>
<script src="https://unpkg.com/3d-force-graph"></script>
<script>
const data = {data};
const Graph = ForceGraph3D({{ controlType: 'orbit' }})(document.getElementById('graph'))
  .graphData(data)
  .nodeLabel('title')
  .nodeColor('color')
  .nodeVal('val')
  .nodeOpacity(1)
  .linkWidth('width')
  .linkColor('color')
  .linkOpacity(0.55)
  .linkDirectionalArrowLength(4)
  .linkDirectionalArrowRelPos(1)
  .backgroundColor('#0b0f14')
  .showNavInfo(false);
Graph.d3Force('charge').strength(-80);
Graph.cameraPosition({{ x: 180, y: 120, z: 220 }});
Graph.controls().autoRotate = false;
Graph.controls().autoRotateSpeed = 0.0;
Graph.controls().enableDamping = true;
Graph.controls().dampingFactor = 0.08;
window.addEventListener('resize', () => Graph.width(window.innerWidth).height(window.innerHeight));
</script>
</body>
</html>
""".format(
        step=step,
        nodes=len(nodes),
        edges=len(links),
        data=json.dumps(data, ensure_ascii=False),
    )
    path = out_dir / f"step_3d_{step:03d}.html"
    path.write_text(html, encoding="utf-8")
    return path


def render_graph_plotly(model: DevelopmentalGraph, step: int, out_dir: Path = OUT) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    graph = model.graph
    degree = dict(graph.degree())
    max_degree = max(degree.values()) if degree else 1
    new_ids = {b["id"] for b in model.births if b["at"] > step - 10}

    node_x, node_y, node_z = [], [], []
    node_colors, node_sizes, node_labels, node_hover = [], [], [], []
    for idx in graph.nodes:
        if idx in new_ids:
            color = "#2ecc71"
            group = "new"
        elif idx in (0, 1):
            color = "#f1c40f"
            group = "input"
        elif idx == 9:
            color = "#9b59b6"
            group = "output"
        else:
            color = "#3498db"
            group = "alive"
        node = model.nodes[idx]
        node_x.append(node.pos[0])
        node_y.append(node.pos[1])
        node_z.append(node.pos[2])
        node_colors.append(color)
        node_sizes.append(8 + 1.4 * degree.get(idx, 0))
        node_labels.append(str(idx))
        node_hover.append(
            f"node {idx}<br>degree {degree.get(idx, 0)}<br>group {group}<br>pos {[round(v, 1) for v in node.pos]}"
        )

    ghost_x, ghost_y, ghost_z, ghost_labels, ghost_hover = [], [], [], [], []
    for ghost in model.ghosts:
        ghost_x.append(ghost.pos[0])
        ghost_y.append(ghost.pos[1])
        ghost_z.append(ghost.pos[2])
        ghost_labels.append(f"{ghost.idx}†")
        ghost_hover.append(f"PRUNED node {ghost.idx}<br>at cycle {ghost.pruned_at}<br>degree {ghost.degree}")

    bins = {
        "light": {"x": [], "y": [], "z": [], "width": 1.5, "color": "#6f8f80"},
        "medium": {"x": [], "y": [], "z": [], "width": 2.8, "color": "#8aa99a"},
        "heavy": {"x": [], "y": [], "z": [], "width": 5.0, "color": "#b9d6c4"},
    }
    for u, v, data in graph.edges(data=True):
        weight = float(data.get("weight", 0.1))
        key = "light" if weight < 0.35 else ("medium" if weight < 0.65 else "heavy")
        pu, pv = model.nodes[u].pos, model.nodes[v].pos
        bins[key]["x"].extend([pu[0], pv[0], None])
        bins[key]["y"].extend([pu[1], pv[1], None])
        bins[key]["z"].extend([pu[2], pv[2], None])

    fig = go.Figure()
    for item in bins.values():
        if item["x"]:
            fig.add_trace(
                go.Scatter3d(
                    x=item["x"], y=item["y"], z=item["z"], mode="lines",
                    line=dict(color=item["color"], width=item["width"]),
                    hoverinfo="skip", showlegend=False,
                )
            )
    fig.add_trace(
        go.Scatter3d(
            x=node_x, y=node_y, z=node_z, mode="markers+text",
            text=node_labels, textposition="top center",
            textfont=dict(size=9, color="#d8e7de"),
            hovertext=node_hover, hoverinfo="text",
            marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color="#173d2c")),
            showlegend=False,
        )
    )
    if ghost_x:
        fig.add_trace(
            go.Scatter3d(
                x=ghost_x, y=ghost_y, z=ghost_z, mode="markers+text",
                text=ghost_labels, textposition="top center",
                textfont=dict(size=8, color="#e74c3c"),
                hovertext=ghost_hover, hoverinfo="text",
                marker=dict(size=7, color="#e74c3c", symbol="diamond"),
                showlegend=False,
            )
        )
    fig.update_layout(
        title=dict(
            text=f"Silicon Hunger / 3D Plotly — step {step:03d} · nodes {len(node_x)} · edges {len(graph.edges)}",
            font=dict(color="#d8e7de", size=15),
        ),
        paper_bgcolor="#0b0f14",
        plot_bgcolor="#0b0f14",
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            bgcolor="#0b0f14",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="cube",
            dragmode="orbit",
            camera=dict(eye=dict(x=2.0, y=1.5, z=1.4)),
        ),
        showlegend=False,
        font=dict(color="#d8e7de"),
    )

    plotly_js = Path(__import__("plotly").__file__).parent / "package_data" / "plotly.min.js"
    target_js = out_dir / "plotly.min.js"
    if plotly_js.exists() and not target_js.exists():
        shutil.copyfile(plotly_js, target_js)

    path = out_dir / f"step_3d_{step:03d}.html"
    fig.write_html(
        str(path),
        include_plotlyjs="directory",
        full_html=True,
        div_id="graph",
        config={
            "displayModeBar": False,
            "responsive": True,
            "scrollZoom": True,
            "doubleClick": "reset",
        },
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--render-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    model = DevelopmentalGraph(seed=args.seed)
    log = []

    for step in range(1, args.cycles + 1):
        if step == CATASTROPHE_STEP:
            env = list(EXTREME)
            model.enter_catastrophe()
            print(f"CATASTROPHE_START step={step} env={env}")
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
            print(f"CATASTROPHE_END step={step}")

        if step % 10 == 0:
            log.append({"step": step, **event, "mutation": mutation, "summary": model.summary()})
            print(
                f"step={step:03d} nodes={event['nodes']} edges={event['edges']} "
                f"max_degree={model.summary()['max_degree']} hub={model.summary()['hub']} "
                f"hub_degree={model.summary()['hub_degree']} "
                f"births={len(model.births)} deaths={len(model.deaths)} mutations={len(model.mutations)}"
            )
        if step % args.render_every == 0 or step == args.cycles:
            path = render_graph(model, step)
            path3d = render_graph_plotly(model, step)
            print("RENDER", path)
            print("RENDER3D", path3d)

    (ROOT / "phase0_log.json").write_text(
        json.dumps({"log": log, "final": model.to_dict()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("FINAL", json.dumps(model.summary(), ensure_ascii=False))


if __name__ == "__main__":
    main()
