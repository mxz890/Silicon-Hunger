#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a standalone Plotly dashboard for the fair 10k comparison."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
VIS = ROOT / "visualizations"
DATA = ROOT / "fair_10k_result.json"


def main() -> None:
    d = json.loads(DATA.read_text(encoding="utf-8"))
    rows = d["results"]
    names = [r["name"] for r in rows]
    math = [r["math_mae"] for r in rows]
    chinese = [r["chinese_accuracy"] for r in rows]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Math test MAE (lower is better)", "Chinese next-character accuracy"),
        horizontal_spacing=0.18,
    )
    colors = ["#8aa99a", "#2f7d5b"]
    fig.add_trace(
        go.Bar(x=names, y=math, marker_color=colors, text=[f"{v:.4f}" for v in math], textposition="outside", name="Math MAE"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=names, y=chinese, marker_color=colors, text=[f"{v:.4f}" for v in chinese], textposition="outside", name="Chinese acc"),
        row=1, col=2,
    )
    fig.add_hline(y=1/64, line_dash="dot", line_color="#e74c3c", row=1, col=2, annotation_text="chance 1/64", annotation_position="top right")
    fig.update_layout(
        paper_bgcolor="#0b0f14", plot_bgcolor="#0b0f14",
        font=dict(color="#d8e7de"),
        showlegend=False, margin=dict(l=30, r=30, t=70, b=40),
        height=430,
    )
    fig.update_xaxes(gridcolor="#26352d", tickfont=dict(color="#d8e7de"))
    fig.update_yaxes(gridcolor="#26352d", tickfont=dict(color="#d8e7de"))
    chart = fig.to_html(
        include_plotlyjs="directory", full_html=False, div_id="charts",
        config={"displayModeBar": False, "responsive": True},
    )

    cards = "".join(
        f'<div class="card"><div class="name">{r["name"]}</div>'
        f'<div class="metric">math MAE <b>{r["math_mae"]:.4f}</b></div>'
        f'<div class="metric">Chinese acc <b>{r["chinese_accuracy"]:.4f}</b></div>'
        f'<div class="small">edges {r["edges"]} · output indegree {r["output_indegree"]}</div></div>'
        for r in rows
    )

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Fair 10k comparison</title>
<style>
html,body{{margin:0;background:#0b0f14;color:#d8e7de;font-family:system-ui,sans-serif}}
.wrap{{max-width:1200px;margin:0 auto;padding:28px 22px 40px}}
h1{{font-size:24px;margin:0 0 6px}}
.sub{{color:#8aa99a;font-size:13px;line-height:1.6;margin-bottom:18px}}
.cards{{display:flex;gap:14px;flex-wrap:wrap;margin:18px 0}}
.card{{flex:1 1 260px;background:#111a16;border:1px solid #2f7d5b;border-radius:12px;padding:14px 16px}}
.name{{font-weight:700;color:#5fd39a;margin-bottom:8px}}
.metric{{font-size:14px;color:#cfe1d6;margin:4px 0}}
.metric b{{color:#fff;font-size:18px}}
.small{{font-size:12px;color:#8aa99a;margin-top:8px}}
.note{{background:#111a16;border-left:3px solid #f1c40f;padding:12px 14px;margin:18px 0;color:#cfe1d6;font-size:13px;line-height:1.7}}
a{{color:#5fd39a}}
.links{{font-size:13px;line-height:1.9;margin-top:14px}}
</style>
</head>
<body>
<div class="wrap">
  <h1>10k Fair Comparison</h1>
  <div class="sub">
    控制变量：10000 节点 · 56212 边 · output indegree 8002 · 相同数据 / readout / bias 与 input_gain 随机种子。<br>
    对比：从 2000 seed 长出来的 10k vs 从零独立生成的 10k。
  </div>
  <div class="cards">{cards}</div>
  {chart}
  <div class="note">
    <b>结论：</b>控制变量后 fresh 的数学 MAE 反而更低（0.2183 vs 0.8741）。<br>
    中文两者都接近随机水平：fresh 0.0867，seed-grown 0.0700；随机基线是 1/64 ≈ 0.0156。<br>
    之前“fresh 更差”的结论受到 output 入度、边数和随机种子的混杂影响。
  </div>
  <div class="links">
    <a href="thought_chain.html">打开 3D thinking chain</a> ·
    <a href="../FAIR_10K_REPORT.md">Fair 10k 报告</a> ·
    <a href="../fair_10k_result.json">原始 JSON</a>
  </div>
</div>
</body>
</html>"""
    VIS.mkdir(exist_ok=True)
    out = VIS / "fair_10k_dashboard.html"
    out.write_text(page, encoding="utf-8")
    print("SAVED", out)


if __name__ == "__main__":
    main()
