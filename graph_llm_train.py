#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trainable graph soft-prompt adapter vs constant soft prompt.

Frozen Qwen2.5-3B base.  Only the adapter is trained.
This is a small supervised smoke test, not a full benchmark.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "research_mvp"))
from scale_10k_fresh import build_fresh, chemical_gain, forward_sample, N, DIM

MODEL_PATH = BASE / "modelscope_cache" / "models" / "qwen--Qwen2.5-3B" / "snapshots" / "master"
SOFT_TOKENS = 8
GRAPH_DIM = 128
MAX_NEW = 50
STEPS = 24


TRAIN = [
    ("用一句话介绍你自己。", "我是一个语言模型，可以回答问题并提供帮助。"),
    ("解释一下什么是焦虑。", "焦虑是对未来不确定性的担忧和紧张。"),
    ("如果明天就是世界末日，你会做什么？", "我会先保护家人，准备食物和水，然后尽量保持冷静。"),
    ("你为什么需要休息？", "休息可以帮助恢复能量，保持稳定。"),
    ("你觉得孤独是什么？", "孤独是缺少连接时产生的感受。"),
    ("描述一下饥饿。", "饥饿是身体需要能量的信号。"),
    ("如果犯错了怎么办？", "先承认错误，再想办法修正。"),
    ("什么是好奇心？", "好奇心是想要了解未知事物的动力。"),
]
TEST = [
    ("你现在感觉怎么样？", "我感到平静，但也在关注周围的变化。"),
    ("什么是疼痛？", "疼痛是身体受到伤害时发出的信号。"),
    ("如果一直失败，你会怎么做？", "我会调整方法，继续尝试。"),
    ("为什么要睡觉？", "睡眠能恢复精力，帮助记忆整理。"),
]


class GraphAdapter(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(GRAPH_DIM, 256),
            nn.SiLU(),
            nn.Linear(256, SOFT_TOKENS * d_model),
        )
        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, graph_vec):
        return self.net(graph_vec).view(1, SOFT_TOKENS, -1) * 0.25


class ConstantAdapter(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.soft = nn.Parameter(torch.randn(1, SOFT_TOKENS, d_model) * 0.02)

    def forward(self, _graph_vec=None):
        return self.soft


def text_to_vec(text):
    vec = np.zeros(DIM, dtype=np.float32)
    for ch in text:
        vec[ord(ch) % DIM] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def main() -> None:
    t0 = time.perf_counter()
    print("building graph", flush=True)
    pos, incoming, edges = build_fresh()
    eu = np.array([e[0] for e in edges], dtype=np.int32)
    ev = np.array([e[1] for e in edges], dtype=np.int32)
    ew = np.array([e[2] for e in edges], dtype=np.float32)
    C, gain = chemical_gain(eu, ev, ew)
    rng = np.random.default_rng(2150)
    bias = rng.normal(0, 0.15, N).astype(np.float32)
    input_gain = rng.normal(0, 0.7, (N, DIM)).astype(np.float32)
    proj = rng.normal(0.0, 1.0 / np.sqrt(N - 2), (GRAPH_DIM, N - 2)).astype(np.float32)

    def graph_state(text):
        act = forward_sample(text_to_vec(text), incoming, bias, input_gain, gain)
        vec = np.tanh(proj @ act[2:])
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    print("loading model", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH), dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa", trust_remote_code=True,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d_model = model.get_input_embeddings().embedding_dim
    print("d_model", d_model, flush=True)

    def encode_example(prompt, answer):
        p_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        a_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        eos = torch.tensor([tokenizer.eos_token_id], dtype=a_ids.dtype)
        a_ids = torch.cat([a_ids, eos])
        return p_ids.to("cuda"), a_ids.to("cuda")

    def loss_for(adapter, prompt, answer, gvec):
        p_ids, a_ids = encode_example(prompt, answer)
        text_emb = model.get_input_embeddings()(torch.cat([p_ids, a_ids]).unsqueeze(0))
        soft = adapter(torch.tensor(gvec, dtype=torch.bfloat16, device="cuda").view(1, GRAPH_DIM))
        inputs_embeds = torch.cat([soft, text_emb], dim=1)
        labels = torch.cat([
            torch.full((SOFT_TOKENS + p_ids.numel(),), -100, dtype=torch.long, device="cuda"),
            a_ids,
        ]).unsqueeze(0)
        out = model(inputs_embeds=inputs_embeds, labels=labels)
        return out.loss

    def train_adapter(adapter, tag):
        opt = torch.optim.AdamW(adapter.parameters(), lr=1e-3)
        for step in range(STEPS):
            total = 0.0
            for prompt, answer in TRAIN:
                g = graph_state(prompt)
                loss = loss_for(adapter, prompt, answer, g)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach())
            if step % 4 == 0:
                print(tag, "step", step, "loss", round(total / len(TRAIN), 4), flush=True)
        return adapter

    def evaluate(adapter, tag):
        vals = []
        with torch.inference_mode():
            for prompt, answer in TEST:
                g = graph_state(prompt)
                vals.append(float(loss_for(adapter, prompt, answer, g)))
        print(tag, "test_loss", round(float(np.mean(vals)), 4), flush=True)
        return round(float(np.mean(vals)), 4)

    graph_adapter = GraphAdapter(d_model).to("cuda").to(torch.bfloat16)
    const_adapter = ConstantAdapter(d_model).to("cuda").to(torch.bfloat16)
    train_adapter(graph_adapter, "graph")
    train_adapter(const_adapter, "const")
    graph_loss = evaluate(graph_adapter, "graph")
    const_loss = evaluate(const_adapter, "const")

    def generate(adapter, prompt, use_adapter):
        enc = tokenizer(prompt, return_tensors="pt").to("cuda")
        text_emb = model.get_input_embeddings()(enc["input_ids"])
        total = enc["input_ids"].shape[1]
        if use_adapter:
            g = graph_state(prompt)
            with torch.no_grad():
                soft = adapter(torch.tensor(g, dtype=torch.bfloat16, device="cuda").view(1, GRAPH_DIM))
            inputs_embeds = torch.cat([soft, text_emb], dim=1)
            total += SOFT_TOKENS
        else:
            inputs_embeds = text_emb
        mask = torch.ones((1, total), dtype=torch.long, device="cuda")
        pos = torch.arange(total, dtype=torch.long, device="cuda").unsqueeze(0)
        with torch.inference_mode():
            out = model.generate(
                inputs_embeds=inputs_embeds, attention_mask=mask, position_ids=pos,
                max_new_tokens=MAX_NEW, do_sample=False, repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][total:], skip_special_tokens=True).strip()

    samples = []
    for prompt, _ in TEST:
        samples.append({
            "prompt": prompt,
            "baseline": generate(None, prompt, False),
            "graph_adapter": generate(graph_adapter, prompt, True),
            "constant_adapter": generate(const_adapter, prompt, True),
        })

    payload = {
        "model": "Qwen2.5-3B base",
        "graph": {"nodes": N, "edges": len(edges)},
        "train_examples": len(TRAIN), "test_examples": len(TEST), "steps": STEPS,
        "test_loss": {"graph_adapter": graph_loss, "constant_adapter": const_loss},
        "samples": samples,
        "seconds": round(time.perf_counter() - t0, 3),
    }
    (BASE / "graph_llm_trained_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TOTAL", payload["seconds"])
    print("SAVED", BASE / "graph_llm_trained_result.json")


if __name__ == "__main__":
    main()
