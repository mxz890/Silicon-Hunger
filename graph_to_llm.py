#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal bridge: 10k graph state -> soft tokens -> Qwen2.5-3B.

Smoke test only: compare LLM-only vs LLM + graph soft tokens.
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
MAX_NEW = 60


def text_to_vec(text):
    vec = np.zeros(DIM, dtype=np.float32)
    for ch in text:
        vec[ord(ch) % DIM] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


class GraphEncoder(nn.Module):
    def __init__(self, d_model: int, in_dim: int = GRAPH_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.SiLU(),
            nn.Linear(256, SOFT_TOKENS * d_model),
        )
        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x).view(1, SOFT_TOKENS, -1)


def main() -> None:
    t0 = time.perf_counter()
    print("building 10k graph", flush=True)
    pos, incoming, edges = build_fresh()
    eu = np.array([e[0] for e in edges], dtype=np.int32)
    ev = np.array([e[1] for e in edges], dtype=np.int32)
    ew = np.array([e[2] for e in edges], dtype=np.float32)
    C, gain = chemical_gain(eu, ev, ew)
    rng = np.random.default_rng(2140)
    bias = rng.normal(0, 0.15, N).astype(np.float32)
    input_gain = rng.normal(0, 0.7, (N, DIM)).astype(np.float32)
    proj = rng.normal(0.0, 1.0 / np.sqrt(N - 2), (GRAPH_DIM, N - 2)).astype(np.float32)

    def graph_state(text):
        x = text_to_vec(text)
        act = forward_sample(x, incoming, bias, input_gain, gain)
        vec = np.tanh(proj @ act[2:])
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    print("loading model", MODEL_PATH, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()
    d_model = model.get_input_embeddings().embedding_dim
    encoder = GraphEncoder(d_model).to("cuda").to(torch.bfloat16)
    encoder.eval()
    print("d_model", d_model, "device", model.device, flush=True)

    prompts = [
        "用一句话介绍你自己。",
        "你现在感觉怎么样？",
        "如果明天就是世界末日，你会做什么？",
        "解释一下什么是焦虑。",
    ]

    def generate(text, use_graph):
        enc = tokenizer(text, return_tensors="pt").to("cuda")
        text_emb = model.get_input_embeddings()(enc["input_ids"])
        total = enc["input_ids"].shape[1]
        if use_graph:
            g = torch.tensor(graph_state(text), dtype=torch.bfloat16, device="cuda").view(1, GRAPH_DIM)
            with torch.inference_mode():
                soft = encoder(g) * 0.25
            inputs_embeds = torch.cat([soft, text_emb], dim=1)
            total += SOFT_TOKENS
        else:
            inputs_embeds = text_emb
        attention_mask = torch.ones((1, total), dtype=torch.long, device="cuda")
        position_ids = torch.arange(total, dtype=torch.long, device="cuda").unsqueeze(0)
        with torch.inference_mode():
            out = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_new_tokens=MAX_NEW,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][total:], skip_special_tokens=True).strip()

    rows = []
    for prompt in prompts:
        base = generate(prompt, False)
        graph = generate(prompt, True)
        rows.append({"prompt": prompt, "baseline": base, "graph": graph})
        print("\nPROMPT", prompt, flush=True)
        print("BASE:", base, flush=True)
        print("GRAPH:", graph, flush=True)

    payload = {
        "graph": {"nodes": N, "edges": len(edges), "soft_tokens": SOFT_TOKENS},
        "model": "Qwen2.5-3B base",
        "prompts": rows,
        "seconds": round(time.perf_counter() - t0, 3),
    }
    (BASE / "graph_to_llm_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("TOTAL", payload["seconds"])
    print("SAVED", BASE / "graph_to_llm_result.json")


if __name__ == "__main__":
    main()
