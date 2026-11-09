#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe: inject 7D state as continuous soft tokens into Qwen base inputs_embeds."""

from __future__ import annotations

import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(BASE / "hf_cache"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-3B"
SOFT_TOKENS = 8
STATE_DIM = 7


class StateEncoder(nn.Module):
    def __init__(self, d_model: int, k: int = SOFT_TOKENS):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 256),
            nn.SiLU(),
            nn.Linear(256, k * d_model),
        )
        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).view(1, self.k, self.d_model)


def resolve_model_path() -> str:
    explicit = os.environ.get("MODEL_PATH")
    if explicit and Path(explicit).exists():
        return explicit
    candidates = [
        BASE / "modelscope_cache" / "models" / "qwen--Qwen2.5-3B" / "snapshots" / "master",
        BASE / "modelscope_cache" / "qwen" / "Qwen2.5-3B",
        BASE / "modelscope_cache" / "Qwen" / "Qwen2.5-3B",
    ]
    for candidate in candidates:
        if (candidate / "config.json").exists():
            return str(candidate)
    return snapshot_download(MODEL_ID, local_files_only=True)


def main() -> None:
    path = resolve_model_path()
    print("loading", MODEL_ID, "from", path, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()
    d_model = model.get_input_embeddings().embedding_dim
    encoder = StateEncoder(d_model).to("cuda").to(torch.bfloat16)
    encoder.eval()
    print("d_model", d_model, "soft", SOFT_TOKENS, "device", model.device, flush=True)

    prompt = "User: ...\nAssistant:"
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to("cuda")
    text_emb = model.get_input_embeddings()(input_ids)

    states = {
        "low": [1.00, 0.12, 0.40, 0.50, 0.05, 0.03, 0.01],
        "high": [0.50, 1.00, 0.40, 0.50, 0.50, 1.00, 1.00],
    }

    for name, values in states.items():
        state = torch.tensor(values, dtype=torch.bfloat16, device="cuda").view(1, STATE_DIM)
        with torch.inference_mode():
            soft = encoder(state)
            inputs_embeds = torch.cat([soft, text_emb], dim=1)
            attention_mask = torch.ones(
                (1, SOFT_TOKENS + input_ids.shape[1]), dtype=torch.long, device="cuda"
            )
            position_ids = torch.arange(
                SOFT_TOKENS + input_ids.shape[1], dtype=torch.long, device="cuda"
            ).unsqueeze(0)
            print(name, "soft_norm", float(soft.float().norm().item()), flush=True)
            output = model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                max_new_tokens=80,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        print("=== " + name + " ===", flush=True)
        print(text, flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
