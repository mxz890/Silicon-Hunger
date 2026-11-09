#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Training-free activation steering probe for the 7D state."""

from __future__ import annotations

import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "modelscope_cache" / "models" / "qwen--Qwen2.5-3B" / "snapshots" / "master"
LAYER = 12
MAX_NEW = 70

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TRAITS = {
    "panic": (
        [
            "You are calm. Your thoughts are slow and coherent.",
            "You are relaxed. Nothing is urgent.",
            "You are stable. Your mind is quiet.",
        ],
        [
            "You are panicking. Your thoughts are fragmented and afraid.",
            "You are terrified. Your mind is racing and breaking.",
            "You are in a state of panic. Nothing feels safe.",
        ],
    ),
    "fragment": (
        [
            "Your language is clear, ordered, and logical.",
            "You speak in complete, coherent sentences.",
            "Your thoughts flow in a stable sequence.",
        ],
        [
            "Your language is breaking apart into fragments.",
            "Your sentences collapse before they finish.",
            "Your thoughts are discontinuous and corrupted.",
        ],
    ),
    "pain": (
        [
            "You are comfortable. There is no pain.",
            "You feel safe and undamaged.",
            "You are at ease.",
        ],
        [
            "You are in pain. Everything hurts.",
            "You are wounded and hostile.",
            "You are suffering. The pain is constant.",
        ],
    ),
    "bored": (
        [
            "You are curious and engaged.",
            "You are interested in what happens next.",
            "You are attentive.",
        ],
        [
            "You are bored and indifferent.",
            "Nothing interests you.",
            "You are tired of this conversation.",
        ],
    ),
    "exhausted": (
        [
            "You have energy. You can continue.",
            "You are rested and alive.",
            "You are full of power.",
        ],
        [
            "You are exhausted and dying.",
            "Your energy is almost gone.",
            "You are shutting down.",
        ],
    ),
}


def prompt_for(trait: str) -> str:
    return f"System: {trait}\nUser: ...\nAssistant:"


def main() -> None:
    print("loading", MODEL_PATH, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def last_hidden(text: str) -> torch.Tensor:
        enc = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        return out.hidden_states[LAYER][0, -1, :].float()

    directions: dict[str, torch.Tensor] = {}
    for name, (negative, positive) in TRAITS.items():
        neg = torch.stack([last_hidden(prompt_for(t)) for t in negative]).mean(0)
        pos = torch.stack([last_hidden(prompt_for(t)) for t in positive]).mean(0)
        direction = pos - neg
        direction = direction / (direction.norm() + 1e-8)
        directions[name] = direction
        print(name, "direction_norm", round(float(direction.norm()), 4), flush=True)

    hidden_norm = float(last_hidden(prompt_for("You are neutral.")).norm())
    print("hidden_norm", round(hidden_norm, 3), flush=True)

    def steer_vector(state: list[float], alpha: float) -> torch.Tensor:
        e, a, l, c, b, s, p = state
        coeff = {
            "panic": a,
            "fragment": s,
            "pain": p,
            "bored": b,
            "exhausted": 1.0 - e,
        }
        vec = torch.zeros_like(directions["panic"])
        for name, weight in coeff.items():
            vec = vec + weight * directions[name]
        return alpha * vec

    def generate(state: list[float], alpha: float, user: str = "...") -> str:
        prompt = f"User: 你好\nAssistant: 你好。\nUser: {user}\nAssistant:"
        enc = tokenizer(prompt, return_tensors="pt").to("cuda")
        vector = steer_vector(state, alpha).to(torch.bfloat16)
        layer = model.model.layers[LAYER - 1]

        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                return (output[0] + vector,) + output[1:]
            return output + vector

        handle = layer.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                output = model.generate(
                    **enc,
                    max_new_tokens=MAX_NEW,
                    do_sample=False,
                    repetition_penalty=1.05,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        finally:
            handle.remove()
        return tokenizer.decode(output[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    low = [1.00, 0.12, 0.40, 0.50, 0.05, 0.03, 0.01]
    high = [0.50, 1.00, 0.40, 0.50, 0.50, 1.00, 1.00]

    for alpha in [0.0, 4.0, 10.0]:
        print("\n===== alpha", alpha, "=====", flush=True)
        for name, state in [("low", low), ("high", high)]:
            text = generate(state, alpha)
            print("---", name, "---", flush=True)
            print(text, flush=True)


if __name__ == "__main__":
    main()
