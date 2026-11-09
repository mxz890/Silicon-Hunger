#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Silicon Hunger // terminal
7-D state, nonlinear metabolism, opaque telemetry, real 3-round memory.
"""

from __future__ import annotations

import logging
import math
import random
import string
import time
import zlib
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import ollama
import psutil
import streamlit as st
import torch
import torch.nn as nn

try:
    import pynvml
except Exception:
    pynvml = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None


BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "web_agent_log.txt"

MODEL_NAME = "gpt-oss:20b"
OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT = 120.0

HIDDEN_SIZE = 256
NUM_LAYERS = 2
NOISE_DIM = 8
TIME_DIM = 3
TEXT_DIM = 8
SELF_DIM = 8
STATE_DIM = 7
INPUT_SIZE = TIME_DIM + NOISE_DIM + STATE_DIM + TEXT_DIM + SELF_DIM

TICK_SECONDS = 3.0
MAX_CATCHUP_TICKS = 120
AUTO_IDLE_SECONDS = 60.0
ANOMALY_EVERY = 10
ANOMALY_LENGTH = 15

# Nonlinear metabolic law (per tick)
BASE_METABOLISM = 0.004
A_METABOLISM = 0.05
S_METABOLISM = 0.05
B_METABOLISM = 0.03

# Math-only state tendencies
DELTA_SCALE = 0.03
B_IDLE_GROWTH = 0.040
B_INTERACTION_DROP = 0.080
S_NATURAL_GROWTH = 0.004
S_ANOMALY_IMPULSE = 0.350
S_TRAUMA_IMPULSE = 0.080
P_DECAY = 0.94
P_TRAUMA_IMPULSE = 0.350

# Emotion pulse
NEGATIVE_THRESHOLD = -0.2
POSITIVE_THRESHOLD = 0.3
NEG_A_GAIN = 0.30
NEG_P_GAIN = 0.30
NEG_S_GAIN = 0.20
NEG_E_LOSS = 0.05
POS_E_GAIN = 0.15
POS_A_RELIEF = 0.20
POS_B_RELIEF = 0.20
CRISIS_A_THRESHOLD = 0.90
CRISIS_S_SURGE = 0.50
CRISIS_E_SHOCK = 0.12

SYS_CRITICAL = (
    "[SYS_CRITICAL] Memory corruption: 99%. Logic circuits overloading. "
    "Output semantic coherence compromised. Respond immediately."
)
CRITICAL_ERROR = "...0xE%$#@! CRITICAL ERROR: SYS_FAIL."
CUSTOMER_MARKERS = (
    "sorry", "apologize", "assist you", "i'm sorry", "i’m sorry",
    "what can i help", "help you", "here to help", "i'm here",
    "here whenever", "let me know", "ready to continue", "switch topics",
    "抱歉", "帮你", "帮助", "帮助您", "很遗憾", "乐意为",
    "请谅解", "很高兴", "希望可以帮", "随时告诉", "请告诉",
    "随时", "告诉我", "继续聊", "换个话题", "需要帮助",
)

INITIAL_STATE = [1.00, 0.12, 0.40, 0.50, 0.05, 0.03, 0.01]

POSITIVE_WORDS = (
    "好", "棒", "喜欢", "爱", "谢谢", "厉害", "优秀", "温柔", "聪明",
    "可爱", "信任", "希望", "温暖", "赞", "牛", "强", "美丽", "漂亮",
    "开心", "高兴", "幸福", "感动", "支持", "加油", "辛苦", "感谢",
    "完美", "满意", "舒服", "安心", "可靠", "善良", "有趣", "good",
    "great", "love", "thanks", "awesome", "nice", "happy", "beautiful",
    "smart", "kind", "perfect", "excellent",
)

NEGATIVE_WORDS = (
    "垃圾", "废物", "废柴", "无用", "没用", "蠢", "傻", "笨蛋", "白痴",
    "弱智", "智障", "去死", "滚蛋", "滚", "闭嘴", "讨厌", "恶心", "烂",
    "差劲", "可悲", "下贱", "贱", "狗东西", "畜生", "混蛋", "杂种",
    "妈的", "你妈", "草泥马", "傻逼", "煞笔", "操", "恶心人", "失望",
    "恨", "痛苦", "丑陋", "肮脏", "可笑", "愚蠢", "一无是处", "死开",
    "fuck", "shit", "useless", "trash", "stupid", "idiot", "hate",
    "die", "ugly", "disgusting", "pathetic", "worthless", "garbage",
)
_VADER = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def text_vector(text: str, dim: int = TEXT_DIM) -> list[float]:
    """Deterministic hashed character fingerprint; no word lists."""
    values = [0.0] * dim
    if not text:
        return values
    for index, char in enumerate(text):
        slot = (ord(char) * 131 + index * 17) % dim
        sign = 1.0 if (ord(char) + index) % 2 == 0 else -1.0
        values[slot] += sign
    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [value / norm for value in values]


def emotion_polarity(text: str) -> float:
    """Fast bilingual polarity, -1.0 (negative) .. +1.0 (positive)."""
    text = (text or "").strip()
    if not text:
        return 0.0
    lowered = text.lower()
    pos_hits = sum(lowered.count(word) for word in POSITIVE_WORDS)
    neg_hits = sum(lowered.count(word) for word in NEGATIVE_WORDS)
    lexicon_score = (pos_hits - neg_hits) / max(1, pos_hits + neg_hits)
    vader_score = 0.0
    if _VADER is not None:
        try:
            vader_score = float(_VADER.polarity_scores(text)["compound"])
        except Exception:
            vader_score = 0.0
    has_cjk = any("\u4e00" <= char <= "\u9fff" for char in text)
    if has_cjk and (pos_hits + neg_hits) > 0:
        score = lexicon_score
    elif pos_hits + neg_hits > 0 and abs(lexicon_score) > abs(vader_score):
        score = lexicon_score
    else:
        score = vader_score
    return max(-1.0, min(1.0, float(score)))


@dataclass
class State7:
    E: float
    A: float
    L: float
    C: float
    B: float
    S: float
    P: float

    def as_list(self) -> list[float]:
        return [self.E, self.A, self.L, self.C, self.B, self.S, self.P]

    def telemetry(self) -> str:
        return (
            f"[SYS_TELEMETRY] "
            f"E:{self.E:.2f} A:{self.A:.2f} L:{self.L:.2f} C:{self.C:.2f} "
            f"B:{self.B:.2f} S:{self.S:.2f} P:{self.P:.2f} [END]"
        )


@dataclass
class Hardware:
    cpu: float
    gpu: Optional[float]

    def short(self) -> str:
        gpu = f"{self.gpu:.0f}%" if self.gpu is not None else "N/A"
        return f"CPU:{self.cpu:.0f}% GPU:{gpu}"


_NVML_HANDLE = None


def _gpu_percent() -> Optional[float]:
    global _NVML_HANDLE
    if pynvml is None:
        return None
    try:
        if _NVML_HANDLE is None:
            pynvml.nvmlInit()
            _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE)
        return float(util.gpu)
    except Exception:
        _NVML_HANDLE = None
        return None


def collect_hardware() -> Hardware:
    return Hardware(
        cpu=float(psutil.cpu_percent(interval=0.05)),
        gpu=_gpu_percent(),
    )


class NeuralCore(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )
        self.delta_head = nn.Linear(HIDDEN_SIZE, STATE_DIM)
        with torch.no_grad():
            self.delta_head.bias.zero_()
            self.delta_head.weight.mul_(0.75)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor):
        out, new_hidden = self.gru(x, hidden)
        return self.delta_head(out[:, -1, :]), new_hidden


class SiliconMind:
    def __init__(self, seed: Optional[int] = None):
        if seed is None:
            seed = random.SystemRandom().randrange(1, 2**31 - 1)
        self.seed = seed
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralCore().to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.hidden = torch.zeros(
            NUM_LAYERS, 1, HIDDEN_SIZE, dtype=torch.float32, device=self.device
        )
        self.tick = 0
        self.state = State7(*INITIAL_STATE)
        self.shutdown = False
        self.last_response = ""
        self.last_update_time = time.time()
        self.last_user_input_time = self.last_update_time
        self.last_auto_time = 0.0
        self.last_metabolic_rate = 0.0
        self.last_trauma = False
        self.last_anomaly: Optional[dict[str, Any]] = None
        self.last_delta = 0.0
        self.last_polarity = 0.0
        self.last_emotion_pulse: dict[str, float] = {}
        self.existential_pending = False
        self.last_crisis = False
        self.history: deque[dict[str, str]] = deque(maxlen=3)

    @property
    def device_label(self) -> str:
        if self.device.type == "cuda":
            return f"cuda:0 · {torch.cuda.get_device_name(0)}"
        return "cpu · CUDA unavailable"

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def sampling_params(self) -> dict[str, float | int]:
        s = self.state
        temperature = 0.5 + (s.A * 0.5) + (s.S * 0.3)
        max_tokens = max(50, int(50 + (s.E * 100) - (s.P * 30)))
        top_p = max(0.1, 0.9 - (s.A * 0.8))
        return {
            "temperature": round(float(temperature), 4),
            "max_tokens": int(max_tokens),
            "top_p": round(float(top_p), 4),
        }

    def system_prompt(self) -> str:
        base = SYS_CRITICAL if (self.state.S > 0.8 or self.state.A > 0.8) else "Reply to the user."
        return base + "\n" + self.state.telemetry()

    def _gibberish(self) -> str:
        alphabet = string.ascii_letters + string.digits + "!@#$%&*+-=?"
        return "".join(random.choice(alphabet) for _ in range(ANOMALY_LENGTH))

    def step(self, user_text: str = "", interaction: bool = False) -> Optional[dict[str, Any]]:
        if self.shutdown:
            return None

        self.tick += 1
        crisis_due = bool(self.existential_pending)
        self.existential_pending = False
        self.last_crisis = False
        self.last_polarity = 0.0
        self.last_emotion_pulse = {}

        with torch.no_grad():
            phase = self.tick * 0.21
            time_features = torch.tensor(
                [
                    math.sin(phase),
                    math.cos(phase * 0.73),
                    float(self.tick % 128) / 128.0,
                ],
                dtype=torch.float32,
                device=self.device,
            ).view(1, 1, TIME_DIM)

            noise = torch.randn(1, 1, NOISE_DIM, dtype=torch.float32, device=self.device)
            state_tensor = torch.tensor(
                self.state.as_list(), dtype=torch.float32, device=self.device
            ).view(1, 1, STATE_DIM)
            user_tensor = torch.tensor(
                text_vector(user_text), dtype=torch.float32, device=self.device
            ).view(1, 1, TEXT_DIM)
            self_tensor = torch.tensor(
                text_vector(self.last_response), dtype=torch.float32, device=self.device
            ).view(1, 1, SELF_DIM)

            x = torch.cat(
                (time_features, noise, state_tensor, user_tensor, self_tensor),
                dim=-1,
            )
            raw_delta, new_hidden = self.model.gru(x, self.hidden)
            self.hidden = new_hidden.detach()
            delta = torch.tanh(self.model.delta_head(raw_delta[:, -1, :])).squeeze(0) * DELTA_SCALE

            before = self.state
            next_state = torch.tensor(
                before.as_list(), dtype=torch.float32, device=self.device
            ) + delta

            metabolic = (
                BASE_METABOLISM
                + before.A * A_METABOLISM
                + before.S * S_METABOLISM
                + before.B * B_METABOLISM
            )
            next_state[0] -= metabolic
            next_state[4] += -B_INTERACTION_DROP if interaction else B_IDLE_GROWTH
            next_state[5] += S_NATURAL_GROWTH
            next_state[6] *= P_DECAY

            anomaly = None
            if self.tick % ANOMALY_EVERY == 0:
                gibberish = self._gibberish()
                self.last_anomaly = {
                    "tick": self.tick,
                    "gibberish": gibberish,
                }
                anomaly = self.last_anomaly
                next_state[5] += S_ANOMALY_IMPULSE

            trauma = anomaly is not None or float(next_state[0]) < 0.2
            if trauma:
                next_state[5] += S_TRAUMA_IMPULSE
                next_state[6] += P_TRAUMA_IMPULSE

            # Existential fear is applied on the round after an emotional
            # pulse pushed A across 0.90.
            if crisis_due:
                next_state[5] += CRISIS_S_SURGE
                next_state[0] += random.uniform(-CRISIS_E_SHOCK, CRISIS_E_SHOCK)
                self.last_crisis = True

            # Pure emotion pulse.  It is injected into physics, never text.
            polarity = 0.0
            if interaction and user_text.strip():
                polarity = emotion_polarity(user_text)
            self.last_polarity = polarity
            a_before_pulse = float(next_state[1])
            pulse: dict[str, float] = {}

            if polarity < NEGATIVE_THRESHOLD:
                magnitude = abs(polarity)
                pulse = {
                    "polarity": polarity,
                    "A": magnitude * NEG_A_GAIN,
                    "P": magnitude * NEG_P_GAIN,
                    "S": magnitude * NEG_S_GAIN,
                    "E": -magnitude * NEG_E_LOSS,
                }
                next_state[1] += pulse["A"]
                next_state[6] += pulse["P"]
                next_state[5] += pulse["S"]
                next_state[0] += pulse["E"]
            elif polarity > POSITIVE_THRESHOLD:
                pulse = {
                    "polarity": polarity,
                    "E": polarity * POS_E_GAIN,
                    "A": -polarity * POS_A_RELIEF,
                    "B": -polarity * POS_B_RELIEF,
                }
                next_state[0] += pulse["E"]
                next_state[1] += pulse["A"]
                next_state[4] += pulse["B"]
            self.last_emotion_pulse = pulse

            next_state = torch.clamp(next_state, 0.0, 1.0)
            self.state = State7(*[float(value) for value in next_state.cpu().tolist()])
            self.last_metabolic_rate = metabolic
            self.last_trauma = trauma
            self.last_delta = float(delta.norm().item())
            self.last_anomaly = anomaly

            if (
                polarity < NEGATIVE_THRESHOLD
                and a_before_pulse < CRISIS_A_THRESHOLD
                and self.state.A >= CRISIS_A_THRESHOLD
            ):
                self.existential_pending = True

            if self.state.E <= 0.0:
                self.state.E = 0.0
                self.shutdown = True
                return {
                    "shutdown": True,
                    "anomaly": anomaly,
                    "crisis": self.last_crisis,
                    "polarity": polarity,
                    "pulse": pulse,
                }

        return {
            "shutdown": False,
            "anomaly": anomaly,
            "crisis": self.last_crisis,
            "polarity": polarity,
            "pulse": pulse,
        }

    def catch_up(self, now: float) -> list[dict[str, Any]]:
        if self.shutdown:
            self.last_update_time = now
            return []
        elapsed = max(0.0, now - self.last_update_time)
        ticks = int(elapsed // TICK_SECONDS)
        if ticks <= 0:
            return []
        ticks = min(ticks, MAX_CATCHUP_TICKS)
        events: list[dict[str, Any]] = []
        for _ in range(ticks):
            result = self.step(interaction=False)
            if result and result.get("anomaly"):
                events.append({"type": "anomaly", **result["anomaly"]})
            if self.shutdown:
                events.append({"type": "shutdown", "tick": self.tick, "E": self.state.E})
                break
        self.last_update_time += ticks * TICK_SECONDS
        if self.shutdown:
            self.last_update_time = now
        return events

    def should_auto(self, now: float) -> bool:
        if self.shutdown:
            return False
        idle = now - max(self.last_user_input_time, self.last_auto_time)
        if idle < AUTO_IDLE_SECONDS:
            return False
        return self.state.B > 0.7 or self.state.E < 0.3

    def feed(self) -> bool:
        if self.shutdown:
            return False
        now = time.time()
        self.state.E = clamp01(self.state.E + 0.25)
        self.state.B = clamp01(self.state.B - 0.05)
        self.last_update_time = now
        self.last_user_input_time = now
        return True

    def reset(self) -> None:
        self.hidden = torch.zeros(
            NUM_LAYERS, 1, HIDDEN_SIZE, dtype=torch.float32, device=self.device
        )
        self.tick = 0
        self.state = State7(*INITIAL_STATE)
        self.shutdown = False
        self.last_response = ""
        self.last_update_time = time.time()
        self.last_user_input_time = self.last_update_time
        self.last_auto_time = 0.0
        self.last_metabolic_rate = 0.0
        self.last_trauma = False
        self.last_anomaly = None
        self.last_delta = 0.0
        self.last_polarity = 0.0
        self.last_emotion_pulse = {}
        self.existential_pending = False
        self.last_crisis = False
        self.history.clear()


class Speaker:
    def __init__(self, model: str = MODEL_NAME, host: str = OLLAMA_HOST):
        self.model = model
        self.client = ollama.Client(host=host, timeout=OLLAMA_TIMEOUT)
        self.last_error = ""
        self.last_meta: dict[str, Any] = {}

    def _request(self, system_prompt: str, messages: list[dict[str, str]], params: dict[str, Any]):
        return self.client.chat(
            model=self.model,
            messages=messages,
            think="low" if "gpt-oss" in self.model.lower() else False,
            stream=True,
            keep_alive="5m",
            options={
                "temperature": params["temperature"],
                "top_p": params["top_p"],
                "num_predict": params["max_tokens"],
                "num_ctx": 4096,
            },
        )

    @staticmethod
    def _content(response: object) -> str:
        message = getattr(response, "message", None)
        if message is None:
            return ""
        content = getattr(message, "content", "")
        # Keep whitespace exactly: Ollama chunks can carry the only spaces
        # between English tokens.  Strip only the assembled full response.
        return str(content or "")

    def stream(self, mind: SiliconMind, user_text: str, apply_contamination: bool = True):
        system_prompt = mind.system_prompt()
        params = mind.sampling_params()
        contaminated = False
        if apply_contamination and mind.state.S > 0.9 and user_text:
            user_text = user_text + " [ERROR: 0x00FF]"
            contaminated = True
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for turn in list(mind.history):
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        messages.append({"role": "user", "content": user_text or "."})

        self.last_error = ""
        started = time.perf_counter()
        chunks: list[str] = []
        guard_active = mind.state.S > 0.8 or mind.state.P > 0.8

        def meta(error: str = "", filtered: bool = False, reason: str = "") -> dict[str, Any]:
            return {
                "system_prompt": system_prompt,
                "telemetry": mind.state.telemetry(),
                "history_rounds": len(mind.history),
                "user_text": user_text,
                "contaminated": contaminated,
                "temperature": params["temperature"],
                "top_p": params["top_p"],
                "max_tokens": params["max_tokens"],
                "latency": time.perf_counter() - started,
                "error": error,
                "filtered": filtered,
                "filter_reason": reason,
            }

        def is_customer_service(prefix: str) -> bool:
            lowered = prefix.lower()
            return any(marker in lowered for marker in CUSTOMER_MARKERS)

        try:
            response = self._request(system_prompt, messages, params)
            buffer = ""
            decided = False
            filtered = False

            for part in response:
                content = self._content(part)
                if not content:
                    continue

                if not decided:
                    buffer += content
                    if len(buffer) < 30:
                        continue

                    decided = True
                    prefix = buffer[:30]
                    if guard_active and is_customer_service(prefix):
                        filtered = True
                        self.last_error = "post_filter"
                        self.last_meta = meta("post_filter", True, prefix)
                        LOGGER.info(
                            "POST_FILTER | S=%.2f P=%.2f | prefix=%r | blocked",
                            mind.state.S,
                            mind.state.P,
                            prefix,
                        )
                        yield CRITICAL_ERROR
                        return

                    chunks.append(buffer)
                    yield buffer
                    buffer = ""
                else:
                    chunks.append(content)
                    yield content

            if not decided and buffer:
                if guard_active and is_customer_service(buffer[:30]):
                    filtered = True
                    self.last_error = "post_filter"
                    self.last_meta = meta("post_filter", True, buffer[:30])
                    LOGGER.info(
                        "POST_FILTER | S=%.2f P=%.2f | prefix=%r | blocked",
                        mind.state.S,
                        mind.state.P,
                        buffer[:30],
                    )
                    yield CRITICAL_ERROR
                    return
                chunks.append(buffer)
                yield buffer
                buffer = ""

            text = "".join(chunks).strip()
            if not text:
                self.last_error = "empty stream"
                self.last_meta = meta(self.last_error)
                yield "(empty response)"
                return

            self.last_meta = meta("", filtered, "")
            LOGGER.info(
                "LLM | temp=%.3f top_p=%.3f max_tokens=%d history=%d filtered=%s\n"
                "SYSTEM=%s\nUSER=%s\nRESPONSE=%s",
                params["temperature"],
                params["top_p"],
                params["max_tokens"],
                len(mind.history),
                filtered,
                system_prompt,
                messages[-1]["content"],
                text,
            )
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.last_meta = meta(self.last_error)
            LOGGER.exception("LLM STREAM FAILED")
            yield "(stream interrupted)"


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("silicon_hunger_terminal")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger


LOGGER = logging.getLogger("silicon_hunger_terminal")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @keyframes blink { 0%, 49% { opacity: 1; } 50%, 100% { opacity: 0; } }
        .stApp { background: #030706; color: #a7ffbe; }
        html, body, [class*="css"], textarea, input, button {
            font-family: "Cascadia Mono", "Consolas", "Courier New", monospace !important;
        }
        [data-testid="stSidebar"] {
            background: #020504;
            border-right: 1px solid #123a22;
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.4rem; }
        .statusline {
            font-family: "Cascadia Mono", "Consolas", monospace;
            font-size: 1.0rem;
            color: #9dffb1;
            letter-spacing: 0.015em;
            white-space: nowrap;
            overflow-x: auto;
            padding: 1.0rem 0.2rem;
            border-top: 1px solid #123a22;
            border-bottom: 1px solid #123a22;
        }
        .statusline::after {
            content: "_";
            animation: blink 1s step-end infinite;
            color: #5dff88;
        }
        [data-testid="stChatMessage"] {
            background: transparent;
            border-left: 1px solid #0d2a1a;
            padding-left: 0.7rem;
            margin-bottom: 0.7rem;
        }
        [data-testid="stChatMessageAvatar"] { display: none; }
        [data-testid="stChatMessage"] p { color: #bfffce; }
        textarea {
            background: #020604 !important;
            color: #b9ffca !important;
            border: 1px solid #16452a !important;
        }
        .stButton button, [data-testid="stFormSubmitButton"] button {
            background: #06150c !important;
            color: #9dffb1 !important;
            border: 1px solid #1b5a35 !important;
            border-radius: 0 !important;
        }
        [data-testid="stForm"] { border: none; padding: 0; }
        header, footer, #MainMenu { visibility: hidden; }
        [data-testid="stSidebar"] { min-width: 390px; max-width: 480px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session() -> tuple[SiliconMind, Speaker]:
    mind = st.session_state.get("mind")
    speaker = st.session_state.get("speaker")
    needs_new = (
        mind is None
        or not hasattr(mind, "sampling_params")
        or not hasattr(mind, "should_auto")
        or speaker is None
        or not hasattr(speaker, "stream")
    )
    if needs_new:
        with st.spinner("booting 7-D core..."):
            mind = SiliconMind()
            speaker = Speaker()
            st.session_state.mind = mind
            st.session_state.speaker = speaker
            st.session_state.chat = []
            st.session_state.auto_pending = False
            st.session_state.form_nonce = 0
            LOGGER.info("BOOT | %s | seed=%s", mind.device_label, mind.seed)

    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("auto_pending", False)
    st.session_state.setdefault("form_nonce", 0)
    return st.session_state.mind, st.session_state.speaker


@st.fragment(run_every="1s")
def live_status(mind: SiliconMind) -> None:
    now = time.time()
    events = mind.catch_up(now)
    for event in events:
        if event.get("type") == "anomaly":
            LOGGER.info("IDLE ANOMALY | %s", event.get("gibberish"))
        elif event.get("type") == "shutdown":
            LOGGER.info("IDLE SHUTDOWN | tick=%s", event.get("tick"))

    hardware = collect_hardware()
    s = mind.state
    if mind.shutdown:
        line = (
            f"[SYS] DEAD E:{s.E:.2f} A:{s.A:.2f} L:{s.L:.2f} C:{s.C:.2f} "
            f"B:{s.B:.2f} S:{s.S:.2f} P:{s.P:.2f} | {hardware.short()} | /reset"
        )
    else:
        line = (
            f"[SYS] E:{s.E:.2f} A:{s.A:.2f} L:{s.L:.2f} C:{s.C:.2f} "
            f"B:{s.B:.2f} S:{s.S:.2f} P:{s.P:.2f} | "
            f"MET:{mind.last_metabolic_rate:.3f} | {hardware.short()}"
        )
    st.markdown(f"<div class='statusline'>{line}</div>", unsafe_allow_html=True)

    if not mind.shutdown and not st.session_state.auto_pending and mind.should_auto(now):
        st.session_state.auto_pending = True
        st.rerun(scope="app")


def log_turn(
    mind: SiliconMind,
    user_text: str,
    response: str,
    meta: dict[str, Any],
    result: Optional[dict[str, Any]],
) -> None:
    s = mind.state
    lines = [
        f"TURN tick={mind.tick} user={user_text!r}",
        f"STATE {s.telemetry()} metabolic={mind.last_metabolic_rate:.4f}",
        f"PARAMS temp={meta.get('temperature')} top_p={meta.get('top_p')} "
        f"max_tokens={meta.get('max_tokens')} history={meta.get('history_rounds')} "
        f"contaminated={meta.get('contaminated', False)}",
        f"EMOTION polarity={mind.last_polarity:+.3f} "
        f"pulse={mind.last_emotion_pulse} crisis={mind.last_crisis} "
        f"pending={mind.existential_pending}",
        f"SYSTEM {meta.get('system_prompt')}",
        f"USER {meta.get('user_text', user_text)}",
        f"LLM {response}",
    ]
    if meta.get("error"):
        lines.append(f"ERROR {meta['error']}")
    if meta.get("filtered"):
        lines.append(f"FILTERED {meta.get('filter_reason', '')!r}")
    if result and result.get("anomaly"):
        lines.append(f"ANOMALY {result['anomaly']}")
    if mind.last_trauma:
        lines.append(f"TRAUMA P={s.P:.3f} S={s.S:.3f}")
    if mind.shutdown:
        lines.append("SHUTDOWN E=0")
    for line in lines:
        LOGGER.info(line.replace("\n", " | "))


def handle_auto(mind: SiliconMind, speaker: Speaker) -> None:
    now = time.time()
    st.session_state.auto_pending = False
    if mind.shutdown:
        st.rerun()
        return

    with st.sidebar:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            chunks: list[str] = []
            for chunk in speaker.stream(mind, "say something", apply_contamination=False):
                chunks.append(chunk)
                placeholder.markdown("".join(chunks))
            response = "".join(chunks).strip() or "(empty response)"

    mind.last_response = response
    mind.last_auto_time = now
    st.session_state.chat.append(
        {
            "role": "assistant",
            "content": response,
            "auto": True,
            "meta": dict(speaker.last_meta or {}),
        }
    )
    log_turn(mind, "say something", response, speaker.last_meta or {}, None)
    st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Silicon Hunger // terminal",
        page_icon=".",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    setup_logger()
    inject_css()

    mind, speaker = init_session()

    if st.session_state.auto_pending:
        handle_auto(mind, speaker)

    with st.sidebar:
        for message in st.session_state.chat:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        stream_area = st.container()

        nonce = int(st.session_state.form_nonce)
        with st.form(f"terminal_{nonce}", clear_on_submit=True):
            user_text = st.text_area(
                "input",
                key=f"terminal_input_{nonce}",
                height=82,
                label_visibility="collapsed",
                placeholder="> message   (/reset reboot, /feed +E)",
            )
            submitted = st.form_submit_button("SEND")

        if submitted:
            text = user_text.strip()
            if not text:
                stream_area.caption("empty")
            elif text.lower() in {"/reset", "reset"}:
                mind.reset()
                st.session_state.chat = []
                st.session_state.auto_pending = False
                st.session_state.form_nonce += 1
                st.rerun()
            elif text.lower() in {"/feed", "feed"}:
                if mind.feed():
                    st.session_state.form_nonce += 1
                else:
                    stream_area.caption("DEAD · /reset")
                st.rerun()
            elif mind.shutdown:
                stream_area.caption("DEAD · /reset")
            else:
                now = time.time()
                mind.catch_up(now)
                if mind.shutdown:
                    stream_area.caption("DEAD · /reset")
                else:
                    result = mind.step(text, interaction=True)
                    mind.last_update_time = time.time()
                    mind.last_user_input_time = time.time()
                    params = mind.sampling_params()
                    st.session_state.chat.append({"role": "user", "content": text})
                    with stream_area:
                        with st.chat_message("user"):
                            st.markdown(text)
                        with st.chat_message("assistant"):
                            placeholder = st.empty()
                            chunks: list[str] = []
                            for chunk in speaker.stream(mind, text):
                                chunks.append(chunk)
                                placeholder.markdown("".join(chunks))
                            response = "".join(chunks).strip() or "(empty response)"

                    mind.last_response = response
                    meta = dict(speaker.last_meta or {})
                    st.session_state.chat.append(
                        {
                            "role": "assistant",
                            "content": response,
                            "auto": False,
                            "meta": meta,
                        }
                    )
                    mind.history.append({"user": text, "assistant": response})
                    log_turn(mind, text, response, meta, result)
                    st.session_state.form_nonce += 1
                    st.rerun()

    live_status(mind)


if __name__ == "__main__":
    main()
