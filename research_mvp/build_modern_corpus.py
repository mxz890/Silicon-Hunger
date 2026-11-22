#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a richer modern Chinese prose corpus from local saved HTML pages."""

from __future__ import annotations

import html
import re
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


def read_any(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def plain(path: Path) -> str:
    t = read_any(path)
    t = re.sub(r"<script.*?</script>", "", t, flags=re.S | re.I)
    t = re.sub(r"<style.*?</style>", "", t, flags=re.S | re.I)
    t = re.sub(r"<[^>]+>", "", t)
    t = html.unescape(t)
    return re.sub(r"[ \t\r\f\v]+", "", t)


def slice_between(path: Path, start: str, end: str) -> str:
    text = plain(path)
    i = text.find(start)
    if i < 0:
        raise ValueError(f"start not found in {path}: {start}")
    j = text.rfind(end, i)
    if j < 0:
        j = len(text)
    return text[i:j].strip()


SPECS = [
    ("luxun_corpus.txt", "鲁迅作品", None, None),
    ("chun.html", "朱自清《春》", "盼望着，盼望着", "有的是希望"),
    ("beijing.html", "朱自清《背影》", "那年冬天，祖母死了", "我不知何时再能与他相见"),
    ("jinan.html", "老舍《济南的冬天》", "对于一个在北平住惯的人", "这就是冬天的济南"),
    ("beijing_spring.html", "老舍《北京的春节》", "照北京的老规矩", "这的确是美好快乐的日子"),
    ("congcong.html", "朱自清《匆匆》", "燕子去了", "我们的日子为什么一去不复返呢"),
]


def main() -> None:
    blocks = []
    for filename, title, start, end in SPECS:
        path = DATA / filename
        if not path.exists():
            print("MISSING", path)
            continue
        if start is None:
            text = read_any(path)
        else:
            text = slice_between(path, start, end)
        text = re.sub(r"\s+", "", text)
        blocks.append(f"\n\n=== {title} ===\n{text}")
        print(title, "chars", len(text))
    corpus = "\n".join(blocks)
    (DATA / "modern_chinese_corpus.txt").write_text(corpus, encoding="utf-8")
    print("TOTAL", len(corpus), "SAVED", DATA / "modern_chinese_corpus.txt")


if __name__ == "__main__":
    main()
