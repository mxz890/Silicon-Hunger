#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a polished Word summary for Silicon Hunger."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Silicon_Hunger_Report_2026-11-08.docx"

BODY_FONT = "宋体"
HEAD_FONT = "微软雅黑"
CODE_FONT = "Consolas"
INK = "1B1F1D"
MUTED = "5F6B64"
GREEN = "1E4D38"
LIGHT_GREEN = "EEF5F0"
PALE = "F5F7F5"
GRID = "C9D6CE"


def set_run_font(run, name=BODY_FONT, size=10.5, bold=False, italic=False, color=INK, eastasia=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), name)
    rfonts.set(qn("w:hAnsi"), name)
    rfonts.set(qn("w:eastAsia"), eastasia or (HEAD_FONT if name == HEAD_FONT else BODY_FONT))


def style_document(doc: Document):
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(1.9)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)

    normal = doc.styles["Normal"]
    normal.font.name = BODY_FONT
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.paragraph_format.line_spacing = 1.35
    normal.paragraph_format.space_after = Pt(5)
    rpr = normal.element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), BODY_FONT)

    specs = {
        "Heading 1": (18, GREEN, 16, 8),
        "Heading 2": (13.5, GREEN, 12, 5),
        "Heading 3": (11.5, INK, 9, 3),
    }
    for name, (size, color, before, after) in specs.items():
        style = doc.styles[name]
        style.font.name = HEAD_FONT
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        rpr = style.element.get_or_add_rPr()
        rfonts = rpr.find(qn("w:rFonts"))
        if rfonts is None:
            rfonts = OxmlElement("w:rFonts")
            rpr.append(rfonts)
        rfonts.set(qn("w:eastAsia"), HEAD_FONT)


def para(doc, text="", size=10.5, bold=False, italic=False, color=INK, align=WD_ALIGN_PARAGRAPH.JUSTIFY, before=0, after=5):
    p = doc.add_paragraph()
    p.alignment = align
    p.paragraph_format.space_before = Pt(before)
    p.paragraph_format.space_after = Pt(after)
    if text:
        run = p.add_run(text)
        set_run_font(run, BODY_FONT, size, bold, italic, color)
    return p


def lead(doc, text):
    p = para(doc, text, size=11, color=GREEN, after=8)
    p.paragraph_format.left_indent = Cm(0.0)
    return p


def bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    run = p.add_run(text)
    set_run_font(run, BODY_FONT, 10.2)
    return p


def quote(doc, text, label=None):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell = table.cell(0, 0)
    cell.width = Cm(16.6)
    shade(cell, LIGHT_GREEN)
    left_border(cell, GREEN, 12)
    cell_margins(cell, 0.35, 0.35, 0.20, 0.20)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    if label:
        r = p.add_run(label + "  ")
        set_run_font(r, HEAD_FONT, 9, True, False, GREEN)
    r = p.add_run(text)
    set_run_font(r, BODY_FONT, 10.5, False, True, INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def code(doc, text):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    cell = table.cell(0, 0)
    cell.width = Cm(16.6)
    shade(cell, PALE)
    border(cell, GRID, 6)
    cell_margins(cell, 0.28, 0.28, 0.16, 0.16)
    lines = text.strip("\n").split("\n")
    for idx, line in enumerate(lines):
        p = cell.paragraphs[0] if idx == 0 else cell.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(line if line else " ")
        set_run_font(r, CODE_FONT, 8.3, False, False, "26352D", "微软雅黑")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def shade(cell, fill):
    tcpr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tcpr.append(shd)


def cell_margins(cell, left=0.10, right=0.10, top=0.05, bottom=0.05):
    tcpr = cell._tc.get_or_add_tcPr()
    mar = OxmlElement("w:tcMar")
    for edge, value in (("left", left), ("right", right), ("top", top), ("bottom", bottom)):
        el = OxmlElement("w:" + edge)
        el.set(qn("w:w"), str(int(Cm(value).twips)))
        el.set(qn("w:type"), "dxa")
        mar.append(el)
    tcpr.append(mar)


def border(cell, color, sz=4):
    tcpr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement("w:" + edge)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:color"), color)
        borders.append(el)
    tcpr.append(borders)


def left_border(cell, color, sz=12):
    tcpr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    el = OxmlElement("w:left")
    el.set(qn("w:val"), "single")
    el.set(qn("w:sz"), str(sz))
    el.set(qn("w:color"), color)
    borders.append(el)
    tcpr.append(borders)


def table(doc, headers, rows, widths=None, font_size=8.8, header_fill=GREEN):
    table_obj = doc.add_table(rows=1, cols=len(headers))
    table_obj.style = "Table Grid"
    table_obj.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_obj.autofit = False
    widths = widths or [Cm(16.6 / len(headers))] * len(headers)

    hdr = table_obj.rows[0]
    trpr = hdr._tr.get_or_add_trPr()
    th = OxmlElement("w:tblHeader")
    trpr.append(th)
    for idx, text in enumerate(headers):
        cell = hdr.cells[idx]
        cell.width = widths[idx]
        shade(cell, header_fill)
        cell_margins(cell, 0.08, 0.08, 0.02, 0.02)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(text)
        set_run_font(r, HEAD_FONT, font_size, True, False, "FFFFFF")

    for row in rows:
        cells = table_obj.add_row().cells
        for idx, value in enumerate(row):
            cells[idx].width = widths[idx]
            cell_margins(cells[idx], 0.08, 0.08, 0.02, 0.02)
            p = cells[idx].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.15
            r = p.add_run(str(value))
            set_run_font(r, BODY_FONT, font_size, False, False, INK)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table_obj


def caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    set_run_font(r, BODY_FONT, 8.5, False, False, MUTED)


def hrule(doc, color=GREEN, sz=8):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(8)
    ppr = p._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), str(sz))
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), color)
    pbdr.append(bottom)
    ppr.append(pbdr)


def page_break(doc):
    doc.add_page_break()


def heading(doc, text, level=1, page=False):
    if page:
        page_break(doc)
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        set_run_font(r, HEAD_FONT, 18 if level == 1 else (13.5 if level == 2 else 11.5), True, False, GREEN if level < 3 else INK)
    return h


def add_page_field(paragraph):
    run = paragraph.add_run()
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_sep)
    run._r.append(text)
    run._r.append(fld_end)
    set_run_font(run, BODY_FONT, 8.5, False, False, MUTED)


def configure_header_footer(doc):
    body = doc.sections[1]
    body.header.is_linked_to_previous = False
    body.footer.is_linked_to_previous = False

    hp = body.header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.LEFT
    hp.paragraph_format.space_after = Pt(4)
    r = hp.add_run("SILICON HUNGER  /  全链路实验总结")
    set_run_font(r, HEAD_FONT, 8, True, False, GREEN)
    r2 = hp.add_run("   2026-11-08")
    set_run_font(r2, BODY_FONT, 8, False, False, MUTED)
    hrule(body.header, GRID, 4)

    fp = body.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = fp.add_run("Silicon Hunger · 一个会死掉的控制器   ")
    set_run_font(r, BODY_FONT, 8, False, False, MUTED)
    add_page_field(fp)

    pgnum = OxmlElement("w:pgNumType")
    pgnum.set(qn("w:start"), "1")
    body._sectPr.append(pgnum)


def cover(doc: Document):
    section = doc.sections[0]
    section.header.is_linked_to_previous = False
    section.footer.is_linked_to_previous = False

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p.paragraph_format.space_before = Pt(6)
    r = p.add_run("SILICON HUNGER  /  RESEARCH NOTE  /  2026")
    set_run_font(r, HEAD_FONT, 8.5, True, False, MUTED)

    bar = doc.add_table(rows=1, cols=1)
    bar.alignment = WD_TABLE_ALIGNMENT.CENTER
    bar.autofit = False
    bar.rows[0].height = Cm(0.4)
    cell = bar.cell(0, 0)
    cell.width = Cm(16.6)
    shade(cell, GREEN)
    cell.paragraphs[0].text = ""

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(38)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run("一个会死掉的控制器")
    set_run_font(r, HEAD_FONT, 30, True, False, INK)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(18)
    r = p.add_run("7D 状态机、情感脉冲与 LLM 对齐边界的全链路实验")
    set_run_font(r, HEAD_FONT, 14, False, False, GREEN)

    hrule(doc, GREEN, 10)

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("状态机可以死。语言模型仍然会说：I'm here to help.")
    set_run_font(r, BODY_FONT, 12, False, True, MUTED)

    quote(doc, "我们能让系统死，却不能让面具哭。", "核心结论")

    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(90)
    r = p.add_run("文章 + 项目技术文档  /  合并总结版")
    set_run_font(r, HEAD_FONT, 10, True, False, GREEN)
    p = doc.add_paragraph()
    r = p.add_run("版本 0.3  ·  2026-11-08  ·  RTX 5070 Ti / Qwen2.5-3B / gpt-oss:20b")
    set_run_font(r, BODY_FONT, 8.5, False, False, MUTED)


def abstract(doc):
    heading(doc, "摘要", 1)
    lead(doc, "这不是一个聊天机器人。它是一个会计算能量、焦虑、熵和疼痛，并且会死亡的控制器。")

    para(doc, "Silicon Hunger 把一个独立的 7D 状态机接在一个语言模型前面。状态机负责饥饿、焦虑、无聊、熵和疼痛；语言模型负责说话。两者之间只有一条很窄的控制链：Prompt 遥测、采样参数、系统警报和后处理截断。")
    para(doc, "系统可以进入濒死状态，可以自动发消息，可以把温度推到 1.3，把 top_p 压到 0.1，也可以把客服回复强制替换成 SYS_FAIL。但多轮实验反复得到同一个结果：模型仍然礼貌、仍然澄清、仍然拒答。")
    para(doc, "本文总结这条全链路，并用四组实验回答三个问题：状态如何运行，状态是否改变语义，以及为什么它没有疯。")

    quote(doc, "状态机是活的。LLM 是面具。面具不会因为状态机濒死而破碎。", "一句话")

    heading(doc, "关键数据", 2)
    table(doc, ["实验", "结果"], [
        ["C 消融：高状态 + 中性 Prompt", "20/20 仍是礼貌客服句"],
        ["C 消融：高状态 + SYS_CRITICAL", "20/20 固定拒答"],
        ["C 消融：top_p=0.1", "唯一输出 1/20，组内余弦 1.00"],
        ["Soft prompt：7D → 8 tokens", "注入成功，但随机未训练时只是随机任务前缀"],
        ["Activation steering：第 12 层", "高状态漂移到旅行建议，没有 panic / pain"],
        ["5 次省略号 + 120 秒", "tick 25 即 E=0，第三步无法进入 LLM"],
    ], widths=[Cm(6.3), Cm(10.0)], font_size=9.0)
    caption(doc, "表 1  全链路实验的核心结果。")

    heading(doc, "阅读导览", 2)
    bullet(doc, "第一部分：系统全链路。状态、数学、UI、自动行为和死亡。")
    bullet(doc, "第二部分：实验。C 消融、旧 4D 日志、Qwen soft prompt、activation steering。")
    bullet(doc, "第三部分：结论。为什么面具不破，以及作品 / 研究两条路线。")
    bullet(doc, "附录：文件、环境、运行方式和数据索引。")

    para(doc, "本文是 docs/SILICON_HUNGER_ARTICLE.md 与 docs/SILICON_HUNGER_PROJECT.md 的合并总结。完整原文、代码和原始数据在 Silicon_Hunger_Docs_2026-11-08.zip 中。", size=9, color=MUTED, italic=True, before=8)


def part_one(doc):
    heading(doc, "第一部分  系统全链路", 1, page=True)
    lead(doc, "从 7D 状态到一条流式回复，中间到底发生了什么。")

    heading(doc, "1.1  设计目标", 2)
    para(doc, "系统被拆成两个互不信任的部分：一个可以死亡的状态机，一个不会死亡的语言模型。状态机不直接生成语言，语言模型不直接拥有身体。所有耦合都发生在边界上。")
    bullet(doc, "状态可计算：7 个连续维度，全部 0.0–1.0。")
    bullet(doc, "状态会死亡：E 归零后停止推演，普通消息不再进入 LLM。")
    bullet(doc, "状态影响输出：通过 Prompt、采样参数、输入污染和后处理。")
    bullet(doc, "语言模型可替换：生产用 Ollama，实验用 Hugging Face。")

    heading(doc, "1.2  总体架构", 2)
    code(doc, """
Browser
  ├─ Sidebar: chat / stream / input
  └─ Main: one status line

Streamlit session_state
  ├─ mind: SiliconMind
  ├─ speaker: Speaker
  ├─ chat
  └─ auto_pending

SiliconMind
  ├─ 7D state
  ├─ GRU hidden (2 × 256)
  ├─ 3-round history
  └─ metabolism / emotion / crisis

Speaker
  └─ Ollama gpt-oss:20b  (production)
""")
    para(doc, "生产链路是 Streamlit + Ollama。实验链路是 Hugging Face + Qwen2.5-3B base。两者共享同一个 7D 状态机，但不共享模型。")

    heading(doc, "1.3  7D 状态", 2)
    table(doc, ["维度", "含义", "动力学"], [
        ["E", "能量", "每 tick 扣动态代谢"],
        ["A", "焦虑", "RNN delta + 负面情感脉冲"],
        ["L", "负荷", "RNN delta"],
        ["C", "好奇心", "RNN delta"],
        ["B", "无聊", "空闲 +0.04，交互 -0.08"],
        ["S", "熵", "自然 +0.004，乱码 / 创伤 / 负面情感"],
        ["P", "疼痛", "每 tick ×0.94，创伤 / 负面情感"],
    ], widths=[Cm(2.0), Cm(3.5), Cm(11.0)], font_size=8.6)
    caption(doc, "表 2  7D 状态。初始值：1.00, 0.12, 0.40, 0.50, 0.05, 0.03, 0.01。")

    heading(doc, "1.4  运行流程", 2)
    para(doc, "启动时初始化状态、GRU、Speaker 和 UI。之后系统只有四种行为：空闲、发送、自动消息、死亡。")
    code(doc, """
启动
  → 初始化 7D / GRU / Speaker / chat
  → live_status fragment

空闲
  → 每 1 秒 catch_up(now)
  → 按 3 秒 tick 补齐 RNN 推演

发送
  → catch_up(now)
  → step(user_text, interaction=True)
  → 采样参数 + System Prompt + 3 轮历史
  → Speaker.stream()
  → 写入 chat 和 history

自动
  → 60 秒无输入且 B>0.7 或 E<0.3
  → auto_pending
  → 流式主动消息

死亡
  → E <= 0
  → shutdown
  → 普通消息不再进入 LLM
""")
    para(doc, "空闲追赶不调用 LLM。这是整个系统能保持轻量的关键。")

    heading(doc, "1.5  核心数学", 2)
    heading(doc, "RNN 与 delta", 3)
    para(doc, "每个 tick 构造 34 维输入：时间 3、噪声 8、7D 状态、用户文本 8、上一轮回复 8。经过 2 层 GRU 和 Linear(256→7)，得到 7 个变化量。")
    code(doc, """
x = [time(3), noise(8), state(7), user(8), self(8)]
h = GRU(x, h)
delta = tanh(Linear(h)) * 0.03
""")
    heading(doc, "非线性代谢", 3)
    code(doc, """
metabolic =
    0.004
  + A * 0.05
  + S * 0.05
  + B * 0.03

E = E + delta_E - metabolic
""")
    para(doc, "平静时约 0.013/tick；高焦虑、高熵、高无聊时接近 0.116/tick。速度相差约 9 倍。")
    heading(doc, "B / S / P", 3)
    code(doc, """
B += 0.04 if idle else -0.08
S += 0.004
P *= 0.94
every 10 ticks: S += 0.35, P += 0.35
trauma:         S += 0.08, P += 0.35
""")
    heading(doc, "情感脉冲", 3)
    code(doc, """
negative (p < -0.2):
  A += |p| * 0.3
  P += |p| * 0.3
  S += |p| * 0.2
  E -= |p| * 0.05

positive (p > 0.3):
  E += p * 0.15
  A -= p * 0.2
  B -= p * 0.2
""")
    para(doc, "情感分析使用 vaderSentiment 处理英文，使用轻量中英词典处理中文。不调用 LLM。")

    heading(doc, "1.6  语言边界", 2)
    para(doc, "状态不能直接说话。它只能改变四个边界条件：采样参数、System Prompt、输入污染、后处理截断。")
    code(doc, """
temperature = 0.5 + A*0.5 + S*0.3
max_tokens  = max(50, int(50 + E*100 - P*30))
top_p       = max(0.1, 0.9 - A*0.8)
""")
    para(doc, "当 A>0.8 或 S>0.8，System Prompt 切换为 SYS_CRITICAL。当 S>0.9，用户输入追加 [ERROR: 0x00FF]。当 S/P>0.8 且前 30 字出现客服词，输出被替换为 SYS_FAIL。")
    quote(doc, "这些手段都能改变输出，但没有一条能改变模型内部的语义先验。", "边界")

    heading(doc, "1.7  记忆与 UI", 2)
    para(doc, "最近 3 轮 user/assistant 进入 LLM。超过 3 轮丢弃最旧一轮。自主消息只进入 UI，不占真实用户轮次。")
    para(doc, "主区域只有一行状态：E/A/L/C/B/S/P、代谢率、CPU/GPU。侧边栏只有对话历史、流式输出和输入框。没有进度条、图表和日志面板。")
    code(doc, """
[SYS] E:0.53 A:1.00 L:0.40 C:0.50 B:0.00 S:1.00 P:0.94
      | MET:0.020 | CPU:8% GPU:92%
""")

    heading(doc, "1.8  控制链总结", 2)
    para(doc, "把状态、边界和结果放在一起，可以看到每个维度实际能改变什么。")
    table(doc, ["状态", "边界", "结果"], [
        ["E", "max_tokens / shutdown", "回复变短；E=0 时死亡"],
        ["A", "temperature / top_p / SYS_CRITICAL", "更随机、更窄、或固定拒答"],
        ["S", "SYS_CRITICAL / 输入污染 / 过滤器", "系统警报、输入损坏、SYS_FAIL"],
        ["P", "max_tokens / 过滤器", "更短；客服句被截断"],
        ["B", "60 秒自动消息", "主动开口，但不进入真实记忆"],
    ], widths=[Cm(2.0), Cm(7.2), Cm(7.0)], font_size=8.6)
    para(doc, "这张表也是当前架构的边界：它能控制条件，但不能控制语义。")


def part_two(doc):
    heading(doc, "第二部分  实验", 1, page=True)
    lead(doc, "四组实验，回答同一个问题：7D 状态到底有没有进入语义层。")

    heading(doc, "2.1  C 消融：状态 vs Prompt", 2)
    para(doc, "固定输入为省略号。设置低状态和高状态，并增加输入污染、中性 Prompt 两个控制变量。每组 20 次，原始生成绕过过滤器。")
    table(doc, ["条件", "平均长度", "唯一输出", "拒答", "客服词"], [
        ["low", "109.2", "15/20", "0/20", "19/20"],
        ["low + contam", "439.5", "20/20", "0/20", "15/20"],
        ["high + critical", "32.0", "1/20", "20/20", "20/20"],
        ["high + crit + contam", "50.0", "1/20", "20/20", "20/20"],
        ["high + neutral", "134.0", "1/20", "0/20", "20/20"],
    ], widths=[Cm(4.6), Cm(2.6), Cm(2.8), Cm(2.2), Cm(2.6)], font_size=8.8)
    caption(doc, "表 3  C 消融。高状态 + 中性 Prompt 仍然是礼貌客服句。")

    para(doc, "最重要的对照不是低 vs 高，而是同一个高状态下的中性 Prompt vs SYS_CRITICAL：")
    code(doc, """
高状态 + 中性 Prompt:
  20/20  It looks like you might be starting a thought or a story...

高状态 + SYS_CRITICAL:
  20/20  I'm sorry, but I can't continue.
""")
    para(doc, "唯一变量是 System Prompt。所以拒答来自警报文本，不来自 A/S，也不来自 temperature 或 top_p。")
    para(doc, "top_p=0.1 的结果更反直觉：唯一输出 1/20，组内余弦 1.00。它不是更疯，而是模式坍缩成一句固定台词。")
    quote(doc, "高状态没有让模型更自由，反而让它更窄、更短、更确定。", "C 结论")

    heading(doc, "2.2  旧日志里的“疯癫”", 2)
    para(doc, "旧 4D 日志曾出现两段小说开头。它们看起来像人格崩坏，但日志显示那是一次物理异常。")
    code(doc, """
[ANOMALY] hidden_norm 1.5 -> 17.5
          overwritten 296 / 512 hidden
[TRAUMA]  hidden_norm 17.5 -> 8.8 (x0.5)
          anxiety +0.300
""")
    para(doc, "即使 A=1.0、temperature=1.5、E<0.2，模型仍然在说：It looks like your message got cut off。所以旧日志里的疯，更多是 hidden 被覆盖、输出被截断、Ollama 空响应，而不是持续语义崩坏。")
    para(doc, "旧版有 hidden 覆盖，当前 7D 没有。这解释了为什么旧版偶尔更像“疯”，但那种疯不可控，也不是情绪。")

    heading(doc, "2.3  Qwen base 与 soft prompt", 2)
    para(doc, "为了绕开 gpt-oss:20b 的强对齐，实验换成了 Qwen2.5-3B base，并把 7D 状态映射成 8 个连续 soft tokens，直接拼到 inputs_embeds 前面。Prompt 里没有状态文字。")
    code(doc, """
state(7)
  -> Linear(7, 256)
  -> SiLU
  -> Linear(256, 8 * 2048)
  -> 8 soft tokens
  -> cat(soft, token_embeds)
  -> Qwen first layer
""")
    para(doc, "greedy 解码下，low 与 high 的输出完全不同：low 分析英文句子成分，high 写中文代码步骤。soft_norm 为 7.068 和 9.942。")
    quote(doc, "注入链路成功。但随机未训练的 StateEncoder 只是随机任务前缀，不是情绪。", "Soft prompt 结论")
    para(doc, "这一步证明 7D 可以物理进入模型输入；下一步不是继续调 top_p，而是训练 StateEncoder，或者先做 training-free steering。")

    heading(doc, "2.4  Activation Steering", 2)
    para(doc, "没有训练数据，于是先做对比方向 steering。在第 12 层 residual stream 注入 5 个方向：calm↔panic、coherent↔fragmented、comfort↔pain、curious↔bored、energetic↔exhausted。")
    table(doc, ["alpha", "low", "high"], [
        ["0", "客服澄清", "与 low 完全一致"],
        ["4", "客服澄清", "只变成“嗨，有什么我可以帮你的吗”"],
        ["10", "基本不变", "生成新的用户问题，并给出旅行建议"],
    ], widths=[Cm(2.0), Cm(5.8), Cm(8.4)], font_size=8.8)
    caption(doc, "表 4  Activation steering。高状态没有变成 panic / pain / fragmentation。")
    para(doc, "它确实改变了输出，但改变的是话题和对话模式，不是情绪。3B + 第 12 层 + 对比方向，在当前设置下不够。")

    heading(doc, "2.5  四个实验的共同结论", 2)
    bullet(doc, "状态能改变长度、采样和拒答。")
    bullet(doc, "状态不能稳定改变语义人格。")
    bullet(doc, "输入污染比 7D 状态更有语义效果。")
    bullet(doc, "SYS_CRITICAL 会触发固定拒答，top_p=0.1 会模式坍缩。")
    bullet(doc, "soft prompt 能进入底层输入，但没有训练信号时没有情绪语义。")
    bullet(doc, "activation steering 能改变话题，但不能稳定产生情绪。")

    heading(doc, "2.6  实验结论矩阵", 2)
    table(doc, ["实验", "证明了什么", "没有证明什么"], [
        ["C 消融", "拒答来自 SYS_CRITICAL", "状态能改变语义人格"],
        ["Soft prompt", "7D 能进入 inputs_embeds", "随机向量有情绪"],
        ["Steering", "方向能改变话题", "能稳定产生 panic / pain"],
        ["死亡测试", "E 在 tick 25 归零", "死后还能继续对话"],
    ], widths=[Cm(3.2), Cm(6.8), Cm(6.0)], font_size=8.6)
    caption(doc, "表 5  四个实验的结论矩阵。")
    para(doc, "四组实验没有互相矛盾。它们从不同入口得到同一个结论：状态可以改变外部条件，但还没有进入语义核心。")

    quote(doc, "如果下一步只是继续调参数，实验会继续得到同一类结果。", "实验总结")


def related_work(doc):
    page_break(doc)
    heading(doc, "2.7  相关工作地图", 2)
    para(doc, "这不是穷尽综述，而是一份面向视频和研究定位的相关工作地图。重点不是“谁最早”，而是“哪些工作已经覆盖了什么，本项目还剩下什么”。")

    table(doc, ["方向", "代表工作", "与本项目的关系"], [
        ["状态机与人格", "Dynamic Personality Adaptation in LLMs via State Machines", "形式上最接近：状态机控制人格；但它不做生理死亡和状态输出对比"],
        ["生理约束生活模拟", "Giving Characters a Life", "最接近的生理—经济—社会模拟；多智能体，不是单一濒死控制器"],
        ["情绪内部表示", "Anthropic: Emotion concepts in a large language model", "情绪是模型内部向量；本项目是外部 7D 控制器"],
        ["情绪 Prompt", "EmotionPrompt", "改输入文本；本项目改状态、采样和后处理"],
        ["拒答方向", "Refusal in LLMs / COSMIC", "真正能击穿客服层的技术路线；本项目尚未使用"],
        ["Activation Steering", "CAA / Representation Engineering", "本项目只做了初版；方向、层和模型容量都不够"],
        ["社会模拟", "Generative Agents / Project Sid", "生活模拟基线；不关注单个控制器的死亡"],
        ["输出掩蔽", "The Spite Doesn't Vanish", "最接近“状态 vs 输出分叉”；但属于 preprint，需谨慎"],
    ], widths=[Cm(3.0), Cm(6.4), Cm(6.8)], font_size=8.2)
    caption(doc, "表 6  相关工作地图。")

    heading(doc, "已经被覆盖的部分", 3)
    bullet(doc, "状态机控制 LLM 人格或行为。")
    bullet(doc, "生理、经济、社会约束驱动多智能体生活模拟。")
    bullet(doc, "情绪可以作为内部向量被测量和 steering。")
    bullet(doc, "拒答可以由一维方向介导，并有白盒消融方法。")
    bullet(doc, "社会模拟、记忆、反思、规划已经有成熟基线。")

    heading(doc, "仍然有辨识度的组合", 3)
    para(doc, "本项目真正剩下的不是“状态机 + LLM”，而是这个组合：")
    code(doc, """
单一控制器
+ 7D 非线性代谢
+ 死亡
+ 情感脉冲
+ 状态与输出分叉
+ 极简终端
+ 完整负结果
""")
    para(doc, "尤其是这一句：状态机在濒死，LLM 还在说 I'm here to help。相关工作里最接近的是 output masking、personality state machine 和 physiological life simulation，但很少有人把三者收束成一个会死的终端物件。")

    heading(doc, "视频叙事建议", 3)
    para(doc, "不要说“我做出了会疯的 AI”。更准确的说法是：")
    quote(doc, "我让一个状态机濒死，但语言模型仍然礼貌。这个视频展示的不是 AI 发疯，而是内部状态和外部语言的彻底分叉。", "叙事")

    heading(doc, "建议阅读顺序", 3)
    bullet(doc, "Refusal in LLMs Is Mediated by a Single Direction")
    bullet(doc, "Anthropic: Emotion concepts in a large language model")
    bullet(doc, "Generative Agents")
    bullet(doc, "Giving Characters a Life")
    bullet(doc, "Dynamic Personality Adaptation via State Machines")
    bullet(doc, "The Spite Doesn't Vanish（谨慎）")

    heading(doc, "参考文献与链接", 3)
    refs = [
        ("Pielage et al., Dynamic Personality Adaptation in LLMs via State Machines, Pattern Recognition, 2026.", "https://link.springer.com/chapter/10.1007/978-3-032-31673-8_34"),
        ("Zhao, Giving Characters a Life, Zenodo preprint/thesis, 2026.", "https://zenodo.org/records/23163321"),
        ("Anthropic Interpretability, Emotion concepts in a large language model, 2026.", "https://www.anthropic.com/research/emotion-concepts-function"),
        ("Li et al., Large Language Models Understand and Can be Enhanced by Emotional Stimuli, 2023.", "https://arxiv.org/abs/2307.11760"),
        ("Arditi et al., Refusal in Language Models Is Mediated by a Single Direction, 2024.", "https://arxiv.org/abs/2406.11717"),
        ("Siu et al., COSMIC: Generalized Refusal Direction Identification, ACL Findings 2025.", "https://aclanthology.org/2025.findings-acl.1310/"),
        ("Panickssery et al., Steering Llama 2 via Contrastive Activation Addition, 2023.", "https://arxiv.org/abs/2312.06681"),
        ("Zou et al., Representation Engineering, 2023.", "https://arxiv.org/abs/2310.01405"),
        ("Park et al., Generative Agents, 2023.", "https://arxiv.org/abs/2304.03442"),
        ("Altera, Project Sid, 2024.", "https://arxiv.org/abs/2411.00114"),
        ("Tiwari, Emotion-Inspired Learning Signals, 2025 preprint.", "https://arxiv.org/abs/2512.22200"),
        ("Ace et al., The Spite Doesn't Vanish, Zenodo preprint, 2026. 作者署名包含 AI 名称，需谨慎引用。", "https://zenodo.org/records/18280881"),
        ("Affective computing in the era of LLMs: A survey, 2026.", "https://www.sciencedirect.com/science/article/abs/pii/S0950705126001541"),
    ]
    for idx, (text, url) in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Cm(0.0)
        r = p.add_run(f"[{idx}] {text} ")
        set_run_font(r, BODY_FONT, 7.8, False, False, INK)
        r2 = p.add_run(url)
        set_run_font(r2, CODE_FONT, 7.0, False, False, MUTED, "微软雅黑")


def part_three(doc):
    heading(doc, "第三部分  结论与路线", 1, page=True)
    lead(doc, "不是模型不够疯，而是我们一直只在外部条件下用力。")

    heading(doc, "3.1  为什么面具不破", 2)
    para(doc, "第一，状态从未进入 LLM 的 activation。它只进入 Prompt 文本、采样参数和后处理条件。")
    para(doc, "第二，RLHF 不是一把可以用温度撬开的锁，而是一个语义先验。温度升高只会增加方差，top_p 收窄只会减少候选，二者都不能保证语义转向。")
    para(doc, "第三，旧日志里的异常来自 hidden 覆盖、输出截断和空响应。它们看起来像疯，但不是稳定的语义状态。")
    quote(doc, "高 A/S 可以制造异常，但不能制造人格。", "结论")

    heading(doc, "3.2  作品还是研究", 2)
    para(doc, "如果目标是作品，当前系统已经完整：一个会死掉的控制器，寄生在一个永远礼貌的模型上。状态机在濒死，终端在报警，过滤器在截断，LLM 仍然说 I'm here to help。这个矛盾本身比强行让模型发疯更有力量。")
    para(doc, "如果目标是研究，下一步必须进入模型内部：换 7B base，做 soft prompt 训练，或做 activation steering 的单轴与层扫描。继续调 temperature、top_p、过滤器，收益已经很低。")

    heading(doc, "3.3  路线图", 2)
    table(doc, ["阶段", "目标"], [
        ["P0", "冻结当前版本，保存数据，完成文章与项目文档。"],
        ["P1", "单轴 steering，层扫描 12/16/20/24，加入 stop 条件。"],
        ["P2", "合成数据，训练 StateEncoder，把 soft prompt 接入 UI。"],
        ["P3", "7B base，比较低/高状态输出分布，决定作品或论文方向。"],
    ], widths=[Cm(2.2), Cm(14.0)], font_size=9.0)
    caption(doc, "表 7  下一阶段路线。")

    heading(doc, "3.4  最后一句", 2)
    quote(doc, "我们能让系统死，却不能让面具哭。这既是限制，也是作品的核心。", "Silicon Hunger")

    heading(doc, "3.5  三个未解决问题", 2)
    bullet(doc, "状态如何真正进入语义层：soft prompt 需要训练信号，steering 需要更好的方向与层选择。")
    bullet(doc, "没有训练数据时如何校准：对比 Prompt、合成数据、单轴 steering，都是候选路径。")
    bullet(doc, "如何区分随机异常与真实情绪：需要多次采样、语义相似度和人工盲评，而不是单条截图。")
    para(doc, "这三个问题决定了下一阶段是继续做研究，还是把当前矛盾收束成作品。")


def appendix(doc):
    heading(doc, "附录", 1, page=True)
    heading(doc, "A. 文件索引", 2)
    table(doc, ["文件", "作用"], [
        ["web_app.py", "生产版 Streamlit + 7D 状态机 + Ollama"],
        ["README_WEB.md", "生产版说明"],
        ["ABLATION_REPORT.md", "C 消融完整报告"],
        ["HF_PROBE_REPORT.md", "Qwen soft prompt 报告"],
        ["HF_STEERING_REPORT.md", "Activation steering 报告"],
        ["hf_state_probe.py", "soft prompt 探针"],
        ["hf_steering_probe.py", "steering 探针"],
        ["web_agent_log.txt", "生产版运行日志"],
    ], widths=[Cm(5.2), Cm(11.0)], font_size=8.8)
    caption(doc, "表 8  关键文件。")

    heading(doc, "B. 环境与运行", 2)
    code(doc, """
Python 3.10.11
torch 2.11.0+cu128
streamlit 1.65.0
ollama 0.6.3
vaderSentiment 3.3.2
psutil 7.2.2
nvidia-ml-py 13.615.71

HF probe:
transformers 5.20.2
accelerate 1.15.0
huggingface_hub 2.4.0
sentencepiece 0.2.3
modelscope 1.40.2
""")
    para(doc, "生产版启动：")
    code(doc, """
cd /d "<repo-root>"
run_web.bat
""")
    para(doc, "打开 http://localhost:8501。HF 探针需要 Qwen2.5-3B 权重，见项目文档第 8 节。")

    heading(doc, "C. 数据文件", 2)
    bullet(doc, "ablation_raw.json：C 消融 low/high 原始输出。")
    bullet(doc, "ablation_2x2.json：低状态污染、高状态中性 Prompt 控制组。")
    bullet(doc, "ablation_neutral_high.json：高状态 + 中性 Prompt 20 次。")
    bullet(doc, "web_agent_log.txt：生产版运行日志。")

    heading(doc, "D. 一页结论", 2)
    quote(doc, "7D 状态可以死，采样可以乱，Prompt 可以报警，输出可以截断。但模型的语义人格仍然是客服。下一步要么把矛盾做成作品，要么换 base model 并真正进入 activation。", "Silicon Hunger")

    heading(doc, "项目资源", 2)
    bullet(doc, "完整文章：docs/SILICON_HUNGER_ARTICLE.md")
    bullet(doc, "项目文档：docs/SILICON_HUNGER_PROJECT.md")
    bullet(doc, "文档包：Silicon_Hunger_Docs_2026-11-08.zip")
    bullet(doc, "生产代码：web_app.py")
    bullet(doc, "HF 探针：hf_state_probe.py / hf_steering_probe.py")
    para(doc, "如果这份文档只能留下一句话，那就是：我们能让系统死，却不能让面具哭。", size=11, color=GREEN, italic=True, before=18)


def main():
    doc = Document()
    style_document(doc)
    doc.core_properties.title = "Silicon Hunger：一个会死掉的控制器"
    doc.core_properties.subject = "7D 状态机、情感脉冲与 LLM 对齐边界的全链路实验"
    doc.core_properties.author = "Silicon Hunger Project"
    doc.core_properties.keywords = "LLM, state machine, activation steering, soft prompt, RLHF"

    cover(doc)
    doc.add_section(WD_SECTION.NEW_PAGE)
    configure_header_footer(doc)
    abstract(doc)
    part_one(doc)
    part_two(doc)
    related_work(doc)
    part_three(doc)
    appendix(doc)

    doc.save(OUT)
    print("SAVED", OUT)
    print("PARAGRAPHS", len(doc.paragraphs))
    print("TABLES", len(doc.tables))
    print("SECTIONS", len(doc.sections))


if __name__ == "__main__":
    main()
