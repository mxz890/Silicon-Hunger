# Archive — Non-Embodied Work

Status: **archived / frozen**
Not deleted. Do not use as current baseline unless doing a historical comparison.

---

## 1. 7D state machine + LLM

Files:

- web_app.py
- hf_state_probe.py
- hf_steering_probe.py
- ABLATION_REPORT.md
- HF_PROBE_REPORT.md
- HF_STEERING_REPORT.md

Original question:

> Can a 7D state machine make an LLM hungry, anxious, or unstable?

Conclusion:

> It can change length, sampling, refusal, and some activation behavior;
> it cannot establish stable emotional personality.

---

## 2. 10k graph reservoir / math / Chinese

Files:

- research_mvp/scale_10k_real.py
- research_mvp/scale_10k_fresh.py
- research_mvp/scale_10k_richer_chinese.py
- research_mvp/linear_algebra_10k.py
- research_mvp/SCALE_10K_REAL_REPORT.md
- research_mvp/SCALE_10K_FRESH_REPORT.md
- research_mvp/SCALE_10K_RICHER_REPORT.md
- research_mvp/LINEAR_ALGEBRA_10K_REPORT.md

Conclusion:

> Simple algebra works; Chinese next-character is weak; hard linear algebra fails.
> The 10k graph is a fixed reservoir, not a learned language model.

---

## 3. Graph growth / chemistry / dynamic edges

Files:

- research_mvp/grow_train_loop*.py
- research_mvp/dynamic_chem_edges.py
- research_mvp/credit_chem_ablation.py
- research_mvp/forgetting_growth.py
- research_mvp/PHASE_REPORT.md
- research_mvp/GROW_TRAIN_LOOP_V2_REPORT.md
- research_mvp/GROW_V2_MULTISEED_REPORT.md
- research_mvp/GROW_TRAIN_LOOP_V3_REPORT.md
- research_mvp/DYNAMIC_CHEM_EDGES_REPORT.md
- research_mvp/CREDIT_CHEM_ABLATION_REPORT.md
- research_mvp/FORGETTING_GROWTH_REPORT.md

Conclusion:

> Growth and chemistry change structure and training dynamics, but do not
> automatically improve capability. Seed variance is large.

---

## 4. 7D / 10k visualizations

Archived:

- research_mvp/visualizations/fair_10k_dashboard.html
- research_mvp/visualizations/thought_chain.html
- research_mvp/visualizations/temporal_timeline.html

These remain viewable, but are not part of the active embodied loop.

---

## 5. Where old results live

Full index:

```text
research_mvp/RESULTS_INDEX.md
```

10k baseline definition:

```text
research_mvp/10K_BASELINE.md
```

New active focus:

```text
CURRENT_FOCUS.md
embodied_sim/
```
