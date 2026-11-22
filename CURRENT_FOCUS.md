# Current Focus — Embodied Intelligence

Status: **active**

The project focus is now **embodied intelligence in simulation**.

---

## Active work

```text
MuJoCo
→ point mass / robot
→ vision or state
→ graph controller
→ action
→ training / evaluation
```

Directory:

```text
embodied_sim/
```

Current simulator:

- MuJoCo 3.2.7
- custom point-mass environment
- 2 DOF
- optional obstacles
- 64×64 camera
- 500 / 2000 node graph policies

Current results:

| Experiment | Result |
|---|---:|
| PD baseline | 1.00 success |
| 500-node graph BC + DAgger | 1.00 success |
| 2000-node vision BC (no obstacle) | 0.9667 success |
| 2000-node vision BC + PPO (1 obstacle) | 0.0333 success |

Main next problem:

> Make Stage 1 (one fixed obstacle) work before adding more modules.

---

## Archived work

The following are archived, not deleted:

- 7D state machine + LLM
- Qwen soft prompt / activation steering
- 10k graph reservoir
- math / Chinese next-character experiments
- dynamic chemistry / edge gating
- credit chemistry ablation
- graph growth / catastrophe / mutation
- old visualizations

Archive index:

```text
archive/README.md
```

Do not mix old 10k math results into embodied conclusions.

---

## Next milestones

1. Stage 1 single fixed obstacle > 80%
2. 8–16 parallel envs
3. sim-to-sim target domain > 50%
4. 2000-node graph with stable training
5. only then consider real robot
