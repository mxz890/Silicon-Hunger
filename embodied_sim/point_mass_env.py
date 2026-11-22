#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal MuJoCo point-mass reaching task.

Observation: [x, y, vx, vy, tx, ty]
Action:      [fx, fy] in [-1, 1]
Reward:      -distance to target
Success:     distance < 0.12
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco

XML = """
<mujoco model="point_mass">
  <option timestep="0.01" gravity="0 0 0" integrator="RK4"/>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.12 0.16 0.13 1"/>
    <body name="agent" pos="0 0 0.3">
      <joint name="jx" type="slide" axis="1 0 0" damping="1.2"/>
      <joint name="jy" type="slide" axis="0 1 0" damping="1.2"/>
      <geom name="agent_geom" type="sphere" size="0.10" rgba="0.18 0.80 0.45 1"/>
    </body>
    <body name="target" pos="1 1 0.3">
      <geom name="target_geom" type="sphere" size="0.08" rgba="0.95 0.35 0.30 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="jx" ctrlrange="-1 1" gear="10"/>
    <motor joint="jy" ctrlrange="-1 1" gear="10"/>
  </actuator>
</mujoco>
"""


@dataclass
class StepResult:
    obs: np.ndarray
    reward: float
    done: bool
    success: bool


class PointMassEnv:
    def __init__(self, seed: int = 0):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.jx = self.model.joint("jx").qposadr[0]
        self.jy = self.model.joint("jy").qposadr[0]
        self.target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.steps = 0
        self.reset()

    def reset(self, target=None):
        mujoco.mj_resetData(self.model, self.data)
        xy = self.rng.uniform(-1.0, 1.0, 2)
        if target is None:
            target = self.rng.uniform(-1.0, 1.0, 2)
        self.data.qpos[self.jx:self.jx + 2] = xy
        self.data.qvel[:] = 0.0
        self.model.body_pos[self.target_id, :2] = target
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self.obs()

    def obs(self):
        p = self.data.qpos[self.jx:self.jx + 2].copy()
        v = self.data.qvel[:2].copy()
        t = self.model.body_pos[self.target_id, :2].copy()
        return np.concatenate([p, v, t]).astype(np.float32)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        p = self.data.qpos[self.jx:self.jx + 2]
        t = self.model.body_pos[self.target_id, :2]
        dist = float(np.linalg.norm(p - t))
        success = dist < 0.12
        done = success or self.steps >= 250
        return StepResult(self.obs(), -dist, done, success)


def pd_action(obs, kp=4.0, kd=1.5):
    p, v, t = obs[:2], obs[2:4], obs[4:6]
    return np.clip(kp * (t - p) - kd * v, -1.0, 1.0)
