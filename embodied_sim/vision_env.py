#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MuJoCo point-mass + obstacles with a top-down RGB camera.

The policy receives:
  image: 64x64x3
  state: 12-d vector
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import mujoco

XML = """
<mujoco model="point_mass_vision">
  <option timestep="0.01" gravity="0 0 0" integrator="RK4"/>
  <worldbody>
    <camera name="top" pos="0 0 4" xyaxes="1 0 0 0 1 0"/>
    <geom name="floor" type="plane" size="3 3 0.1" rgba="0.12 0.16 0.13 1"/>
    <body name="agent" pos="0 0 0.3">
      <joint name="jx" type="slide" axis="1 0 0" damping="1.2"/>
      <joint name="jy" type="slide" axis="0 1 0" damping="1.2"/>
      <geom name="agent_geom" type="sphere" size="0.10" rgba="0.18 0.80 0.45 1"/>
    </body>
    <body name="target" pos="1 1 0.3">
      <geom name="target_geom" type="sphere" size="0.08" rgba="0.95 0.35 0.30 1" contype="0" conaffinity="0"/>
    </body>
    <body name="obs1" pos="0.3 0.3 0.3"><geom type="sphere" size="0.14" rgba="0.16 0.38 0.75 1"/></body>
    <body name="obs2" pos="-0.4 0.4 0.3"><geom type="sphere" size="0.14" rgba="0.16 0.38 0.75 1"/></body>
    <body name="obs3" pos="0.4 -0.4 0.3"><geom type="sphere" size="0.14" rgba="0.16 0.38 0.75 1"/></body>
  </worldbody>
  <actuator>
    <motor joint="jx" ctrlrange="-1 1" gear="10"/>
    <motor joint="jy" ctrlrange="-1 1" gear="10"/>
  </actuator>
</mujoco>
"""


@dataclass
class VisionStep:
    image: np.ndarray
    state: np.ndarray
    reward: float
    done: bool
    success: bool
    collision: bool


class VisionPointMassEnv:
    def __init__(
        self,
        seed: int = 0,
        domain: str = "source",
        size: int = 64,
        n_obstacles: int = 3,
        target_range: float = 1.6,
        start_range: float = 1.3,
        obstacle_range: float = 1.1,
        render: bool = True,
        target_bias: tuple[float, float] = (0.0, 0.0),
        obs_noise: float = 0.0,
        fixed_obstacles: tuple | None = None,
    ):
        self.model = mujoco.MjModel.from_xml_string(XML)
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.size = size
        self.n_obstacles = n_obstacles
        self.target_range = target_range
        self.start_range = start_range
        self.obstacle_range = obstacle_range
        self.target_bias = np.asarray(target_bias, dtype=np.float32)
        self.obs_noise = float(obs_noise)
        self.fixed_obstacles = None if fixed_obstacles is None else np.asarray(fixed_obstacles, dtype=np.float32)
        self.render = render
        self.renderer = mujoco.Renderer(self.model, height=size, width=size) if render else None
        self.jx = self.model.joint("jx").qposadr[0]
        self.jy = self.model.joint("jy").qposadr[0]
        self.target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.obs_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obs{i}") for i in (1, 2, 3)]
        self.active_obs = self.obs_ids[:n_obstacles]
        for bid in self.obs_ids[n_obstacles:]:
            gid = self.model.body_geomadr[bid]
            self.model.geom_contype[gid] = 0
            self.model.geom_conaffinity[gid] = 0
            self.model.geom_rgba[gid] = [0.0, 0.0, 0.0, 0.0]
            self.model.body_pos[bid, :2] = [10.0, 10.0]
        self.domain = domain
        if domain == "target":
            for bid in self.active_obs:
                gid = self.model.body_geomadr[bid]
                self.model.geom_rgba[gid] = [0.85, 0.25, 0.20, 1.0]
            self.model.dof_damping[:] = 0.8
        self.steps = 0
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for _ in range(100):
            xy = self.rng.uniform(-self.start_range, self.start_range, 2)
            target = self.rng.uniform(-self.target_range, self.target_range, 2) + self.target_bias
            if self.fixed_obstacles is not None:
                obstacles = self.fixed_obstacles[:len(self.active_obs)].copy()
            else:
                obstacles = self.rng.uniform(-self.obstacle_range, self.obstacle_range, (len(self.active_obs), 2)) if self.active_obs else np.zeros((0, 2))
            if np.linalg.norm(xy - target) < 0.4:
                continue
            if len(obstacles) and min(np.linalg.norm(xy - o) for o in obstacles) < 0.30:
                continue
            if len(obstacles) and min(np.linalg.norm(target - o) for o in obstacles) < 0.25:
                continue
            break
        self.data.qpos[self.jx:self.jx + 2] = xy
        self.data.qvel[:] = 0.0
        self.model.body_pos[self.target_id, :2] = target
        for bid, p in zip(self.active_obs, obstacles):
            self.model.body_pos[bid, :2] = p
        for bid in self.obs_ids[len(self.active_obs):]:
            self.model.body_pos[bid, :2] = [10.0, 10.0]
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self._obs()

    def _obs(self):
        p = self.data.qpos[self.jx:self.jx + 2].copy()
        v = self.data.qvel[:2].copy()
        t = self.model.body_pos[self.target_id, :2].copy()
        rel = []
        for bid in self.obs_ids:
            if bid in self.active_obs:
                rel.append(self.model.body_pos[bid, :2] - p)
            else:
                rel.append(np.zeros(2, dtype=np.float32))
        state = np.concatenate([p, v, t] + rel).astype(np.float32)
        if self.obs_noise > 0:
            state = state + self.rng.normal(0.0, self.obs_noise, state.shape).astype(np.float32)
        if self.renderer is not None:
            self.renderer.update_scene(self.data, camera="top")
            image = self.renderer.render().copy()
        else:
            image = np.zeros((1, 1, 3), dtype=np.uint8)
        return image, state

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        p = self.data.qpos[self.jx:self.jx + 2]
        t = self.model.body_pos[self.target_id, :2]
        dist = float(np.linalg.norm(p - t))
        collision = self.data.ncon > 0
        success = dist < 0.14
        reward = -dist - (0.3 if collision else 0.0) + (2.0 if success else 0.0)
        done = success or self.steps >= 250
        image, state = self._obs()
        return VisionStep(image, state, reward, done, success, collision)
