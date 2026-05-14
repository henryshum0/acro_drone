# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import math as math_utils

from isaaclab.sensors import TiledCamera
from isaaclab_contrib.assets import Multirotor
from isaaclab_contrib.assets.multirotor.multirotor_data import MultirotorData
from isaaclab_tasks.direct.acro_drone.trajectory.acro_templates import PowerloopTemplate


from .controllers.ctbr_pd import CtbrPdController
from .acro_drone_env_cfg import AcroDroneEnvCfg
from .trajectory.trj_interface import TrajectoryInterface
from .controllers.mpc import MPCControllerWrapper

class AcroDroneEnv(DirectRLEnv):
    cfg: AcroDroneEnvCfg

    def __init__(self, cfg: AcroDroneEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._ctbr_controller = CtbrPdController(
            self.cfg.controller_cfg, num_envs=self.num_envs, device=self.device
        )
        self.trj_interface = TrajectoryInterface(num_envs=self.num_envs, cfg=self.cfg.trj_interface_cfg)
        self.mpc = MPCControllerWrapper(self.cfg.mpc_cfg, num_envs=self.num_envs)

        self._action_scale = torch.tensor(self.cfg.action_scale, device=self.device, dtype=torch.float32)
        masses = self.robot.data.default_mass
        if masses is None:
            self._mass = torch.full((self.num_envs, 1), float(self.cfg.total_mass), device=self.device)
        else:
            self._mass = masses.sum(dim=1, keepdim=True)
        self._hover_accel = float(self.cfg.gravity)

        self._sim_dt = float(self.cfg.sim.dt)
        self._sim_rate_hz = float(self.cfg.sim_rate_hz)
        self._mpc_rate_hz = float(self.cfg.mpc_rate_hz)
        self._camera_rate_hz = float(self.cfg.camera_rate_hz)

        self._mpc_decimation = int(round(self._sim_rate_hz / self._mpc_rate_hz))
        self._camera_decimation = int(round(self._sim_rate_hz / self._camera_rate_hz))

        self._step_count = 0
        self._mpc_target = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self._traj_end_time = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self._camera_cache = None

    def _setup_scene(self):
        self.robot = Multirotor(self.cfg.robot_cfg)


        self._fisheye_camera = TiledCamera(self.cfg.fisheye_camera)
        # add ground plane
        # spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=True)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        # add articulation and sensors to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["fisheye_camera"] = self._fisheye_camera
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._step_count += 1
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        if self._step_count % self._mpc_decimation == 0:
            env_ids = list(range(self.num_envs))
            t_now = (self.episode_length_buf * self._sim_dt).detach().cpu().numpy()
            t_end = t_now + self.cfg.mpc_cfg.horizon * self.cfg.mpc_cfg.horizon_dt
            horizon = self.trj_interface.get_trajectory_window(
                env_ids=env_ids,
                t_start=t_now,
                t_end=t_end,
                time_step=self.cfg.mpc_cfg.horizon_dt,
            )

            root_pos = self.robot.data.root_pos_w - self.scene.env_origins
            root_quat = math_utils.convert_quat(self.robot.data.root_quat_w, to="xyzw")
            root_lin_vel = self.robot.data.root_lin_vel_b
            root_ang_vel = self.robot.data.root_ang_vel_b
            x0 = torch.cat((root_pos, root_quat, root_lin_vel, root_ang_vel), dim=-1)
            mpc_u = self.mpc.make_step(x0.detach().cpu().numpy(), horizon)
            self._mpc_target = torch.as_tensor(mpc_u, device=self.device, dtype=torch.float32).squeeze(-1)

        target_ctbr = self._mpc_target
        target_ctbr[:, 0:1] = target_ctbr[:, 0:1] * self._mass

        multirotor_data = cast(MultirotorData, self.robot.data)
        collective_thrust = multirotor_data.applied_thrust.sum(dim=1, keepdim=True)
        body_rates = self.robot.data.root_ang_vel_b
        current_ctbr = torch.cat((collective_thrust, body_rates), dim=1)
        motor_thrusts = self._ctbr_controller.compute(target_ctbr, current_ctbr)
        
        self.robot.set_thrust_target(motor_thrusts)

    def _get_observations(self) -> dict:
        lin_vel = self.robot.data.root_lin_vel_b
        ang_vel = self.robot.data.root_ang_vel_b
        obs = torch.cat((lin_vel, ang_vel), dim=-1)

        camera_data = self._fisheye_camera.data.output["rgb"] / 255.0
        if self._camera_cache is None:
            self._camera_cache = camera_data.clone()
        if self._step_count % self._camera_decimation == 0:
            self._camera_cache = camera_data.clone()

        observations = {"policy": obs, "camera": self._camera_cache}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        return torch.ones_like(self.reset_terminated, dtype=torch.float32)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        t_now = self.episode_length_buf * self._sim_dt
        time_out = t_now >= self._traj_end_time
        out_of_bounds = torch.zeros_like(time_out, dtype=torch.bool)
        return out_of_bounds, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids_seq = list(range(self.num_envs))
            env_ids_tensor = torch.arange(self.num_envs, device=self.device)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
            env_ids_seq = env_ids_tensor.tolist()
        else:
            env_ids_seq = list(env_ids)
            env_ids_tensor = torch.tensor(env_ids_seq, device=self.device, dtype=torch.long)

        super()._reset_idx(env_ids_seq)

        default_root_state = self.robot.data.default_root_state[env_ids_tensor]
        default_root_state[:, :3] += self.scene.env_origins[env_ids_tensor]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], cast(Sequence[int], env_ids_tensor))
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], cast(Sequence[int], env_ids_tensor))

        self.trj_interface.reset_idx(env_ids_seq=env_ids_seq, templates=self.cfg.templates)
        self._ctbr_controller.reset_idx(env_ids_seq)
        x0 = self.trj_interface.get_state_at_time(env_ids=env_ids_seq, t=0.0)
        horizon_0 = self.trj_interface.get_trajectory_window(
            env_ids=env_ids_seq,
            t_start=0.0,
            t_end=self.cfg.mpc_cfg.horizon * self.cfg.mpc_cfg.horizon_dt,
            time_step=self.cfg.mpc_cfg.horizon_dt,
        )
        self.mpc.reset_idx(env_ids=env_ids_seq, x0=x0, horizon=horizon_0)
        for i, env_id in enumerate(env_ids_seq):
            traj_time = self.trj_interface.get_trj_time(env_ids=[env_id])
            self._traj_end_time[env_ids_tensor[i]] = traj_time
        if len(env_ids_seq) == self.num_envs:
            self._step_count = 0
        if self._camera_cache is not None:
            self._camera_cache[env_ids_tensor] = 0.0