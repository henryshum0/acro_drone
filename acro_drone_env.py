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
from .sensors.noisy_imu import NoisyImu
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
        masses = self.robot.root_physx_view.get_masses()

        if masses is None:
            self._mass = torch.full((self.num_envs, 1), float(self.cfg.total_mass), device=self.device)
        else:
            self._mass = masses.sum(dim=1, keepdim=True).to(device=self.device, dtype=torch.float32)
        self._hover_accel = float(self.cfg.gravity)

        self._sim_dt = float(self.cfg.sim.dt)
        self._sim_rate_hz = float(self.cfg.sim_rate_hz)
        self._mpc_rate_hz = float(self.cfg.mpc_rate_hz)
        self._camera_rate_hz = float(self.cfg.camera_rate_hz)
        self._imu_rate_hz = float(self.cfg.imu_rate_hz)
        self._time_horizon_s = float(self.cfg.time_horizon_s)

        self._mpc_decimation = int(round(self._sim_rate_hz / self._mpc_rate_hz))
        self._camera_decimation = int(round(self._sim_rate_hz / self._camera_rate_hz))
        self._imu_samples_per_step = max(1, int(round(self._imu_rate_hz / self._sim_rate_hz)))
        self._imu_dt = self._sim_dt / float(self._imu_samples_per_step)

        self._camera_buffer_len = max(1, int(self._time_horizon_s * self._camera_rate_hz))
        self._imu_buffer_len = max(1, int(self._time_horizon_s * self._imu_rate_hz))
        self._mpc_buffer_len = max(1, int(self._time_horizon_s * self._mpc_rate_hz))
        self._mpc_horizon_steps = int(self.cfg.mpc_cfg.horizon) + 1

        self._step_count = 0
        self._mpc_target = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self._traj_end_time = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self._camera_cache = None
        self._camera_buffer = None
        self._imu_buffer = torch.zeros((self.num_envs, self._imu_buffer_len, 6), device=self.device)
        self._mpc_horizon_buffer = torch.zeros(
            (self.num_envs, self._mpc_buffer_len, self._mpc_horizon_steps, 13), device=self.device
        )
        self._mpc_action_buffer = torch.zeros((self.num_envs, self._mpc_buffer_len, 4), device=self.device)

    def _setup_scene(self):
        self.robot = Multirotor(self.cfg.robot_cfg)


        self._fisheye_camera = TiledCamera(self.cfg.fisheye_camera)
        self._noisy_imu = NoisyImu(self.cfg.noisy_imu)
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
        self.scene.sensors["noisy_imu"] = self._noisy_imu
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
            root_lin_vel = self.robot.data.root_lin_vel_w
            root_ang_vel = self.robot.data.root_ang_vel_b
            x0 = torch.cat((root_pos, root_quat, root_lin_vel, root_ang_vel), dim=-1)
            mpc_u = self.mpc.make_step(x0.detach().cpu().numpy(), horizon)
            horizon_tensor = torch.as_tensor(horizon, device=self.device, dtype=torch.float32)
            self._mpc_target = torch.as_tensor(mpc_u, device=self.device, dtype=torch.float32).squeeze(-1)
            self._push_mpc_buffers(horizon_tensor, self._mpc_target)

        target_ctbr = self._mpc_target.clone()
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
        if self._camera_buffer is None:
            self._camera_buffer = self._init_camera_buffer(camera_data)
            self._camera_cache = self._camera_buffer[:, -1]
        if self._step_count % self._camera_decimation == 0:
            self._push_fifo(self._camera_buffer, camera_data)
            self._camera_cache = camera_data.clone()
        else:
            self._camera_cache = self._camera_buffer[:, -1]

        for _ in range(self._imu_samples_per_step):
            self._noisy_imu.update(self._imu_dt, force_recompute=True)
            imu_data = self._noisy_imu.data
            imu_sample = torch.cat((imu_data.lin_acc_b, imu_data.ang_vel_b), dim=-1)
            self._push_fifo(self._imu_buffer, imu_sample)

        observations = {
            "policy": obs,
            "camera": self._camera_cache,
            "camera_seq": self._camera_buffer,
            "imu_seq": self._imu_buffer,
            "mpc_horizon_seq": self._mpc_horizon_buffer,
            "mpc_action_seq": self._mpc_action_buffer,
        }
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
        if self._camera_buffer is not None:
            self._camera_buffer[env_ids_tensor] = 0.0
        if self._imu_buffer is not None:
            self._imu_buffer[env_ids_tensor] = 0.0
        if self._mpc_horizon_buffer is not None:
            self._mpc_horizon_buffer[env_ids_tensor] = 0.0
        if self._mpc_action_buffer is not None:
            self._mpc_action_buffer[env_ids_tensor] = 0.0

    def _init_camera_buffer(self, camera_data: torch.Tensor) -> torch.Tensor:
        buffer = torch.zeros(
            (self.num_envs, self._camera_buffer_len) + camera_data.shape[1:],
            device=self.device,
            dtype=camera_data.dtype,
        )
        buffer[:] = camera_data.unsqueeze(1)
        return buffer

    def _push_fifo(
        self, buffer: torch.Tensor, new_data: torch.Tensor, env_ids: Sequence[int] | torch.Tensor | slice | None = None
    ) -> None:
        if env_ids is None:
            env_ids = slice(None)
        buffer[env_ids, :-1].copy_(buffer[env_ids, 1:].clone())
        buffer[env_ids, -1] = new_data

    def _push_mpc_buffers(self, horizon: torch.Tensor, actions: torch.Tensor) -> None:
        self._push_fifo(self._mpc_horizon_buffer, horizon)
        self._push_fifo(self._mpc_action_buffer, actions)