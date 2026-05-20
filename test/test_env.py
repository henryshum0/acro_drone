"""End-to-end acro drone environment test.

Runs one episode and plots MPC output vs actual state and actual state vs IMU readings.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(headless=False, enable_cameras=True)
simulation_app = app_launcher.app

from isaaclab_contrib.assets.multirotor.multirotor_data import MultirotorData
from isaaclab_tasks.direct.acro_drone.acro_drone_env import AcroDroneEnv
from isaaclab_tasks.direct.acro_drone.acro_drone_env_cfg import AcroDroneEnvCfg


def _plot_xyz(ax, t: np.ndarray, series: np.ndarray, label_prefix: str, linestyle: str = "solid") -> None:
	ax.plot(t, series[:, 0], label=f"{label_prefix} x", linestyle=linestyle)
	ax.plot(t, series[:, 1], label=f"{label_prefix} y", linestyle=linestyle)
	ax.plot(t, series[:, 2], label=f"{label_prefix} z", linestyle=linestyle)
	if series.shape[1] > 3:
		ax.plot(t, series[:, 3], label=f"{label_prefix} w", linestyle=linestyle)


def main() -> None:


	cfg = AcroDroneEnvCfg()
	cfg.scene.num_envs = 1

	env = AcroDroneEnv(cfg, render_mode=None)
	obs, _ = env.reset()

	dt = float(env.step_dt)
	prev_lin_vel = env.robot.data.root_lin_vel_b[0].clone()

	times: list[float] = []
	mpc_thrusts: list[float] = []
	actual_thrusts: list[float] = []
	mpc_body_rates: list[np.ndarray] = []
	actual_body_rates: list[np.ndarray] = []
	actual_lin_accs: list[np.ndarray] = []
	imu_lin_accs: list[np.ndarray] = []
	imu_ang_vels: list[np.ndarray] = []
	actual_positions: list[np.ndarray] = []
	ref_positions: list[np.ndarray] = []
	actual_velocities: list[np.ndarray] = []
	ref_velocities: list[np.ndarray] = []
	actual_quaternions: list[np.ndarray] = []
	ref_quaternions: list[np.ndarray] = []

	mass = float(env._mass[0, 0])
	step_idx = 0

	print("trj time:", env.trj_interface.get_trj_time(env_ids=[0]))
	while True:
		actions = torch.zeros((env.num_envs, cfg.action_space), device=env.device)
		obs, _, _, time_outs, _ = env.step(actions)

		mpc_action = obs["mpc_action_seq"][0, -1].detach().cpu().numpy()
		mpc_thrusts.append(mpc_action[0] * mass)
		mpc_body_rates.append(mpc_action[1:4].copy())

		multirotor_data = cast(MultirotorData, env.robot.data)
		actual_thrusts.append(multirotor_data.applied_thrust.sum(dim=1)[0].item())
		actual_body_rates.append(env.robot.data.root_ang_vel_b[0].detach().cpu().numpy())
		actual_positions.append((env.robot.data.root_pos_w - env.scene.env_origins)[0].detach().cpu().numpy())
		actual_velocities.append(env.robot.data.root_lin_vel_b[0].detach().cpu().numpy())
		actual_quaternions.append(env.robot.data.root_quat_w[0].detach().cpu().numpy())
		t_now = step_idx * dt
		ref_state = env.trj_interface.get_state_at_time(env_ids=[0], t=float(t_now))[0]
		ref_positions.append(ref_state[:3].copy())
		ref_velocities.append(ref_state[7:10].copy())
		ref_quaternions.append(ref_state[3:7].copy())

		lin_vel = env.robot.data.root_lin_vel_b[0]
		lin_acc = ((lin_vel - prev_lin_vel) / dt).detach().cpu().numpy()
		prev_lin_vel = lin_vel.clone()
		actual_lin_accs.append(lin_acc)

		imu_sample = obs["imu_seq"][0, -1].detach().cpu().numpy()
		imu_lin_accs.append(imu_sample[:3].copy())
		imu_ang_vels.append(imu_sample[3:].copy())

		times.append(step_idx * dt)
		step_idx += 1

		if bool(time_outs[0].item()):
			break

	t = np.asarray(times, dtype=float)
	mpc_body_rates_arr = np.asarray(mpc_body_rates, dtype=float)
	actual_body_rates_arr = np.asarray(actual_body_rates, dtype=float)
	actual_lin_accs_arr = np.asarray(actual_lin_accs, dtype=float)
	imu_lin_accs_arr = np.asarray(imu_lin_accs, dtype=float)
	imu_ang_vels_arr = np.asarray(imu_ang_vels, dtype=float)
	actual_positions_arr = np.asarray(actual_positions, dtype=float)
	ref_positions_arr = np.asarray(ref_positions, dtype=float)
	actual_velocities_arr = np.asarray(actual_velocities, dtype=float)
	ref_velocities_arr = np.asarray(ref_velocities, dtype=float)
	actual_quaternions_arr = np.asarray(actual_quaternions, dtype=float)
	ref_quaternions_arr = np.asarray(ref_quaternions, dtype=float)

	fig_mpc, ax_mpc = plt.subplots(4, sharex=True, figsize=(14, 9))
	fig_mpc.suptitle("MPC Output vs Actual State")

	ax_mpc[0].plot(t, mpc_thrusts, label="MPC thrust target", linestyle="dashed")
	ax_mpc[0].plot(t, actual_thrusts, label="Actual collective thrust")
	ax_mpc[0].set_ylabel("Thrust (N)")
	ax_mpc[0].grid(True)
	ax_mpc[0].legend()

	_plot_xyz(ax_mpc[1], t, mpc_body_rates_arr, "MPC body rate", linestyle="dashed")
	_plot_xyz(ax_mpc[1], t, actual_body_rates_arr, "Actual body rate")
	ax_mpc[1].set_ylabel("Body rates (rad/s)")
	ax_mpc[1].grid(True)
	ax_mpc[1].legend()

	ax_mpc[2].plot(t, mpc_body_rates_arr[:, 0] - actual_body_rates_arr[:, 0], label="Roll rate error")
	ax_mpc[2].plot(t, mpc_body_rates_arr[:, 1] - actual_body_rates_arr[:, 1], label="Pitch rate error")

	ax_mpc[2].plot(t, mpc_body_rates_arr[:, 2] - actual_body_rates_arr[:, 2], label="Yaw rate error")
	ax_mpc[2].set_ylabel("Rate error (rad/s)")
	ax_mpc[2].grid(True)
	ax_mpc[2].legend()

	ax_mpc[3].plot(t, np.asarray(mpc_thrusts) - np.asarray(actual_thrusts), label="Thrust error")
	ax_mpc[3].set_ylabel("Thrust error (N)")
	ax_mpc[3].set_xlabel("Time (s)")
	ax_mpc[3].grid(True)
	ax_mpc[3].legend()

	fig_imu, ax_imu = plt.subplots(2, sharex=True, figsize=(14, 7))
	fig_imu.suptitle("Actual State vs IMU")

	_plot_xyz(ax_imu[0], t, actual_lin_accs_arr, "Actual lin acc")
	_plot_xyz(ax_imu[0], t, imu_lin_accs_arr, "IMU lin acc")
	ax_imu[0].set_ylabel("Linear acc (m/s^2)")
	ax_imu[0].grid(True)
	ax_imu[0].legend()

	_plot_xyz(ax_imu[1], t, actual_body_rates_arr, "Actual ang vel")
	_plot_xyz(ax_imu[1], t, imu_ang_vels_arr, "IMU ang vel")
	ax_imu[1].set_ylabel("Angular vel (rad/s)")
	ax_imu[1].set_xlabel("Time (s)")
	ax_imu[1].grid(True)
	ax_imu[1].legend()

	output_dir = Path(__file__).resolve().parent
	fig_mpc.savefig(output_dir / "test_env_mpc_vs_actual.png", dpi=150)
	fig_imu.savefig(output_dir / "test_env_actual_vs_imu.png", dpi=150)

	fig_pos_vel, ax_pos_vel = plt.subplots(3, sharex=True, figsize=(14, 11))
	fig_pos_vel.suptitle("Reference vs Ground Truth")

	_plot_xyz(ax_pos_vel[0], t, ref_positions_arr, "Ref position", linestyle="dashed")
	_plot_xyz(ax_pos_vel[0], t, actual_positions_arr, "Actual position")
	ax_pos_vel[0].set_ylabel("Position (m)")
	ax_pos_vel[0].grid(True)
	ax_pos_vel[0].legend()

	_plot_xyz(ax_pos_vel[1], t, ref_velocities_arr, "Ref velocity", linestyle="dashed")
	_plot_xyz(ax_pos_vel[1], t, actual_velocities_arr, "Actual velocity")
	ax_pos_vel[1].set_ylabel("Velocity (m/s)")
	ax_pos_vel[1].grid(True)
	ax_pos_vel[1].legend()

	print("ref quaternions:", ref_quaternions_arr[:5])
	_plot_xyz(ax_pos_vel[2], t, ref_quaternions_arr, "Ref quat", linestyle="dashed")
	_plot_xyz(ax_pos_vel[2], t, actual_quaternions_arr, "Actual quat")
	ax_pos_vel[2].set_ylabel("Quaternion (xyzw)")
	ax_pos_vel[2].set_xlabel("Time (s)")
	ax_pos_vel[2].grid(True)
	ax_pos_vel[2].legend()

	fig_pos_vel.savefig(output_dir / "test_env_ref_vs_actual_pos_vel.png", dpi=150)

	if not args_cli.headless:
		plt.show()

	simulation_app.close()


if __name__ == "__main__":
	main()


