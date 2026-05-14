"""PD tuning script for the acro drone CTBR controller.

Generates random collective thrust/body-rate targets every N env steps and records
the commanded targets along with the actual body rates and collective thrust.
"""

from __future__ import annotations

import argparse
from typing import cast

import numpy as np
import torch
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher

def _sample_ctbr(num_envs: int, thrust_range: tuple[float, float], rate_range: float, device: str) -> torch.Tensor:
	"""Sample [collective thrust, p, q, r] targets."""
	# thrust = torch.rand(num_envs, 1, device=device) * (thrust_range[1] - thrust_range[0]) + thrust_range[0]
	# rates = (torch.rand(num_envs, 3, device=device) * 2.0 - 1.0) * rate_range
	thrust = torch.zeros(num_envs, 1, device=device)
	rates = torch.zeros(num_envs, 3, device=device)
	rates[:, 2] = rate_range * -0.75
	return torch.cat((thrust, rates), dim=1)


def main() -> None:
	parser = argparse.ArgumentParser(description="Tune CTBR PD controller with random commands.")
	AppLauncher.add_app_launcher_args(parser)
	parser.add_argument("--num_envs", type=int, default=64, help="Number of parallel environments.")
	parser.add_argument("--num_steps", type=int, default=500, help="Number of environment steps to run.")
	parser.add_argument("--command_interval", type=int, default=50, help="Steps between random commands.")
	parser.add_argument("--thrust_min", type=float, default=0.0, help="Minimum normalized collective thrust.")
	parser.add_argument("--thrust_max", type=float, default=0.0, help="Maximum normalized collective thrust.")
	parser.add_argument("--rate_range", type=float, default=1, help="Body-rate.")
	args_cli = parser.parse_args()

	if hasattr(args_cli, "enable_cameras") and not args_cli.enable_cameras:
		args_cli.enable_cameras = True

	app_launcher = AppLauncher(args_cli)
	simulation_app = app_launcher.app

	from isaaclab_contrib.assets.multirotor.multirotor_data import MultirotorData
	from isaaclab_tasks.direct.acro_drone.acro_drone_env import AcroDroneEnv
	from isaaclab_tasks.direct.acro_drone.acro_drone_env_cfg import AcroDroneEnvCfg

	cfg = AcroDroneEnvCfg()
	cfg.scene.num_envs = args_cli.num_envs

	env = AcroDroneEnv(cfg)
	env.reset()

	device = env.device
	num_envs = env.num_envs
	command_interval = max(1, args_cli.command_interval)
	thrust_range = (args_cli.thrust_min, args_cli.thrust_max)

	current_cmd = _sample_ctbr(num_envs, thrust_range, args_cli.rate_range, device)
	cmd_log: list[torch.Tensor] = []
	actual_log: list[torch.Tensor] = []

	for step in range(args_cli.num_steps):
		if step % command_interval == 0:
			current_cmd = _sample_ctbr(num_envs, thrust_range, args_cli.rate_range, device)

		env.step(current_cmd)

		multirotor_data = cast(MultirotorData, env.robot.data)
		collective_thrust = multirotor_data.applied_thrust.sum(dim=1, keepdim=True)
		body_rates = env.robot.data.root_ang_vel_b
		actual_ctbr = torch.cat((collective_thrust, body_rates), dim=1)

		# Replicate the scaling that _apply_action applies so logged cmd matches
		# what the controller actually receives as its target.
		action_scale_t = torch.tensor(cfg.action_scale, device="cpu", dtype=torch.float32)
		cmd_scaled = current_cmd.detach().cpu() * action_scale_t
		cmd_scaled[:, 0] *= cfg.total_mass  # _apply_action only scales thrust by mass
		cmd_log.append(cmd_scaled)
		actual_log.append(actual_ctbr.detach().cpu())

	cmd_arr = torch.stack(cmd_log, dim=0).numpy()
	actual_arr = torch.stack(actual_log, dim=0).numpy()
	# err_arr = actual_arr - cmd_arr

	# Plot mean across envs to keep plots readable.
	cmd_mean = cmd_arr.mean(axis=1)
	actual_mean = actual_arr.mean(axis=1)
	# err_mean = err_arr.mean(axis=1)

	labels = ["collective_thrust", "p", "q", "r"]
	steps = np.arange(cmd_mean.shape[0])

	fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
	for i in range(4):
		axes[i].plot(steps, cmd_mean[:, i], label="command")
		axes[i].plot(steps, actual_mean[:, i], label="actual")
		# axes[i].plot(steps, err_mean[:, i], label="error")
		axes[i].set_ylabel(labels[i])
		axes[i].grid(True, alpha=0.3)
		if i == 0:
			axes[i].legend(loc="upper right")

	axes[-1].set_xlabel("step")
	fig.tight_layout()
	plt.show()

	env.close()
	simulation_app.close()


if __name__ == "__main__":
	main()
