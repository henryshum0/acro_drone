import argparse
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# ========== 你的测试代码放在这里 ==========
from isaaclab_tasks.direct.acro_drone.trajectory.acro_templates import HeartTemplate, PowerloopTemplate, SplitSLeftTemplate
from isaaclab_tasks.direct.acro_drone.trajectory.trj_interface import TrajectoryInterface
from isaaclab_tasks.direct.acro_drone.trajectory.trj_interface_cfg import TrajectoryInterfaceCfg


def main():
	cfg = TrajectoryInterfaceCfg(
		sampling_rate=50.0,
		optimize_time=True,
		time_penalty=100,
		max_velocity=20.0,
		max_normalized_thrust=60.0,
	)
	interface = TrajectoryInterface(num_envs=2, cfg=cfg)

	templates = [HeartTemplate()]
	interface.reset_idx(env_ids_seq=[0,1], templates=templates)

	state = interface.get_state_at_time(env_ids=[0], t=0.5)
	print("state:", state)

	state_window = interface.get_trajectory_window(
		env_ids=[1],
		t_start=0.0,
		t_end=2.0,
		time_step=0.1,
	)
	pos_window = state_window[0, :, :3]
	quat_window = state_window[0, :, 3:7]
	vel_window = state_window[0, :, 7:10]
	body_rate_window = state_window[0, :, 10:13]
	print("total time:", interface.get_trj_time(env_ids=[1]))
	print("window_pos:", pos_window.shape)
	print("window_quat:", quat_window)
	print("window_vel:", vel_window)
	print("window_body_rate:", body_rate_window)

	assert pos_window.shape[1] == 3
	assert quat_window.shape[1] == 4
	assert vel_window.shape[1] == 3
	assert body_rate_window.shape[1] == 3
	assert np.isfinite(state).all()


if __name__ == "__main__":
	main()



# ==========================================

simulation_app.close()
