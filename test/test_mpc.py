import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import numpy as np
import do_mpc
from typing import List

from isaaclab_tasks.direct.acro_drone.controllers.mpc import MpcCfg, MPCControllerWrapper

from isaaclab_tasks.direct.acro_drone.trajectory.acro_templates import HeartTemplate, PowerloopTemplate, SplitSLeftTemplate, BarrelRollLeftTemplate, StraightLineTemplate
from isaaclab_tasks.direct.acro_drone.trajectory.base_templates import WaypointTemplate
from isaaclab_tasks.direct.acro_drone.trajectory.trj_interface import TrajectoryInterface
from isaaclab_tasks.direct.acro_drone.trajectory.trj_interface_cfg import TrajectoryInterfaceCfg


cfg = MpcCfg(
    horizon=50,
    solver_max_iter=100,
    solver_tol=1e-3,
    horizon_dt=1/50,
    dt=1/50,
    w_pos=1.5,
    w_quat=2.0,
    w_vel=0.5,
    w_body_rate=0.1,
    w_output=1.0,
    w_output_derivative=1.5,
    max_normalized_thrust=60.0,
    max_roll_pitch_rate=20.0,
    max_yaw_rate=10.0,
)
num_envs = 2
env_ids = list(range(num_envs))
mpc_controller = MPCControllerWrapper(cfg, num_envs=num_envs)

trj_interface_cfg = TrajectoryInterfaceCfg(
	sampling_rate=300,
	time_penalty=10,
	max_velocity=20,
	max_normalized_thrust=50,
)
trj_interface = TrajectoryInterface(num_envs=num_envs, cfg=trj_interface_cfg)
templates: List[WaypointTemplate] = [PowerloopTemplate()]
trj_interface.reset_idx(env_ids_seq=env_ids, templates=templates)
print("Trajectory total time:", trj_interface.get_trj_time(env_ids=env_ids))

x0 = np.zeros((num_envs, 10), dtype=float)
x0[: , 3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
# x0 = trj_interface.get_state_at_time(env_ids=env_ids, t=0.0)
print(cfg.horizon)

horizon_0 = np.zeros((num_envs, cfg.horizon + 1, 13), dtype=float)
horizon_0 = trj_interface.get_trajectory_window(
	env_ids=env_ids,
	t_start=0.0,
	t_end=cfg.horizon * cfg.horizon_dt,
	time_step=cfg.horizon_dt,
)
mpc_controller.reset_idx(env_ids=env_ids, x0=x0, horizon=horizon_0)

import matplotlib.pyplot as plt
import matplotlib as mpl
# Customizing Matplotlib:
mpl.rcParams['font.size'] = 9
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

simulator = do_mpc.simulator.Simulator(mpc_controller.mpc_controllers[0].mpc.model)
simulator.set_param(t_step=cfg.dt)
sim_tvp_temp = simulator.get_tvp_template()

def sim_tvp_fun(t_now):
	state = trj_interface.get_state_at_time(env_ids=[0], t=t_now)[0]
	pos_ref = state[0:3]
	quat_ref = state[3:7]
	v_ref = state[7:10]
	body_rate_ref = state[10:13]
	sim_tvp_temp["p_ref"] = pos_ref.reshape(3,)
	sim_tvp_temp["quat_ref"] = quat_ref.reshape(4,)
	sim_tvp_temp["v_ref"] = v_ref.reshape(3,)
	sim_tvp_temp["body_rate_ref"] = body_rate_ref.reshape(3,)
    
	return sim_tvp_temp

simulator.set_tvp_fun(sim_tvp_fun)
simulator.setup()
simulator.x0 = x0[0, :10]

mpc_graphics = do_mpc.graphics.Graphics(mpc_controller.mpc_controllers[0].mpc.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)


def quat_xyzw_to_rotmat(quat_xyzw):
	quat_xyzw = np.asarray(quat_xyzw, dtype=float)
	x, y, z, w = quat_xyzw
	xx = x * x
	yy = y * y
	zz = z * z
	xy = x * y
	xz = x * z
	yz = y * z
	wx = w * x
	wy = w * y
	wz = w * z

	return np.array(
		[
			[1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
			[2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
			[2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
		],
		dtype=float,
	)


def set_equal_3d_axes(ax, points):
	points = np.asarray(points, dtype=float)
	x_min, y_min, z_min = np.min(points, axis=0)
	x_max, y_max, z_max = np.max(points, axis=0)

	x_mid = 0.5 * (x_min + x_max)
	y_mid = 0.5 * (y_min + y_max)
	z_mid = 0.5 * (z_min + z_max)

	half_range = 0.5 * max(x_max - x_min, y_max - y_min, z_max - z_min)
	if half_range == 0:
		half_range = 1.0

	ax.set_xlim(x_mid - half_range, x_mid + half_range)
	ax.set_ylim(y_mid - half_range, y_mid + half_range)
	ax.set_zlim(z_mid - half_range, z_mid + half_range)


# We just want to create the plot and not show it right now. This "inline magic" supresses the output.
fig, ax = plt.subplots(4, sharex=True, figsize=(16,9))
fig.align_ylabels()


for g in [sim_graphics]:
	g.add_line(var_type="_x", var_name="p", axis=ax[0], label="Position")
	g.add_line(var_type="_x", var_name="v", axis=ax[1], label="Velocity")
	g.add_line(var_type="_x", var_name="quat", axis=ax[2], label="Quaternion")
	# g.add_line(var_type="_x", var_name="body_rate", axis=ax[3], label="Angular Velocity")
	# g.add_line(var_type="_u", var_name="thrust", axis=ax[4], label="Thrust")
	# g.add_line(var_type="_u", var_name="w", axis=ax[5], label="Body Rate Command")
	
	g.add_line(var_type="_tvp", var_name="p_ref", axis=ax[0], label="Position Ref", linestyle="dashed")
	g.add_line(var_type="_tvp", var_name="v_ref", axis=ax[1], label="Velocity Ref", linestyle="dashed")
	g.add_line(var_type="_tvp", var_name="quat_ref", axis=ax[2], label="Quaternion Ref", linestyle="dashed")
	# g.add_line(var_type="_tvp", var_name="body_rate_ref", axis=ax[3], label="Angular Velocity Ref", linestyle="dashed")

ax[0].set_ylabel("Position (m)")
ax[1].set_ylabel("Velocity (m/s)")
ax[2].set_ylabel("Quaternion (xyzw)")
ax[3].set_ylabel("Angular Velocity (rad/s)")
ax[3].set_xlabel("Time (s)")
t = 0.0
sim_state = x0[0].copy()
traj_total_time = trj_interface.get_trj_time(env_ids=[0])
num_steps = 150

sim_positions = [sim_state[0:3].copy()]
sim_quats = [sim_state[3:7].copy()]
mpc_actions = []


for i in range(num_steps):
	horizon = np.zeros((num_envs, cfg.horizon + 1, 13), dtype=float)
	horizon = trj_interface.get_trajectory_window(
		env_ids=env_ids,
		t_start=t,
		t_end=t + cfg.horizon * cfg.horizon_dt,
		time_step=cfg.horizon_dt,
	)
	u0 = mpc_controller.make_step(x0, horizon=horizon)
	mpc_actions.append(u0[0].copy())
	sim_state = np.asarray(simulator.make_step(u0[0])).reshape(10,)
	x0[0] = sim_state
	sim_positions.append(sim_state[0:3].copy())
	sim_quats.append(sim_state[3:7].copy())
	t += cfg.dt

sim_graphics.plot_results()
sim_graphics.reset_axes()
fig.legend(loc="upper right")
fig.show()

mpc_actions = np.asarray(mpc_actions, dtype=float)
time_axis = np.arange(len(mpc_actions)) * cfg.dt

fig_actions, ax_actions = plt.subplots(4, sharex=True, figsize=(10, 9))
fig_actions.suptitle("MPC Output")

ax_actions[0].plot(time_axis, mpc_actions[:, 0], label="Thrust")
ax_actions[0].set_ylabel("Thrust")
ax_actions[0].grid(True)
ax_actions[0].legend()

ax_actions[1].plot(time_axis, mpc_actions[:, 1], label="p")
ax_actions[1].set_ylabel("p (rad/s)")
ax_actions[1].grid(True)
ax_actions[1].legend()

ax_actions[2].plot(time_axis, mpc_actions[:, 2], label="q")
ax_actions[2].set_ylabel("q (rad/s)")
ax_actions[2].grid(True)
ax_actions[2].legend()

ax_actions[3].plot(time_axis, mpc_actions[:, 3], label="r")
ax_actions[3].set_ylabel("r (rad/s)")
ax_actions[3].set_xlabel("Time (s)")
ax_actions[3].grid(True)
ax_actions[3].legend()
fig_actions.show()

sim_positions = np.asarray(sim_positions)
sim_quats = np.asarray(sim_quats)
rot_mats = np.stack([quat_xyzw_to_rotmat(q) for q in sim_quats], axis=0)

body_x = rot_mats[:, :, 0]
body_y = rot_mats[:, :, 1]
body_z = rot_mats[:, :, 2]

fig_orient = plt.figure(figsize=(8, 6))
ax_orient = fig_orient.add_subplot(111, projection="3d")
ax_orient.plot(sim_positions[:, 0], sim_positions[:, 1], sim_positions[:, 2], linewidth=2)

stride = max(1, len(sim_positions) // 20)
idx = np.arange(0, len(sim_positions), stride)
scale = 0.4

ax_orient.quiver(
	sim_positions[idx, 0], sim_positions[idx, 1], sim_positions[idx, 2],
	body_x[idx, 0], body_x[idx, 1], body_x[idx, 2],
	length=scale,
	normalize=True,
	color="tab:blue",
)
ax_orient.quiver(
	sim_positions[idx, 0], sim_positions[idx, 1], sim_positions[idx, 2],
	body_y[idx, 0], body_y[idx, 1], body_y[idx, 2],
	length=scale,
	normalize=True,
	color="tab:green",
)
ax_orient.quiver(
	sim_positions[idx, 0], sim_positions[idx, 1], sim_positions[idx, 2],
	body_z[idx, 0], body_z[idx, 1], body_z[idx, 2],
	length=scale,
	normalize=True,
	color="tab:red",
)

ax_orient.set_xlabel("X")
ax_orient.set_ylabel("Y")
ax_orient.set_zlabel("Z")
ax_orient.set_title("MPC Trajectory with Body Axes")
set_equal_3d_axes(ax_orient, sim_positions)
ax_orient.set_box_aspect((1, 1, 1))
fig_orient.show()
input()
# for i in range(100):
# 	x0 = np.concatenate(trj_interface.get_state_at_time(env_id=0, t=t)).reshape(13,)
# 	horizon = np.hstack(trj_interface.get_trajectory_window(
# 		env_id=0,
# 		t_start=t,
# 		t_end=t + cfg.horizon * cfg.horizon_dt,
# 		time_step=cfg.horizon_dt,
# 	))
# 	mpc_controller.set_horizon(horizon)
# 	u = mpc_controller.mpc.make_step(x0)
simulation_app.close()