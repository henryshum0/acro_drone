import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# ========== 你的测试代码放在这里 ==========
from isaaclab_tasks.direct.acro_drone.trajectory.acro_templates import *

TRAIN_TEMPLATES2 = [
	PowerloopTemplate,
	SplitSLeftTemplate,
	SplitSRightTemplate,
	BarrelRollRightTemplate,
	BarrelRollLeftTemplate,
]

from isaaclab_tasks.direct.acro_drone.trajectory.trajectory_generation import build_trajectory_from_template, trajectory_from_template, sample_discrete_trajectory
from isaaclab_tasks.direct.acro_drone.trajectory.trajectory_optimize import optimize_trj_time
template = SplitSLeftTemplate()
trajectory = build_trajectory_from_template(template)
optimized_traj, optimized_time, min_result = optimize_trj_time(
    trajectory,
    time_penalty=np.array([100 for seg in trajectory._segments]),
    preserve_total_time=False,
    max_velocity=20,
    max_normalized_thrust=60,
    report_peaks=True,
)
print("optimized_time:", optimized_time)
optimized_traj.visualize(show=True)
# ==========================================

simulation_app.close()
