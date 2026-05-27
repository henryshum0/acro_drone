
from acro_drone.trajectory.acro_templates import *

TRAIN_TEMPLATES2 = [
	PowerloopTemplate,
	SplitSLeftTemplate,
	SplitSRightTemplate,
	BarrelRollRightTemplate,
	BarrelRollLeftTemplate,
]

from acro_drone.trajectory.trajectory_generation import build_trajectory_from_template, trajectory_from_template, sample_discrete_trajectory
from acro_drone.trajectory.trajectory_optimize import optimize_trj_time
template = PowerloopTemplate()
trajectory = build_trajectory_from_template(template)
optimized_traj, optimized_time, min_result = optimize_trj_time(
    trajectory,
    time_penalty=np.array([100 for seg in trajectory._segments]),
    preserve_total_time=False,
    max_velocity=20,
    max_normalized_thrust=50,
    report_peaks=True,
)
print("optimized_time:", optimized_time)
optimized_traj.visualize(show=True)
