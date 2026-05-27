from __future__ import annotations

import numpy as np

from .base_templates import WaypointTemplate
from .trajectory import Node, Segment, Trajectory
from .trajectory_optimize import optimize_trj_time


def sample_template(template:WaypointTemplate) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[bool], float]:
    """Sample a waypoint template and return waypoint positions, orientations, velocities, durations, and accelerations."""
    xyzs = np.asarray(template.waypoints_xyzs, dtype=float)
    psis = np.asarray(template.waypoints_psis, dtype=float)
    durations = np.asarray(template.waypoints_durations, dtype=float).reshape(-1)

    # Try to get velocities and accelerations, with fallback defaults
    vels = template.waypoints_vels
    accs = template.waypoints_accs

    if (
        xyzs.shape[0] != psis.shape[0]
        or xyzs.shape[0] != len(vels)
        or xyzs.shape[0] != len(accs)
        or xyzs.shape[0] != durations.shape[0]
    ):
        raise ValueError("Template sample must provide matching xyzs, psis, vels, accs, and durations lengths")

    return xyzs, psis, vels, accs, durations


def build_trajectory_from_template(
    template: WaypointTemplate
) -> Trajectory:
    """Build a trajectory from a sampled waypoint template.
    """
    xyzs, psis, vels, accs, durations = sample_template(template)

    if xyzs.shape[0] < 2:
        raise ValueError("A trajectory needs at least two waypoints")

    nodes = []
    for xyz, psi, vel, acc in zip(xyzs, psis, vels, accs):
        nodes.append(Node(pos=xyz, psi=float(psi), con_vel=vel, con_acc=acc))

    segments = []
    for i in range(len(nodes) - 1):
        duration = durations[i+1]
        segments.append(Segment(nodes[i], nodes[i + 1], duration=duration))

    return Trajectory(segments)

def build_trajectory_from_given_points(
    xyzs: list,
    psis: list,
    vels: list,
    accs: list,
    durations: list,
) -> Trajectory:
    """Build a trajectory from given waypoint positions, orientations, velocities, accelerations, and durations."""
    if xyzs.shape[0] < 2:
        raise ValueError("A trajectory needs at least two waypoints")

    nodes = []
    for xyz, psi, vel, acc in zip(xyzs, psis, vels, accs):
        nodes.append(Node(pos=xyz, psi=float(psi), con_vel=vel, con_acc=acc))

    segments = []
    for i in range(len(nodes) - 1):
        duration = durations[i+1]
        segments.append(Segment(nodes[i], nodes[i + 1], duration=duration))

    return Trajectory(segments)


def trajectory_from_template(template: WaypointTemplate):
    """Backward-friendly alias for build_trajectory_from_template()."""
    return build_trajectory_from_template(template)


def sample_discrete_trajectory(
    trajectory: Trajectory,
    sampling_rate: float,
    gravity_vector: np.ndarray = np.array([0.0, 0.0, -9.81]),
) -> dict:

    trj = trajectory.sample_full_state(
        sampling_rate=sampling_rate,
        gravity_vector=gravity_vector,
        include_terminal=True,
    )

    return {
        "t": trj["t"],
        "pos": trj["pos"],
        "quat": trj["quat"],
        "vel": trj["vel"],
        "body_rate": trj["body_rate"],
    }

if __name__ == "__main__":
    from .acro_templates import\
        PowerloopTemplate, SplitSLeftTemplate, SplitSRightTemplate, BarrelRollLeftTemplate, BarrelRollRightTemplate,\
        HeartTemplate
    from .trajectory_optimize import optimize_trj_time
    template = BarrelRollLeftTemplate()
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