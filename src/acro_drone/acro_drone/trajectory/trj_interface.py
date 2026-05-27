"""Trajectory interface for managing reference trajectories across environments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation

from .trajectory_generation import build_trajectory_from_given_points, sample_discrete_trajectory, build_trajectory_from_template
from .trajectory_optimize import optimize_trj_time
from .trj_interface_data import TrajectoryData
from .trj_interface_cfg import TrajectoryInterfaceCfg
from .trajectory import Trajectory
from .base_templates import WaypointTemplate


class TrajectoryInterface:
    """Interface for managing and querying reference trajectories for multiple environments.
    
    This class handles:
    - Storing full trajectory references for all environments
    - Resetting trajectories for specific environments
    - Querying reference states at arbitrary times with interpolation
    - Batch querying of trajectory windows with specified time steps
    
    The interface uses linear interpolation for position and velocity,
    and spherical linear interpolation (SLERP) for quaternions.
    """
    
    def __init__(self, num_envs: int, cfg: TrajectoryInterfaceCfg | None = None):
        """Initialize the trajectory interface.
        
        Args:
            num_envs: Number of environments.
            cfg: Configuration for trajectory interface. Defaults to TrajectoryInterfaceCfg().
        """
        self.num_envs = num_envs
        self.cfg = cfg if cfg is not None else TrajectoryInterfaceCfg()
        
        # Storage for trajectory data for each environment
        self._trajectories: dict[int, TrajectoryData | None] = {i: None for i in range(num_envs)}
        
        # Storage for interpolators for each environment
        self._pos_interp: dict[int, interp1d | None] = {i: None for i in range(num_envs)}
        self._vel_interp: dict[int, interp1d | None] = {i: None for i in range(num_envs)}
        self._quat_interp: dict[int, Slerp | None] = {i: None for i in range(num_envs)}
        self._body_rate_interp: dict[int, interp1d | None] = {i: None for i in range(num_envs)}
    
    def reset_idx(self, env_ids_seq: list[int], templates: list[WaypointTemplate]=None, points:dict=None) -> None:
        """Reset the reference trajectory for a specific environment.
        
        Randomly samples a template from the provided list (or uses the single template),
        builds trajectory from it, applies time optimization, samples it, and stores the result.
        
        Args:
            env_ids_seq: List of environment indices.
            templates: Waypoint template(s) to build trajectory from. Can be a single template
                      or a list of templates to randomly sample from.
        
        Raises:
            ValueError: If env_id is out of bounds or templates list is empty.
        """
        for env_id in env_ids_seq:
            if not 0 <= env_id < self.num_envs:
                raise ValueError(f"env_id {env_id} out of bounds [0, {self.num_envs})")
        
            # Handle single template or list of templates
            if isinstance(templates, list):
                if len(templates) == 0:
                    raise ValueError("templates list cannot be empty")
                template = templates[np.random.randint(len(templates))]
            
            # Build trajectory from randomly selected template
            if template is not None:
                trajectory = build_trajectory_from_template(template)
            elif points is not None:
                trajectory = build_trajectory_from_given_points(**points)
            else:
                raise ValueError("Either templates or points must be provided to build a trajectory.")
            
            # Apply time optimization if configured
            if self.cfg.optimize_time:
                time_penalty = np.ones(len(trajectory._segments), dtype=float) * self.cfg.time_penalty
                
                trajectory, _, _ = optimize_trj_time(
                    trajectory,
                    time_penalty=time_penalty,
                    preserve_total_time=False,
                    max_velocity=self.cfg.max_velocity,
                    max_normalized_thrust=self.cfg.max_normalized_thrust,
                    report_peaks=False,
                )
            
            # Sample the trajectory
            discrete_traj = sample_discrete_trajectory(
                trajectory,
                sampling_rate=self.cfg.sampling_rate,
                gravity_vector=np.array(self.cfg.gravity_vector),
            )
            
            # Create trajectory data container
            traj_data = TrajectoryData(
                t=discrete_traj["t"],
                pos=discrete_traj["pos"],
                quat=discrete_traj["quat"],
                vel=discrete_traj["vel"],
                body_rate=discrete_traj["body_rate"],
            )
            
            self._trajectories[env_id] = traj_data
            
            # Build interpolators
            self._build_interpolators(env_id, traj_data)

    def get_trj_time(self, env_ids: list[int]) -> float:
        return np.hstack([self._trajectories[env_id].t[-1] for env_id in env_ids if self._trajectories[env_id] is not None]).max()

    def _build_interpolators(self, env_id: int, traj_data: TrajectoryData) -> None:
        """Build interpolators for a trajectory.
        
        Args:
            env_id: Environment index.
            traj_data: Trajectory data to build interpolators for.
        """
        t = traj_data.t
        
        # Linear interpolators for position, velocity, and body_rate
        self._pos_interp[env_id] = interp1d(
            t, traj_data.pos, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate"
        )
        self._vel_interp[env_id] = interp1d(
            t, traj_data.vel, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate"
        )
        self._body_rate_interp[env_id] = interp1d(
            t, traj_data.body_rate, axis=0, kind="linear", bounds_error=False, fill_value="extrapolate"
        )
        
        # SLERP for quaternions
        rotations = Rotation.from_quat(traj_data.quat)
        self._quat_interp[env_id] = Slerp(t, rotations)
    
    def get_state_at_time(self, env_ids: list[int], t: float):
        """Get the reference state at a specific time.
        
        Args:
            env_ids: List of environment indices.
            t: Time [s] at which to query the state.
        
        Returns:
            state vector (num_envs, 13).
        
        Raises:
            ValueError: If trajectory not set for the environment or env_id is invalid.
        """
        for env_id in env_ids:
            if not 0 <= env_id < self.num_envs:
                raise ValueError(f"env_id {env_id} out of bounds [0, {self.num_envs})")
        
        # Check if trajectories are set for all requested environments
        for env_id in env_ids:
            if self._trajectories[env_id] is None:
                raise ValueError(f"Trajectory not set for environment {env_id}")

        # Initialize lists to store interpolated values
        state_list = []

        for env_id in env_ids:
            traj_data = self._trajectories[env_id]
            assert traj_data is not None

            pos_interp = self._pos_interp[env_id]
            quat_interp = self._quat_interp[env_id]
            vel_interp = self._vel_interp[env_id]
            body_rate_interp = self._body_rate_interp[env_id]
            if pos_interp is None or quat_interp is None or vel_interp is None or body_rate_interp is None:
                raise RuntimeError(f"Interpolators are not initialized for environment {env_id}")

            # Clamp query time so requests beyond trajectory duration return endpoint state.
            t_query = min(float(t), float(traj_data.t[-1]))
            
            # Query all interpolators
            pos = np.array(pos_interp(t_query)).flatten()
            quat_rotation = quat_interp(t_query)
            quat = quat_rotation.as_quat()  # Returns [x, y, z, w] format
            vel = np.array(vel_interp(t_query)).flatten()
            body_rate = np.array(body_rate_interp(t_query)).flatten()
            state_list.append(np.hstack((pos, quat, vel, body_rate)))
        
        return np.array(state_list)
    
    def get_trajectory_window(
        self,
        env_ids: list[int],
        t_start: float | list[float] | np.ndarray,
        t_end: float | list[float] | np.ndarray,
        time_step: float,
    ) -> np.ndarray:
        """Get a window of reference states over a time interval.
        
        Args:
            env_ids: List of environment indices.
            t_start: Start time [s].
            t_end: End time [s].
            time_step: Time step [s] for sampling within the window.
        
        Returns:
            Array of shape (N, M, 13) where:
            - N: Number of environments
            - M: Number of time steps
            - 13: Dimensions of the state vector (3 pos + 4 quat + 3 vel + 3 body_rate)
        
        Raises:
            ValueError: If trajectory not set or invalid parameters.
        """
        for env_id in env_ids:
            if not 0 <= env_id < self.num_envs:
                raise ValueError(f"env_id {env_id} out of bounds [0, {self.num_envs})")
        
        for env_id in env_ids:
            if self._trajectories[env_id] is None:
                raise ValueError(f"Trajectory not set for environment {env_id}")

        if time_step <= 0:
            raise ValueError(f"time_step ({time_step}) must be positive")

        t_start_arr = np.asarray(t_start, dtype=float)
        t_end_arr = np.asarray(t_end, dtype=float)
        if t_start_arr.ndim == 0:
            t_start_arr = np.full(len(env_ids), float(t_start_arr))
        if t_end_arr.ndim == 0:
            t_end_arr = np.full(len(env_ids), float(t_end_arr))
        if t_start_arr.shape[0] != len(env_ids) or t_end_arr.shape[0] != len(env_ids):
            raise ValueError("t_start and t_end must be scalars or match env_ids length.")

        # Precompute time arrays and verify consistent length.
        time_arrays = []
        expected_len = None
        for idx, env_id in enumerate(env_ids):
            if t_end_arr[idx] <= t_start_arr[idx]:
                raise ValueError(f"t_end ({t_end_arr[idx]}) must be greater than t_start ({t_start_arr[idx]}).")
            t_array = np.arange(t_start_arr[idx], t_end_arr[idx] + time_step / 2, time_step)
            if expected_len is None:
                expected_len = len(t_array)
            elif len(t_array) != expected_len:
                raise ValueError("Batched t_start/t_end must produce the same number of steps.")
            time_arrays.append(t_array)

        # Initialize lists to store interpolated values
        state_list = []

        for idx, env_id in enumerate(env_ids):
            traj_data = self._trajectories[env_id]
            assert traj_data is not None

            pos_interp = self._pos_interp[env_id]
            quat_interp = self._quat_interp[env_id]
            vel_interp = self._vel_interp[env_id]
            body_rate_interp = self._body_rate_interp[env_id]
            if pos_interp is None or quat_interp is None or vel_interp is None or body_rate_interp is None:
                raise RuntimeError(f"Interpolators are not initialized for environment {env_id}")
            
            t_array = time_arrays[idx]

            # Clamp query times so requests beyond trajectory duration return endpoint state.
            t_clamped = np.minimum(t_array, float(traj_data.t[-1]))
            
            # Query all interpolators
            pos = np.array(pos_interp(t_clamped))
            vel = np.array(vel_interp(t_clamped))
            body_rate = np.array(body_rate_interp(t_clamped))
            
            # Query quaternions
            quat_rotations = quat_interp(t_clamped)
            quat = quat_rotations.as_quat()  # Returns [x, y, z, w] format
            
            state = np.hstack((pos, quat, vel, body_rate))  # Shape (M, 3+4+3+3) = (M, 13)
            state_list.append(state)
        return np.array(state_list)  # Shape (N, M, 13)
    
    def get_trajectory(self, env_id: int) -> TrajectoryData | None:
        """Get the full trajectory data for an environment.
        
        Args:
            env_id: Environment index.
        
        Returns:
            TrajectoryData if set, None otherwise.
        
        Raises:
            ValueError: If env_id is out of bounds.
        """
        if not 0 <= env_id < self.num_envs:
            raise ValueError(f"env_id {env_id} out of bounds [0, {self.num_envs})")
        
        return self._trajectories[env_id]
