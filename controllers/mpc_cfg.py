
from __future__ import annotations
from dataclasses import MISSING

from isaaclab.utils import configclass

@configclass
class MpcCfg:
	"""Configuration for MPC controller."""
	# Prediction horizon in seconds.
	horizon: int = MISSING
	horizon_dt: float = MISSING
	
	# Control timestep in seconds.
	dt: float = MISSING
	
	# Weights for MPC cost terms.
	w_pos:float = MISSING
	w_quat:float = MISSING
	w_vel:float = MISSING
	w_output:float = MISSING
	w_output_derivative:float = MISSING
	
	# constraints.
	max_normalized_thrust: float = MISSING
	max_roll_pitch_rate: float = MISSING
	max_yaw_rate: float = MISSING

	# Optional MPC solver settings.
	solver_max_iter: int = 80
	solver_tol: float = 1e-4

	gravity: float = 9.81