"""Configuration for collective-thrust/body-rate PD controller."""

from __future__ import annotations
from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class CtbrPdControllerCfg:
    """PD controller config for [collective thrust, body rate]."""
    # Derivative term timestep in seconds.
    dt: float = MISSING
    # Gains are ordered as: [collective_thrust, p, q, r]
    kp: tuple[float, float, float] = MISSING # shape: (3,)
    kd: tuple[float, float, float] = MISSING # shape: (3,)
    # Allocation matrix maps motor thrusts -> [collective thrust, p, q, r].
    allocation_matrix: tuple[tuple[float, float, float, float], ...] = MISSING  # shape: (4, 4)
    # Optional motor thrust saturation.
    thrust_min: float | None = None
    thrust_max: float | None = None
    
