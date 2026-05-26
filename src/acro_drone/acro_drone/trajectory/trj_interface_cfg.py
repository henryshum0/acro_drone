"""Configuration for trajectory interface."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrajectoryInterfaceCfg:
    """Configuration for trajectory interface.
    
    Attributes:
        sampling_rate: Sampling rate in Hz for trajectory discretization.
        gravity_vector: Gravity vector for trajectory computation [m/s^2].
        optimize_time: Whether to apply time optimization during trajectory reset.
        time_penalty: Time penalty weights for optimization. Per-segment penalty.
        max_velocity: Maximum allowed velocity [m/s]. None to disable constraint.
        max_normalized_thrust: Maximum allowed normalized thrust. None to disable constraint.
    """
    
    sampling_rate: float = 100.0
    gravity_vector: list[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])
    optimize_time: bool = True
    time_penalty: float = 100
    max_velocity: float | None = None
    max_normalized_thrust: float | None = None
