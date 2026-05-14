"""Data structures for trajectory interface."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrajectoryData:
    """Container for discretized trajectory data.
    
    Attributes:
        t: Time array [s] of shape (N,).
        pos: Position array [m] of shape (N, 3).
        quat: Quaternion array (xyzw format) of shape (N, 4).
        vel: Velocity array [m/s] of shape (N, 3).
        body_rate: Body rate array [rad/s] of shape (N, 3).
    """
    
    t: np.ndarray
    pos: np.ndarray
    quat: np.ndarray
    vel: np.ndarray
    body_rate: np.ndarray
    
    @property
    def total_time(self) -> float:
        """Total trajectory duration in seconds."""
        return float(self.t[-1] - self.t[0])
    
    @property
    def num_samples(self) -> int:
        """Number of trajectory samples."""
        return len(self.t)
