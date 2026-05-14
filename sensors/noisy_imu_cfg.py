from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.sensors.imu import ImuCfg

from .noisy_imu import NoisyImu


@configclass
class NoisyImuCfg(ImuCfg):
	"""Configuration for a noisy IMU sensor.

	The noise stds are resampled uniformly from the configured ranges on every reset.
	"""

	class_type: type = NoisyImu

	init_accel_bias_std_range: tuple[tuple[float, float, float], tuple[float, float, float]] = (
		(0.0, 0.0, 0.0),
		(0.0, 0.0, 0.0),
	)
	"""Uniform range for sampling accel bias std at reset (min, max) per axis."""

	init_gyro_bias_std_range: tuple[tuple[float, float, float], tuple[float, float, float]] = (
		(0.0, 0.0, 0.0),
		(0.0, 0.0, 0.0),
	)
	"""Uniform range for sampling gyro bias std at reset (min, max) per axis."""

	rw_accel_bias_std_range: tuple[tuple[float, float, float], tuple[float, float, float]] = (
		(0.0, 0.0, 0.0),
		(0.0, 0.0, 0.0),
	)
	"""Uniform range for sampling accel random-walk std at reset (min, max) per axis."""

	rw_gyro_bias_std_range: tuple[tuple[float, float, float], tuple[float, float, float]] = (
		(0.0, 0.0, 0.0),
		(0.0, 0.0, 0.0),
	)
	"""Uniform range for sampling gyro random-walk std at reset (min, max) per axis."""
