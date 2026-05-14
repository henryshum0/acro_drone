from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.sensors.imu import Imu

if TYPE_CHECKING:
	from .noisy_imu_cfg import NoisyImuCfg


class NoisyImu(Imu):
	"""IMU sensor with resampled noise stds and random-walk bias."""

	cfg: NoisyImuCfg

	def _initialize_buffers_impl(self):
		super()._initialize_buffers_impl()

		self._init_accel_bias = torch.zeros(self._view.count, 3, device=self._device)
		self._init_gyro_bias = torch.zeros(self._view.count, 3, device=self._device)
		self._rw_accel_bias = torch.zeros(self._view.count, 3, device=self._device)
		self._rw_gyro_bias = torch.zeros(self._view.count, 3, device=self._device)

		self._init_accel_std = torch.zeros(self._view.count, 3, device=self._device)
		self._init_gyro_std = torch.zeros(self._view.count, 3, device=self._device)
		self._rw_accel_std = torch.zeros(self._view.count, 3, device=self._device)
		self._rw_gyro_std = torch.zeros(self._view.count, 3, device=self._device)

	def reset(self, env_ids: Sequence[int] | None = None):
		super().reset(env_ids)
		if env_ids is None:
			env_ids = slice(None)

		self._init_accel_std[env_ids] = self._sample_std_range(self.cfg.init_accel_bias_std_range, env_ids)
		self._init_gyro_std[env_ids] = self._sample_std_range(self.cfg.init_gyro_bias_std_range, env_ids)
		self._rw_accel_std[env_ids] = self._sample_std_range(self.cfg.rw_accel_bias_std_range, env_ids)
		self._rw_gyro_std[env_ids] = self._sample_std_range(self.cfg.rw_gyro_bias_std_range, env_ids)

		self._init_accel_bias[env_ids] = torch.randn_like(self._init_accel_bias[env_ids]) * self._init_accel_std[
			env_ids
		]
		self._init_gyro_bias[env_ids] = torch.randn_like(self._init_gyro_bias[env_ids]) * self._init_gyro_std[
			env_ids
		]
		self._rw_accel_bias[env_ids] = 0.0
		self._rw_gyro_bias[env_ids] = 0.0

	def _update_buffers_impl(self, env_ids: Sequence[int]):
		super()._update_buffers_impl(env_ids)

		dt = float(self._dt) if hasattr(self, "_dt") else float(self._sim_physics_dt)
		rw_scale = dt**0.5

		self._rw_accel_bias[env_ids] += (
			torch.randn_like(self._rw_accel_bias[env_ids]) * self._rw_accel_std[env_ids] * rw_scale
		)
		self._rw_gyro_bias[env_ids] += (
			torch.randn_like(self._rw_gyro_bias[env_ids]) * self._rw_gyro_std[env_ids] * rw_scale
		)

		self._data.lin_acc_b[env_ids] += self._init_accel_bias[env_ids] + self._rw_accel_bias[env_ids]
		self._data.ang_vel_b[env_ids] += self._init_gyro_bias[env_ids] + self._rw_gyro_bias[env_ids]

	def _sample_std_range(
		self,
		std_range: tuple[tuple[float, float, float], tuple[float, float, float]],
		env_ids: Sequence[int] | slice,
	) -> torch.Tensor:
		min_std = torch.tensor(std_range[0], device=self._device, dtype=torch.float32)
		max_std = torch.tensor(std_range[1], device=self._device, dtype=torch.float32)
		shape = self._init_accel_std[env_ids].shape
		return torch.rand(shape, device=self._device) * (max_std - min_std) + min_std
