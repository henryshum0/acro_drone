from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from px4_msgs.msg import TimesyncStatus, VehicleRatesSetpoint
from rclpy.node import Node
from scipy.spatial.transform import Rotation

_SRC_ROOT = Path(__file__).resolve().parents[3]
if str(_SRC_ROOT) not in sys.path:
	sys.path.insert(0, str(_SRC_ROOT))

from acro_drone.controllers.mpc import MPCController
from acro_drone.controllers.mpc_cfg import MpcCfg
from acro_drone.trajectory.acro_templates import PowerloopTemplate
from acro_drone.trajectory.trj_interface import TrajectoryInterface
from acro_drone.trajectory.trj_interface_cfg import TrajectoryInterfaceCfg

@dataclass
class _Inputs:
	pose: Optional[PoseStamped] = None
	twist: Optional[TwistStamped] = None
	timesync: Optional[TimesyncStatus] = None


class MpcRatesNode(Node):
	def __init__(self) -> None:
		super().__init__("mpc_rates_node")

		self._inputs = _Inputs()
		self._has_warned_waiting = False
		self._initialized = False

		self.cfg = MpcCfg(
			horizon=50,
			horizon_dt=1 / 50,
			w_pos=10,
			w_quat=20,
			w_vel=2,
			w_output=0,
			w_output_derivative=0.1,
			max_normalized_thrust=60.0,
			max_roll_pitch_rate=20.0,
			max_yaw_rate=10.0,
		)
		trj_cfg = TrajectoryInterfaceCfg(
			sampling_rate=300,
			time_penalty=10,
			max_velocity=20,
			max_normalized_thrust=50,
		)

		self._trj_interface = TrajectoryInterface(num_envs=1, cfg=trj_cfg)
		self._trj_interface.reset_idx(env_ids_seq=[0], templates=[PowerloopTemplate()])
		self._traj_total_time = float(self._trj_interface.get_trj_time(env_ids=[0]))

		self._start_time = self._now_s()
		self._mpc = MPCController(self.cfg)

		self._pub_rates = self.create_publisher(
			VehicleRatesSetpoint,
			"/fmu/in/vehicle_rates_setpoint",
			10,
		)
		self.create_subscription(
			PoseStamped,
			"drone1/state/pose",
			self._pose_cb,
			10,
		)
		self.create_subscription(
			TwistStamped,
			"drone1/state/twist",
			self._twist_cb,
			10,
		)
		self.create_subscription(
			TimesyncStatus,
			"/fmu/out/timesync_status",
			self._timesync_cb,
			10,
		)

		self.create_timer(1.0 / 50.0, self._control_step)

	def _now_s(self) -> float:
		return self.get_clock().now().nanoseconds * 1e-9

	def _pose_cb(self, msg: PoseStamped) -> None:
		self._inputs.pose = msg

	def _twist_cb(self, msg: TwistStamped) -> None:
		self._inputs.twist = msg

	def _timesync_cb(self, msg: TimesyncStatus) -> None:
		self._inputs.timesync = msg

	def _control_step(self) -> None:
		if self._inputs.pose is None or self._inputs.twist is None or self._inputs.timesync is None:
			if not self._has_warned_waiting:
				self.get_logger().warn("Waiting for pose, twist, and timesync messages...")
				self._has_warned_waiting = True
			return

		self._has_warned_waiting = False

		elapsed = self._now_s() - self._start_time
		if self._traj_total_time > 0.0 and elapsed > self._traj_total_time:
			self._trj_interface.reset_idx(env_ids_seq=[0], templates=[PowerloopTemplate()])
			self._traj_total_time = float(self._trj_interface.get_trj_time(env_ids=[0]))
			self._start_time = self._now_s()
			elapsed = 0.0

		x0 = self._assemble_state()
		if x0 is None:
			return

		horizon = self._trj_interface.get_trajectory_window(
			env_ids=[0],
			t_start=elapsed,
			t_end=elapsed + self.cfg.horizon * self.cfg.horizon_dt,
			time_step=self.cfg.horizon_dt,
		)

		if not self._initialized:
			self._mpc.reset(x0=x0.reshape(-1), horizon=horizon[0])
			self._initialized = True

		u = self._mpc.make_step(x0.reshape(-1), horizon=horizon[0])
		u = np.asarray(u, dtype=float).reshape(-1)
		thrust = float(u[0])
		body_rates = self._body_flu_to_frd(np.asarray(u[1:4], dtype=float))
		thrust_norm = float(np.clip(thrust / self.cfg.max_normalized_thrust, -1.0, 1.0))

		msg = VehicleRatesSetpoint()
		self._set_timestamp(msg, self._inputs.timesync)
		self._set_rates(msg, body_rates)
		self._set_thrust(msg, thrust_norm)
		self._pub_rates.publish(msg)

	def _assemble_state(self) -> Optional[np.ndarray]:
		pose = self._inputs.pose
		twist = self._inputs.twist
		if pose is None or twist is None:
			return None

		pos_ned = np.array(
			[
				pose.pose.position.x,
				pose.pose.position.y,
				pose.pose.position.z,
			],
			dtype=float,
		)
		quat_ned = np.array(
			[
				pose.pose.orientation.x,
				pose.pose.orientation.y,
				pose.pose.orientation.z,
				pose.pose.orientation.w,
			],
			dtype=float,
		)
		vel_frd = np.array(
			[
				twist.twist.linear.x,
				twist.twist.linear.y,
				twist.twist.linear.z,
			],
			dtype=float,
		)
		body_rate_frd = np.array(
			[
				twist.twist.angular.x,
				twist.twist.angular.y,
				twist.twist.angular.z,
			],
			dtype=float,
		)

		pos = self._ned_to_enu_vec(pos_ned)
		quat = self._ned_to_enu_quat(quat_ned)
		vel = self._body_frd_to_flu(vel_frd)
		body_rate = self._body_frd_to_flu(body_rate_frd)

		state = np.hstack((pos, quat, vel, body_rate))
		return state.reshape(1, -1)

	def _set_timestamp(self, msg: VehicleRatesSetpoint, timesync: TimesyncStatus) -> None:
		timestamp = int(timesync.timestamp)
		if hasattr(msg, "timestamp"):
			msg.timestamp = timestamp
		elif hasattr(msg, "timestamp_sample"):
			msg.timestamp_sample = timestamp

	def _set_rates(self, msg: VehicleRatesSetpoint, body_rates: np.ndarray) -> None:
		p_rate, q_rate, r_rate = (float(body_rates[0]), float(body_rates[1]), float(body_rates[2]))
		if hasattr(msg, "roll"):
			msg.roll = p_rate
		if hasattr(msg, "pitch"):
			msg.pitch = q_rate
		if hasattr(msg, "yaw"):
			msg.yaw = r_rate
		if hasattr(msg, "roll_rate"):
			msg.roll_rate = p_rate
		if hasattr(msg, "pitch_rate"):
			msg.pitch_rate = q_rate
		if hasattr(msg, "yaw_rate"):
			msg.yaw_rate = r_rate

	def _set_thrust(self, msg: VehicleRatesSetpoint, thrust_norm: float) -> None:
		if hasattr(msg, "thrust_body"):
			msg.thrust_body = [0.0, 0.0, -thrust_norm]
		elif hasattr(msg, "thrust"):
			msg.thrust = thrust_norm

	@staticmethod
	def _ned_to_enu_vec(vec_ned: np.ndarray) -> np.ndarray:
		return np.array([vec_ned[1], vec_ned[0], -vec_ned[2]], dtype=float)

	@staticmethod
	def _body_frd_to_flu(vec_frd: np.ndarray) -> np.ndarray:
		return np.array([vec_frd[0], -vec_frd[1], -vec_frd[2]], dtype=float)

	@staticmethod
	def _body_flu_to_frd(vec_flu: np.ndarray) -> np.ndarray:
		return np.array([vec_flu[0], -vec_flu[1], -vec_flu[2]], dtype=float)

	def _ned_to_enu_quat(self, quat_ned: np.ndarray) -> np.ndarray:
		t = np.array(
			[
				[0.0, 1.0, 0.0],
				[1.0, 0.0, 0.0],
				[0.0, 0.0, -1.0],
			],
			dtype=float,
		)
		rot_ned = Rotation.from_quat(quat_ned).as_matrix()
		rot_enu = t @ rot_ned @ t.T
		quat_enu = Rotation.from_matrix(rot_enu).as_quat()
		return quat_enu.astype(float)


def main() -> None:
	rclpy.init()
	node = MpcRatesNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main()
