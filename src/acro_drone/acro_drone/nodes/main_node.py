from __future__ import annotations

from typing import Optional

import rclpy
from px4_msgs.msg import OffboardControlMode, VehicleCommand, VehicleStatus
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from acro_drone.nodes.privileged_agent import MpcRatesNode
from acro_drone.nodes.return_node import PX4ReturnNode


class PX4MainNode(Node):
	def __init__(self, executor: Optional[MultiThreadedExecutor] = None) -> None:
		super().__init__("px4_main_node")

		self._executor = executor
		self._privileged_agent: Optional[MpcRatesNode] = None
		self._return_node: Optional[PX4ReturnNode] = None
		self._agent_started = False
		self._return_started = False
		self._offboard_setpoint_counter = 0
		self._vehicle_status = VehicleStatus()
		self._start_acro = bool(self.declare_parameter("start_acro", False).value)
		self.add_on_set_parameters_callback(self._on_param_update)

		self._acro_start_ns: Optional[int] = None
		self._acro_duration_s: Optional[float] = None

		self._last_offboard_cmd_ns = 0
		self._last_arm_cmd_ns = 0
		self._command_interval_ns = int(1.0 * 1e9)

		qos_profile = QoSProfile(
			reliability=ReliabilityPolicy.BEST_EFFORT,
			durability=DurabilityPolicy.TRANSIENT_LOCAL,
			history=HistoryPolicy.KEEP_LAST,
			depth=1,
		)

		self._offboard_control_mode_pub = self.create_publisher(
			OffboardControlMode,
			"/fmu/in/offboard_control_mode",
			qos_profile,
		)
		self._vehicle_command_pub = self.create_publisher(
			VehicleCommand,
			"/fmu/in/vehicle_command",
			qos_profile,
		)
		self.create_subscription(
			VehicleStatus,
			"/fmu/out/vehicle_status",
			self._vehicle_status_cb,
			qos_profile,
		)

		self.create_timer(0.05, self._timer_cb)

	def _vehicle_status_cb(self, msg: VehicleStatus) -> None:
		self._vehicle_status = msg

	def _timer_cb(self) -> None:
		if self._offboard_setpoint_counter < 10:
			self._publish_offboard_control_mode(body_rate=True)
			self._offboard_setpoint_counter += 1
			return

		if self._vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
			self._send_offboard_request()
			return

		if self._vehicle_status.arming_state != VehicleStatus.ARMING_STATE_ARMED:
			self._send_arm_request()
			return

		if self._start_acro:
			if self._return_started:
				self._stop_return_node()
			if not self._agent_started:
				self._start_privileged_agent()
		else:
			if self._agent_started:
				self._stop_privileged_agent()
			if not self._return_started:
				self._start_return_node()

		if self._agent_started:
			self._publish_offboard_control_mode(body_rate=True)
		elif self._return_started:
			self._publish_offboard_control_mode(body_rate=False)
		else:
			self._publish_offboard_control_mode(body_rate=True)

		if self._agent_started:
			self._check_acro_completion()

	def _publish_offboard_control_mode(self, *, body_rate: bool) -> None:
		msg = OffboardControlMode()
		msg.timestamp = self._now_us()
		msg.position = not body_rate
		msg.velocity = False
		msg.acceleration = False
		msg.attitude = False
		msg.body_rate = body_rate
		self._offboard_control_mode_pub.publish(msg)

	def _send_offboard_request(self) -> None:
		now_ns = self.get_clock().now().nanoseconds
		if now_ns - self._last_offboard_cmd_ns < self._command_interval_ns:
			return
		self._last_offboard_cmd_ns = now_ns
		self.get_logger().info("Requesting Offboard flight mode...")
		self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)

	def _send_arm_request(self) -> None:
		now_ns = self.get_clock().now().nanoseconds
		if now_ns - self._last_arm_cmd_ns < self._command_interval_ns:
			return
		self._last_arm_cmd_ns = now_ns
		self.get_logger().info("Requesting vehicle arming...")
		self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)

	def _publish_vehicle_command(self, command: int, param1: float = 0.0, param2: float = 0.0) -> None:
		msg = VehicleCommand()
		msg.timestamp = self._now_us()
		msg.command = int(command)
		msg.param1 = float(param1)
		msg.param2 = float(param2)
		msg.target_system = 1
		msg.target_component = 1
		msg.source_system = 1
		msg.source_component = 1
		msg.from_external = True
		self._vehicle_command_pub.publish(msg)

	def _start_privileged_agent(self) -> None:
		if self._executor is None:
			self.get_logger().error("Executor unavailable; cannot start privileged_agent")
			return
		self._privileged_agent = MpcRatesNode()
		self._executor.add_node(self._privileged_agent)
		self._agent_started = True
		self._acro_start_ns = self.get_clock().now().nanoseconds
		self._acro_duration_s = float(getattr(self._privileged_agent, "_traj_total_time", 0.0)) or None
		self.get_logger().info("privileged_agent started")

	def _stop_privileged_agent(self) -> None:
		if self._privileged_agent is None:
			return
		if self._executor is not None:
			self._executor.remove_node(self._privileged_agent)
		self._privileged_agent.destroy_node()
		self._privileged_agent = None
		self._agent_started = False
		self._acro_start_ns = None
		self._acro_duration_s = None
		self.get_logger().info("privileged_agent stopped")

	def _start_return_node(self) -> None:
		if self._executor is None:
			self.get_logger().error("Executor unavailable; cannot start return_node")
			return
		self._return_node = PX4ReturnNode()
		self._executor.add_node(self._return_node)
		self._return_started = True
		self.get_logger().info("return_node started")

	def _stop_return_node(self) -> None:
		if self._return_node is None:
			return
		if self._executor is not None:
			self._executor.remove_node(self._return_node)
		self._return_node.destroy_node()
		self._return_node = None
		self._return_started = False
		self.get_logger().info("return_node stopped")

	def _check_acro_completion(self) -> None:
		if self._acro_start_ns is None or self._acro_duration_s is None:
			return
		elapsed_s = (self.get_clock().now().nanoseconds - self._acro_start_ns) * 1e-9
		if elapsed_s < self._acro_duration_s:
			return
		self.get_logger().info("Acro trajectory complete; stopping privileged_agent")
		self.set_parameters(
			[Parameter("start_acro", Parameter.Type.BOOL, False)]
		)

	def _on_param_update(self, params):
		for param in params:
			if param.name == "start_acro":
				if param.type_ != param.Type.BOOL:
					return SetParametersResult(successful=False, reason="start_acro must be boolean")
				self._start_acro = bool(param.value)
		return SetParametersResult(successful=True)

	def destroy_children(self) -> None:
		if self._privileged_agent is not None:
			self._privileged_agent.destroy_node()
			self._privileged_agent = None
		if self._return_node is not None:
			self._return_node.destroy_node()
			self._return_node = None

	def _now_us(self) -> int:
		return int(self.get_clock().now().nanoseconds / 1000)


def main() -> None:
	rclpy.init()
	executor = MultiThreadedExecutor()
	node = PX4MainNode(executor)
	executor.add_node(node)
	try:
		executor.spin()
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_children()
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main()
