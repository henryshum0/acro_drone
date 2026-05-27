from __future__ import annotations

import rclpy
from px4_msgs.msg import TrajectorySetpoint
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, HistoryPolicy, ReliabilityPolicy


class PX4ReturnNode(Node):
	def __init__(self) -> None:
		super().__init__("px4_return_node")

		self._origin_altitude_m = float(self.declare_parameter("return_origin_altitude_m", 2.0).value)
		self.get_logger().info(f"Return setpoint height: {self._origin_altitude_m} m (NED z={-self._origin_altitude_m})")

		qos_profile = QoSProfile(
			reliability=ReliabilityPolicy.BEST_EFFORT,
			durability=DurabilityPolicy.TRANSIENT_LOCAL,
			history=HistoryPolicy.KEEP_LAST,
			depth=1,
		)

		self._trajectory_setpoint_pub = self.create_publisher(
			TrajectorySetpoint,
			"/fmu/in/trajectory_setpoint",
			qos_profile,
		)
		self.create_timer(0.05, self._timer_cb)

	def _timer_cb(self) -> None:
		self._publish_trajectory_setpoint()

	def _publish_trajectory_setpoint(self) -> None:
		msg = TrajectorySetpoint()
		msg.timestamp = self._now_us()
		msg.position = [0.0, 0.0, -self._origin_altitude_m]
		msg.yaw = 0.0
		self._trajectory_setpoint_pub.publish(msg)

	def _now_us(self) -> int:
		return int(self.get_clock().now().nanoseconds / 1000)


def main() -> None:
	rclpy.init()
	node = PX4ReturnNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main()
