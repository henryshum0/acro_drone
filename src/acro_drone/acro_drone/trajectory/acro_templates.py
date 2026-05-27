import numpy as np

from .base_templates import *


def _offset_waypoints(waypoints_xyzs: np.ndarray) -> np.ndarray:
	waypoints_xyzs = np.asarray(waypoints_xyzs, dtype=float)
	return waypoints_xyzs - waypoints_xyzs[0]

class StraightLineTemplate(WaypointTemplate):
	def __init__(self):
		waypoints_xyzs = np.array([
			[0.0, 0.0, 1],
			[6.0, 0.0, 1],
		])
		waypoints_xyzs = _offset_waypoints(waypoints_xyzs)
		waypoints_psi = np.array([
			0,
			0,
		])
		waypoints_vels = [
			[0, 0, 0],
			[0, 0, 0],
		]
		waypoints_accs = [
			[0, 0, 0],
			[0, 0, 0],
		]
		waypoints_durations = np.array([
			1,
			1
		])


		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_psis=waypoints_psi,
			waypoints_vels=waypoints_vels,
			waypoints_durations=waypoints_durations,
			waypoints_accs=waypoints_accs,
		)
		self.waypoints_accs = waypoints_accs


class HeartTemplate(WaypointTemplate):

	def __init__(self):
		waypoints_xyzs = np.array([
			[-2, 0.0, 1],
			[0., 0.0, 1.],
			[2, 0, 3],
			[1, 0.0, 4],
			[-0.5, 0, 3],
			[0, 0., 2],
			[0.5, 0, 3],
			[-1., 0.0, 4],
			[-2., 0.0, 3],
			[0., 0.0, 1.],
		])
		waypoints_xyzs = _offset_waypoints(waypoints_xyzs)
		self.waypoints_psis = np.array([
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
			0,
		])
		waypoints_vels = [
			[0, 0, 0],
			None,
			None,
			None,
			None,
			None,
			None,
			None,
			None,
			[0, 0, 0],
		]
		waypoints_accs = [
			None,
			None,
			None,
			[0., 0., -10.],
			[10., 0., 0.],
			[0., 0., 10.],
			[-10., 0., 0.],
			[0., 0., -10.],
			None,
			None,
		]
		waypoints_durations = np.array([
			0.,
			1,
			1,
			1,
			1,
			1,
			1,
			1,
			1,
			1
		])

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_psis=self.waypoints_psis,
			waypoints_vels=waypoints_vels,
			waypoints_durations=waypoints_durations,
			waypoints_accs=waypoints_accs,
		)

class PowerloopTemplate(WaypointTemplate):
	"""Back-roll pair: same XY, first waypoint lower than second."""

	def __init__(self):
		waypoints_xyzs = np.array([
			[-2, 0.0, 1],
			[0., 0.0, 1.],
			[1, 0, 2],
			[0., 0.0, 3],
			[-1, 0., 2],
			[0., 0.0, 1],
			[2., 0.0, 1],
		])
		waypoints_xyzs = _offset_waypoints(waypoints_xyzs)
		waypoints_psis = np.array([
			0,
			0,
			0,
			0,
			0,
			0,
			0,
		])
		waypoints_vels = [
			[0, 0, 0],
			None,
			None,
			None,
			None,
			None,
			[0, 0, 0],
		]
		waypoints_accs = [
			[0, 0, 0],
			None,
			None,
			[0., 0., -10.],
			None,
			None,
			[0, 0, 0],
		]
		waypoints_durations = np.array([
			0.,
			0.5,
			0.5,
			0.5,
			0.5,
			0.5,
			0.5
		])

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_psis=waypoints_psis,
			waypoints_vels=waypoints_vels,
			waypoints_accs=waypoints_accs,
			waypoints_durations=waypoints_durations,
		)


class SplitSLeftTemplate(WaypointTemplate):
	"""Split-S: invert, descend, and exit in the opposite heading."""

	def __init__(self):
		waypoints_xyzs = np.array([
			[1.0, 0.0, 1.5],
			[2, -1, 1.5],
			[3., 1 , 1.5],
			[1, 0 , 0.5],
			
		])
		waypoints_xyzs = _offset_waypoints(waypoints_xyzs)
		waypoints_vels = [
			[0, 0, 0],
			None,
			None,
			[0, 0, 0],
		]
		waypoints_accs = [None, None, None, None]
		waypoints_durations = np.array([
			1.00,
			1.10,
			0.90,
			1.10,
			
		])
		waypoints_psis = np.array([
			0,
			0,
			0,
			0,	
		])

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_vels=waypoints_vels,
			waypoints_accs=waypoints_accs,
			waypoints_durations=waypoints_durations,
			waypoints_psis=waypoints_psis,
		)
		
class SplitSRightTemplate(WaypointTemplate):
	"""Split-S: invert, descend, and exit in the opposite heading."""

	def __init__(self):
		waypoints_xyzs = np.array([
			[1.0, 0.0, 1.5],
			[2, 1, 1.5],
			[3., -1 , 1.5],
			[1, 0 , 0.5],
			
		])
		waypoints_xyzs = _offset_waypoints(waypoints_xyzs)
		waypoints_vels = [
			[0, 0, 0],
			None,
			None,
			[0, 0, 0],
			
		]
		waypoints_durations = np.array([
			1.00,
			1.10,
			0.90,
			1.10,
			
		])
		waypoints_psis = np.array([
			0,
			0,
			0,
			0,
		])

		waypoints_accs = [None, None, None, None]

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_psis=waypoints_psis,
			waypoints_vels=waypoints_vels,
			waypoints_accs=waypoints_accs,
			waypoints_durations=waypoints_durations,
		)

class BarrelRollLeftTemplate(WaypointTemplate):
	"""Barrel roll progression with forward travels through the roll."""

	def __init__(self):
		waypoints_xyzs = np.array([
			[0.0, 0., 1],
			[1.0, 0., 1],
			[2, 1., 2.5],
			[2.5, 2., 1],
			[5, 2., 1],

		])
		waypoints_xyzs = _offset_waypoints(waypoints_xyzs)
		waypoints_vels = [
			[0, 0, 0],
			None,
			None,
			None,
			[0, 0, 0],
		]
		waypoints_durations = np.array([
			1.00,
			1.00,
			0.5,
			1,
			1.0
		])
		waypoints_accs = [
			None,
			None,
			[0., 0., -10.],
			None,
			None,
		]
		waypoints_psis = np.array([
			0,
			0,
			0,
			0,
			0,
		])

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_psis=waypoints_psis,
			waypoints_vels=waypoints_vels,
			waypoints_accs=waypoints_accs,
			waypoints_durations=waypoints_durations,
		)

class BarrelRollRightTemplate(WaypointTemplate):
	"""Barrel roll progression with forward travels through the roll."""

	def __init__(self):
		waypoints_xyzs = np.array([
			[0.0, 0., 1],
			[1.0, 0., 1],
			[2, -1., 2.5],
			[2.5, -2., 1],
			[5, -2., 1],

		])
		waypoints_xyzs = _offset_waypoints(waypoints_xyzs)
		waypoints_vels = [
			[0, 0, 0],
			None,
			None,
			None,
			[0, 0, 0],
		]
		waypoints_durations = np.array([
			1.00,
			1.00,
			0.5,
			1,
			1.0
		])
		waypoints_accs = [
			None,
			None,
			[0., 0., -10.],
			None,
			None,
		]
		waypoints_psis = np.array([
			0,
			0,
			0,
			0,
			0,
		])

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_psis=waypoints_psis,
			waypoints_vels=waypoints_vels,
			waypoints_accs=waypoints_accs,
			waypoints_durations=waypoints_durations,
		)