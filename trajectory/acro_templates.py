import numpy as np

from .base_templates import *


def _default_spawn():
	return [
		{
			"pos": np.array([0, 0.0, 0.0]),
			"vels": np.array([0.0, 0.0, 0.0]),
			"acc": np.array([0.0, 0.0, 0.0]),
			"rpy": np.array([0.0, 0.0, 0.0]),
			"next_waypoints": [0, 1],
		}
	]


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
		waypoints_rpys = np.array([
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],

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

		waypoints_rpys_choices = waypoints_rpys
		waypoints_scale = [2,2]

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_rpys=waypoints_rpys,
			waypoints_vels=waypoints_vels,
			waypoints_durations=waypoints_durations,
			waypoints_accs=waypoints_accs,
			spawns=_default_spawn(),
			rpy_choices=waypoints_rpys_choices,
			waypoints_scale=waypoints_scale,
			max_dist=7,
			difficulty="easy",
			repeat=0,
			time_limit_sec=5,
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
		waypoints_rpys = np.array([
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],
			[0, np.pi, 0],
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],
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

		waypoints_rpys_choices = waypoints_rpys
		waypoints_scale = [2,2]

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_rpys=waypoints_rpys,
			waypoints_vels=waypoints_vels,
			waypoints_durations=waypoints_durations,
			waypoints_accs=waypoints_accs,
			spawns=_default_spawn(),
			rpy_choices=waypoints_rpys_choices,
			waypoints_scale=waypoints_scale,
			max_dist=7,
			difficulty="easy",
			repeat=0,
			time_limit_sec=5,
		)
		self.waypoints_accs = waypoints_accs

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
		waypoints_rpys = np.array([
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],
			[0, np.pi, 0],
			[0.0, 0.0, 0],
			[0.0, 0.0, 0],
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
			None,
			None,
			None,
			[0., 0., -10.],
			None,
			None,
			None,
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

		waypoints_rpys_choices = waypoints_rpys
		waypoints_scale = [2,2]

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_rpys=waypoints_rpys,
			waypoints_vels=waypoints_vels,
			waypoints_accs=waypoints_accs,
			waypoints_durations=waypoints_durations,
			spawns=_default_spawn(),
			rpy_choices=waypoints_rpys_choices,
			waypoints_scale=waypoints_scale,
			max_dist=7,
			difficulty="easy",
			repeat=0,
			time_limit_sec=5,
		)
		self.waypoints_accs = waypoints_accs


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
		waypoints_rpys = np.array([
			RPY_FRONT_UP,
			RPY_FRONT_UP,
			RPY_LEFT_BACK,
			RPY_BACK_UP,
			
		])


		waypoints_rpys_choices = waypoints_rpys
		waypoints_scale = [1,2]

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_rpys=waypoints_rpys,
			waypoints_vels=waypoints_vels,
			waypoints_accs=waypoints_accs,
			waypoints_durations=waypoints_durations,
			spawns=_default_spawn(),
			rpy_choices=waypoints_rpys_choices,
			waypoints_scale=waypoints_scale,
			max_dist=7,
			difficulty="easy",
			repeat=0,
			time_limit_sec=7,
		)
		self.waypoints_accs = waypoints_accs
		
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
		waypoints_rpys = np.array([
			RPY_FRONT_UP,
			RPY_FRONT_UP,
			RPY_RIGHT_BACK,
			RPY_BACK_UP,
			
		])



		waypoints_accs = [None, None, None, None]
		waypoints_rpys_choices = waypoints_rpys
		waypoints_scale = [2, 3]

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_rpys=waypoints_rpys,
			waypoints_vels=waypoints_vels,
			waypoints_accs=waypoints_accs,
			waypoints_durations=waypoints_durations,
			spawns=_default_spawn(),
			rpy_choices=waypoints_rpys_choices,
			waypoints_scale=waypoints_scale,
			max_dist=7,
			difficulty="easy",
			repeat=0,
			time_limit_sec=7,
		)
		self.waypoints_accs = waypoints_accs

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
		waypoints_rpys = np.array([
			RPY_FRONT_UP,
			RPY_FRONT_UP,
			RPY_FRONT_DOWN,
			RPY_FRONT_UP,
			RPY_FRONT_UP,

		])

		waypoints_rpys_choices = waypoints_rpys
		waypoints_scale = [1,2]

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_rpys=waypoints_rpys,
			waypoints_vels=waypoints_vels,
			waypoints_accs=waypoints_accs,
			waypoints_durations=waypoints_durations,
			spawns=_default_spawn(),
			rpy_choices=waypoints_rpys_choices,
			waypoints_scale=waypoints_scale,
			max_dist=7,
			difficulty="easy",
			repeat=0,
			time_limit_sec=7,
		)
		self.waypoints_accs = waypoints_accs

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
		waypoints_rpys = np.array([
			RPY_FRONT_UP,
			RPY_FRONT_UP,
			RPY_FRONT_DOWN,
			RPY_FRONT_UP,
			RPY_FRONT_UP,

		])

		waypoints_rpys_choices = waypoints_rpys
		waypoints_scale = [1,2]

		super().__init__(
			waypoints_xyzs=waypoints_xyzs,
			waypoints_rpys=waypoints_rpys,
			waypoints_vels=waypoints_vels,
			waypoints_accs=waypoints_accs,
			waypoints_durations=waypoints_durations,
			spawns=_default_spawn(),
			rpy_choices=waypoints_rpys_choices,
			waypoints_scale=waypoints_scale,
			max_dist=7,
			difficulty="easy",
			repeat=0,
			time_limit_sec=7,
		)
		self.waypoints_accs = waypoints_accs
		



TRAIN_TEMPLATES2 = [
	PowerloopTemplate,
	SplitSLeftTemplate,
	SplitSRightTemplate,
	BarrelRollRightTemplate,
	BarrelRollLeftTemplate,
]


def visualize_all_templates(randomized=False, show_orientation=True, cols=3):
	"""Visualize all templates in TRAIN_TEMPLATES2 in a single figure grid."""
	import math
	import matplotlib.pyplot as plt

	total = len(TRAIN_TEMPLATES2)
	cols = max(1, int(cols))
	rows = int(math.ceil(total / cols))
	fig = plt.figure(figsize=(5 * cols, 4.5 * rows))

	for i, template_cls in enumerate(TRAIN_TEMPLATES2):
		ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
		template = template_cls()
		template.visualize_waypoints(
			randomized=randomized,
			show_orientation=show_orientation,
			show_spawn=True,
			show=False,
			ax=ax,
			title=template_cls.__name__,
		)

	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	visualize_all_templates(randomized=True, show_orientation=True, cols=3)