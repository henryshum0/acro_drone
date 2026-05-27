import numpy as np
from transforms3d.euler import euler2quat

RPY_FRONT_UP = np.array([0, 0, 0])
RPY_FRONT_DOWN = np.array([np.pi, 0, 0])
RPY_FRONT_LEFT = np.array([-np.pi/2, 0, 0])
RPY_FRONT_RIGHT = np.array([np.pi/2, 0, 0])
RPY_BACK_UP = np.array([0, 0, np.pi])
RPY_BACK_DOWN = np.array([np.pi, 0, np.pi])
RPY_BACK_LEFT = np.array([np.pi/2, 0, np.pi])
RPY_BACK_RIGHT = np.array([-np.pi/2, 0, np.pi])
RPY_LEFT_UP = np.array([0, 0, np.pi/2])
RPY_LEFT_DOWN = np.array([np.pi, 0, np.pi/2])
RPY_LEFT_FRONT = np.array([np.pi/2, 0, np.pi/2])
RPY_LEFT_BACK = np.array([-np.pi/2, 0, np.pi/2])
RPY_RIGHT_UP = np.array([0, 0, -np.pi/2])
RPY_RIGHT_DOWN = np.array([np.pi, 0, -np.pi/2])
RPY_RIGHT_FRONT = np.array([-np.pi/2, 0, -np.pi/2])
RPY_RIGHT_BACK = np.array([np.pi/2, 0, -np.pi/2])
RPY_DOWN_FRONT = np.array([0, np.pi/2, 0])
RPY_DOWN_BACK = np.array([np.pi, np.pi/2, 0])
RPY_DOWN_LEFT = np.array([-np.pi/2, np.pi/2, 0])
RPY_DOWN_RIGHT = np.array([np.pi/2, np.pi/2, 0])

class WaypointTemplate():
    def __init__(
    self, 
    waypoints_xyzs,
    waypoints_psis,
    waypoints_vels,
    waypoints_accs,
    waypoints_durations,
    ):
        self.waypoints_xyzs = waypoints_xyzs
        self.waypoints_psis = waypoints_psis
        self.waypoints_vels = waypoints_vels
        self.waypoints_accs = waypoints_accs
        self.waypoints_durations = waypoints_durations
    

