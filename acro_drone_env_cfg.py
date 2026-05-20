# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_contrib.actuators import ThrusterCfg
from isaaclab_contrib.assets import MultirotorCfg

from .controllers.ctbr_pd_cfg import CtbrPdControllerCfg
from .controllers.mpc import MpcCfg
from .sensors.noisy_imu_cfg import NoisyImuCfg
from .trajectory.trj_interface_cfg import TrajectoryInterfaceCfg
from .trajectory.acro_templates import *

@configclass
class RacerCfg(MultirotorCfg):
    """Multirotor configuration for the acro drone racer."""

    prim_path: str = "/World/envs/env_.*/racer"

    spawn: sim_utils.UrdfFileCfg = sim_utils.UrdfFileCfg(
        asset_path=str(Path(__file__).resolve().parent / "assets" / "racer.urdf"),
        fix_base=False,
        merge_fixed_joints=False,
        make_instanceable=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            metallic=0.0,
            roughness=0.5,
            diffuse_color=(1.0, 0.1, 0.1),
        ),
    )

    init_state: MultirotorCfg.InitialStateCfg = MultirotorCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        rps={".*": 200.0},
    )

    actuators: dict[str, ThrusterCfg] = {
        "thrusters": ThrusterCfg(
            dt=1 / 300,
            max_thrust_rate=1000000.0,
            thrust_range=(0.0, 20.0),
            thrust_const_range=(2.e-8, 3e-8),
            tau_inc_range=(0.00, 0.0),
            tau_dec_range=(0.00, 0.0),
            torque_to_thrust_ratio=0.0,
            thruster_names_expr=["prop0_link", "prop1_link", "prop2_link", "prop3_link"],
        )
    }

    allocation_matrix = [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0675, 0.0675, -0.0675, -0.0675],
        [-0.085, 0.085, 0.085, -0.085],
        [0.013, -0.013, 0.013, -0.013],
    ]

    rotor_directions = [1, -1, 1, -1]


@configclass
class AcroDroneEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 1
    episode_length_s = 5.0
    sim_rate_hz = 300.0
    mpc_rate_hz = 100.0
    pd_rate_hz = 300.0
    camera_rate_hz = 30.0
    imu_rate_hz = mpc_rate_hz * 2.0
    time_horizon_s = 1.0
    # - spaces definition
    action_space = 4
    observation_space = 6
    state_space = 0

    templates = [
        # HeartTemplate(), 
        # PowerloopTemplate(), 
        # SplitSLeftTemplate(),
        # SplitSRightTemplate(),
        BarrelRollLeftTemplate(),
        # BarrelRollRightTemplate(),    
        # StraightLineTemplate(),
    ]

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / sim_rate_hz, render_interval=decimation)

    # robot(s)
    robot_cfg: MultirotorCfg = RacerCfg()
    robot_cfg.actuators["thrusters"].dt = 1 / sim_rate_hz

    # controller
    controller_cfg: CtbrPdControllerCfg = CtbrPdControllerCfg(
        dt=1 / pd_rate_hz,
        kp=(0.3, 0.3, 0.1),
        kd=(0.0, 0.0, 0.0),
        allocation_matrix=(
            (1.0, 1.0, 1.0, 1.0),
            (0.0675, 0.0675, -0.0675, -0.0675),
            (-0.085, 0.085, 0.085, -0.085),
            (0.013, -0.013, 0.013, -0.013),
        ),
        thrust_min=0.0,
        thrust_max=20.0,
    )

    # MPC controller
    mpc_cfg = MpcCfg(
        horizon=10,
        solver_max_iter=100,
        solver_tol=1e-3,
        horizon_dt=0.05,
        dt=1/mpc_rate_hz,
        w_pos=10.0,
        w_quat=20.0,
        w_vel=2,
        w_output=0.0,
        w_output_derivative=0.1,
        max_normalized_thrust=60.0,
        max_roll_pitch_rate=20.0,
        max_yaw_rate=10.0,
    )

    # trajectory reference for mpc
    trj_interface_cfg = TrajectoryInterfaceCfg(
        sampling_rate=300,
        time_penalty=100,
        max_velocity=20,
        max_normalized_thrust=50,
    )

    # camera
    fisheye_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path=f"{robot_cfg.prim_path}/base_link/FisheyeCamera",
        update_period=1 / camera_rate_hz,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.1, 0.0, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.FisheyeCameraCfg(
            projection_type="fisheyePolynomial",
            focal_length=5.0,
            focus_distance=400.0,
            clipping_range=(0.1, 50.0),
            horizontal_aperture=20.955,
        ),
        width=160,
        height=120,
    )

    noisy_imu: NoisyImuCfg = NoisyImuCfg(
        prim_path=f"{robot_cfg.prim_path}/base_link",
        update_period=1 / imu_rate_hz,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=20.0, replicate_physics=True)

    # custom parameters/scales
    # - controllable joint
    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"
    # - action scale
    action_scale = (60, 5*torch.pi, 5*torch.pi, 2*torch.pi)
    # - thrust normalization
    gravity = 9.81
    total_mass = 0.86
    # - reward scales
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005
    # - reset states/conditions
    initial_pole_angle_range = [-0.25, 0.25]  # pole angle sample range on reset [rad]
    max_cart_pos = 3.0  # reset if cart exceeds this position [m]