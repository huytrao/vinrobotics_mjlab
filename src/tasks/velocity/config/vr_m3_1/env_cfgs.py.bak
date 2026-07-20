# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""VinRobotics VR M3 1 velocity environment configurations.

Self-contained: this module builds the full `ManagerBasedRlEnvCfg` inline
rather than deriving from `src.tasks.velocity.velocity_env_cfg`, so changes
to the shared factory do not silently affect VR M3 1 training.
"""

import math
from dataclasses import dataclass, replace
from mjlab.envs import ManagerBasedRlEnvCfg, mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig
import src.tasks.velocity.mdp as mdp
from src.assets.robots import VR_M3_1_ACTION_SCALE, get_vr_m3_1_implicit_actuator_robot_cfg
from src.tasks.velocity.terrains.config import TERRAINS_CFG

##
# VR M3 1 specific constants
##

ROOT_BODY = "pelvis"
TORSO_BODY = "waist_yaw_link"

FOOT_SITES = ("left_foot", "right_foot")
FOOT_LINK_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")
# Same foot geometry as the fullbody XML: 15 collision geoms per ankle_roll_link.
FOOT_GEOMS = tuple(
    f"{side}_ankle_roll_link_collision_{i}"
    for side in ("left", "right")
    for i in range(1, 9)
)
# Pose reward stds — only has leg joints. Hip roll/yaw stay tight to
# approximate Isaac's `joint_deviation_l1` on those joints (which keeps the
# legs aligned forward).
POSE_STD_STANDING: dict[str, float] = {".*": 0.05}
POSE_STD_WALKING: dict[str, float] = {
    # Lower body.
    r".*hip_pitch.*": 0.5,
    r".*hip_roll.*": 0.5,
    r".*hip_yaw.*": 0.1,
    r".*knee.*": 0.5,
    r".*ankle_pitch.*": 0.5,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.15,
    # Arms.
    r".*shoulder_pitch.*": 0.1,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
}
POSE_STD_RUNNING: dict[str, float] = {
    r".*hip_pitch.*": 1.0,
    r".*hip_roll.*": 1.0,
    r".*hip_yaw.*": 0.1,
    r".*knee.*": 1.0,
    r".*ankle_pitch.*": 1.0,
    r".*ankle_roll.*": 0.1,
    # Waist.
    r".*waist_yaw.*": 0.25,
    # Arms.
    r".*shoulder_pitch.*": 0.1,
    r".*shoulder_roll.*": 0.1,
    r".*shoulder_yaw.*": 0.1,
    r".*elbow.*": 0.1,
    r".*wrist.*": 0.1,
}
GAIT_CYCLE_PERIOD = 0.9


@dataclass
class VelocityEnvCfg(ManagerBasedRlEnvCfg):

    def __post_init__(self):
        pass


def vr_m3_1_rough_env_cfg(play: bool = False) -> VelocityEnvCfg:
    """Create VR M3 1 terrain velocity configuration."""
    ##
    # Sensors
    ##
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
        history_length=4,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    ##
    # Observations
    ##

    actor_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            scale=0.25,
            params={"sensor_name": "robot/imu_ang_vel"},
            noise=Unoise(n_min=-0.25, n_max=0.25),
            history_length=5,
            flatten_history_dim=True,
        ),
        "projected_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=5,
            flatten_history_dim=True,
        ),
        "velocity_commands": ObservationTermCfg(
            func=mdp.generated_commands,
            params={"command_name": "twist"},
            history_length=5,
            flatten_history_dim=True,
        ),
        "joint_pos_rel": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=5,
            flatten_history_dim=True,
        ),
        "joint_vel_rel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=5,
            flatten_history_dim=True,
        ),
        "last_action": ObservationTermCfg(
            func=mdp.last_action, history_length=5, flatten_history_dim=True
        ),
        "gait_phase": ObservationTermCfg(
            func=mdp.phase,
            params={"period": GAIT_CYCLE_PERIOD, "command_name": "twist"},
            history_length=5,
            flatten_history_dim=True,
        ),
    }

    critic_terms = {
        **actor_terms,
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
            history_length=5,
            flatten_history_dim=True,
        ),
        "true_base_velocity": ObservationTermCfg(
            func=envs_mdp.base_lin_vel,
            history_length=5,
            flatten_history_dim=True,
        ),
        "foot_height": ObservationTermCfg(
            func=mdp.foot_height,
            params={"asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES)},
            history_length=5,
            flatten_history_dim=True,
        ),
        "foot_air_time": ObservationTermCfg(
            func=mdp.foot_air_time,
            params={"sensor_name": "feet_ground_contact"},
            history_length=5,
            flatten_history_dim=True,
        ),
        "foot_contact": ObservationTermCfg(
            func=mdp.foot_contact,
            params={"sensor_name": "feet_ground_contact"},
            history_length=5,
            flatten_history_dim=True,
        ),
        "foot_contact_forces": ObservationTermCfg(
            func=mdp.foot_contact_forces,
            params={"sensor_name": "feet_ground_contact"},
            history_length=5,
            flatten_history_dim=True,
        ),
        "knee_reference": ObservationTermCfg(
            func=mdp.get_desired_knee_angle,
            params={
                "period": GAIT_CYCLE_PERIOD,
                "offset": [0.0, 0.5],
                "command_name": "twist",
            },
            history_length=5,
            flatten_history_dim=True,
        ),
    }

    observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
            nan_policy="sanitize",
            nan_check_per_term=True,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
            nan_policy="sanitize",
            nan_check_per_term=True,
        ),
    }

    ##
    # Metrics
    ##

    metrics = {
        "lin_vel_error_x": MetricsTermCfg(
            func=mdp.lin_vel_error_x,
            params={"command_name": "twist"},
        ),
        "lin_vel_error_y": MetricsTermCfg(
            func=mdp.lin_vel_error_y,
            params={"command_name": "twist"},
        ),
        "ang_vel_error_z": MetricsTermCfg(
            func=mdp.ang_vel_error_z,
            params={"command_name": "twist"},
        ),
        "waist_roll": MetricsTermCfg(
            func=mdp.body_link_roll,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[TORSO_BODY])},
        ),
        "waist_pitch": MetricsTermCfg(
            func=mdp.body_link_pitch,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[TORSO_BODY])},
        ),
    }

    ##
    # Actions
    ##

    actions: dict[str, ActionTermCfg] = {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=VR_M3_1_ACTION_SCALE,
            use_default_offset=True,
        )
    }

    ##
    # Commands
    ##

    commands: dict[str, CommandTermCfg] = {
        "twist": UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(3.0, 8.0),
            rel_standing_envs=0.1,
            heading_command=False,
            heading_control_stiffness=0.5,
            debug_vis=True,
            ranges=UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 2.0),
                lin_vel_y=(-0.5, 0.5),
                ang_vel_z=(-1.0, 1.0),
                heading=None,
            ),
        ),
    }
    twist_cmd = commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 1.15

    ##
    # Events
    ##

    events = {
        "reset_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.0, 0.0),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.1, 0.1),
                "velocity_range": (-0.5, 0.5),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(5.0, 6.0),
            params={
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.4, 0.4),
                    "roll": (-0.52, 0.52),
                    "pitch": (-0.52, 0.52),
                    "yaw": (-0.78, 0.78),
                },
            },
        ),
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS),
                "operation": "abs",
                "ranges": (0.3, 1.6),
                "shared_random": True,
            },
        ),
        "encoder_bias": EventTermCfg(
            mode="startup",
            func=dr.encoder_bias,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "bias_range": (-0.015, 0.015),
            },
        ),
        "base_com": EventTermCfg(
            mode="startup",
            func=dr.body_com_offset,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(TORSO_BODY,)),
                "operation": "add",
                "ranges": {
                    0: (-0.05, 0.05),
                    1: (-0.05, 0.05),
                    2: (-0.05, 0.05),
                },
            },
        ),
        "body_com": EventTermCfg(
            mode="startup",
            func=dr.body_ipos,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=(f"^(?!{ROOT_BODY}$).*",)
                ),
                "operation": "add",
                "ranges": {
                    0: (-0.03, 0.03),
                    1: (-0.03, 0.03),
                    2: (-0.03, 0.03),
                },
            },
        ),
        "base_mass": EventTermCfg(
            mode="startup",
            func=dr.body_mass,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(TORSO_BODY)),
                "operation": "add",
                "ranges": (-4.0, 4.0),
            },
        ),
        "body_mass": EventTermCfg(
            mode="startup",
            func=dr.body_mass,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=(f"^(?!{ROOT_BODY}$).*",)
                ),
                "operation": "scale",
                "ranges": (0.9, 1.1),
            },
        ),
        "randomize_actuator_gains": EventTermCfg(
            func=dr.pd_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "operation": "scale",
                "kp_range": (0.7, 1.1),
                "kd_range": (0.7, 1.1),
            },
        ),
        "randomize_joint_limit": EventTermCfg(
            mode="startup",
            func=dr.joint_limits,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "operation": "scale",
                "ranges": (0.94, 1.06),
            },
        ),
        "randomize_joint_stiffness": EventTermCfg(
            mode="startup",
            func=dr.joint_stiffness,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "operation": "scale",
                "ranges": (0.7, 1.3),
            },
        ),
        "randomize_joint_frictionloss": EventTermCfg(
            mode="startup",
            func=dr.joint_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "operation": "scale",
                "ranges": (0.1, 1.3),
            },
        ),
        "randomize_joint_armature": EventTermCfg(
            mode="startup",
            func=dr.joint_armature,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "operation": "scale",
                "ranges": (0.2, 5.0),
            },
        ),
        "randomize_joint_damping": EventTermCfg(
            mode="startup",
            func=dr.joint_damping,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                "operation": "scale",
                "ranges": (0.7, 1.3),
            },
        ),
    }

    ##
    # Rewards
    ##

    rewards = {
        "track_linear_x": RewardTermCfg(
            func=mdp.track_linear_x,
            weight=2.0,
            params={
                "command_name": "twist",
                "std_list": [0.5, 0.25],
                "vel_list": [0.0, 0.1, 2.0],
                "add_penalize": True,
                "penalize_scale": 1.0,
            },
        ),
        "track_linear_y": RewardTermCfg(
            func=mdp.track_linear_y,
            weight=2.0,
            params={
                "command_name": "twist",
                "std_list": [0.5, 0.25],
                "vel_list": [0.0, 0.1, 2.0],
                "add_penalize": True,
                "penalize_scale": 1.0,
            },
        ),
        "track_angular_z": RewardTermCfg(
            func=mdp.track_angular_z,
            weight=2.0,
            params={
                "command_name": "twist",
                "std_list": [0.5, 0.25],
                "vel_list": [0.0, 0.1, 2.0],
                "add_penalize": True,
                "penalize_scale": 1.0,
            },
        ),
        "linear_vel_z": RewardTermCfg(
            func=mdp.lin_vel_z_l2,
            weight=-2.0,
        ),
        "angular_vel_xy": RewardTermCfg(
            func=mdp.ang_vel_xy_l2,
            weight=-0.05,
        ),
        "flat_orientation": RewardTermCfg(func=mdp.flat_orientation_l2, weight=-5.0),
        "link_orientation": RewardTermCfg(
            func=mdp.link_orientation,
            weight=-5.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=(TORSO_BODY,))},
        ),
        "pelvis_roll_pitch": RewardTermCfg(
            func=mdp.body_roll_pitch_penalty,
            weight=-1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=(ROOT_BODY,)),
                "weight_list": [100.0],
                "velocity_list": [0.0, 2.0],
                "command_name": "twist",
            },
        ),
        "pose": RewardTermCfg(
            func=mdp.variable_posture,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "command_name": "twist",
                "std_standing": POSE_STD_STANDING,
                "std_walking": POSE_STD_WALKING,
                "std_running": POSE_STD_RUNNING,
                "walking_threshold": 0.1,
                "running_threshold": 1.5,
            },
        ),
        "feet_air_time_biped": RewardTermCfg(
            func=mdp.feet_air_time_positive_biped,
            weight=1.5,
            params={
                "command_name": "twist",
                "sensor_name": "feet_ground_contact",
                "threshold": 0.52,
            },
        ),
        "foot_clearance": RewardTermCfg(
            func=mdp.feet_clearance,
            weight=-1.0,
            params={
                "target_height": 0.10,
                "command_name": "twist",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
            },
        ),
        "foot_slip": RewardTermCfg(
            func=mdp.feet_slip,
            weight=-0.25,
            params={
                "sensor_name": "feet_ground_contact",
                "command_name": "twist",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES),
            },
        ),
        "feet_close_xy": RewardTermCfg(
            func=mdp.feet_close_xy_gauss,
            weight=1.0,
            params={
                "threshold": 0.18,
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
                ),
                "std": math.sqrt(0.05),
            },
        ),
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.5),
        "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
        "dof_torques_l2": RewardTermCfg(
            func=mdp.joint_torques_l2,
            weight=-1.0e-7,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"),
                )
            },
        ),
        "body_ang_vel": RewardTermCfg(
            func=mdp.body_angular_velocity_penalty,
            weight=-0.1,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=(TORSO_BODY,))},
        ),
        "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-100.0),
        "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-5.0),
        "self_collisions": RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-1.0,
            params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
        ),
        "stand_still": RewardTermCfg(
            func=mdp.stand_still,
            weight=-1.0,
            params={
                "command_name": "twist",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            },
        ),
        "dont_wait": RewardTermCfg(
            func=mdp.dont_wait,
            weight=-0.5,
            params={
                "command_name": "twist",
                "cmd_threshold": 0.1,
            },
        ),
        "knee_motion": RewardTermCfg(
            func=mdp.knee_joint_motion,
            weight=0.5,
            params={
                "reward_limit": 1.0,
                "command_name": "twist",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*knee.*"]),
            },
        ),
    }
    ##
    # Terminations
    ##

    terminations = {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
        "root_height": TerminationTermCfg(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.4},
        ),
        "nan_term": TerminationTermCfg(
            func=mdp.nan_detection,
            time_out=False,
        ),
    }

    ##
    # Curriculum — 4-stage velocity ramp to 2 m/s forward.
    #
    # At num_steps_per_env=24 and max_iterations≈10000, total env-steps ≈ 240k.
    # Stage triggers are expressed in env-steps (step * num_steps_per_env).
    ##

    curriculum = {
        "terrain_levels": CurriculumTermCfg(
            func=mdp.terrain_levels_vel,
            params={"command_name": "twist"},
        ),
        "command_vel": CurriculumTermCfg(
            func=mdp.commands_vel,
            params={
                "command_name": "twist",
                "velocity_stages": [
                    {
                        "step": 0,
                        "lin_vel_x": (-0.3, 0.5),
                        "lin_vel_y": (-0.3, 0.3),
                        "ang_vel_z": (-0.5, 0.5),
                    },
                    {
                        "step": 5000 * 32,
                        "lin_vel_x": (-0.5, 1.0),
                        "lin_vel_y": (-0.5, 0.5),
                        "ang_vel_z": (-0.8, 0.8),
                    },
                    {
                        "step": 10000 * 32,
                        "lin_vel_x": (-1.0, 1.5),
                        "lin_vel_y": (-0.5, 0.5),
                        "ang_vel_z": (-1.0, 1.0),
                    },
                    {
                        "step": 15000 * 32,
                        "lin_vel_x": (-1.0, 2.0),
                        "lin_vel_y": (-0.5, 0.5),
                        "ang_vel_z": (-1.0, 1.0),
                    },
                ],
            },
        ),
    }

    ##
    # Scene + terrain
    ##

    terrain_generator = replace(TERRAINS_CFG)
    terrain_generator.curriculum = True
    terrain = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=terrain_generator,
        max_init_terrain_level=5,
    )

    scene = SceneCfg(
        terrain=terrain,
        entities={"robot": get_vr_m3_1_implicit_actuator_robot_cfg()},
        sensors=(feet_ground_cfg, self_collision_cfg),
        num_envs=1,
        extent=2.0,
    )

    ##
    # Assemble
    ##

    cfg = VelocityEnvCfg(
        scene=scene,
        observations=observations,
        actions=actions,
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        curriculum=curriculum,
        metrics=metrics,
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="robot",
            body_name=TORSO_BODY,
            distance=3.0,
            elevation=-5.0,
            azimuth=90.0,
        ),
        sim=SimulationCfg(
            nconmax=256,
            njmax=700,
            contact_sensor_maxmatch=500,
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                ccd_iterations=50,
            ),
        ),
        decimation=4,
        episode_length_s=20.0,
    )

    # Play mode overrides.
    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )
        assert cfg.scene.terrain is not None
        if cfg.scene.terrain.terrain_generator is not None:
            cfg.scene.terrain.terrain_generator.curriculum = False
            cfg.scene.terrain.terrain_generator.num_cols = 10
            cfg.scene.terrain.terrain_generator.num_rows = 10
            cfg.scene.terrain.terrain_generator.border_width = 10.0

        # Play at the full learned range so we can probe 2 m/s behavior.
        play_twist = cfg.commands["twist"]
        assert isinstance(play_twist, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (0.0, 2.0)
        twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
        twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

    return cfg


def vr_m3_1_flat_env_cfg(play: bool = False) -> VelocityEnvCfg:
    """Create VR M3.1 flat terrain velocity configuration."""
    cfg = vr_m3_1_rough_env_cfg(play=play)
    cfg.sim.njmax = 700
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 256

    # Switch to flat terrain.
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    cfg.curriculum.pop("terrain_levels", None)

    if play:
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (0.0, 2.0)
        twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
        twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

    return cfg
