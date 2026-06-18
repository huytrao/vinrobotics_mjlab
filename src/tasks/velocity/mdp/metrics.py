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

"""Raw locomotion metrics (unshaped) for tuning objectives.

Unlike rewards, these are physical quantities — m/s, rad, m — never passed
through exp/std/clip. They are accumulated per step by MetricsManager and
reported as ``Episode_Metrics/<name>`` (true per-step average over the
episode), which is what hyperparameter tuning should optimize.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from mjlab.entity import Entity
from mjlab.envs.mdp.metrics import mean_action_acc  # noqa: F401  re-export
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_WAIST_ROLL_CFG = SceneEntityCfg("robot", body_names=["waist_roll_link"])


def lin_vel_error_x(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Absolute forward velocity tracking error (m/s)."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    return torch.abs(command[:, 0] - asset.data.root_link_lin_vel_b[:, 0])


def lin_vel_error_y(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Absolute lateral velocity tracking error (m/s)."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    return torch.abs(command[:, 1] - asset.data.root_link_lin_vel_b[:, 1])


def ang_vel_error_z(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Absolute error between commanded and actual yaw rate (rad/s)."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    return torch.abs(command[:, 2] - asset.data.root_link_ang_vel_b[:, 2])


def imu_roll(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Absolute roll angle (rad) — IMU-style tilt around body x-axis.

    Derived from projected gravity in body frame:
        roll = atan2(g_y, -g_z)   (upright robot → g_b ≈ (0, 0, -1) → roll ≈ 0)
    Captures combined static lean + oscillation amplitude.
    """
    asset: Entity = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    return torch.abs(torch.atan2(g[:, 1], -g[:, 2]))


def imu_pitch(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Absolute pitch angle (rad) — IMU-style tilt around body y-axis.

    pitch = atan2(-g_x, -g_z).
    """
    asset: Entity = env.scene[asset_cfg.name]
    g = asset.data.projected_gravity_b
    return torch.abs(torch.atan2(-g[:, 0], -g[:, 2]))


def imu_roll_rate(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Absolute roll angular velocity (rad/s) — pure vibration around body x.

    This is what the IMU gyro outputs. Zero for static tilt, large for wobble.
    """
    asset: Entity = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_link_ang_vel_b[:, 0])


def imu_pitch_rate(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Absolute pitch angular velocity (rad/s) — pure vibration around body y."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.abs(asset.data.root_link_ang_vel_b[:, 1])


def base_height(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """World-frame base z height (m)."""
    asset: Entity = env.scene[asset_cfg.name]
    return asset.data.root_link_pos_w[:, 2]


def joint_power(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Mechanical power proxy Σ |τ · q̇| over actuated joints (W)."""
    asset: Entity = env.scene[asset_cfg.name]
    tau = asset.data.actuator_force
    qd = asset.data.joint_vel
    return torch.sum(torch.abs(tau * qd), dim=-1)


def alive(env: ManagerBasedRlEnv) -> torch.Tensor:
    """1.0 for envs still alive this step, 0.0 for terminated (non-timeout).

    Episode average ∈ [0, 1] — interpret as "fraction of episode the agent was
    not crashed". Higher is better.
    """
    return 1.0 - env.termination_manager.terminated.float()


def body_link_roll(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_WAIST_ROLL_CFG,
) -> torch.Tensor:
    """Absolute roll angle (rad) of waist_roll_link relative to world frame.

    Gravity is projected into the link frame; roll = atan2(g_y, -g_z).
    Upright link → g_link ≈ (0, 0, -1) → roll ≈ 0.
    """
    asset: Entity = env.scene[asset_cfg.name]
    body_names = asset_cfg.body_names if asset_cfg.body_names else ["waist_roll_link"]
    body_idx = asset.find_bodies(body_names)[0][0]
    link_quat_w = asset.data.body_link_quat_w[:, body_idx, :]  # [B, 4]
    g = quat_apply_inverse(link_quat_w, asset.data.gravity_vec_w)  # [B, 3]
    return torch.abs(torch.atan2(g[:, 1], -g[:, 2]))


def body_link_pitch(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_WAIST_ROLL_CFG,
) -> torch.Tensor:
    """Absolute pitch angle (rad) of waist_roll_link relative to world frame.

    Gravity is projected into the link frame; pitch = atan2(-g_x, -g_z).
    """
    asset: Entity = env.scene[asset_cfg.name]
    body_names = asset_cfg.body_names if asset_cfg.body_names else ["waist_roll_link"]
    body_idx = asset.find_bodies(body_names)[0][0]
    link_quat_w = asset.data.body_link_quat_w[:, body_idx, :]  # [B, 4]
    g = quat_apply_inverse(link_quat_w, asset.data.gravity_vec_w)  # [B, 3]
    return torch.abs(torch.atan2(-g[:, 0], -g[:, 2]))
