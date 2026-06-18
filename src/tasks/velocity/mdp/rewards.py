# This file contains code adapted from https://github.com/unitreerobotics/unitree_rl_mjlab
# Original Project Copyright 2026 Unitree
# Original Project License: Apache License 2.0
#
# --------------------------------------------------------------------------
# Modifications Copyright 2026 VinRobotics
#
# This file has been modified. Changes and additions are licensed under  Apache 2.0

"""Reward terms for the velocity task MDP."""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.sensor.raycast_sensor import RayCastSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import resolve_matching_names_values

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for tracking the commanded base linear velocity.

    The commanded z velocity is assumed to be zero.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_lin_vel_b
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    lin_vel_error = xy_error + (2 * z_error)
    return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward heading error for heading-controlled envs, angular velocity for others.

    The commanded xy angular velocities are assumed to be zero.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_ang_vel_b
    z_error = torch.square(command[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    ang_vel_error = z_error + (0.05 * xy_error)
    return torch.exp(-ang_vel_error / std**2)


def body_orientation_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward flat base orientation (robot being upright).

    If asset_cfg has body_ids specified, computes the projected gravity
    for that specific body. Otherwise, uses the root link projected gravity.
    """
    asset: Entity = env.scene[asset_cfg.name]

    # If body_ids are specified, compute projected gravity for that body.
    if asset_cfg.body_ids:
        body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
        body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
        gravity_w = asset.data.gravity_vec_w  # [3]
        projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
        xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    else:
        # Use root link projected gravity.
        xy_squared = torch.sum(
            torch.square(asset.data.projected_gravity_b[:, :2]), dim=1
        )
    return xy_squared


def self_collision_cost(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Penalize self-collisions.

    When the sensor provides force history (from ``history_length > 0``),
    counts substeps where any contact force exceeds *force_threshold*.
    Falls back to the instantaneous ``found`` count otherwise.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    data = sensor.data
    if data.force_history is not None:
        # force_history: [B, N, H, 3]
        force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
        hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
        return hit.sum(dim=-1).float()  # [B]
    assert data.found is not None
    return data.found.squeeze(-1)


def body_roll_pitch_penalty(
    env: ManagerBasedRlEnv,
    weight_list: list[float] | None = None,
    velocity_list: list[float] | None = None,
    command_name: str | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize roll and pitch of a body link relative to the world frame.

    Projects the gravity vector into the body frame (via the body's world-frame
    quaternion) and derives roll/pitch angles via atan2 — the same convention
    used by ``imu_roll_pitch_penalty`` for the root link.

    Use ``asset_cfg.body_names = ["waist_roll_link"]`` to target that body.
    """
    asset: Entity = env.scene[asset_cfg.name]
    if not asset_cfg.body_ids or len(asset_cfg.body_ids) != 1:
        raise ValueError(
            f"body_roll_pitch_penalty expects exactly one body, got body_ids={asset_cfg.body_ids}"
        )
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]  # [B, 4]
    gravity_w = asset.data.gravity_vec_w
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    roll = torch.atan2(projected_gravity_b[:, 1], -projected_gravity_b[:, 2])
    pitch = torch.atan2(-projected_gravity_b[:, 0], -projected_gravity_b[:, 2])
    roll_pitch_error = torch.square(roll) + torch.square(pitch)
    if weight_list is not None and velocity_list is not None:
        command = env.command_manager.get_command(command_name)
        assert command is not None, f"Command '{command_name}' not found."
        vx_cmd = torch.abs(command[:, 0])
        boundaries = torch.tensor(
            velocity_list[1:], device=vx_cmd.device, dtype=vx_cmd.dtype
        )
        bin_idx = torch.bucketize(vx_cmd.contiguous(), boundaries).clamp(
            max=len(weight_list) - 1
        )
        weight_t = torch.tensor(weight_list, device=vx_cmd.device, dtype=vx_cmd.dtype)
        weights = weight_t[bin_idx]
        roll_pitch_error = roll_pitch_error * weights
    return roll_pitch_error


def body_angular_velocity_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize excessive body angular velocities."""
    asset: Entity = env.scene[asset_cfg.name]
    ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
    ang_vel = ang_vel.squeeze(1)
    ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
    return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Penalize whole-body angular momentum to encourage natural arm swing."""
    angmom_sensor: BuiltinSensor = env.scene[sensor_name]
    angmom = angmom_sensor.data
    angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
    angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
    env.extras["log"]["Episode_Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
    return angmom_magnitude_sq


def feet_air_time(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float = 0.4,
    command_name: str | None = None,
    command_threshold: float = 0.1,
    action_name: str | None = None,
) -> torch.Tensor:
    """Reward feet air time."""
    if action_name is not None:
        gait_term = env.action_manager.get_term(action_name)
        freq = gait_term.processed_frequency  # (B,)
        threshold = (1.0 - 0.6) * freq

    sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = sensor.data
    air_time = sensor_data.current_air_time
    contact_time = sensor_data.current_contact_time
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
    mode_time = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
    )[0]
    error = torch.abs(mode_time - threshold)
    reward = torch.clamp(threshold - error, min=0.0)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


def feet_clearance(
    env: ManagerBasedRlEnv,
    target_height: float,
    command_name: str | None = None,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize deviation from target clearance height, weighted by foot velocity."""
    asset: Entity = env.scene[asset_cfg.name]
    foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
    vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
    delta = torch.abs(foot_z - target_height)  # [B, N]
    cost = torch.sum(delta * vel_norm, dim=1)  # [B]
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            active = (total_command > command_threshold).float()
            cost = cost * active
    return cost


def feet_gait(
    env: ManagerBasedRlEnv,
    period: float,
    offset: list[float],
    threshold: float,
    command_threshold: float,
    command_name: str,
    sensor_name: str,
    action_name: str | None = None,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    if action_name is not None:
        # Use the learnable phase φ_t from GaitFrequencyAction instead of
        # the fixed-period time-based phase.
        gait_term = env.action_manager.get_term(action_name)
        global_phase = gait_term.phase.unsqueeze(1)  # [B, 1]
    else:
        global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(
        1, -1
    )
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = leg_phase < threshold
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


class feet_swing_height:
    """Penalize deviation from target swing height, evaluated at landing."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.sensor_name = cfg.params["sensor_name"]
        self.site_names = cfg.params["asset_cfg"].site_names
        self.peak_heights = torch.zeros(
            (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
        )
        self.step_dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        target_height: float,
        command_name: str,
        command_threshold: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        asset: Entity = env.scene[asset_cfg.name]
        contact_sensor: ContactSensor = env.scene[sensor_name]
        command = env.command_manager.get_command(command_name)
        assert command is not None
        foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
        in_air = contact_sensor.data.found == 0
        self.peak_heights = torch.where(
            in_air,
            torch.maximum(self.peak_heights, foot_heights),
            self.peak_heights,
        )
        first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        total_command = linear_norm + angular_norm
        active = (total_command > command_threshold).float()
        error = self.peak_heights / target_height - 1.0
        cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
        num_landings = torch.sum(first_contact.float())
        peak_heights_at_landing = self.peak_heights * first_contact.float()
        mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
            num_landings, min=1
        )
        env.extras["log"]["Episode_Metrics/peak_height_mean"] = mean_peak_height
        self.peak_heights = torch.where(
            first_contact,
            torch.zeros_like(self.peak_heights),
            self.peak_heights,
        )
        return cost


def feet_slip(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize foot sliding (xy velocity while in contact)."""
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    assert contact_sensor.data.found is not None
    in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
    vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
    vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
    cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
    num_in_contact = torch.sum(in_contact)
    mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
        num_in_contact, min=1
    )
    env.extras["log"]["Episode_Metrics/slip_velocity_mean"] = mean_slip_vel
    return cost


def soft_landing(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str | None = None,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize high impact forces at landing to encourage soft footfalls."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = contact_sensor.data
    assert sensor_data.force is not None
    forces = sensor_data.force  # [B, N, 3]
    force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
    first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
    landing_impact = force_magnitude * first_contact.float()  # [B, N]
    cost = torch.sum(landing_impact, dim=1)  # [B]
    num_landings = torch.sum(first_contact.float())
    mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
    env.extras["log"]["Episode_Metrics/landing_force_mean"] = mean_landing_force
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            active = (total_command > command_threshold).float()
            cost = cost * active
    return cost


class variable_posture:
    """Penalize deviation from default pose with speed-dependent tolerance.

    Uses per-joint standard deviations to control how much each joint can deviate
    from default pose. Smaller std = stricter (less deviation allowed), larger
    std = more forgiving. The reward is: exp(-mean(error² / std²))

    Three speed regimes (based on linear + angular command velocity):
      - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
      - std_walking (walking_threshold <= speed < running_threshold): Moderate.
      - std_running (speed >= running_threshold): Loose tolerance for large motion.

    Tune std values per joint based on how much motion that joint needs at each
    speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset: Entity = env.scene[cfg.params["asset_cfg"].name]
        default_joint_pos = asset.data.default_joint_pos
        assert default_joint_pos is not None
        self.default_joint_pos = default_joint_pos

        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

        _, _, std_standing = resolve_matching_names_values(
            data=cfg.params["std_standing"],
            list_of_strings=joint_names,
        )
        self.std_standing = torch.tensor(
            std_standing, device=env.device, dtype=torch.float32
        )

        _, _, std_walking = resolve_matching_names_values(
            data=cfg.params["std_walking"],
            list_of_strings=joint_names,
        )
        self.std_walking = torch.tensor(
            std_walking, device=env.device, dtype=torch.float32
        )

        _, _, std_running = resolve_matching_names_values(
            data=cfg.params["std_running"],
            list_of_strings=joint_names,
        )
        self.std_running = torch.tensor(
            std_running, device=env.device, dtype=torch.float32
        )

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std_standing,
        std_walking,
        std_running,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        walking_threshold: float = 0.5,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:
        del std_standing, std_walking, std_running  # Unused.

        asset: Entity = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        assert command is not None

        linear_speed = torch.norm(command[:, :2], dim=1)
        angular_speed = torch.abs(command[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < walking_threshold).float()
        walking_mask = (
            (total_speed >= walking_threshold) & (total_speed < running_threshold)
        ).float()
        running_mask = (total_speed >= running_threshold).float()

        std = (
            self.std_standing * standing_mask.unsqueeze(1)
            + self.std_walking * walking_mask.unsqueeze(1)
            + self.std_running * running_mask.unsqueeze(1)
        )

        current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
        error_squared = torch.square(current_joint_pos - desired_joint_pos)

        return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def stand_still(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward *= scale
    return reward


def dont_wait(
    env: ManagerBasedRlEnv,
    command_name: str,
    cmd_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize standing still when there is a forward velocity command."""
    asset: Entity = env.scene[asset_cfg.name]
    lin_vel_cmd_x = env.command_manager.get_command(command_name)[:, 0]
    lin_vel_x = asset.data.root_link_lin_vel_b[:, 0]

    v1 = cmd_threshold * 0.5
    v2 = 0.0  # stationary
    v3 = -cmd_threshold * 0.5

    return (lin_vel_cmd_x > cmd_threshold) * (
        (lin_vel_x < v1).float() + (lin_vel_x < v2).float() + (lin_vel_x < v3).float()
    )


def feet_double_support(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    period: float = 0.64,
    offset: list[float] = [0.0, 0.5],
    threshold: float = 0.5,
    command_name: str | None = None,
    cmd_threshold: float = 0.1,
    ds_match_weight: float = 1.0,
    non_ds_match_weight: float = 0.0,
    action_name: str | None = None,
) -> torch.Tensor:
    """Reward matching double-support timing."""
    contact_sensor: ContactSensor = env.scene[sensor_name]

    is_contact = contact_sensor.data.current_contact_time > 0.0  # [B, 2]

    if action_name is not None:
        gait_term = env.action_manager.get_term(action_name)
        global_phase = gait_term.phase.unsqueeze(1)  # [B, 1]
    else:
        global_phase = (
            (env.episode_length_buf * env.step_dt) % period / period
        ).unsqueeze(
            1
        )  # (num_envs, 1)

    phases = [(global_phase + off) % 1.0 for off in offset]
    leg_phase = torch.cat(phases, dim=1)  # (num_envs, num_feet)

    desired_stance = leg_phase < threshold
    desired_ds = desired_stance.all(dim=1)
    actual_ds = is_contact.all(dim=1)

    ds_match = desired_ds == actual_ds
    w = torch.where(desired_ds, ds_match_weight, non_ds_match_weight)
    reward = w * ds_match.float()

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= (cmd_norm > cmd_threshold).float()

    return reward


def feet_close_xy_gauss(
    env: ManagerBasedRlEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    std: float = 0.1,
) -> torch.Tensor:
    """Penalize when feet are too close together in the y distance."""
    asset: Entity = env.scene[asset_cfg.name]
    body_link_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]

    left_foot_xy = body_link_pos_w[:, 0, :2]
    right_foot_xy = body_link_pos_w[:, 1, :2]
    heading_w = asset.data.heading_w

    cos_heading = torch.cos(heading_w)
    sin_heading = torch.sin(heading_w)

    left_y = -sin_heading * left_foot_xy[:, 0] + cos_heading * left_foot_xy[:, 1]
    right_y = -sin_heading * right_foot_xy[:, 0] + cos_heading * right_foot_xy[:, 1]
    feet_distance_y = torch.abs(left_y - right_y)

    return torch.exp(-torch.clamp(threshold - feet_distance_y, min=0.0) / std**2) - 1


def feet_air_time_positive_biped(
    env: ManagerBasedRlEnv,
    command_name: str,
    threshold: float,
    sensor_name: str,
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene[sensor_name]
    air_time = contact_sensor.data.current_air_time
    contact_time = contact_sensor.data.current_contact_time
    # compute the reward
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
    )[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    )
    return reward


# ======================================
# CUSTOM REWARDS FROM HUMANOID_RAB #####
# ======================================


def _minjerk(u: torch.Tensor) -> torch.Tensor:
    """Compute the standard minimum-jerk polynomial interpolation s(u) in [0,1]."""
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def _get_knee_knots_deg(command: torch.Tensor) -> torch.Tensor:
    """Return knee angle knots (in degrees) at key gait events for each env: [N, 8]."""
    v = torch.abs(command[:, 0])
    N = v.shape[0]

    q_OT = (-3.6449 * v * v) + (17.6143 * v) + 6.0109
    q_FA = (0.2867 * v * v) + (3.0904 * v) + 59.6052

    q = [
        torch.full((N,), 7.0, device=v.device),  # t_IC (0%)
        q_OT,  # t_OT (12%)
        torch.full((N,), 10.0, device=v.device),  # t_HR (31%)
        torch.full((N,), 23.0, device=v.device),  # t_OI (50%)
        torch.full((N,), 32.0, device=v.device),  # t_TO (62%)
        q_FA,  # t_FA (73%)
        torch.full((N,), 2.0, device=v.device),  # t_KMF (99%)
        torch.full((N,), 7.0, device=v.device),  # wrap to next t_IC (100%)
    ]
    return torch.stack(q, dim=1)  # [N, 8]


def get_desired_knee_angle(
    env: ManagerBasedRlEnv,
    period: float,
    offset: list,
    command_name: str = "twist",
    action_name: str | None = None,
) -> torch.Tensor:
    """Compute desired knee angle per leg over gait cycle (radians): [N, 2].

    Uses minimum-jerk spline interpolation over velocity-dependent knee knots,
    with separate phase offsets per leg.

    Args:
        env: The environment instance.
        period: Gait cycle period in seconds.
        offset: Phase offsets for each leg (e.g. [0.0, 0.5]).
        command_name: Name of the velocity command to look up.

    Returns:
        Tensor of shape [N, 2] with desired knee angles in radians.
    """
    if action_name is not None:
        # Use the learnable phase φ_t from GaitFrequencyAction instead of
        # the fixed-period time-based phase.
        gait_term = env.action_manager.get_term(action_name)
        global_phase = gait_term.phase.unsqueeze(1)  # [B, 1]
    else:
        global_phase = (
            ((env.episode_length_buf * env.step_dt) % period) / period
        ).unsqueeze(1)

    phases = [(global_phase + off) % 1.0 for off in offset]
    leg_phase = torch.cat(phases, dim=-1)

    t = torch.tensor(
        [0.00, 0.12, 0.31, 0.50, 0.62, 0.73, 0.99, 1.00], device=leg_phase.device
    )
    K = t.numel() - 1

    t0 = t[:-1].view(1, 1, K)  # [1, 1, K]
    t1 = t[1:].view(1, 1, K)  # [1, 1, K]

    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."

    q_rad = _get_knee_knots_deg(command) * torch.pi / 180.0
    q0 = q_rad[:, :-1].unsqueeze(1)  # [N, 1, K]
    q1 = q_rad[:, 1:].unsqueeze(1)  # [N, 1, K]

    tau = leg_phase.unsqueeze(-1)  # [N, 2, 1]

    seg_mask = (tau >= t0) & (tau < t1)  # [N, 2, K]

    at_one = tau >= 1.0 - 1e-6  # [N, 2, 1]
    if at_one.any():
        last = torch.zeros_like(seg_mask)
        last[..., -1] = at_one.squeeze(-1)
        empty = ~seg_mask.any(dim=-1, keepdim=True)
        seg_mask = torch.where(empty, last, seg_mask)

    denom = (t1 - t0).clamp_min(1e-6)  # [1, 1, K]
    u = ((tau - t0) / denom).clamp(0.0, 1.0)  # [N, 2, K]
    s = _minjerk(u)  # [N, 2, K]

    y_seg = q0 + (q1 - q0) * s  # [N, 2, K]
    y = (y_seg * seg_mask).sum(dim=-1)  # [N, 2]

    stationary_mask = torch.norm(command[:, :2], dim=1) < 0.1
    if stationary_mask.any():
        stand_theta = q_rad[:, 3]  # [N]
        y[stationary_mask] = stand_theta[stationary_mask].unsqueeze(-1).expand(-1, 2)

    return torch.nan_to_num(y, nan=0.0)  # [N, 2]


def _get_terrain_height(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    """Get terrain surface height from a single-ray RayCastSensor. Returns [B]."""
    sensor: RayCastSensor = env.scene[sensor_name]
    hit_z = sensor.data.hit_pos_w[:, 0, 2]  # [B] single ray
    # Handle miss (distance == -1) or NaN/inf in hit_z: fall back to env origin z.
    fallback = env.scene.env_origins[:, 2]
    invalid = (
        (sensor.data.distances[:, 0] < 0) | torch.isnan(hit_z) | torch.isinf(hit_z)
    )
    hit_z = torch.where(invalid, fallback, hit_z)
    return hit_z


def terrain_foot_clearance_cost(
    env: ManagerBasedRlEnv,
    target_clearance: float,
    left_terrain_sensor: str,
    right_terrain_sensor: str,
    command_name: str | None = None,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize deviation from target clearance above local terrain surface.

    Computes terrain-relative foot height using per-foot RayCastSensors, then
    penalizes deviation from ``target_clearance`` weighted by foot velocity
    (swing phase detection).

    Args:
      env: The environment instance.
      target_clearance: Target height above terrain surface (m).
      left_terrain_sensor: Name of single-ray RayCastSensor under left foot.
      right_terrain_sensor: Name of single-ray RayCastSensor under right foot.
      command_name: If set, gate reward by command magnitude.
      command_threshold: Minimum command norm to activate reward.
      asset_cfg: Robot asset config with ``site_names`` for foot sites.

    Returns:
      Cost tensor [B] (use with negative weight).
    """
    asset: Entity = env.scene[asset_cfg.name]
    foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, 2]

    # Terrain height under each foot.
    left_z = _get_terrain_height(env, left_terrain_sensor)  # [B]
    right_z = _get_terrain_height(env, right_terrain_sensor)  # [B]
    terrain_z = torch.stack([left_z, right_z], dim=1)  # [B, 2]

    # Terrain-relative clearance.
    clearance = foot_z - terrain_z  # [B, 2]

    # Swing detection via foot velocity.
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, 2, 2]
    vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, 2]

    # Cost: deviation from target clearance, weighted by velocity.
    delta = torch.abs(clearance - target_clearance)  # [B, 2]
    cost = torch.sum(delta * vel_norm, dim=1)  # [B]

    # Gate by command magnitude.
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            active = (total_command > command_threshold).float()
            cost = cost * active

    return torch.nan_to_num(cost, nan=0.0, posinf=0.0, neginf=0.0)


def terrain_stumble_cost(
    env: ManagerBasedRlEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    velocity_threshold: float = 0.5,
) -> torch.Tensor:
    """Penalize foot-terrain contact during swing phase (foot moving fast).

    Extends the existing ``feet_stumble`` pattern by gating on swing phase:
    only penalizes when lateral forces exceed 4x vertical AND the foot is
    moving faster than ``velocity_threshold``.

    Args:
      env: The environment instance.
      sensor_cfg: Contact sensor config with ``body_ids`` for feet.
      asset_cfg: Robot asset config with ``body_ids`` for foot bodies.
      velocity_threshold: Minimum foot speed (m/s) to classify as swinging.

    Returns:
      Penalty tensor [B] (use with negative weight).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Entity = env.scene[asset_cfg.name]

    # Foot velocity magnitude (xy plane).
    foot_vel = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]  # [B, N, 3]
    foot_speed = torch.norm(foot_vel[:, :, :2], dim=-1)  # [B, N]
    is_swinging = foot_speed > velocity_threshold  # [B, N]

    # Force ratio check (lateral > 4x vertical).
    forces_z = torch.abs(contact_sensor.data.force[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.norm(contact_sensor.data.force[:, sensor_cfg.body_ids, :2], dim=2)
    stumble = forces_xy > 4 * forces_z  # [B, N]

    # Gate by swing phase.
    penalty = torch.any(stumble & is_swinging, dim=1).float()  # [B]
    return penalty


def step_accuracy(
    env: ManagerBasedRlEnv,
    sensor_cfg: SceneEntityCfg,
    patch_name: str = "target",
    std: float = 0.25,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward foot placement near flat patches on discrete terrain.

    Uses ``TerrainEntity.flat_patches`` (pre-computed during terrain generation)
    to find the nearest flat surface for each foot, and rewards proximity
    when the foot is in contact with the ground.

    Args:
      env: The environment instance.
      sensor_cfg: Contact sensor config with ``body_ids`` for feet.
      patch_name: Key in ``terrain.flat_patches`` dict.
      std: Standard deviation for exponential distance kernel.
      asset_cfg: Robot asset config with ``body_ids`` for foot bodies.

    Returns:
      Reward tensor [B] (use with positive weight).
    """
    terrain = env.scene.terrain
    if terrain is None or patch_name not in terrain.flat_patches:
        return torch.zeros(env.num_envs, device=env.device)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Entity = env.scene[asset_cfg.name]

    # flat_patches[patch_name]: [num_rows, num_cols, num_patches, 3]
    patches = terrain.flat_patches[patch_name]
    # Per-env patches based on terrain level/type.
    env_patches = patches[terrain.terrain_levels, terrain.terrain_types]  # [B, P, 3]

    # Foot positions.
    foot_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]  # [B, N, 3]

    # Distance from each foot to nearest patch (xy only).
    # foot_pos: [B, N, 1, 2], env_patches: [B, 1, P, 2]
    dist = torch.norm(
        foot_pos[:, :, None, :2] - env_patches[:, None, :, :2], dim=-1
    )  # [B, N, P]
    min_dist = dist.min(dim=-1).values  # [B, N]

    # Reward on contact.
    in_contact = (
        contact_sensor.data.found[:, sensor_cfg.body_ids] > 0
    ).float()  # [B, N]
    reward = torch.sum(torch.exp(-min_dist / std) * in_contact, dim=1)  # [B]
    return reward


def terrain_base_height_l2_cost(
    env: ManagerBasedRlEnv,
    target_height: float,
    base_terrain_sensor: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize base height deviation from target above local terrain surface.

    Uses a base-mounted RayCastSensor (``hit_pos_w``, NOT ``ray_hits_w``)
    to query terrain height beneath the robot, then computes L2 penalty
    against the terrain-relative target height.

    Args:
      env: The environment instance.
      target_height: Target height above terrain surface (m).
      base_terrain_sensor: Name of single-ray RayCastSensor under base.
      asset_cfg: Robot asset config.

    Returns:
      Penalty tensor [B] (use with negative weight).
    """
    asset: Entity = env.scene[asset_cfg.name]
    terrain_z = _get_terrain_height(env, base_terrain_sensor)  # [B]
    adjusted_target = target_height + terrain_z  # [B]
    cost = torch.square(asset.data.root_link_pos_w[:, 2] - adjusted_target)  # [B]
    return torch.nan_to_num(cost, nan=0.0, posinf=0.0, neginf=0.0)


def foot_edge_cost(
    env: ManagerBasedRlEnv,
    sensor_cfg: SceneEntityCfg,
    left_gradient_sensor: str,
    right_gradient_sensor: str,
    gradient_threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize foot placement at terrain edges (high gradient zones).

    Uses a 3x2 grid RayCastSensor (nx=3 along x, ny=2 along y) at each foot
    to compute terrain height gradients via finite differences. Penalizes when
    gradient magnitude exceeds threshold and foot is in contact with the ground.

    Grid layout (size=(0.2, 0.1), resolution=0.1, ordering="xy"):
      x in {-0.1, 0.0, 0.1}, y in {-0.05, 0.05} → 6 rays, row-major [ny, nx].

    Args:
      env: The environment instance.
      sensor_cfg: Contact sensor config with ``body_ids`` for feet.
      left_gradient_sensor: Name of 6-ray (3x2 grid) RayCastSensor at left foot.
      right_gradient_sensor: Name of 6-ray (3x2 grid) RayCastSensor at right foot.
      gradient_threshold: Maximum gradient magnitude before penalty applies.
      asset_cfg: Robot asset config with ``body_ids`` for foot bodies.

    Returns:
      Penalty tensor [B] (use with negative weight).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    grad_mags = []
    for grad_sensor_name in (left_gradient_sensor, right_gradient_sensor):
        sensor: RayCastSensor = env.scene[grad_sensor_name]
        hit_z = sensor.data.hit_pos_w[..., 2]  # [B, 6]

        # Handle ray misses (distance == -1) or NaN/inf: replace with zeros.
        invalid = (sensor.data.distances < 0) | torch.isnan(hit_z) | torch.isinf(hit_z)
        hit_z = torch.where(invalid, torch.zeros_like(hit_z), hit_z)

        # Reshape to [B, ny=2, nx=3] (row-major, "xy" ordering).
        grid = hit_z.reshape(-1, 2, 3)

        # dz/dx: central difference along x (cols), outer points 0.2m apart.
        dz_dx = (grid[:, :, 2] - grid[:, :, 0]) / 0.2  # [B, 2]
        # dz/dy: forward difference along y (rows), spacing 0.1m.
        dz_dy = (grid[:, 1, :] - grid[:, 0, :]) / 0.1  # [B, 3]

        # Mean gradient magnitude across the grid.
        sum_sq = torch.mean(dz_dx**2, dim=1) + torch.mean(dz_dy**2, dim=1)
        grad_mag = torch.sqrt(sum_sq.clamp(min=0.0))  # [B]
        grad_mags.append(grad_mag)

    # Stack left/right: [B, 2]
    grad_mags_t = torch.stack(grad_mags, dim=1)  # [B, 2]

    # Contact mask.
    in_contact = (
        contact_sensor.data.found[:, sensor_cfg.body_ids] > 0
    ).float()  # [B, 2]

    # Penalty: penalize when gradient exceeds threshold AND foot is in contact.
    penalty = torch.sum(
        (grad_mags_t > gradient_threshold).float() * in_contact, dim=1
    )  # [B]
    return torch.nan_to_num(penalty, nan=0.0, posinf=0.0, neginf=0.0)


def root_height_below_minimum_terrain(
    env: ManagerBasedRlEnv,
    minimum_height: float,
    terrain_sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate when the asset's root height above terrain is below minimum."""
    asset: Entity = env.scene[asset_cfg.name]
    terrain_z = _get_terrain_height(env, terrain_sensor_name)
    height_above_terrain = asset.data.root_link_pos_w[:, 2] - terrain_z
    return height_above_terrain < minimum_height


def knee_joint_motion(
    env: ManagerBasedRlEnv,
    reward_limit: float,
    command_name: str | None = None,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward knee joint movement (velocity magnitude), capped at reward_limit.

    Encourages the knee joints to actively move without over-rewarding
    extreme velocities. The sum of squared knee joint velocities is clamped
    to reward_limit, so the reward saturates once the knees move sufficiently.

    Args:
        reward_limit: Maximum reward per step (clamp ceiling).
        command_name: If set, gate reward by command magnitude > command_threshold.
        command_threshold: Minimum total command norm to activate reward.
        asset_cfg: Asset config with joint_ids resolved to knee joints.
    """
    asset: Entity = env.scene[asset_cfg.name]
    knee_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]  # [B, N]
    motion = torch.sum(torch.square(knee_vel), dim=1)  # [B]
    reward = torch.clamp(motion, max=reward_limit)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            reward = reward * (total_command > command_threshold).float()
    return reward


def gait_action_limit(env: ManagerBasedRlEnv, action_name: str) -> torch.Tensor:
    """Penalty for gait-frequency actions outside the valid frequency range.

    Returns 1.0 for envs where the raw policy output falls outside
    [min_frequency, max_frequency], 0.0 otherwise. Encourages the policy to
    stay within the operating band without hard clamping gradients.
    """
    gait_term = env.action_manager.get_term(action_name)
    raw = gait_term.raw_action[:, 0]
    out_of_bounds = (raw < gait_term.cfg.min_frequency) | (
        raw > gait_term.cfg.max_frequency
    )
    return out_of_bounds.float()


def _track_velocity_axis(
    env: ManagerBasedRlEnv,
    std_list: list[float],
    vel_list: list[float],
    command_name: str,
    cmd_axis: int,
    actual_vel_axis: torch.Tensor,
    add_penalize: bool,
    penalize_scale: float,
) -> torch.Tensor:
    """Exponential-kernel reward for a single velocity axis.

    Returns exp(-e²/σ²) in [0, 1], or (1+scale)*exp(-e²/σ²) - scale in [-scale, 1]
    when add_penalize is True.

    σ is a step function of |command|: segment [vel_list[i], vel_list[i+1]) maps
    to std_list[i], matching the bucketize convention used by body_roll_pitch_penalty.
    vel_list must have exactly len(std_list)+1 entries.
    """
    if len(vel_list) != len(std_list) + 1:
        raise ValueError(
            f"vel_list must have len(std_list)+1 entries, got vel_list={len(vel_list)} vs std_list={len(std_list)}"
        )
    command = env.command_manager.get_command(command_name)
    if command is None:
        raise ValueError(f"Command '{command_name}' not found.")
    cmd_abs = torch.abs(command[:, cmd_axis])
    boundaries = torch.tensor(vel_list[1:], device=cmd_abs.device, dtype=cmd_abs.dtype)
    bin_idx = torch.bucketize(cmd_abs.contiguous(), boundaries).clamp(
        max=len(std_list) - 1
    )
    std_t = torch.tensor(std_list, device=cmd_abs.device, dtype=cmd_abs.dtype)[bin_idx]
    error = torch.square(command[:, cmd_axis] - actual_vel_axis)
    r = torch.exp(-error / std_t**2)
    if add_penalize:
        return (1.0 + penalize_scale) * r - penalize_scale
    return r


def track_linear_x(
    env: ManagerBasedRlEnv,
    std_list: list[float],
    vel_list: list[float],
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    add_penalize: bool = False,
    penalize_scale: float = 1.0,
) -> torch.Tensor:
    """Reward tracking of linear x velocity command using an exponential kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    return _track_velocity_axis(
        env,
        std_list,
        vel_list,
        command_name,
        0,
        asset.data.root_link_lin_vel_b[:, 0],
        add_penalize,
        penalize_scale,
    )


def track_linear_y(
    env: ManagerBasedRlEnv,
    std_list: list[float],
    vel_list: list[float],
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    add_penalize: bool = False,
    penalize_scale: float = 1.0,
) -> torch.Tensor:
    """Reward tracking of linear y velocity command using an exponential kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    return _track_velocity_axis(
        env,
        std_list,
        vel_list,
        command_name,
        1,
        asset.data.root_link_lin_vel_b[:, 1],
        add_penalize,
        penalize_scale,
    )


def track_angular_z(
    env: ManagerBasedRlEnv,
    std_list: list[float],
    vel_list: list[float],
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    add_penalize: bool = False,
    penalize_scale: float = 1.0,
) -> torch.Tensor:
    """Reward tracking of angular z velocity command using an exponential kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    return _track_velocity_axis(
        env,
        std_list,
        vel_list,
        command_name,
        2,
        asset.data.root_link_ang_vel_b[:, 2],
        add_penalize,
        penalize_scale,
    )


def ang_vel_xy_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def lin_vel_z_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def link_orientation(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    link_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], :]
    link_projected_gravity = quat_apply_inverse(link_quat, asset.data.gravity_vec_w)
    return torch.sum(torch.square(link_projected_gravity[:, :2]), dim=1)
