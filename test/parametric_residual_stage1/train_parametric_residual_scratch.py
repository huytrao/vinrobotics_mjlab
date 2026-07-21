"""Train VR-M3-1 from scratch with parametric-gait residual actions.

This entrypoint follows the attached scratch-training plan:

  q_des = q_default + alpha * q_gait(v, phase) + s_res * tanh(a_policy)

The action semantics are active from iteration 0. No warm-start checkpoint,
distillation, or leg-only action space is used.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
WARP_CACHE_DIR = WORKSPACE_ROOT / ".cache" / "warp"
WARP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("WARP_CACHE_PATH", str(WARP_CACHE_DIR))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import mjlab.tasks  # noqa: E402,F401
import src.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg  # noqa: E402
from mjlab.envs.mdp.events import resolve_env_ids  # noqa: E402
from mjlab.managers.curriculum_manager import CurriculumTermCfg  # noqa: E402
from mjlab.managers.event_manager import EventTermCfg  # noqa: E402
from mjlab.managers.observation_manager import ObservationTermCfg  # noqa: E402
from mjlab.managers.reward_manager import RewardTermCfg  # noqa: E402
from mjlab.managers.scene_entity_config import SceneEntityCfg  # noqa: E402
from mjlab.utils.lab_api.math import quat_apply  # noqa: E402
from src.assets.robots import VR_M3_1_ACTION_SCALE  # noqa: E402
from src.tasks.velocity.config.vr_m3_1 import env_cfgs as vr_env_cfgs  # noqa: E402
from scripts.train import TrainConfig, launch_training  # noqa: E402


TASK_ID = "VR-M3-1-Flat"
EXPERIMENT_NAME = "vr_m3_1_parametric_residual_scratch"
RUN_NAME = "stage1_flat_bootstrap"
DEFAULT_LOGGER = "wandb"
DEFAULT_WANDB_PROJECT = EXPERIMENT_NAME

RESIDUAL_ACTION_SCALE = {
    ".*_hip_pitch_joint": 0.38,
    ".*_hip_roll_joint": 0.20,
    ".*_hip_yaw_joint": 0.15,
    ".*_knee_pitch_joint": 0.45,
    ".*_ankle_pitch_joint": 0.28,
    ".*_ankle_roll_joint": 0.15,
    "waist_yaw_joint": 0.20,
    ".*_shoulder_pitch_joint": 0.32,
    ".*_shoulder_roll_joint": 0.18,
    ".*_shoulder_yaw_joint": 0.18,
    ".*_elbow_pitch_joint": 0.28,
    ".*_wrist_yaw_joint": 0.10,
    ".*_wrist_roll_joint": 0.10,
    ".*_wrist_pitch_joint": 0.10,
}

HEAVY_RANDOMIZATION_EVENTS = (
    "push_robot",
    "foot_friction",
    "encoder_bias",
    "base_com",
    "body_com",
    "base_mass",
    "body_mass",
    "randomize_actuator_gains",
    "randomize_joint_limit",
    "randomize_joint_stiffness",
    "randomize_joint_frictionloss",
    "randomize_joint_armature",
    "randomize_joint_damping",
)


def smoothstep01(x: torch.Tensor) -> torch.Tensor:
    x = torch.clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def effective_command_speed(command: torch.Tensor) -> torch.Tensor:
    return (
        torch.abs(command[:, 0])
        + 0.6 * torch.abs(command[:, 1])
        + 0.25 * torch.abs(command[:, 2])
    )


def swing_envelope(leg_phase: torch.Tensor, duty: torch.Tensor) -> torch.Tensor:
    swing_active = leg_phase >= duty
    u = (leg_phase - duty) / torch.clamp(1.0 - duty, min=1.0e-4)
    u = torch.clamp(u, 0.0, 1.0)
    return torch.sin(math.pi * u).square() * swing_active.float()


@dataclass(kw_only=True)
class ParametricResidualJointPositionActionCfg(JointPositionActionCfg):
    initial_alpha: float = 0.55
    min_frequency: float = 0.8
    max_frequency: float = 1.1
    duty_factor: float = 0.66
    min_duty_factor: float = 0.58
    max_duty_factor: float = 0.70
    command_filter_alpha: float = 0.85
    command_name: str = "twist"
    command_threshold: float = 0.08

    def build(self, env: ManagerBasedRlEnv) -> "ParametricResidualJointPositionAction":
        return ParametricResidualJointPositionAction(self, env)


class ParametricResidualJointPositionAction(JointPositionAction):
    cfg: ParametricResidualJointPositionActionCfg

    def __init__(self, cfg: ParametricResidualJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)
        self._alpha = torch.full((self.num_envs, 1), cfg.initial_alpha, device=self.device)
        self._gait_offsets = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._phase = torch.zeros(self.num_envs, device=self.device)
        self._frequency = torch.full((self.num_envs,), cfg.min_frequency, device=self.device)
        self._duty_factor = torch.full((self.num_envs,), cfg.duty_factor, device=self.device)
        self._gait_strength = torch.zeros(self.num_envs, device=self.device)
        self._filtered_command = torch.zeros(self.num_envs, 3, device=self.device)
        self._min_frequency = float(cfg.min_frequency)
        self._max_frequency = float(cfg.max_frequency)

    @property
    def gait_offsets(self) -> torch.Tensor:
        return self._gait_offsets

    @property
    def alpha(self) -> torch.Tensor:
        return self._alpha

    @property
    def phase(self) -> torch.Tensor:
        return self._phase

    @property
    def frequency(self) -> torch.Tensor:
        return self._frequency

    @property
    def duty_factor(self) -> torch.Tensor:
        return self._duty_factor

    @property
    def gait_strength(self) -> torch.Tensor:
        return self._gait_strength

    def set_alpha(self, value: float) -> None:
        self._alpha[:] = float(value)

    def set_frequency_range(self, min_frequency: float, max_frequency: float) -> None:
        self._min_frequency = float(min_frequency)
        self._max_frequency = float(max_frequency)

    def reset_gait_state(
        self,
        env_ids: torch.Tensor,
        phase: torch.Tensor,
        gait_strength: torch.Tensor,
    ) -> None:
        self._phase[env_ids] = phase
        self._gait_strength[env_ids] = gait_strength
        command = self._env.command_manager.get_command(self.cfg.command_name)
        self._filtered_command[env_ids] = command[env_ids]

    def _update_gait_state(self) -> None:
        command = self._env.command_manager.get_command(self.cfg.command_name)
        self._filtered_command = (
            self.cfg.command_filter_alpha * self._filtered_command
            + (1.0 - self.cfg.command_filter_alpha) * command
        )
        effective_speed = effective_command_speed(self._filtered_command)
        strength_u = (effective_speed - 0.05) / (0.20 - 0.05)
        self._gait_strength = smoothstep01(strength_u)
        self._frequency = torch.clamp(
            0.80 + 0.55 * effective_speed,
            min=self._min_frequency,
            max=self._max_frequency,
        )
        self._duty_factor = torch.clamp(
            0.68 - 0.10 * effective_speed,
            min=self.cfg.min_duty_factor,
            max=self.cfg.max_duty_factor,
        )
        self._phase = (
            self._phase + self._frequency * self._env.step_dt * self._gait_strength
        ) % 1.0

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        residual = torch.tanh(self._raw_actions) * self._scale
        self._update_gait_state()
        command = self._filtered_command
        speed_ratio = torch.clamp(effective_command_speed(command) / 1.0, 0.0, 1.0)
        self._gait_offsets = compute_parametric_gait_offsets_from_phase(
            self._env,
            self._target_names,
            phase=self._phase,
            duty=self._duty_factor,
            gait_strength=self._gait_strength,
            speed_ratio=speed_ratio,
            command=command,
        )
        default_target = self._entity.data.default_joint_pos[:, self._target_ids]
        self._processed_actions = default_target + self._alpha * self._gait_offsets + residual
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


def compute_parametric_gait_offsets_from_phase(
    env: ManagerBasedRlEnv,
    joint_names: Sequence[str],
    phase: torch.Tensor,
    duty: torch.Tensor,
    gait_strength: torch.Tensor,
    speed_ratio: torch.Tensor,
    command: torch.Tensor | None = None,
) -> torch.Tensor:
    left_phase = phase
    right_phase = (phase + 0.5) % 1.0
    left_wave = torch.sin(2.0 * math.pi * left_phase)
    right_wave = torch.sin(2.0 * math.pi * right_phase)
    left_swing = swing_envelope(left_phase, duty)
    right_swing = swing_envelope(right_phase, duty)

    leg_amp = (0.18 + 0.12 * speed_ratio) * gait_strength
    knee_amp = (0.28 + 0.18 * speed_ratio) * gait_strength
    ankle_amp = (0.12 + 0.10 * speed_ratio) * gait_strength
    arm_amp = (0.18 + 0.18 * speed_ratio) * gait_strength
    waist_amp = (0.04 + 0.05 * speed_ratio) * gait_strength
    turn = torch.zeros_like(phase) if command is None else command[:, 2]

    offsets = torch.zeros(phase.shape[0], len(joint_names), device=env.device)
    for i, name in enumerate(joint_names):
        n = name.lower()
        is_left = n.startswith("left_")
        wave = left_wave if is_left else right_wave
        swing = left_swing if is_left else right_swing
        opposite_wave = right_wave if is_left else left_wave

        if "hip_pitch" in n:
            offsets[:, i] = -leg_amp * wave
        elif "hip_roll" in n:
            offsets[:, i] = (0.04 * gait_strength) * (1.0 if is_left else -1.0) * torch.cos(2.0 * math.pi * phase)
        elif "hip_yaw" in n:
            offsets[:, i] = (0.04 * gait_strength) * (1.0 if is_left else -1.0) * wave + 0.06 * turn
        elif "knee" in n:
            offsets[:, i] = knee_amp * swing
        elif "ankle_pitch" in n:
            offsets[:, i] = -ankle_amp * swing + 0.05 * leg_amp * wave
        elif "ankle_roll" in n:
            offsets[:, i] = (0.03 * gait_strength) * (-1.0 if is_left else 1.0) * torch.cos(2.0 * math.pi * phase)
        elif "waist_yaw" in n:
            offsets[:, i] = waist_amp * torch.sin(2.0 * math.pi * phase) + 0.12 * turn
        elif "shoulder_pitch" in n:
            offsets[:, i] = arm_amp * opposite_wave
        elif "elbow" in n:
            offsets[:, i] = 0.10 * gait_strength * torch.clamp(opposite_wave, min=0.0)
        elif "shoulder_roll" in n or "shoulder_yaw" in n:
            offsets[:, i] = 0.05 * gait_strength * opposite_wave
        elif "wrist" in n:
            offsets[:, i] = 0.03 * gait_strength * opposite_wave
    return offsets


def gait_frequency_obs(
    env: ManagerBasedRlEnv,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    term = env.action_manager.get_term(action_name)
    return term.frequency.unsqueeze(1)


def gait_duty_factor_obs(env: ManagerBasedRlEnv, action_name: str = "joint_pos") -> torch.Tensor:
    term = env.action_manager.get_term(action_name)
    return term.duty_factor.unsqueeze(1)


def gait_strength_obs(env: ManagerBasedRlEnv, action_name: str = "joint_pos") -> torch.Tensor:
    term = env.action_manager.get_term(action_name)
    return term.gait_strength.unsqueeze(1)


def gait_alpha_obs(env: ManagerBasedRlEnv, action_name: str = "joint_pos") -> torch.Tensor:
    term = env.action_manager.get_term(action_name)
    return term.alpha


def dynamic_gait_phase_obs(env: ManagerBasedRlEnv, action_name: str = "joint_pos") -> torch.Tensor:
    term = env.action_manager.get_term(action_name)
    phase = term.phase
    return torch.stack(
        (torch.sin(2.0 * math.pi * phase), torch.cos(2.0 * math.pi * phase)),
        dim=1,
    )


def target_foot_contacts_obs(env: ManagerBasedRlEnv, action_name: str = "joint_pos") -> torch.Tensor:
    term = env.action_manager.get_term(action_name)
    phase = term.phase
    duty = term.duty_factor
    strength = term.gait_strength
    contacts = torch.stack((phase < duty, ((phase + 0.5) % 1.0) < duty), dim=1).float()
    double_support = torch.ones_like(contacts)
    return strength.unsqueeze(1) * contacts + (1.0 - strength).unsqueeze(1) * double_support


def parametric_joint_error_obs(
    env: ManagerBasedRlEnv,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    term = env.action_manager.get_term(action_name)
    current = term._entity.data.joint_pos[:, term.target_ids]
    default = term._entity.data.default_joint_pos[:, term.target_ids]
    return current - (default + term.alpha * term.gait_offsets)


def track_parametric_pose(
    env: ManagerBasedRlEnv,
    action_name: str,
    joint_regex: tuple[str, ...],
    std: float,
) -> torch.Tensor:
    term = env.action_manager.get_term(action_name)
    names = term.target_names
    ids = [i for i, name in enumerate(names) if any(__import__("re").match(p, name) for p in joint_regex)]
    if not ids:
        return torch.zeros(env.num_envs, device=env.device)
    current = term._entity.data.joint_pos[:, term.target_ids[ids]]
    default = term._entity.data.default_joint_pos[:, term.target_ids[ids]]
    target = default + term.alpha * term.gait_offsets[:, ids]
    return torch.exp(-torch.mean(torch.square(current - target), dim=1) / (std * std))


def track_gait_contacts(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    term = env.action_manager.get_term(action_name)
    sensor = env.scene[sensor_name]
    found = sensor.data.found
    assert found is not None
    actual = (found.squeeze(-1) > 0).float()
    phase = term.phase
    duty = term.duty_factor
    strength = term.gait_strength
    target = torch.stack((phase < duty, ((phase + 0.5) % 1.0) < duty), dim=1).float()
    walking_match = 1.0 - torch.mean(torch.abs(actual - target), dim=1)
    standing_match = actual.all(dim=1).float()
    return strength * walking_match + (1.0 - strength) * standing_match


def parametric_alpha_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    action_name: str,
    alpha_stages: list[dict[str, float]],
) -> torch.Tensor:
    del env_ids
    term = env.action_manager.get_term(action_name)
    alpha = term.cfg.initial_alpha
    for stage in alpha_stages:
        if env.common_step_counter > int(stage["step"]):
            alpha = float(stage["alpha"])
    term.set_alpha(alpha)
    return torch.tensor([alpha], device=env.device)


def parametric_frequency_curriculum(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    action_name: str,
    frequency_stages: list[dict[str, float]],
) -> torch.Tensor:
    del env_ids
    term = env.action_manager.get_term(action_name)
    min_frequency = term.cfg.min_frequency
    max_frequency = term.cfg.max_frequency
    for stage in frequency_stages:
        if env.common_step_counter > int(stage["step"]):
            min_frequency = float(stage["min_frequency"])
            max_frequency = float(stage["max_frequency"])
    term.set_frequency_range(min_frequency, max_frequency)
    return torch.tensor(max_frequency, device=env.device)


def reset_from_parametric_gait(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    action_name: str = "joint_pos",
    gait_reset_prob: float = 0.70,
    alpha_reset: float = 0.55,
    joint_noise: float = 0.04,
    velocity_noise: float = 0.10,
    root_velocity_scale: float = 0.40,
) -> None:
    env_ids = resolve_env_ids(env, env_ids)
    term = env.action_manager.get_term(action_name)
    asset = term._entity

    num = len(env_ids)
    command = env.command_manager.get_command(term.cfg.command_name)
    command_eff = effective_command_speed(command[env_ids])
    command_active = command_eff > term.cfg.command_threshold
    use_gait = (torch.rand(num, device=env.device) < gait_reset_prob) & command_active
    phase = torch.rand(num, device=env.device)
    speed_ratio = torch.clamp(command_eff / 1.0, min=0.0, max=1.0)
    reset_strength = smoothstep01((command_eff - 0.05) / (0.20 - 0.05))
    gait_strength = reset_strength * use_gait.float()
    duty = torch.clamp(
        0.68 - 0.10 * command_eff,
        min=term.cfg.min_duty_factor,
        max=term.cfg.max_duty_factor,
    )
    offsets = compute_parametric_gait_offsets_from_phase(
        env,
        term.target_names,
        phase=phase,
        duty=duty,
        gait_strength=gait_strength,
        speed_ratio=speed_ratio,
        command=command[env_ids],
    )
    eps = 1.0e-3
    offsets_next = compute_parametric_gait_offsets_from_phase(
        env,
        term.target_names,
        phase=(phase + eps) % 1.0,
        duty=duty,
        gait_strength=gait_strength,
        speed_ratio=speed_ratio,
        command=command[env_ids],
    )
    reset_frequency = torch.clamp(
        0.80 + 0.55 * command_eff,
        min=term._min_frequency,
        max=term._max_frequency,
    )
    gait_vel = (offsets_next - offsets) / eps * reset_frequency.unsqueeze(1)

    default_joint_pos = asset.data.default_joint_pos[env_ids][:, term.target_ids]
    default_joint_vel = asset.data.default_joint_vel[env_ids][:, term.target_ids]
    joint_pos = default_joint_pos + alpha_reset * offsets
    joint_pos += (2.0 * torch.rand_like(joint_pos) - 1.0) * joint_noise
    joint_vel = default_joint_vel + 0.4 * alpha_reset * gait_vel
    joint_vel += (2.0 * torch.rand_like(default_joint_vel) - 1.0) * velocity_noise

    soft_limits = asset.data.soft_joint_pos_limits[env_ids][:, term.target_ids]
    joint_pos = joint_pos.clamp(soft_limits[..., 0], soft_limits[..., 1])
    asset.write_joint_state_to_sim(
        joint_pos,
        joint_vel,
        env_ids=env_ids,
        joint_ids=term.target_ids,
    )

    term.reset_gait_state(env_ids, phase=phase, gait_strength=gait_strength)

    lin_vel_b = torch.zeros(num, 3, device=env.device)
    lin_vel_b[:, 0] = root_velocity_scale * command[env_ids, 0] * gait_strength
    lin_vel_b[:, 1] = root_velocity_scale * command[env_ids, 1] * gait_strength
    lin_vel_w = quat_apply(asset.data.root_link_quat_w[env_ids], lin_vel_b)
    root_vel_w = torch.zeros(num, 6, device=env.device)
    root_vel_w[:, :3] = lin_vel_w
    root_vel_w[:, 5] = root_velocity_scale * command[env_ids, 2] * gait_strength
    asset.write_root_link_velocity_to_sim(root_vel_w, env_ids=env_ids)


def patch_parametric_scratch_config(
    cfg: TrainConfig,
    num_envs: int,
    max_iterations: int | None,
    logger: str,
    wandb_project: str,
) -> TrainConfig:
    env = cfg.env
    agent = cfg.agent

    env.scene.num_envs = num_envs
    agent.experiment_name = EXPERIMENT_NAME
    agent.run_name = RUN_NAME
    agent.logger = logger
    agent.wandb_project = wandb_project
    agent.resume = False
    agent.upload_model = False
    agent.actor.distribution_cfg["init_std"] = 0.4
    if max_iterations is not None:
        agent.max_iterations = max_iterations

    env.actions["joint_pos"] = ParametricResidualJointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=RESIDUAL_ACTION_SCALE,
        use_default_offset=True,
        initial_alpha=0.55,
        min_frequency=0.8,
        max_frequency=1.1,
        duty_factor=0.66,
        min_duty_factor=0.62,
        max_duty_factor=0.70,
        command_filter_alpha=0.85,
        command_name="twist",
    )

    for group in env.observations.values():
        group.enable_corruption = False
        group.terms["gait_phase"] = ObservationTermCfg(
            func=dynamic_gait_phase_obs,
            params={"action_name": "joint_pos"},
            history_length=1,
            flatten_history_dim=True,
        )
        group.terms["gait_frequency"] = ObservationTermCfg(
            func=gait_frequency_obs,
            params={"action_name": "joint_pos"},
        )
        group.terms["gait_duty_factor"] = ObservationTermCfg(
            func=gait_duty_factor_obs,
            params={"action_name": "joint_pos"},
        )
        group.terms["gait_strength"] = ObservationTermCfg(
            func=gait_strength_obs,
            params={"action_name": "joint_pos"},
        )
        group.terms["gait_alpha"] = ObservationTermCfg(
            func=gait_alpha_obs,
            params={"action_name": "joint_pos"},
        )
        group.terms["target_foot_contacts"] = ObservationTermCfg(
            func=target_foot_contacts_obs,
            params={"action_name": "joint_pos"},
        )
        group.terms["parametric_joint_error"] = ObservationTermCfg(
            func=parametric_joint_error_obs,
            history_length=3,
            flatten_history_dim=True,
        )

    # Stage 1 flat bootstrap: no strong randomization, no negative vx.
    if env.scene.terrain is not None:
        env.scene.terrain.terrain_type = "plane"
        env.scene.terrain.terrain_generator = None
    env.curriculum.pop("terrain_levels", None)
    for name in HEAVY_RANDOMIZATION_EVENTS:
        env.events.pop(name, None)
    env.events.pop("reset_robot_joints", None)
    env.events["reset_from_parametric_gait"] = EventTermCfg(
        func=reset_from_parametric_gait,
        mode="reset",
        params={
            "action_name": "joint_pos",
            "gait_reset_prob": 0.70,
            "alpha_reset": 0.55,
            "joint_noise": 0.04,
            "velocity_noise": 0.10,
            "root_velocity_scale": 0.40,
        },
    )

    twist = env.commands["twist"]
    twist.rel_standing_envs = 0.12
    twist.resampling_time_range = (3.0, 6.0)
    twist.ranges.lin_vel_x = (0.15, 0.60)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)

    if "command_vel" in env.curriculum:
        env.curriculum["command_vel"].params["velocity_stages"] = [
            {"step": 0, "lin_vel_x": (0.15, 0.60), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (0.0, 0.0)},
        ]

    pose_params = env.rewards["pose"].params
    pose_params["std_walking"].update({
        r".*shoulder_pitch.*": 0.60,
        r".*shoulder_roll.*": 0.30,
        r".*shoulder_yaw.*": 0.30,
        r".*elbow.*": 0.45,
        r".*wrist.*": 0.20,
        r".*waist_yaw.*": 0.35,
    })
    pose_params["std_running"].update({
        r".*shoulder_pitch.*": 0.80,
        r".*shoulder_roll.*": 0.35,
        r".*shoulder_yaw.*": 0.35,
        r".*elbow.*": 0.55,
        r".*wrist.*": 0.25,
        r".*waist_yaw.*": 0.45,
    })

    rewards = env.rewards
    rewards["pose"].weight = 0.15
    rewards["track_linear_x"].weight = 2.0
    rewards["track_linear_y"].weight = 0.5
    rewards["track_angular_z"].weight = 0.5
    rewards["feet_air_time_biped"].weight = 0.0
    rewards["knee_motion"].weight = 0.08
    rewards["action_rate_l2"].weight = -0.15
    rewards["joint_acc_l2"].weight = -1.0e-7
    rewards["dof_torques_l2"].weight = -5.0e-8
    rewards["is_terminated"].weight = -100.0
    rewards["flat_orientation"].weight = -1.0
    rewards.pop("link_orientation", None)
    rewards.pop("pelvis_roll_pitch", None)
    rewards.pop("base_roll_penalty", None)
    rewards.pop("body_ang_vel", None)

    rewards["track_parametric_leg_pose"] = RewardTermCfg(
        func=track_parametric_pose,
        weight=0.4,
        params={
            "action_name": "joint_pos",
            "joint_regex": (r".*_hip_.*", r".*_knee_.*", r".*_ankle_.*"),
            "std": 0.35,
        },
    )
    rewards["track_parametric_upper_pose"] = RewardTermCfg(
        func=track_parametric_pose,
        weight=0.7,
        params={
            "action_name": "joint_pos",
            "joint_regex": (r".*shoulder.*", r".*elbow.*", r".*wrist.*", r".*waist_yaw.*"),
            "std": 0.30,
        },
    )
    rewards["track_gait_contacts"] = RewardTermCfg(
        func=track_gait_contacts,
        weight=1.0,
        params={"sensor_name": "feet_ground_contact", "action_name": "joint_pos"},
    )

    env.curriculum["parametric_alpha"] = CurriculumTermCfg(
        func=parametric_alpha_curriculum,
        params={
            "action_name": "joint_pos",
            "alpha_stages": [
                {"step": 0, "alpha": 0.55},
            ],
        },
    )
    env.curriculum["parametric_frequency"] = CurriculumTermCfg(
        func=parametric_frequency_curriculum,
        params={
            "action_name": "joint_pos",
            "frequency_stages": [
                {"step": 0, "min_frequency": 0.8, "max_frequency": 1.1},
            ],
        },
    )

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=1600)
    parser.add_argument("--max-iterations", type=int, default=2500)
    parser.add_argument("--gpu-ids", type=str, default="0")
    parser.add_argument("--logger", choices=("wandb", "tensorboard"), default=DEFAULT_LOGGER)
    parser.add_argument("--wandb-project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("WANDB_SILENT", "true")
    if args.wandb_mode is not None:
        os.environ["WANDB_MODE"] = args.wandb_mode
    if args.logger == "wandb":
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip()]
    cfg = TrainConfig.from_task(TASK_ID)
    cfg = patch_parametric_scratch_config(
        cfg,
        args.num_envs,
        args.max_iterations,
        args.logger,
        args.wandb_project,
    )
    cfg = TrainConfig(
        env=cfg.env,
        agent=cfg.agent,
        video=False,
        debug=args.debug,
        gpu_ids=gpu_ids,
    )
    print("[INFO] Parametric residual scratch training")
    print("[INFO] No checkpoint / no warm-start / no distillation")
    print("[INFO] Action: q_default + alpha*q_gait + residual_scale*tanh(policy)")
    print("[INFO] init_std:", cfg.agent.actor.distribution_cfg["init_std"])
    print("[INFO] num_envs:", cfg.env.scene.num_envs)
    print("[INFO] max_iterations:", cfg.agent.max_iterations)
    print("[INFO] logger:", cfg.agent.logger)
    print("[INFO] wandb_project:", getattr(cfg.agent, "wandb_project", None))
    print("[INFO] WANDB_MODE:", os.environ.get("WANDB_MODE", "not set"))
    launch_training(TASK_ID, cfg)


if __name__ == "__main__":
    main()
