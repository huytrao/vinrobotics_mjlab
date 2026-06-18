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

import copy
import os
from dataclasses import asdict
import yaml
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction

# from isaaclab.utils import class_to_dict
from mjlab.utils.lab_api.string import resolve_matching_names


def format_value(x):
    if isinstance(x, float):
        return float(f"{x:.3g}")
    elif isinstance(x, list):
        return [format_value(i) for i in x]
    elif isinstance(x, tuple):
        return [format_value(i) for i in x]
    elif isinstance(x, dict):
        return {k: format_value(v) for k, v in x.items()}
    else:
        return x


def export_deploy_cfg(env: ManagerBasedRlEnv, log_dir):
    robot: Entity = env.scene["robot"]
    joint_sdk_names = robot.joint_names
    joint_ids_map, _ = resolve_matching_names(robot.joint_names, joint_sdk_names, preserve_order=True)

    # Verify joint_pos action exists when present (not all envs use it, e.g.
    # hierarchical envs replace it with FrozenLocomotionAction).
    if "joint_pos" in env.action_manager.active_terms:
        joint_action = env.action_manager.get_term("joint_pos")
        assert isinstance(joint_action, JointPositionAction)
    # Build mapping from joint name to actuator ID for natural joint order.
    # Each spec actuator controls exactly one joint (via its target field).
    joint_name_to_ctrl_id = {}
    for actuator in robot.spec.actuators:
        joint_names = actuator.target.split("/")[-1]
        joint_name_to_ctrl_id[joint_names] = actuator.id
    # Get actuator IDs in natural joint order (same order as robot.joint_names).
    ctrl_ids_natural = [
        joint_name_to_ctrl_id[jname]
        for jname in robot.joint_names  # global joint order
        if jname in joint_name_to_ctrl_id  # skip non-actuated joints
    ]
    joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids_natural, 0]
    joint_damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids_natural, 2]

    cfg = {}  # noqa: SIM904
    cfg["joint_names"] = robot.joint_names
    cfg["joint_ids_map"] = joint_ids_map
    cfg["step_dt"] = env.cfg.sim.mujoco.timestep * env.cfg.decimation

    cfg["stiffness"] = joint_stiffness.tolist()
    cfg["damping"] = joint_damping.tolist()
    cfg["default_joint_pos"] = robot.data.default_joint_pos[0].detach().cpu().tolist()

    # --- commands ---
    command_names = env.command_manager.active_terms
    command_terms = zip(command_names, env.command_manager._terms.values())

    cfg["commands"] = {}
    for command_name, command_term in command_terms:
        if "motion" in command_name:
            continue

        if hasattr(command_term.cfg, "limit_ranges"):
            ranges = asdict(command_term.cfg.limit_ranges)
        elif hasattr(command_term.cfg, "ranges"):
            ranges = asdict(command_term.cfg.ranges)
        else:
            continue
        cfg["commands"][command_name] = {}
        for item_name in ["lin_vel_x", "lin_vel_y", "ang_vel_z"]:
            ranges[item_name] = list(ranges[item_name])
        cfg["commands"][command_name]["ranges"] = ranges

    # --- actions ---
    action_names = env.action_manager.active_terms
    action_terms = zip(action_names, env.action_manager._terms.values())
    cfg["actions"] = {}
    for action_name, action_term in action_terms:
        term_cfg = copy.deepcopy(action_term.cfg)

        if hasattr(term_cfg, "scale"):
            if isinstance(term_cfg.scale, float):
                term_cfg.scale = [term_cfg.scale for _ in range(action_term.action_dim)]
            else:  # dict
                term_cfg.scale = action_term._scale[0].detach().cpu().numpy().tolist()

        if hasattr(term_cfg, "clip") and term_cfg.clip is not None:
            term_cfg.clip = action_term._clip[0].detach().cpu().numpy().tolist()

        if action_name in ["joint_pos", "JointPositionAction", "JointVelocityAction"]:
            if term_cfg.use_default_offset:
                term_cfg.offset = action_term._offset[0].detach().cpu().numpy().tolist()
            else:
                term_cfg.offset = [0.0 for _ in range(action_term.action_dim)]

        # clean cfg
        term_cfg = asdict(term_cfg)
        if "actuator_names" in term_cfg:
            term_cfg["joint_names"] = term_cfg.pop("actuator_names")

        for key in ["entity_name", "transmission_type", "preserve_order", "use_default_offset"]:
            term_cfg.pop(key, None)
        cfg["actions"][action_name] = term_cfg

        if hasattr(action_term, "_joint_ids"):
            cfg["actions"][action_name]["joint_ids"] = action_term._joint_ids
        else:
            cfg["actions"][action_name]["joint_ids"] = None

        if "joint_pos" in cfg["actions"]:
            cfg["actions"]["JointPositionAction"] = cfg["actions"].pop("joint_pos")

    # --- observations ---
    obs_names = env.observation_manager.active_terms["actor"]
    obs_cfgs = env.observation_manager._group_obs_term_cfgs["actor"]
    obs_terms = zip(obs_names, obs_cfgs)
    cfg["observations"] = {}
    for obs_name, obs_cfg in obs_terms:
        obs_dims = tuple(obs_cfg.func(env, **obs_cfg.params).shape)
        term_cfg = copy.deepcopy(obs_cfg)
        if term_cfg.scale is not None:
            scale = term_cfg.scale.detach().cpu().numpy().tolist()
            if isinstance(scale, float):
                term_cfg.scale = [scale for _ in range(obs_dims[1])]
            else:
                term_cfg.scale = scale
        else:
            term_cfg.scale = [1.0 for _ in range(obs_dims[1])]
        if term_cfg.clip is not None:
            term_cfg.clip = list(term_cfg.clip)
        if term_cfg.history_length == 0:
            term_cfg.history_length = 1

        # clean cfg
        term_cfg = asdict(term_cfg)
        # for _ in ["func", "modifiers", "noise", "flatten_history_dim"]:
            # del term_cfg[_]

        for _ in [
            "func",
            "noise",
            "flatten_history_dim",
            "delay_min_lag",
            "delay_max_lag",
            "delay_per_env",
            "delay_hold_prob",
            "delay_update_period",
            "delay_per_env_phase",
        ]:
            del term_cfg[_]

        cfg["observations"][obs_name] = term_cfg

    # --- save config file ---
    filename = os.path.join(log_dir, "params", "deploy.yaml")
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    # if not isinstance(cfg, dict):
    #     cfg = class_to_dict(cfg)
    cfg = format_value(cfg)
    with open(filename, "w") as f:
        yaml.dump(cfg, f, default_flow_style=None, sort_keys=False)
