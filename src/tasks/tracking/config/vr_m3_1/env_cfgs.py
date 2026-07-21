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

"""VinRobotics VR M3.1 (full-body) flat-terrain motion-tracking environment.

Mirrors mjlab's own Unitree G1 tracking config
(`mjlab.tasks.tracking.config.g1.env_cfgs`), swapped to the VR M3.1 humanoid
and its teleop mocap data (see `scripts/mocap_csv_to_motion_npz.py`).
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from src.assets.robots import VR_M3_1_ACTION_SCALE, get_vr_m3_1_implicit_actuator_robot_cfg
# Generic (not velocity-task-specific) shaping term, reused across tasks.
from src.tasks.velocity.mdp.rewards import angular_momentum_penalty

ROOT_BODY = "pelvis"
TORSO_BODY = "waist_yaw_link"

# Bodies whose pose/velocity are tracked against the motion reference.
#
# Legs + pelvis/waist only, arms excluded: the teleop motion npz's arm/shoulder
# joints have velocity spikes up to ~30 rad/s (physically implausible -- almost
# certainly mocap noise or angle-wrap artifacts), while the leg joints and root
# height track a plausible, stable walking pattern. This task has never been
# trained before (no checkpoint/video evidence anywhere in this repo), so the
# first pass targets the verified-plausible half of the reference data only,
# rather than fighting noisy arm targets on an already-unvalidated task.
# Re-add the arm bodies below once a legs-only run is confirmed to converge.
TRACKED_BODY_NAMES = (
  "pelvis",
  "left_hip_roll_link",
  "left_knee_pitch_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_pitch_link",
  "right_ankle_roll_link",
  "waist_yaw_link",
)

# Matches TRACKED_BODY_NAMES: wrist end-effectors dropped so the `ee_body_pos`
# termination doesn't end episodes over the noisy arm reference data above.
END_EFFECTOR_BODY_NAMES = (
  "left_ankle_roll_link",
  "right_ankle_roll_link",
)

FOOT_GEOMS = r"^(left|right)_ankle_roll_link_collision_\d+$"


def vr_m3_1_flat_tracking_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create VR M3.1 flat-terrain motion-tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_vr_m3_1_implicit_actuator_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = VR_M3_1_ACTION_SCALE

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = TORSO_BODY
  motion_cmd.body_names = TRACKED_BODY_NAMES

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = FOOT_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = (TORSO_BODY,)

  cfg.terminations["ee_body_pos"].params["body_names"] = END_EFFECTOR_BODY_NAMES

  # Arm-swing shaping: the pose-tracking rewards above only track legs/pelvis/
  # waist (TRACKED_BODY_NAMES) -- arm mocap data is too noisy to track
  # directly. Without any arm objective the policy has no reason to move the
  # arms at all (limp/frozen arms), so penalize whole-body angular momentum
  # instead: swinging the arms opposite the legs is exactly what keeps net
  # angular momentum low during natural human walking, so this pushes toward
  # arm movement without depending on the unreliable arm reference. Sensor is
  # the MJCF's built-in <subtreeangmom name="root_angmom" body="pelvis"/>.
  # Weight is an untuned starting point -- watch
  # Episode_Metrics/angular_momentum_mean during training and adjust if arm
  # motion looks suppressed (too negative) or erratic (too small).
  cfg.rewards["angular_momentum"] = RewardTermCfg(
    func=angular_momentum_penalty,
    weight=-0.01,
    params={"sensor_name": "robot/root_angmom"},
  )

  cfg.viewer.body_name = TORSO_BODY

  # The full-body VR M3.1 has many more self-collision geoms than the base
  # tracking config's G1-sized defaults (nconmax=35, njmax=250) budget for;
  # match the sizing already validated for this robot in the velocity task
  # (src/tasks/velocity/config/vr_m3_1/env_cfgs.py).
  cfg.sim.nconmax = 256
  cfg.sim.njmax = 700
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.mujoco.ccd_iterations = 50

  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg
