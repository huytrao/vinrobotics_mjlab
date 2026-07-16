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

"""Convert the cleaned teleop mocap CSV into an mjlab motion-tracking `.npz`.

`src/tasks/tracking/config/vr_m3_1/env_cfgs.py` registers a
`MotionCommandCfg` for task ``VR-M3-1-Tracking-Flat``. That command loads a
motion file produced by `mjlab.tasks.tracking.mdp.commands.MotionLoader`,
which expects an ``.npz`` with keys ``joint_pos``, ``joint_vel``,
``body_pos_w``, ``body_quat_w``, ``body_lin_vel_w``, ``body_ang_vel_w`` and
``fps`` (mirrors ``mjlab/scripts/csv_to_npz.py``, adapted to read this
repo's already-clean `ts_teleop_*` columns directly instead of a positional
CSV, and to replay through the VR M3.1 humanoid instead of the Unitree G1).

Joint order matches `Entity.joint_names` for
`get_vr_m3_1_implicit_actuator_robot_cfg()` (the 27 actuated joints; the
MJCF's unactuated `head_yaw_joint`/`head_pitch_joint` aren't part of the
entity's joint set and are skipped entirely).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_conjugate, quat_mul

from src.tasks.tracking.config.vr_m3_1.env_cfgs import vr_m3_1_flat_tracking_env_cfg

# MJCF joint order (excludes the free `base_joint`).
JOINT_NAMES = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_pitch_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_pitch_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_pitch_joint",
  "left_wrist_yaw_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_pitch_joint",
  "right_wrist_yaw_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
]
# Joints with no mocap tracker: held fixed at 0 rad throughout the motion.
# (head_yaw_joint/head_pitch_joint exist in the MJCF but have no actuator on
# this robot variant, so they aren't part of the entity's joint set at all.)
UNTRACKED_JOINTS: set[str] = set()

PELVIS_ENTITY = "ts_teleop_pelvis"


def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
  """Angular velocity (B, 3) from a sequence of wxyz quaternions (B, 4)."""
  q_prev, q_next = rotations[:-2], rotations[2:]
  q_rel = quat_mul(q_next, quat_conjugate(q_prev))
  omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
  return torch.cat([omega[:1], omega, omega[-1:]], dim=0)


def load_teleop_csv(
  csv_path: Path, drop_frozen: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
  """Returns (root_pos [T,3], root_quat_wxyz [T,4], joint_pos [T,J], fps)."""
  df = pd.read_csv(csv_path)
  if drop_frozen and "is_frozen" in df.columns:
    df = df[~df["is_frozen"].astype(bool)].reset_index(drop=True)

  root_pos = df[[f"{PELVIS_ENTITY}:position_{i}" for i in range(3)]].to_numpy(
    dtype=np.float32
  )
  root_quat = df[[f"{PELVIS_ENTITY}:quaternion_{i}" for i in range(4)]].to_numpy(
    dtype=np.float32
  )
  root_quat /= np.linalg.norm(root_quat, axis=1, keepdims=True)

  joint_pos = np.zeros((len(df), len(JOINT_NAMES)), dtype=np.float32)
  for j, name in enumerate(JOINT_NAMES):
    if name in UNTRACKED_JOINTS:
      continue
    col = f"ts_teleop_{name}:joint_angular_position"
    joint_pos[:, j] = df[col].to_numpy(dtype=np.float32)

  if "time" in df.columns:
    dt = np.diff(df["time"].to_numpy(dtype=np.float64))
    fps = float(1.0 / np.median(dt[dt > 0]))
  else:
    fps = 47.8  # data_clean.meta.json nominal capture rate.

  return root_pos, root_quat, joint_pos, fps


def main(
  csv_path: str = "huggingface_upload/data/data_clean.csv",
  output_path: str = "data/motion/vr_m3_1_teleop_motion.npz",
  drop_frozen: bool = True,
  device: str = "cpu",
) -> None:
  """Replay the cleaned teleop CSV through the VR M3.1 sim and save a motion npz.

  Args:
    csv_path: Path to `data_clean.csv` (produced by `scripts/clean_mocap_data.py`).
    output_path: Where to write the `.npz` motion file consumed by
      `--motion-file` in `scripts/train.py VR-M3-1-Tracking-Flat`.
    drop_frozen: Skip frames flagged `is_frozen` (repeated mocap-dropout
      frames) instead of replaying them as zero-velocity holds.
    device: torch device for the replay sim ("cpu" works fine; this is a
      single-env, non-realtime replay, not RL training).
  """
  root_pos_np, root_quat_np, joint_pos_np, fps = load_teleop_csv(
    Path(csv_path), drop_frozen=drop_frozen
  )
  num_frames = root_pos_np.shape[0]
  dt = 1.0 / fps
  print(f"[INFO] Loaded {num_frames} frames from {csv_path} at {fps:.2f} fps")

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = dt

  scene = Scene(vr_m3_1_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(JOINT_NAMES, preserve_order=True)[0]

  root_pos = torch.from_numpy(root_pos_np).to(device=device, dtype=torch.float32)
  root_quat = torch.from_numpy(root_quat_np).to(device=device, dtype=torch.float32)
  joint_pos = torch.from_numpy(joint_pos_np).to(device=device, dtype=torch.float32)

  root_lin_vel = torch.gradient(root_pos, spacing=dt, dim=0)[0]
  root_ang_vel = _so3_derivative(root_quat, dt)
  joint_vel = torch.gradient(joint_pos, spacing=dt, dim=0)[0]

  log: dict[str, list[np.ndarray]] = {
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }

  scene.reset()
  for t in range(num_frames):
    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = root_pos[t]
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = root_quat[t]
    root_states[:, 7:10] = root_lin_vel[t]
    root_states[:, 10:] = root_ang_vel[t]
    robot.write_root_state_to_sim(root_states)

    full_joint_pos = robot.data.default_joint_pos.clone()
    full_joint_vel = robot.data.default_joint_vel.clone()
    full_joint_pos[:, robot_joint_indexes] = joint_pos[t]
    full_joint_vel[:, robot_joint_indexes] = joint_vel[t]
    robot.write_joint_state_to_sim(full_joint_pos, full_joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
    log["body_lin_vel_w"].append(
      robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
    )
    log["body_ang_vel_w"].append(
      robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
    )

    if (t + 1) % 500 == 0 or t + 1 == num_frames:
      print(f"[INFO] Replayed {t + 1}/{num_frames} frames")

  out = Path(output_path)
  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    out,
    fps=np.array(fps, dtype=np.float32),
    **{k: np.stack(v, axis=0) for k, v in log.items()},
  )
  print(f"[INFO] Saved motion npz to {out.resolve()}")


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
