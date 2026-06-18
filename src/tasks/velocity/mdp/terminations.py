# This file contains code adapted from https://github.com/unitreerobotics/unitree_rl_mjlab
# Original Project Copyright 2026 Unitree
# Original Project License: Apache License 2.0
#
# --------------------------------------------------------------------------
# Modifications Copyright 2026 VinRobotics
#
# This file has been modified. Changes and additions are licensed under  Apache 2.0

"""Termination terms for the velocity task MDP."""

from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)
