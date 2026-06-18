# This file contains code adapted from https://github.com/unitreerobotics/unitree_rl_mjlab
# Original Project Copyright 2026 Unitree
# Original Project License: Apache License 2.0
#
# --------------------------------------------------------------------------
# Modifications Copyright 2026 VinRobotics
#
# This file has been modified. Changes and additions are licensed under  Apache 2.0

"""On-policy runner for the velocity task."""

import os
import wandb
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    onnx_path = os.path.join(policy_path, filename)
    self.export_policy_to_onnx(policy_path, filename)
    if not os.path.exists(onnx_path):
      return  # export was skipped (residual policy)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    metadata = get_base_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
