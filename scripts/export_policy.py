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

"""Script to batch-export ONNX models from a range of checkpoints."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class ExportConfig:
  checkpoint_dir: str = ""
  """Path to the directory containing model_*.pt checkpoints."""
  start: int = 0
  """First checkpoint iteration to export (n)."""
  end: int = 0
  """Last checkpoint iteration to export (m)."""
  step: int = 100
  """Step between checkpoint iterations (k)."""
  device: str | None = None
  num_envs: int = 1


def run_export(task_id: str, cfg: ExportConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  env_cfg.scene.num_envs = cfg.num_envs

  checkpoint_dir = Path(cfg.checkpoint_dir)
  if not checkpoint_dir.is_dir():
    raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

  # Collect checkpoint files in the requested range.
  checkpoints: list[Path] = []
  for i in range(cfg.start, cfg.end + 1, cfg.step):
    cp = checkpoint_dir / f"model_{i}.pt"
    if cp.exists():
      checkpoints.append(cp)
    else:
      print(f"[WARN] Checkpoint not found, skipping: {cp.name}")

  if not checkpoints:
    print("[ERROR] No checkpoints found in the specified range.")
    return

  print(f"[INFO] Found {len(checkpoints)} checkpoints to export")

  # Create environment and runner once.
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)

  export_dir = checkpoint_dir / "exported"
  os.makedirs(export_dir, exist_ok=True)

  for cp in checkpoints:
    print(f"[INFO] Exporting {cp.name} ...")
    runner.load(str(cp), load_cfg={"actor": True}, strict=True, map_location=device)

    iteration = cp.stem.replace("model_", "")
    runner.export_policy_to_onnx(str(export_dir), filename=f"policy_{iteration}.onnx")
    print(f"  -> {export_dir / f'policy_{iteration}.onnx'}")

  env.close()
  print(f"[INFO] Done. Exported {len(checkpoints)} ONNX models to {export_dir}")


def main():
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    ExportConfig,
    args=remaining_args,
    default=ExportConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )

  run_export(chosen_task, args)


if __name__ == "__main__":
  main()
