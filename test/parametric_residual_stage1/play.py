"""Play the VR-M3-1 parametric-residual checkpoint saved in this folder.

Reuses the exact env/agent patching from train_parametric_residual_scratch.py
(same directory) so the custom action space (parametric gait + residual) and
observation terms match what the checkpoint was trained with.
"""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train_parametric_residual_scratch import (  # noqa: E402
    TASK_ID,
    patch_parametric_scratch_config,
)

import mjlab.tasks  # noqa: E402,F401
import src.tasks  # noqa: E402,F401
from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # noqa: E402
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls  # noqa: E402
from mjlab.utils.torch import configure_torch_backends  # noqa: E402
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer  # noqa: E402

CHECKPOINT_DEFAULT = str(Path(__file__).resolve().parent / "model_2499.pt")


@dataclass(frozen=True)
class PlayConfig:
    checkpoint_file: str = CHECKPOINT_DEFAULT
    num_envs: int = 1
    device: str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"


def main() -> None:
    cfg = tyro.cli(PlayConfig)
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)

    class _PatchCfg:
        pass

    patch_cfg = _PatchCfg()
    patch_cfg.env = env_cfg
    patch_cfg.agent = agent_cfg
    patch_parametric_scratch_config(
        patch_cfg,
        num_envs=cfg.num_envs,
        max_iterations=None,
        logger="tensorboard",
        wandb_project="play",
    )
    env_cfg.scene.num_envs = cfg.num_envs

    # The stage1 patch narrows lin_vel_y/ang_vel_z to (0.0, 0.0), which the viser
    # joystick GUI can't render (its "Max" slider requires a value >= 0.1). Widen
    # just enough for the GUI; command sampling behavior for viewing is unaffected.
    twist_ranges = env_cfg.commands["twist"].ranges
    if twist_ranges.lin_vel_y[1] < 0.1:
        twist_ranges.lin_vel_y = (-0.1, 0.1)
    if twist_ranges.ang_vel_z[1] < 0.1:
        twist_ranges.ang_vel_z = (-0.1, 0.1)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        cfg.checkpoint_file, load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

    if cfg.viewer == "auto":
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        resolved_viewer = "native" if has_display else "viser"
    else:
        resolved_viewer = cfg.viewer

    if resolved_viewer == "native":
        NativeMujocoViewer(env, policy).run()
    elif resolved_viewer == "viser":
        ViserPlayViewer(env, policy).run()
    else:
        raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

    env.close()


if __name__ == "__main__":
    main()
