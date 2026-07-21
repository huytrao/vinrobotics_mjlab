# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, Cursor, etc.) working in this repository. This file is
committed to version control — `CLAUDE.md` is a local-only pointer to this file (it's gitignored, so it never
leaves your machine; keep this file as the actual source of truth).

## What this repo is

`vr_mjlab`: a reinforcement-learning framework for legged-robot locomotion, built on
[`mjlab`](https://github.com/mujocolab/mjlab.git) (GPU-accelerated MuJoCo Warp physics with an Isaac-Lab-style
manager API) and [`rsl_rl`](https://github.com/leggedrobotics/rsl_rl.git) (PPO). It currently supports the
VinRobotics **M3.1** humanoid, in a full-body (27+ DOF) variant and a lower-body-only **12DOF** variant. Policies
are trained in simulation, exported to ONNX, and consumed downstream by `vinrobotics_mjlab_deploy` for sim-to-real.

The codebase is a thin, self-contained task/config layer on top of `mjlab` — most of the physics engine, RL
runner internals, and manager framework live in the `mjlab` and `rsl_rl` packages (installed as dependencies, not
vendored here).

## Workflow: starting a new experiment

`original_code` is the base codebase to branch experiments from — it's a verbatim copy of
[`VinRobotics/vinrobotics_mjlab`](https://github.com/VinRobotics/vinrobotics_mjlab)'s `main` (the upstream/canonical
repo this fork tracks), pushed here as its own branch so it's available without adding a second remote. `main` on
*this* repo (`huytrao/vinrobotics_mjlab`) has diverged significantly from upstream and is not the experiment base.

Every time a new experiment starts, branch from `original_code` (`git checkout -b <experiment-name>
original_code`) rather than committing experiment-specific changes (new task configs, checkpoints, tuning) directly
on `original_code` or `main`.

To refresh `original_code` with newer upstream changes later:
```bash
git fetch https://github.com/VinRobotics/vinrobotics_mjlab.git main
git push origin FETCH_HEAD:original_code
```

## Install

```bash
pip install mjlab==1.4.0 mujoco==3.8.1 mujoco-warp==3.8.1 warp-lang==1.13.0
pip install -e .
```

Requires Python 3.8+ and CUDA 12.x for GPU-accelerated simulation (CPU fallback works but is slow — this repo is
built around massively parallel GPU envs).

## Common commands

```bash
# List all registered task IDs
python scripts/list_envs.py

# Train (basic workflow is Train -> Play)
python scripts/train.py VR-M3-1-12DOF-Flat --env.scene.num-envs=4096

# Train on multiple GPUs
python scripts/train.py VR-M3-1-Flat --gpu-ids '[0, 1]' --env.scene.num-envs=4096

# Resume from last checkpoint
python scripts/train.py VR-M3-1-12DOF-Flat --agent.resume=True

# Visualize a trained policy in MuJoCo
python scripts/play.py VR-M3-1-12DOF-Flat \
  --checkpoint_file=logs/rsl_rl/vr_m3_1_12dof_velocity/<date_time>/model_<iter>.pt

# Batch-export ONNX from a checkpoint range
python scripts/export_policy.py <TASK_ID> --checkpoint-dir=<dir> --start=0 --end=20000 --step=1000

# Convert cleaned teleop mocap CSV into an mjlab motion-tracking .npz (for tracking tasks)
python scripts/mocap_csv_to_motion_npz.py
```

All CLI configs are `tyro`-parsed dataclasses, so any nested field of `env` (`ManagerBasedRlEnvCfg`) or `agent`
(`RslRlBaseRunnerCfg`) can be overridden from the command line, e.g. `--env.episode-length-s=10.0`,
`--agent.max-iterations=2000`, `--agent.algorithm.learning-rate=1e-3`, `--env.sim.mujoco.solver=cg`.

Set `MUJOCO_GL=egl` for headless rendering (required in CI/cloud notebooks without a display, e.g. Kaggle).

### Linting

`flake8` (max-line-length 120, project uses 2-space indentation and black-compatible formatting; see `.flake8`)
and `isort` (`.isort.cfg`, black profile) are configured but there is no test suite or CI runner script in the repo.

### Kaggle training

`kaggle_run.py` is a single-file, paste-into-one-cell pipeline for training on Kaggle's free GPU (T4 recommended;
Pascal GPUs like P100 get an automatic sparse-jacobian/CG-solver workaround — see `detect_pascal_flags()`). It
clones the repo fresh, restores checkpoints from a previous session's `results_logs.zip` if provided as an input
dataset, trains for a configured `MAX_ITERS`, renders a play video, exports ONNX, and re-zips logs for the next
session. Tune `NUM_ENVS`/`MAX_ITERS`/`EPISODE_LENGTH_S` at the top of the file to fit the remaining session time.

## Architecture

### Task registration

Tasks are plain Python packages under `src/tasks/<task_family>/config/<robot>/`, auto-discovered by
`mjlab.utils.lab_api.tasks.importer.import_packages` (see `src/tasks/__init__.py`) — a new task becomes visible to
`scripts/list_envs.py`/`train.py`/`play.py` just by adding a config module with a registration call; there is no
central registry file to edit. Two task families exist:

- **`velocity`** (`src/tasks/velocity/`) — velocity-tracking locomotion (walk at a commanded x/y/yaw velocity).
  Task IDs: `VR-M3-1-Flat`, `VR-M3-1-Rough`, `VR-M3-1-12DOF-Flat`, `VR-M3-1-12DOF-Rough`.
- **`tracking`** (`src/tasks/tracking/`) — motion-tracking (imitate a mocap clip via `MotionCommandCfg`). Task ID:
  `VR-M3-1-Tracking-Flat`, driven by `data/motion/vr_m3_1_teleop_motion.npz`.

Each `config/<robot>/` directory has:
- `env_cfgs.py` — builds the full `ManagerBasedRlEnvCfg` (scene, observations, actions, commands, events/domain
  randomization, rewards, terminations, curriculum, sim params) **inline**, deliberately not deriving from a
  shared base config, so edits to one robot/task's env don't silently affect another.
- `rl_cfg.py` — the PPO/`RslRlOnPolicyRunnerCfg` (network architecture, PPO hyperparameters, `max_iterations`,
  `experiment_name`, logger).

Env configs take a `play: bool` flag: `play=True` disables domain randomization/pushes/curriculum and widens the
terrain grid, used by `scripts/play.py` for evaluation.

### Shared MDP terms

`src/tasks/velocity/mdp/` holds the reusable building blocks referenced by every velocity env config: reward
functions (`rewards.py`), observations (`observations.py`), termination conditions (`terminations.py`), events/DR
(`events.py`), curriculum functions (`curriculums.py`), metrics (`metrics.py`), and the velocity command
generator (`velocity_command.py`). It re-exports `mjlab.envs.mdp` alongside these, so env configs import
everything through `src.tasks.velocity.mdp as mdp`.

### Robot assets

`src/assets/robots/vr_m3_1/` holds the MJCF (`xmls/vr_m3_1.xml` full body, `xmls/vr_m3_1_12dof.xml` lower body),
mesh assets, and per-robot constants/actuator configs (`vr_m3_1_constants.py`, `vr_m3_1_12dof_constants.py`)
exposing `get_vr_m3_1_*_implicit_actuator_robot_cfg()` factory functions and `VR_M3_1*_ACTION_SCALE` used by env
configs to build the `scene.entities["robot"]` and the `JointPositionActionCfg`.

### Reward/curriculum design (velocity task)

Rewards combine velocity tracking (`track_linear_x/y`, `track_angular_z`) with gait-quality shaping terms tuned
for biped walking (`feet_air_time_biped`, `feet_close_xy`, `foot_clearance`, `foot_slip`, `knee_motion`, a
`variable_posture`/`pose` term with separate standing/walking/running joint-deviation targets) and standard
penalties (orientation, joint torque/acceleration, action rate, self-collision, termination). The velocity
command range ramps up over training via a 4-stage curriculum (`commands_vel` in `curriculums.py`), keyed on
`step = iteration * num_steps_per_env` — if you change `num_steps_per_env` or compress `max_iterations` for a
shorter training run, the curriculum stage triggers in `env_cfgs.py` need to be rescaled accordingly or later
stages will never be reached. See `docs/training_strategy_2h_fullbody.md` for a worked example of rescaling this
for a compressed training budget, with the research rationale behind it.

### Custom runner

`src/tasks/velocity/rl/runner.py` (`VelocityOnPolicyRunner`) subclasses `mjlab`'s `MjlabOnPolicyRunner` to also
export `policy.onnx` (with attached deployment metadata) alongside every `.pt` checkpoint save, so every saved
checkpoint is immediately deployable without a separate export step.
