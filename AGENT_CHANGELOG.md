# Agent Changelog

Log of notable changes made to this repository by AI coding agents. Newest entries first. Each entry: date,
agent/tool, what changed, and why — keep entries to a few lines; full detail belongs in commit messages/diffs.

## 2026-07-22 — Claude Code

- Fixed a regression from the previous entry's curriculum rescale in
  `src/tasks/velocity/config/vr_m3_1/env_cfgs.py`: the `500/1000/1500`-iteration thresholds were tuned for a
  one-off `--agent.max-iterations=2000` run but got committed as the permanent default, while `rl_cfg.py` kept its
  default `max_iterations=20_001` — so anyone training `VR-M3-1-Flat`/`-Rough` without that CLI override would
  have the curriculum finish ramping to full speed by ~7.5% of training instead of spreading across it. Replaced
  the hardcoded absolute steps with `int(0.25/0.50/0.75 * max_iterations * num_steps_per_env)`, reading
  `max_iterations`/`num_steps_per_env` directly from `rl_cfg.py`'s `vr_m3_1_ppo_runner_cfg()`, so the two files
  can't drift out of sync again if the default `max_iterations` changes. Verified against `VR-M3-1-Flat` current
  default (20,001 iterations) — stages now land at 160008/320016/480024, i.e. 25/50/75%. Note: this still does not
  auto-adjust for a one-off CLI `--agent.max-iterations=N` override, since `env_cfg` is built before CLI parsing —
  that limitation is unchanged and still requires a matching config override.
- Given the limitation above, also changed `rl_cfg.py`'s checked-in default `max_iterations` for `vr_m3_1` from
  `20_001` to `2_001` (and `save_interval` 1000 → 100) to match this branch's actual target: a ~2h Kaggle run at
  `--agent.max-iterations=2001`. With this change the dynamically-computed curriculum thresholds match the real
  run length by default, with no CLI-vs-config mismatch. This is a real behavior change to the checked-in
  default — anyone running `VR-M3-1-Flat`/`-Rough` without an explicit `--agent.max-iterations` override now gets
  a ~2001-iteration run instead of ~20,001.

## 2026-07-21 — Claude Code (2)

- Swapped `AGENTS.md`/`CLAUDE.md` roles: `AGENTS.md` now holds the full, committed instructions; `CLAUDE.md` is a
  short pointer to it. Reason: `CLAUDE.md` is in `.gitignore` (local-only by repo convention), so content living
  only there would never reach other contributors or agents pulling the repo.
- Downloaded 3 reference papers into `reference/` (see `docs/training_strategy_2h_fullbody.md` for how each was
  evaluated): Rudin et al. 2021 "Learning to Walk in Minutes", Bharthulwar/Tao/Su 2025 "Staggered Environment
  Resets", Seo et al. 2025 (Amazon FAR) "Learning Sim-to-Real Humanoid Locomotion in 15 Minutes".
- Rescaled the velocity curriculum thresholds in `src/tasks/velocity/config/vr_m3_1/env_cfgs.py` (full-body
  `VR-M3-1-Flat`/`VR-M3-1-Rough`) 10x — `5000/10000/15000` iterations → `500/1000/1500` — to match a compressed
  `--agent.max-iterations=2000` run instead of the default 20,001; otherwise the curriculum's later speed stages
  are never reached and the policy silently trains on the slow-walk range for the entire run. Backup of the
  pre-edit file kept at `src/tasks/velocity/config/vr_m3_1/env_cfgs.py.bak`. Full rationale, including why the
  other 2 papers' techniques were evaluated but *not* applied to code this session, is in
  `docs/training_strategy_2h_fullbody.md`.

## 2026-07-21 — Claude Code

- Added `CLAUDE.md` (via the `init` skill): commands (install, train/play/export/list-envs, lint config, Kaggle
  pipeline) and architecture notes (task auto-registration, per-robot `env_cfgs.py`/`rl_cfg.py` split, shared MDP
  terms, reward/curriculum design, custom ONNX-exporting runner).
- Added `AGENTS.md` as a pointer to `CLAUDE.md` for agents that look for that filename specifically.
- Added this changelog file.
