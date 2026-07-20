# Agent Changelog

Log of notable changes made to this repository by AI coding agents. Newest entries first. Each entry: date,
agent/tool, what changed, and why — keep entries to a few lines; full detail belongs in commit messages/diffs.

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
