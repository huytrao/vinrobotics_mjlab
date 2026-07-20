# 2-hour full-body training strategy — rationale

Target: train `VR-M3-1-Flat` (full-body, 27+ DOF) to a basic velocity-tracking walk inside roughly a 2-hour
wall-clock budget, using:

```bash
MUJOCO_GL=egl python scripts/train.py VR-M3-1-Flat \
    --gpu-ids '[0, 1]' \
    --env.scene.num-envs=4096 \
    --agent.max-iterations=2000 \
    --agent.save-interval=100 \
    --agent.seed=42
```

Three papers were pulled as reference for this (saved under `reference/`). Below is what each one says, whether
its technique was applied to this repo's code, and why.

## 1. Learning to Walk in Minutes Using Massively Parallel Deep RL (Rudin et al., ETH Zürich / NVIDIA, 2021)
`reference/2109.11978_learning_to_walk_in_minutes.pdf`

This is the paper the `rsl_rl` library (this repo's RL runner) descends from: run thousands of environment copies
in parallel on a single GPU and update PPO on the aggregated batch, turning what used to be a multi-hour/multi-day
CPU-cluster job into a ~20-minute single-workstation job for a quadruped. The core lesson is that wall-clock
training time is driven by **how much simulated experience you can collect per second**, not by clever
algorithmic tricks — so the parallelism budget (`num_envs`) has to be sized to the task, and any curriculum or
schedule expressed in a *fixed iteration count* has to be re-derived when the iteration budget changes, or it
silently stops doing what it was designed to do.

**Applied — this is the one code change made in this session.** `num_envs=4096` across the requested run already
follows the paper's "scale parallelism" recipe; nothing to change there. What *didn't* follow it was the
curriculum: `src/tasks/velocity/config/vr_m3_1/env_cfgs.py`'s 4-stage velocity-range curriculum triggers speed
increases at absolute env-step counts (`5000*32`, `10000*32`, `15000*32`), tuned for the task's default
`max_iterations=20_001` (see `rl_cfg.py`). At `--agent.max-iterations=2000`, none of those thresholds — the first
one alone requires 160,000 env-steps, i.e. iteration 5000 — are ever reached. The entire 2-hour run would have
stayed on stage 0 (forward speed capped at 0.5 m/s), silently, with no error or warning.

**Fix**: rescaled all three thresholds by `2000 / 20_000 = 0.1` → `500*32`, `1000*32`, `1500*32`. This preserves
the original curriculum's *shape* (the same fraction of total training time is spent at each speed stage) while
fitting it inside the compressed 2000-iteration budget, so by the end of the run the commanded velocity range has
actually ramped up toward the full 2 m/s the reward function and network are being trained against. If
`--agent.max-iterations` is changed again, rescale the three `step` values in that curriculum block by
`(new_max_iterations / 20_000)`.

Backup of the pre-edit file: `src/tasks/velocity/config/vr_m3_1/env_cfgs.py.bak`.

## 2. Staggered Environment Resets Improve Massively Parallel On-Policy RL (Bharthulwar, Tao, Su — Harvard/UCSD, 2025)
`reference/2511.21011_staggered_environment_resets.pdf`

This paper's finding matches this repo's exact regime unusually closely: it studies PPO with a rollout length `K`
much shorter than the episode horizon `H` (here `K = num_steps_per_env = 32`, `H ≈ episode_length_s / (timestep *
decimation) = 20 / (0.005*4) = 1000` control steps — `K/H ≈ 0.03`, deep in the range the paper flags as
problematic). When thousands of parallel envs all reset in lockstep at episode end, every PPO batch becomes
"temporally homogeneous" — early-episode data, then mid-episode, then late-episode, then an abrupt reset back to
early-episode — which the paper shows destabilizes the value function and, worse, causes wall-clock convergence
to *stop improving or regress* as `num_envs` grows past roughly 1000–6000 (measured on their manipulation/
locomotion benchmarks). Their fix is to desynchronize (stagger) each environment's initial phase within the
episode horizon so every batch contains a mix of episode phases instead of a narrow slice.

**Not applied to code.** `train.py` already sets `runner.learn(..., init_at_random_ep_len=True)`, which is
`rsl_rl`'s built-in equivalent of the paper's *initial* staggering step (each env starts training at a randomized
point in its episode rather than all at `t=0`) — so the cheapest, already-available mitigation is already on.
What the paper adds beyond that is staggering resets *continuously* through training (not just at the start),
which requires changing how `rsl_rl`'s/`mjlab`'s environment runner decides which parallel env slots reset when —
that logic lives inside the pinned `mjlab==1.4.0`/`rsl_rl` dependencies, not in this repo's task-config layer.
Patching a version-pinned external library's core rollout loop correctly, without silently breaking checkpoint
compatibility or the multi-GPU (`torchrunx`) path, is not something to do blind inside a 2-hour-budget session.
Flagged here as the highest-value follow-up if `num_envs` is pushed higher (8192+) and wall-clock convergence
stops scaling — that would match this paper's predicted symptom.

## 3. Learning Sim-to-Real Humanoid Locomotion in 15 Minutes (Seo, Sferrazza, Chen, Shi, Duan, Abbeel — Amazon FAR, 2025)
`reference/2512.01996_sim_to_real_humanoid_15min.pdf`

Humanoid-specific (Unitree G1 / Booster T1) and the most aggressive result of the three: full sim-to-real
locomotion policies in 15 minutes on a single RTX 4090. The headline technique is switching from on-policy PPO to
tuned off-policy algorithms (FastSAC/FastTD3) that reuse replay-buffer data instead of discarding it after each
update, which they show consistently beats PPO in wall-clock time once domain randomization gets aggressive
(rough terrain, frequent pushes). Secondary findings: a deliberately minimal reward function (fewer than 10
terms, vs. the ~20 terms in this repo's reward set) and per-episode-length curriculum-ramped penalty weights.

**Not applied to code.** The headline technique is an RL-algorithm swap (PPO → FastSAC/FastTD3), which
`rsl_rl`'s `MjlabOnPolicyRunner` (and this repo's `VelocityOnPolicyRunner` subclass) does not implement — adopting
it would mean integrating a different RL library, not editing this task's config. The reward-simplification
finding is directly relevant but was **not** applied either: this repo's ~20-term reward function is already
tuned for the VR M3.1 (per the `unitree_rl_mjlab`-derived recipe this repo credits in `README.md`), and gutting it
down to <10 terms inside this session, untested, risks producing a *worse* policy than just running the existing
recipe for longer — that's a separate experiment to run deliberately with its own comparison, not a same-session
drive-by edit. Kept here for reference if the rescaled-curriculum run in this session doesn't converge to a
walking gait inside the 2-hour budget.

## Net result for this session

Only change #1 (curriculum rescale) was made to code, because it's the only one of the three that is (a) fully
within this repo's own config layer, (b) low-risk (only changes *when* the existing, already-tuned curriculum
stages trigger, not what they do), and (c) directly derived from what the requested run (`--agent.max-iterations
2000`) needed to not silently under-train. The other two are documented above as evaluated-and-deferred, with the
concrete symptom to watch for that would justify revisiting each.

## Run command

```bash
MUJOCO_GL=egl python scripts/train.py VR-M3-1-Flat \
    --gpu-ids '[0, 1]' \
    --env.scene.num-envs=4096 \
    --agent.max-iterations=2000 \
    --agent.save-interval=100 \
    --agent.seed=42
```

No CLI flags changed — the curriculum rescale lives in the task's `env_cfgs.py`, so it applies automatically
whenever `VR-M3-1-Flat` is trained. `save-interval=100` gives 20 checkpoints across the run; use
`scripts/play.py` on an intermediate one to check gait quality before committing to the full 2000 iterations.
Wall-clock time for 2000 iterations was not benchmarked on this machine — if you have a few minutes to spare
first, run with `--agent.max-iterations=50` once to measure iterations/second and confirm 2000 fits your actual
2-hour window before committing the full run.
