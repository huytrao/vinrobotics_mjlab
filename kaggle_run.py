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

"""VR M3.1 motion-tracking training on Kaggle — single-file, paste-and-run.

Paste this ENTIRE file into ONE Kaggle code cell and run it. It does the
whole pipeline: GPU check -> fetch source from GitHub -> restore previous
checkpoints -> install deps -> train -> render video -> export ONNX -> zip
results to /kaggle/working/results_logs.zip.

Kaggle settings: Accelerator = GPU (T4 recommended; P100 auto-falls back to
the sparse jacobian workaround), Internet = ON.

To resume across sessions: download results_logs.zip from the Output tab,
add it as an Input dataset next session, run again.
"""

import glob
import os
import shutil
import stat
import subprocess
import sys
import zipfile

# ================== CONFIG ==================
TASK = "VR-M3-1-Tracking-Flat"
NUM_ENVS = 1024               # reduce to 256-512 if you hit GPU OOM
MAX_ITERS = 3000              # tune to session time remaining
MOTION_FILE = "data/motion/vr_m3_1_teleop_motion.npz"
GIT_URL = "https://github.com/huytrao/vinrobotics_mjlab.git"
USE_WANDB = False             # True needs Kaggle secret WANDB_API_KEY

LEARNING_RATE = 1.0e-3
SAVE_INTERVAL = 500
EPISODE_LENGTH_S = 10.0
RECORD_VIDEO_DURING_TRAIN = False
VIDEO_INTERVAL = 2000
VIDEO_LENGTH = 200
RUN_PLAY_VIDEO = True         # render an mp4 with the best checkpoint after training
EXPORT_ONNX = True

WORKING = "/kaggle/working"
PROJECT_DIR = f"{WORKING}/vinrobotics_mjlab"
# ============================================


def sh(cmd, **kw):
    """Run a command, streaming output; raise on failure unless check=False."""
    print("+", " ".join(map(str, cmd)), flush=True)
    kw.setdefault("check", True)
    return subprocess.run(list(map(str, cmd)), **kw)


def force_rmtree(path):
    def onerr(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass
    if os.path.exists(path):
        shutil.rmtree(path, onerror=onerr)


# ---------- 1. GPU check ----------
def detect_jacobian():
    jacobian = "auto"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        print("GPU(s):", out)
        for line in out.splitlines():
            gpu_name, cc = [x.strip() for x in line.split(",")]
            if int(cc.split(".")[0]) < 7:
                jacobian = "sparse"
                print(f"[WARN] {gpu_name} is Pascal (cc {cc}) -> forcing "
                      "--env.sim.mujoco.jacobian=sparse (dense tile-Cholesky "
                      "does not compile on sm_60). If training still crashes, "
                      "switch the accelerator to T4 x2.")
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("[WARN] nvidia-smi unavailable — no GPU? Check Kaggle Accelerator setting.")
    print("JACOBIAN =", jacobian)
    return jacobian


# ---------- 2. Fetch source ----------
def clone_fresh():
    os.chdir(WORKING)
    fresh = f"{WORKING}/_fresh_clone"
    force_rmtree(fresh)
    sh(["git", "clone", GIT_URL, fresh])
    force_rmtree(PROJECT_DIR)
    if os.path.exists(PROJECT_DIR):
        shutil.copytree(fresh, PROJECT_DIR, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(".git"))
        force_rmtree(fresh)
    else:
        os.rename(fresh, PROJECT_DIR)


def fetch_source():
    have_setup = os.path.exists(os.path.join(PROJECT_DIR, "setup.py"))
    have_motion = os.path.exists(os.path.join(PROJECT_DIR, MOTION_FILE))
    if have_setup and have_motion:
        print("Source + motion file already present, skipping clone.")
        return
    if have_setup:
        print("Stale checkout without the motion file -> re-cloning...")
    clone_fresh()


# ---------- 3. Restore previous session ----------
def restore_checkpoints():
    prev = glob.glob("/kaggle/input/**/results_logs.zip", recursive=True)
    if not prev:
        print("No previous results_logs.zip input -> training from scratch.")
        return False
    print("Restoring logs from:", prev[0])
    with zipfile.ZipFile(prev[0]) as z:
        z.extractall(PROJECT_DIR)
    ckpts = glob.glob(f"{PROJECT_DIR}/logs/rsl_rl/**/model_*.pt", recursive=True)
    print(f"Restored {len(ckpts)} checkpoints -> training will RESUME." if ckpts
          else "Zip had no checkpoints -> training from scratch.")
    return bool(ckpts)


# ---------- 4. Dependencies ----------
def install_deps():
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    sh(pip + ["--upgrade", "kaggle"], check=False)
    sh(pip + ["mjlab==1.4.0", "mujoco==3.8.1", "mujoco-warp==3.8.1",
              "warp-lang==1.13.0", "prettytable"])
    sh(pip + ["-e", ".", "--no-deps"])
    sh(["apt-get", "-qq", "install", "-y", "libegl1", "libgl1", "libosmesa6"],
       check=False, capture_output=True)

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    if USE_WANDB:
        from kaggle_secrets import UserSecretsClient
        os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")
    else:
        os.environ["WANDB_MODE"] = "offline"

    sh(pip + ["torch==2.5.1", "torchvision", "torchaudio",
              "--index-url", "https://download.pytorch.org/whl/cu121"], check=False)
    sh(pip + ["mlflow"], check=False)

    import torch
    print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
          "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")


# ---------- 5. Train ----------
def train(jacobian, resume):
    cmd = [sys.executable, "scripts/train.py", TASK,
           f"--env.scene.num-envs={NUM_ENVS}",
           f"--env.sim.mujoco.jacobian={jacobian}",
           f"--env.episode-length-s={EPISODE_LENGTH_S}",
           f"--agent.max-iterations={MAX_ITERS}",
           f"--agent.algorithm.learning-rate={LEARNING_RATE}",
           f"--agent.save-interval={SAVE_INTERVAL}",
           "--motion-file", MOTION_FILE,
           "--gpu-ids", "[0]"]
    if resume:
        cmd += ["--agent.resume", "True"]
    if RECORD_VIDEO_DURING_TRAIN:
        cmd += ["--video", "True",
                f"--video-interval={VIDEO_INTERVAL}",
                f"--video-length={VIDEO_LENGTH}"]
    sh(cmd)


def latest_checkpoint():
    ckpts = glob.glob("logs/rsl_rl/**/model_*.pt", recursive=True)
    ckpts.sort(key=os.path.getmtime)
    return ckpts[-1] if ckpts else None


# ---------- 6. Play video ----------
def play_video(ckpt):
    sh([sys.executable, "scripts/play.py", TASK,
        "--checkpoint-file", ckpt,
        "--motion-file", MOTION_FILE,
        "--video", "--video-length", VIDEO_LENGTH],
       check=False, timeout=300)
    vids = sorted(glob.glob("logs/**/*.mp4", recursive=True), key=os.path.getmtime)
    print("Videos:", vids[-3:] if vids else "none rendered")


# ---------- 7. Export ONNX ----------
def export_onnx(ckpt):
    """Inline export (scripts/export_policy.py lacks a --motion-file flag)."""
    import re
    from dataclasses import asdict
    from pathlib import Path

    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.tasks.tracking.mdp import MotionCommandCfg
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    checkpoint_dir = Path(os.path.dirname(ckpt))
    last_iter = int(re.search(r"model_(\d+)", os.path.basename(ckpt)).group(1))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(TASK, play=True)
    agent_cfg = load_rl_cfg(TASK)
    env_cfg.scene.num_envs = 1
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = MOTION_FILE

    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=env_cfg, device=device),
                             clip_actions=agent_cfg.clip_actions)
    runner = (load_runner_cls(TASK) or MjlabOnPolicyRunner)(
        env, asdict(agent_cfg), device=device)

    export_dir = checkpoint_dir / "exported"
    os.makedirs(export_dir, exist_ok=True)
    n = 0
    for i in range(0, last_iter + 1, SAVE_INTERVAL):
        cp = checkpoint_dir / f"model_{i}.pt"
        if not cp.exists():
            continue
        runner.load(str(cp), load_cfg={"actor": True}, strict=True, map_location=device)
        runner.export_policy_to_onnx(str(export_dir), filename=f"policy_{i}.onnx")
        n += 1
    env.close()
    print(f"Exported {n} ONNX models to {export_dir}")


# ---------- 8. Package results ----------
def package_results():
    out = f"{WORKING}/results_logs.zip"
    if os.path.exists(out):
        os.remove(out)
    sh(["zip", "-qr", out, "logs", "-x", "*wandb*"], check=False)
    if os.path.exists(out):
        print(f"Results: {out} ({os.path.getsize(out)/1e6:.1f} MB) — download it "
              "from the Output tab; add it as an Input next session to resume.")


def main():
    jacobian = detect_jacobian()
    fetch_source()
    os.chdir(PROJECT_DIR)

    motion_path = os.path.join(PROJECT_DIR, MOTION_FILE)
    assert os.path.exists(motion_path), (
        f"Motion file missing at {motion_path} even after cloning — check that "
        f"{GIT_URL} has data/motion/*.npz committed.")
    print(f"Motion file OK: {motion_path} ({os.path.getsize(motion_path)/1e6:.1f} MB)")

    resume = restore_checkpoints()
    install_deps()
    train(jacobian, resume)

    ckpt = latest_checkpoint()
    if ckpt is None:
        print("[ERROR] No checkpoint produced — inspect the training output above.")
    else:
        print("Latest checkpoint:", ckpt)
        if RUN_PLAY_VIDEO:
            play_video(ckpt)
        if EXPORT_ONNX:
            export_onnx(ckpt)
    package_results()


main()
