from typing import Optional

import torch


class MjlabEnv:
    """Wrapper for vr_mjlab (mjlab / IsaacLab-style manager API) environments to be
    compatible with FastTD3. Mirrors IsaacLabEnv's interface, but constructs the
    environment directly (no isaaclab.app.AppLauncher) and reads the "actor"/"critic"
    observation groups that vr_mjlab's task configs use instead of "policy"/"critic".
    """

    def __init__(
        self,
        task_name: str,
        device: str,
        num_envs: int,
        seed: int,
        action_bounds: Optional[float] = None,
    ):
        import mjlab.tasks  # noqa: F401  registers upstream mjlab tasks
        import src.tasks  # noqa: F401  registers vr_mjlab (VR-*) tasks
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg

        env_cfg = load_env_cfg(task_name)
        env_cfg.scene.num_envs = num_envs
        env_cfg.seed = seed
        self.seed = seed

        self.envs = ManagerBasedRlEnv(cfg=env_cfg, device=device)

        self.num_envs = self.envs.num_envs
        self.max_episode_steps = self.envs.max_episode_length
        self.action_bounds = action_bounds
        self.num_obs = self.envs.single_observation_space.spaces["actor"].shape[0]
        self.asymmetric_obs = "critic" in self.envs.single_observation_space.spaces
        if self.asymmetric_obs:
            self.num_privileged_obs = self.envs.single_observation_space.spaces[
                "critic"
            ].shape[0]
        else:
            self.num_privileged_obs = 0
        self.num_actions = self.envs.single_action_space.shape[0]

    def reset(self, random_start_init: bool = True) -> torch.Tensor:
        obs_dict, _ = self.envs.reset()
        # NOTE: decorrelate episode horizons like RSL-RL (see mjlab's own
        # train.py, which passes init_at_random_ep_len=True to the PPO runner).
        if random_start_init:
            self.envs.episode_length_buf = torch.randint_like(
                self.envs.episode_length_buf, high=int(self.max_episode_steps)
            )
        return obs_dict["actor"]

    def reset_with_critic_obs(self) -> tuple[torch.Tensor, torch.Tensor]:
        obs_dict, _ = self.envs.reset()
        return obs_dict["actor"], obs_dict["critic"]

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if self.action_bounds is not None:
            actions = torch.clamp(actions, -1.0, 1.0) * self.action_bounds
        obs_dict, rew, terminated, truncated, infos = self.envs.step(actions)
        dones = (terminated | truncated).to(dtype=torch.long)
        obs = obs_dict["actor"]
        critic_obs = obs_dict["critic"] if self.asymmetric_obs else None
        info_ret = {"time_outs": truncated, "observations": {"critic": critic_obs}}
        # NOTE: mirrors IsaacLabEnv — there's no way to get pre-reset raw
        # observations back out of ManagerBasedRlEnv.step's auto-reset, so the
        # post-reset obs is reused as "raw" here too.
        info_ret["observations"]["raw"] = {
            "obs": obs,
            "critic_obs": critic_obs,
        }
        return obs, rew, dones, info_ret

    def render(self):
        raise NotImplementedError("Rendering is not supported for mjlab environments.")
