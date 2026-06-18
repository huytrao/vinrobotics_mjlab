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

"""AMP motion-seeded reset events and command-override events for mjlab environments."""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def zero_velocity_commands(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    command_name: str,
    num_robots_range: tuple[float, float] = (0.3, 1.0),
    zero_duration_range: tuple[float, float] = (2.0, 5.0),
) -> None:
    """Force velocity commands to zero for a random subset of robots for a random duration.

    Designed for use as an ``interval`` mode event.  Each time the event fires:

    1. A random count ``k`` is drawn from
       ``[num_robots_range[0] * len(env_ids), num_robots_range[1] * len(env_ids)]``.
    2. ``k`` robots are randomly selected from ``env_ids``.
    3. Their ``(vx, vy, wz)`` velocity command is set to zero.
    4. Their command-resample timer is set to a random value from
       ``zero_duration_range`` (seconds), so the zero persists until the timer
       expires and normal velocity resampling resumes.

    Args:
        env: The managed RL environment.
        env_ids: Environments whose interval timer fired (the candidate pool).
        command_name: Key of the velocity command term in the command manager
            (e.g. ``"twist"``).
        num_robots_range: ``(min_frac, max_frac)`` fraction of ``env_ids`` to
            zero.  E.g. ``(0.3, 1.0)`` zeros 30–100 % of triggered envs.
        zero_duration_range: ``(min_s, max_s)`` duration in seconds the zero
            command is held before the next natural resample.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    n = len(env_ids)
    if n == 0:
        return

    # Random count of robots to zero from the triggered batch.
    min_k = max(1, int(num_robots_range[0] * n))
    max_k = max(1, int(num_robots_range[1] * n))
    if min_k > max_k:
        max_k = min_k
    k = int(torch.randint(min_k, max_k + 1, (1,)).item())

    # Random subset of env_ids.
    perm = torch.randperm(n, device=env.device)
    selected_ids = env_ids[perm[:k]]

    # Zero the velocity command for the selected envs.
    cmd_term = env.command_manager.get_term(command_name)
    cmd_term.vel_command_b[selected_ids] = 0.0

    # Freeze the resample timer so the zero persists for zero_duration_range seconds.
    cmd_term.time_left[selected_ids] = (
        torch.empty(k, device=env.device).uniform_(*zero_duration_range)
    )
