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

import mjlab.terrains as terrain_gen
import mujoco
from mjlab.terrains import TerrainEntity, TerrainEntityCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=1.0,
    num_rows=10,
    num_cols=10,
    sub_terrains={
        "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.5),
        "gravel": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(0.0, 0.03),
            noise_step=0.005,
            horizontal_scale=0.15,
            vertical_scale=0.005,
        ),
        "box_obstacles": terrain_gen.BoxRandomSpreadTerrainCfg(
            proportion=0.25,
            num_boxes=50,
            box_width_range=(0.1, 0.5),
            box_length_range=(0.1, 0.5),
            box_height_range=(0.01, 0.05),
            platform_width=2.0,
        ),
    },
    add_lights=True,
)


if __name__ == "__main__":
    import mujoco.viewer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    terrain_cfg = TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=TERRAINS_CFG,
    )
    terrain = TerrainEntity(terrain_cfg, device=device)
    mujoco.viewer.launch(terrain.spec.compile())
