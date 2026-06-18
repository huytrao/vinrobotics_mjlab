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

"""VR M3 12DOF robot constants.

This module configures the VR M3 1 12DOF humanoid.
"""

from pathlib import Path
import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg, DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

VR_M3_1_12DOF_XML: Path = Path(__file__).parent / "xmls" / "vr_m3_1_12dof.xml"
assert VR_M3_1_12DOF_XML.exists(), f"VR M3 12DOF XML not found at {VR_M3_1_12DOF_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    return assets


def _name_collision_geoms(body: mujoco.MjsBody) -> None:
    """Assign explicit names to unnamed collision-class geoms.

    The M3 MJCF uses unnamed geoms with ``class="vr_m3_1_collision"``.
    CollisionCfg needs named geoms for regex matching, so we assign names
    following the M3 convention: ``{body_name}_collision_{n}``.
    """
    count = 0
    for geom in body.geoms:
        if geom.classname.name == "vr_m3_1_collision":
            count += 1
            geom.name = f"{body.name}_collision_{count}"
    for child in body.bodies:
        _name_collision_geoms(child)


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(VR_M3_1_12DOF_XML))
    spec.assets = get_assets(spec.meshdir)

    # Name collision geoms so CollisionCfg regex patterns work.
    # _name_collision_geoms(spec.body("pelvis"))

    # Foot sites for clearance/swing/slip rewards.
    for side in ("left", "right"):
        ankle = spec.body(f"{side}_ankle_roll_link")
        ankle.add_site(
            name=f"{side}_foot",
            pos=[0, 0, -0.04],  # Bottom of foot in ankle_roll_link frame
            size=[0.01],
        )

    pelvis = spec.body("pelvis")
    pelvis.add_site(
        name="heightmap_site",
        pos=[0.5, 0, 0],
        size=[0.01],
    )

    return spec


# ==================================================== #
#              VR-M3 Robot Configurations              #
# ==================================================== #


MOTOR_SPECS = {
    "WAIST_MOTOR": {
        "armature": 0.5804369,
        "max_vel": 4.18,
        "max_tau": 102.0,
        "saturation_tau": 102.0 * 1.5,
    },
    "ARM_MOTOR_1": {
        "armature": 0.3896782,
        "max_vel": 4.29,
        "max_tau": 66.0,
        "saturation_tau": 66.0 * 1.5,
    },
    "ARM_MOTOR_2": {
        "armature": 0.1142512,
        "max_vel": 5.13,
        "max_tau": 34.0,
        "saturation_tau": 34.0 * 1.5,
    },
    "ARM_MOTOR_3": {
        "armature": 0.0836482,
        "max_vel": 6.17,
        "max_tau": 11.0,
        "saturation_tau": 11.0 * 1.5,
    },
    "LEG_MOTOR_1": {
        "armature": 0.1404,
        "max_vel": 14.653,
        "max_tau": 360.0,
        "saturation_tau": 816.667,
    },
    "LEG_MOTOR_2": {
        "armature": 0.02864,
        "max_vel": 31.4,
        "max_tau": 130.0,
        "saturation_tau": 389.875,
    },
    "LEG_MOTOR_3": {
        "armature": 0.03006 / 2,
        "max_vel": 16.747,
        "max_tau": 60.0,
        "saturation_tau": 239.985,
    },
}

VR_M3_1_ACTUATOR_NAMES = {
    "hip_pitch": "LEG_MOTOR_1",
    "hip_roll": "LEG_MOTOR_1",
    "hip_yaw": "LEG_MOTOR_2",
    "knee_pitch": "LEG_MOTOR_1",
    "ankle_pitch": "LEG_MOTOR_3",
    "ankle_roll": "LEG_MOTOR_3",
    "waist_yaw": "WAIST_MOTOR",
    "shoulder_pitch": "ARM_MOTOR_1",
    "shoulder_roll": "ARM_MOTOR_1",
    "shoulder_yaw": "ARM_MOTOR_2",
    "elbow_pitch": "ARM_MOTOR_2",
    "wrist_yaw": "ARM_MOTOR_3",
    "wrist_roll": "ARM_MOTOR_3",
    "wrist_pitch": "ARM_MOTOR_3",
}

VR_M3_1_STIFFNESS = {
    "hip_pitch": 150.0,
    "hip_roll": 150.0,
    "hip_yaw": 120.0,
    "knee_pitch": 200.0,
    "ankle_pitch": 200.0,
    "ankle_roll": 200.0,
    "waist_yaw": 367.0,
    "shoulder_pitch": 103.0,
    "shoulder_roll": 94.0,
    "shoulder_yaw": 291.0,
    "elbow_pitch": 291.0,
    "wrist_yaw": 292.0,
    "wrist_roll": 218.0,
    "wrist_pitch": 212.0,
}

VR_M3_1_DAMPING = {
    "hip_pitch": 25.0,
    "hip_roll": 25.0,
    "hip_yaw": 4.0,
    "knee_pitch": 10.0,
    "ankle_pitch": 8.0,
    "ankle_roll": 8.0,
    "waist_yaw": 29.0,
    "shoulder_pitch": 8.0,
    "shoulder_roll": 7.0,
    "shoulder_yaw": 23.0,
    "elbow_pitch": 23.0,
    "wrist_yaw": 12.0,
    "wrist_roll": 9.0,
    "wrist_pitch": 8.0,
}

VR_M3_1_EFFORT_LIMITS = {
    "hip_pitch": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["hip_pitch"]]["max_tau"],
    "hip_roll": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["hip_roll"]]["max_tau"],
    "hip_yaw": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["hip_yaw"]]["max_tau"],
    "knee_pitch": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["knee_pitch"]]["max_tau"],
    "ankle_pitch": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["ankle_pitch"]]["max_tau"] * 2,
    "ankle_roll": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["ankle_roll"]]["max_tau"] * 2,
    "waist_yaw": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["waist_yaw"]]["max_tau"],
    "shoulder_pitch": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["shoulder_pitch"]]["max_tau"],
    "shoulder_roll": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["shoulder_roll"]]["max_tau"],
    "shoulder_yaw": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["shoulder_yaw"]]["max_tau"],
    "elbow_pitch": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["elbow_pitch"]]["max_tau"],
    "wrist_yaw": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["wrist_yaw"]]["max_tau"],
    "wrist_roll": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["wrist_roll"]]["max_tau"],
    "wrist_pitch": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["wrist_pitch"]]["max_tau"],
}

VR_M3_1_ARMATURE = {
    "hip_pitch": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["hip_pitch"]]["armature"],
    "hip_roll": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["hip_roll"]]["armature"],
    "hip_yaw": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["hip_yaw"]]["armature"],
    "knee_pitch": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["knee_pitch"]]["armature"],
    "ankle_pitch": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["ankle_pitch"]]["armature"] * 2,
    "ankle_roll": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["ankle_roll"]]["armature"] * 2,
    "waist_yaw": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["waist_yaw"]]["armature"],
    "shoulder_pitch": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["shoulder_pitch"]]["armature"],
    "shoulder_roll": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["shoulder_roll"]]["armature"],
    "shoulder_yaw": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["shoulder_yaw"]]["armature"],
    "elbow_pitch": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["elbow_pitch"]]["armature"],
    "wrist_yaw": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["wrist_yaw"]]["armature"],
    "wrist_roll": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["wrist_roll"]]["armature"],
    "wrist_pitch": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["wrist_pitch"]]["armature"],
}

VR_M3_1_SATURATION_EFFORTS = {
    "hip_pitch": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["hip_pitch"]]["saturation_tau"],
    "hip_roll": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["hip_roll"]]["saturation_tau"],
    "hip_yaw": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["hip_yaw"]]["saturation_tau"],
    "knee_pitch": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["knee_pitch"]]["saturation_tau"],
    "ankle_pitch": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["ankle_pitch"]]["saturation_tau"],
    "ankle_roll": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["ankle_roll"]]["saturation_tau"],
    "waist_yaw": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["waist_yaw"]]["saturation_tau"],
    "shoulder_pitch": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["shoulder_pitch"]][
        "saturation_tau"
    ],
    "shoulder_roll": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["shoulder_roll"]][
        "saturation_tau"
    ],
    "shoulder_yaw": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["shoulder_yaw"]]["saturation_tau"],
    "elbow_pitch": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["elbow_pitch"]]["saturation_tau"],
    "wrist_yaw": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["wrist_yaw"]]["saturation_tau"],
    "wrist_roll": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["wrist_roll"]]["saturation_tau"],
    "wrist_pitch": MOTOR_SPECS[VR_M3_1_ACTUATOR_NAMES["wrist_pitch"]]["saturation_tau"],
}

# Static proxy = 1.2 * dynamic friction, N*m
VR_M3_1_STATIC_FRICTION = {
    "hip_pitch": 0.1,
    "hip_roll": 0.1,
    "hip_yaw": 0.1,
    "knee_pitch": 0.1,
    "ankle_pitch": 0.1,
    "ankle_roll": 0.1,
    "waist_yaw": 18.0,
    "shoulder_pitch": 3.0,
    "shoulder_roll": 3.0,
    "shoulder_yaw": 3.0,
    "elbow_pitch": 1.2,
    "wrist_yaw": 1.3,
    "wrist_roll": 0.7,
    "wrist_pitch": 0.7,
}
##
# Actuator config.
##

VR_M3_1_12DOF_IMPLICIT_ACTUATORS = {
    "hip_pitch": BuiltinPositionActuatorCfg(
        target_names_expr=(".*_hip_pitch_joint",),
        stiffness=VR_M3_1_STIFFNESS["hip_pitch"],
        damping=VR_M3_1_DAMPING["hip_pitch"],
        effort_limit=VR_M3_1_EFFORT_LIMITS["hip_pitch"],
        armature=VR_M3_1_ARMATURE["hip_pitch"],
        frictionloss=VR_M3_1_STATIC_FRICTION["hip_pitch"],
    ),
    "hip_roll": BuiltinPositionActuatorCfg(
        target_names_expr=(".*_hip_roll_joint",),
        stiffness=VR_M3_1_STIFFNESS["hip_roll"],
        damping=VR_M3_1_DAMPING["hip_roll"],
        effort_limit=VR_M3_1_EFFORT_LIMITS["hip_roll"],
        armature=VR_M3_1_ARMATURE["hip_roll"],
        frictionloss=VR_M3_1_STATIC_FRICTION["hip_roll"],
    ),
    "hip_yaw": BuiltinPositionActuatorCfg(
        target_names_expr=(".*_hip_yaw_joint",),
        stiffness=VR_M3_1_STIFFNESS["hip_yaw"],
        damping=VR_M3_1_DAMPING["hip_yaw"],
        effort_limit=VR_M3_1_EFFORT_LIMITS["hip_yaw"],
        armature=VR_M3_1_ARMATURE["hip_yaw"],
        frictionloss=VR_M3_1_STATIC_FRICTION["hip_yaw"],
    ),
    "knee_pitch": BuiltinPositionActuatorCfg(
        target_names_expr=(".*_knee_pitch_joint",),
        stiffness=VR_M3_1_STIFFNESS["knee_pitch"],
        damping=VR_M3_1_DAMPING["knee_pitch"],
        effort_limit=VR_M3_1_EFFORT_LIMITS["knee_pitch"],
        armature=VR_M3_1_ARMATURE["knee_pitch"],
        frictionloss=VR_M3_1_STATIC_FRICTION["knee_pitch"],
    ),
    "ankle_pitch": BuiltinPositionActuatorCfg(
        target_names_expr=(".*_ankle_pitch_joint",),
        stiffness=VR_M3_1_STIFFNESS["ankle_pitch"],
        damping=VR_M3_1_DAMPING["ankle_pitch"],
        effort_limit=VR_M3_1_EFFORT_LIMITS["ankle_pitch"],
        armature=VR_M3_1_ARMATURE["ankle_pitch"],
        frictionloss=VR_M3_1_STATIC_FRICTION["ankle_pitch"],
    ),
    "ankle_roll": BuiltinPositionActuatorCfg(
        target_names_expr=(".*_ankle_roll_joint",),
        stiffness=VR_M3_1_STIFFNESS["ankle_roll"],
        damping=VR_M3_1_DAMPING["ankle_roll"],
        effort_limit=VR_M3_1_EFFORT_LIMITS["ankle_roll"],
        armature=VR_M3_1_ARMATURE["ankle_roll"],
        frictionloss=VR_M3_1_STATIC_FRICTION["ankle_roll"],
    ),
}

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.9),
    joint_pos={
        # Legs
        ".*_hip_pitch_joint": -0.1,
        ".*_hip_roll_joint": 0.0,
        ".*_hip_yaw_joint": 0.0,
        ".*_knee_pitch_joint": 0.2,
        ".*_ankle_pitch_joint": -0.1,
        ".*_ankle_roll_joint": 0.0,
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
##
# Foot collision geoms: left_ankle_roll_link_collision_{1-15}, etc.
FEET_ONLY_COLLISION = CollisionCfg(
    geom_names_expr=(r"^(left|right)_ankle_roll_link_collision_\d+$",),
    contype=0,
    conaffinity=1,
    condim=3,
    priority=1,
    friction=(0.6,),
)

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision_\\d+",),
    condim={
        # Foot contacts (ankle roll links) get full 3D friction
        r"^(left|right)_ankle_roll_link_collision_\d+$": 3,
        # Everything else gets 1D
        r".*_collision_\d+": 1,
    },
    priority={
        r"^(left|right)_ankle_roll_link_collision_\d+$": 1,
    },
    friction={
        r"^(left|right)_ankle_roll_link_collision_\d+$": (0.6,),
    },
)

##
# Final config.
##

VR_M3_1_12DOF_IMPLICIT_ARTICULATION = EntityArticulationInfoCfg(
    actuators=tuple(VR_M3_1_12DOF_IMPLICIT_ACTUATORS.values()),
    soft_joint_pos_limit_factor=0.9,
)


def get_vr_m3_1_12dof_implicit_actuator_robot_cfg() -> EntityCfg:
    """Get a fresh VR M3 implicit actuator robot configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues when
    the config is shared across multiple places.
    """
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=VR_M3_1_12DOF_IMPLICIT_ARTICULATION,
    )


# Action scale: maps normalized actions to joint position offsets.
VR_M3_1_12DOF_ACTION_SCALE: dict[str, float] = {}
for a in VR_M3_1_12DOF_IMPLICIT_ARTICULATION.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg | DcMotorActuatorCfg)
    e = a.effort_limit
    s = a.stiffness
    names = a.target_names_expr
    assert e is not None
    for n in names:
        VR_M3_1_12DOF_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
    import mujoco.viewer as viewer
    from mjlab.entity.entity import Entity

    robot = Entity(get_vr_m3_1_12dof_implicit_actuator_robot_cfg())
    print(f"Robot has {robot.num_joints} joints")
    print(f"Actuated joints: {robot.num_actuators}")
    print(f"Action scale: {VR_M3_1_12DOF_ACTION_SCALE}")
    # print joints sdk names
    print(f"Joint names: {robot.joint_names}")
    viewer.launch(robot.spec.compile())
