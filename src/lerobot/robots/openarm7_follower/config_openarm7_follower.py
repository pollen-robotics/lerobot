#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""Config for the 7-DOF OpenArm without a gripper on the CAN bus.

Use this when the gripper is controlled via a separate interface (e.g. the
Gripette has its own gRPC service, so the CAN-side driver only needs to
handle the 7 arm joints).
"""

from dataclasses import dataclass, field

from ..config import RobotConfig
from ..openarm_follower.config_openarm_follower import OpenArmFollowerConfigBase

# Default arm joint limits (degrees). These match the standard OpenArm arm.
RIGHT_DEFAULT_JOINTS_LIMITS: dict[str, tuple[float, float]] = {
    "joint_1": (-75.0, 75.0),
    "joint_2": (-9.0, 90.0),
    "joint_3": (-85.0, 85.0),
    "joint_4": (0.0, 135.0),
    "joint_5": (-85.0, 85.0),
    "joint_6": (-40.0, 40.0),
    "joint_7": (-80.0, 80.0),
}

LEFT_DEFAULT_JOINTS_LIMITS: dict[str, tuple[float, float]] = {
    "joint_1": (-75.0, 75.0),
    "joint_2": (-90.0, 9.0),
    "joint_3": (-85.0, 85.0),
    "joint_4": (0.0, 135.0),
    "joint_5": (-85.0, 85.0),
    "joint_6": (-40.0, 40.0),
    "joint_7": (-80.0, 80.0),
}


@dataclass
class OpenArm7FollowerConfigBase(OpenArmFollowerConfigBase):
    """Config for the 7-DOF arm-only variant of the OpenArm follower.

    No gripper motor on the CAN bus — the gripper is expected to be controlled
    via a separate interface (gRPC, USB, etc.).
    """

    # 7 arm joints, no gripper. Same CAN IDs and motor types as the standard OpenArm.
    motor_config: dict[str, tuple[int, int, str]] = field(
        default_factory=lambda: {
            "joint_1": (0x01, 0x11, "dm8009"),
            "joint_2": (0x02, 0x12, "dm8009"),
            "joint_3": (0x03, 0x13, "dm4340"),
            "joint_4": (0x04, 0x14, "dm4340"),
            "joint_5": (0x05, 0x15, "dm4310"),
            "joint_6": (0x06, 0x16, "dm4310"),
            "joint_7": (0x07, 0x17, "dm4310"),
        }
    )

    # MIT control gains for the 7 arm joints (matches OpenArmFollower defaults minus the gripper).
    position_kp: list[float] = field(
        default_factory=lambda: [240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0]
    )
    position_kd: list[float] = field(
        default_factory=lambda: [5.0, 5.0, 3.0, 5.0, 0.3, 0.3, 0.3]
    )

    # Safe default joint limits. Override via CLI or set config.side='left'/'right'.
    joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "joint_1": (-5.0, 5.0),
            "joint_2": (-5.0, 5.0),
            "joint_3": (-5.0, 5.0),
            "joint_4": (0.0, 5.0),
            "joint_5": (-5.0, 5.0),
            "joint_6": (-5.0, 5.0),
            "joint_7": (-5.0, 5.0),
        }
    )


@RobotConfig.register_subclass("openarm7_follower")
@dataclass
class OpenArm7FollowerConfig(RobotConfig, OpenArm7FollowerConfigBase):
    pass
