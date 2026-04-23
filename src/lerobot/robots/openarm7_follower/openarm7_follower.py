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

"""7-DOF OpenArm follower (arm only, no gripper on CAN).

Use when the gripper is controlled via a separate interface (e.g. the Gripette
has its own gRPC service). Inherits from `OpenArmFollower` and overrides only
what's necessary: `send_action` uses a dynamic motor-index built from
`config.motor_config.keys()` so the kp/kd gain lookup works with any number of
motors (7 here vs. 8 for the standard OpenArm).
"""

import logging

from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_not_connected

from ..openarm_follower.openarm_follower import OpenArmFollower
from ..utils import ensure_safe_goal_position
from .config_openarm7_follower import (
    LEFT_DEFAULT_JOINTS_LIMITS,
    RIGHT_DEFAULT_JOINTS_LIMITS,
    OpenArm7FollowerConfig,
)

logger = logging.getLogger(__name__)


class OpenArm7Follower(OpenArmFollower):
    """OpenArm follower with 7 arm joints only (no gripper on CAN)."""

    config_class = OpenArm7FollowerConfig
    name = "openarm7_follower"

    def __init__(self, config: OpenArm7FollowerConfig):
        # Apply Gripette-7-specific side defaults before the base class would
        # apply the standard OpenArm defaults (which include a 'gripper' slot).
        if config.side == "left":
            config.joint_limits = LEFT_DEFAULT_JOINTS_LIMITS
        elif config.side == "right":
            config.joint_limits = RIGHT_DEFAULT_JOINTS_LIMITS
        # Prevent the parent __init__ from overwriting our joint_limits with
        # the standard 8-motor defaults.
        original_side = config.side
        config.side = None
        super().__init__(config)
        config.side = original_side

    # -- Calibration bypass -------------------------------------------------
    # OpenArm motor zeros live in the Damiao motor firmware (set externally
    # via OpenArm's official calibration tool). The Damiao bus never reads
    # the LeRobot calibration fields (homing_offset / drive_mode / range_*),
    # so LeRobot's file-based calibration is purely vestigial for this arm.
    # We bypass it entirely to avoid prompts and accidental zero overwrites.

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        logger.info(
            "LeRobot calibration is disabled for OpenArm7Follower. "
            "Motor zeros are stored in firmware — use OpenArm's calibration tool "
            "to (re)set them. No calibration file will be read or written."
        )

    @check_if_not_connected
    def send_action(
        self,
        action: RobotAction,
        custom_kp: dict[str, float] | None = None,
        custom_kd: dict[str, float] | None = None,
    ) -> RobotAction:
        """Send joint position commands over CAN using MIT control.

        Same as OpenArmFollower.send_action, but builds motor_index dynamically
        from config.motor_config so it matches any motor count.
        """
        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}

        # Joint limit clipping
        for motor_name, position in goal_pos.items():
            if motor_name in self.config.joint_limits:
                min_limit, max_limit = self.config.joint_limits[motor_name]
                clipped = max(min_limit, min(max_limit, position))
                if clipped != position:
                    logger.debug(f"Clipped {motor_name} from {position:.2f}° to {clipped:.2f}°")
                goal_pos[motor_name] = clipped

        # Max-relative-target safety limit
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in goal_pos.items()}
            goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)

        # Motor index built from motor_config order (not hardcoded)
        motor_index = {name: i for i, name in enumerate(self.config.motor_config.keys())}

        # MIT commands
        commands = {}
        for motor_name, position_degrees in goal_pos.items():
            idx = motor_index.get(motor_name, 0)
            if custom_kp is not None and motor_name in custom_kp:
                kp = custom_kp[motor_name]
            else:
                kp = (
                    self.config.position_kp[idx]
                    if isinstance(self.config.position_kp, list)
                    else self.config.position_kp
                )
            if custom_kd is not None and motor_name in custom_kd:
                kd = custom_kd[motor_name]
            else:
                kd = (
                    self.config.position_kd[idx]
                    if isinstance(self.config.position_kd, list)
                    else self.config.position_kd
                )
            commands[motor_name] = (kp, kd, position_degrees, 0.0, 0.0)

        self.bus._mit_control_batch(commands)

        return {f"{motor}.pos": val for motor, val in goal_pos.items()}
