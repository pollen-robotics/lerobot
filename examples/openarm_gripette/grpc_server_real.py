# ruff: noqa: N802  # gRPC servicer methods use PascalCase (protobuf convention)
"""gRPC ArmService server for a real OpenArm (7-DOF, no gripper on CAN).

This server exposes the same ArmService API as `openarm_gripette_simu`, but drives
a real Pollen OpenArm via CAN. It uses the Placo `Kinematics` class and URDF from
`openarm_gripette_model` for bit-for-bit FK/IK compatibility with the simulator.

The **Gripette has its own gRPC service** (same API as `gripper.proto`) running
independently. The eval client connects to two separate endpoints:

  - ArmService     → this server (arm control via CAN)
  - GripperService → the Gripette's own service (camera + gripper motors)

So this script implements ONLY the ArmService.

Behavioral notes:
  - SendCartesianDelta: accumulates on an internal target, runs IK, sends to CAN.
    Identical semantics to the simulator.
  - Reset: interpolates smoothly to the home pose (can't teleport a real arm).
    Cube randomization is a no-op — returns dummy cube coords for API compat.
  - GetSuccessStatus: always returns goal_reached=False (no cube tracking here).

Prerequisites:
  uv sync --locked --extra diffusion --extra kinematics --extra openarms
  uv pip install -e /path/to/openarm_gripette_simu --no-deps
  pip install openarm-gripette-model

Usage:
  uv run python examples/openarm_gripette/grpc_server_real.py \\
      --can_port can0 --side right --arm_port 50052

Then, from the inference machine:
  uv run python examples/openarm_gripette/eval_simulator.py \\
      --checkpoint SteveNguyen/gripette_v3 \\
      --arm_addr <robot-ip>:50052 \\
      --gripper_addr <gripette-ip>:<gripette-port>
"""

import argparse
import logging
import threading
import time
from concurrent import futures

import grpc
import numpy as np

# Kinematics + proto stubs from the simulator package (ensures bit-for-bit match)
from openarm_gripette_simu import Kinematics
from openarm_gripette_simu.kinematics import ARM_JOINT_NAMES as KIN_ARM_JOINT_NAMES
from openarm_gripette_simu.proto import arm_pb2, arm_pb2_grpc
from openarm_gripette_simu.rotation import rotation_6d_to_matrix, rotation_matrix_to_6d

from lerobot.robots.openarm7_follower import OpenArm7Follower, OpenArm7FollowerConfig

logger = logging.getLogger(__name__)

# Default safe home pose (matches simulator START_JOINTS).
HOME_JOINTS_RAD = np.array([0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0])

# Reset interpolation: move from current to home over this many seconds.
RESET_DURATION_S = 3.0
RESET_HZ = 50

# LeRobot's OpenArm driver uses degrees; simulator/policy use radians.
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi


# ---------------------------------------------------------------------------
# Arm wrapper: unit conversion + joint-name mapping
# ---------------------------------------------------------------------------


class ArmInterface:
    """Adapter between the simulator's joint API (radians, r_arm_* names) and
    LeRobot's OpenArm7Follower (degrees, joint_1..joint_7 names).

    Caches the most recent joint read with a short TTL so that GetArmState and
    SendCartesianDelta called within the same control cycle share a single CAN
    refresh. Without this, every inference step triggers two full refreshes
    (one for client-side state, one for IK), doubling bus load and packet drops.
    """

    def __init__(
        self,
        robot: OpenArm7Follower,
        arm_joint_map: dict[str, str],
        state_cache_ttl_s: float = 0.02,
    ):
        self._robot = robot
        self._arm_joint_map = arm_joint_map  # sim_name -> lerobot_name
        self._lerobot_arm_names = [arm_joint_map[n] for n in KIN_ARM_JOINT_NAMES]
        self._lock = threading.Lock()
        self._state_cache_ttl = state_cache_ttl_s
        self._cached_positions_rad: np.ndarray | None = None
        self._cached_positions_ts: float = 0.0

    def get_positions(self) -> np.ndarray:
        """Read arm joint positions in radians, in simulator order (r_arm_pitch, ...).

        Returns a cached value if the last refresh is within state_cache_ttl_s.
        """
        now = time.monotonic()
        if (
            self._cached_positions_rad is not None
            and (now - self._cached_positions_ts) < self._state_cache_ttl
        ):
            return self._cached_positions_rad.copy()
        with self._lock:
            # Double-checked: another thread may have refreshed while we waited.
            now = time.monotonic()
            if (
                self._cached_positions_rad is not None
                and (now - self._cached_positions_ts) < self._state_cache_ttl
            ):
                return self._cached_positions_rad.copy()
            obs = self._robot.get_observation()
            positions_deg = np.array(
                [obs[f"{name}.pos"] for name in self._lerobot_arm_names], dtype=np.float64
            )
            self._cached_positions_rad = positions_deg * DEG_TO_RAD
            self._cached_positions_ts = time.monotonic()
            return self._cached_positions_rad.copy()

    def send_command_rad(self, joint_angles_rad: np.ndarray):
        """Send arm joint commands (radians, in simulator order)."""
        joint_angles_deg = joint_angles_rad * RAD_TO_DEG
        action = {
            f"{self._lerobot_arm_names[i]}.pos": float(joint_angles_deg[i])
            for i in range(len(joint_angles_rad))
        }
        with self._lock:
            self._robot.send_action(action)


# ---------------------------------------------------------------------------
# ArmService — mirrors openarm_gripette_simu/arm_servicer.py behavior
# ---------------------------------------------------------------------------


class ArmServicer(arm_pb2_grpc.ArmServiceServicer):
    """Implements the same delta-target accumulation pattern as the simulator.

    SendCartesianDelta maintains an internal (_target_pos, _target_r6d). Deltas
    accumulate on that target (not on the current FK pose), avoiding drift from
    mechanical tracking errors. After IK, the target is re-synced to the FK of
    the commanded joints so it stays within a reachable neighborhood.
    """

    def __init__(self, arm: ArmInterface, kin: Kinematics, start_time: float):
        self._arm = arm
        self._kin = kin
        self._start_time = start_time
        self._cmd_lock = threading.Lock()
        self._sync_target_from_robot()

    def _sync_target_from_robot(self):
        """Reset internal target from the current robot FK pose."""
        arm_joints = self._arm.get_positions()
        tf = self._kin.forward(arm_joints)
        self._target_pos = tf[:3, 3].copy()
        self._target_r6d = rotation_matrix_to_6d(tf[:3, :3]).copy()

    def SendCartesianDelta(self, request, context):
        try:
            delta_pos = np.array([request.dx, request.dy, request.dz])
            delta_r6d = np.array(request.dr6d)

            if len(delta_r6d) != 6:
                return arm_pb2.ArmCommandResponse(
                    success=False, error=f"dr6d must have 6 values, got {len(delta_r6d)}"
                )

            with self._cmd_lock:
                self._target_pos = self._target_pos + delta_pos
                self._target_r6d = self._target_r6d + delta_r6d

                target_rot = rotation_6d_to_matrix(self._target_r6d)
                target_tf = np.eye(4)
                target_tf[:3, :3] = target_rot
                target_tf[:3, 3] = self._target_pos

                arm_joints = self._arm.get_positions()
                target_joints = self._kin.inverse(target_tf, current_joint_positions=arm_joints)
                self._arm.send_command_rad(target_joints)

                achieved_tf = self._kin.forward(target_joints)
                self._target_pos = achieved_tf[:3, 3].copy()
                self._target_r6d = rotation_matrix_to_6d(achieved_tf[:3, :3]).copy()

            return arm_pb2.ArmCommandResponse(success=True)

        except Exception as e:
            logger.exception("SendCartesianDelta failed")
            return arm_pb2.ArmCommandResponse(success=False, error=str(e))

    def GetArmState(self, request, context):
        """Return the current arm state (FK of measured joints)."""
        arm_joints = self._arm.get_positions()
        tf = self._kin.forward(arm_joints)
        pos = tf[:3, 3]
        r6d = rotation_matrix_to_6d(tf[:3, :3])
        return arm_pb2.ArmState(
            x=float(pos[0]),
            y=float(pos[1]),
            z=float(pos[2]),
            r6d=r6d.tolist(),
            joint_positions=arm_joints.tolist(),
        )

    def Reset(self, request, context):
        """Move the arm smoothly to the home (or specified) joint configuration.

        Cube randomization is a no-op (no physical cube); dummy cube coords are
        returned for API compatibility with the simulator.
        """
        try:
            if len(request.joint_positions) == 7:
                target_joints = np.array(request.joint_positions, dtype=np.float64)
            else:
                target_joints = HOME_JOINTS_RAD.copy()

            with self._cmd_lock:
                start_joints = self._arm.get_positions()

                num_steps = int(RESET_DURATION_S * RESET_HZ)
                dt = 1.0 / RESET_HZ

                logger.info(
                    f"Reset: interpolating over {RESET_DURATION_S}s "
                    f"from {start_joints.round(3).tolist()} to {target_joints.round(3).tolist()}"
                )

                for i in range(1, num_steps + 1):
                    alpha = i / num_steps
                    interp = start_joints * (1 - alpha) + target_joints * alpha
                    self._arm.send_command_rad(interp)
                    time.sleep(dt)

                self._sync_target_from_robot()

            return arm_pb2.ResetResponse(
                success=True,
                cube_x=0.0,
                cube_y=0.0,
                cube_z=0.0,
                error="cube randomization skipped (real robot)",
            )

        except Exception as e:
            logger.exception("Reset failed")
            return arm_pb2.ResetResponse(success=False, error=str(e))

    def GetSuccessStatus(self, request, context):
        """No cube tracking on a real robot — always returns goal_reached=False."""
        return arm_pb2.SuccessStatusResponse(goal_reached=False, cube_displacement=0.0)

    def Ping(self, request, context):
        uptime = time.monotonic() - self._start_time
        return arm_pb2.ArmPingResponse(status="ok", uptime_seconds=uptime)


# ---------------------------------------------------------------------------
# CLI + server setup
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="gRPC ArmService for real OpenArm (7-DOF, no gripper)")
    p.add_argument("--can_port", type=str, default="can0", help="CAN interface name")
    p.add_argument("--side", type=str, default="right", choices=["left", "right"])
    p.add_argument(
        "--max_relative_target",
        type=float,
        default=None,
        help="Max per-step joint motion in degrees (safety limit). When set, "
        "send_action does an extra CAN sync_read each command — costs one full "
        "bus transaction and contributes to packet drops. IK already clips to "
        "joint limits; leave unset unless you specifically need the extra guard.",
    )
    p.add_argument("--arm_port", type=int, default=50052, help="gRPC listen port")
    p.add_argument(
        "--arm_joint_map",
        type=str,
        nargs="+",
        default=[
            "r_arm_pitch=joint_1",
            "r_arm_roll=joint_2",
            "r_arm_yaw=joint_3",
            "r_elbow=joint_4",
            "r_wrist_yaw=joint_5",
            "r_wrist_roll=joint_6",
            "r_wrist_pitch=joint_7",
        ],
        help="Map simulator joint names to LeRobot motor names (format: sim_name=lerobot_name)",
    )
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    arm_joint_map = dict(item.split("=") for item in args.arm_joint_map)
    logger.info(f"Arm joint map (sim → lerobot): {arm_joint_map}")

    for sim_name in KIN_ARM_JOINT_NAMES:
        if sim_name not in arm_joint_map:
            raise ValueError(f"Missing mapping for simulator joint '{sim_name}' in --arm_joint_map")

    # ---- Robot setup (arm only, no gripper on CAN) ----
    logger.info(f"Connecting to OpenArm on {args.can_port}, side={args.side}")
    robot_config = OpenArm7FollowerConfig(
        port=args.can_port,
        side=args.side,
        can_interface="socketcan",
        max_relative_target=args.max_relative_target,
        cameras={},  # no cameras on this driver — camera comes via Gripette's gRPC
    )
    robot = OpenArm7Follower(robot_config)
    # calibrate=False: the server never modifies calibration. If the firmware
    # zeros or the calibration file need updating, run `lerobot-calibrate`
    # explicitly before starting the server.
    robot.connect(calibrate=False)
    logger.info("Robot connected (7-DOF arm, gripper is external)")

    arm_iface = ArmInterface(robot, arm_joint_map)

    # ---- Kinematics (same as simulator) ----
    logger.info("Loading kinematics (placo + URDF from openarm_gripette_model)")
    kin = Kinematics()
    logger.info("Kinematics loaded")

    # ---- gRPC server ----
    start_time = time.monotonic()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    arm_pb2_grpc.add_ArmServiceServicer_to_server(ArmServicer(arm_iface, kin, start_time), server)
    server.add_insecure_port(f"[::]:{args.arm_port}")
    server.start()
    logger.info(f"ArmService listening on port {args.arm_port}")
    logger.info("Press Ctrl+C to stop")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        server.stop(grace=2.0)
        robot.disconnect()
        logger.info("Robot disconnected. Goodbye.")


if __name__ == "__main__":
    main()
