# ruff: noqa: N802  # gRPC servicer methods use PascalCase (protobuf convention)
"""gRPC server that drives a real OpenArm via the same API as the simulator.

This server exposes the exact same ArmService and GripperService as
`openarm_gripette_simu`, but drives a real Pollen OpenArm via CAN bus instead of
MuJoCo. It uses the same Placo `Kinematics` class from the simulator package with
the URDF from `openarm_gripette_model` — so FK/IK behavior is identical to the
simulator by construction.

Drop-in replacement: any client that works with the simulator (e.g.
`eval_simulator.py`) works unchanged against this server — just change the
`--arm_addr` / `--gripper_addr` CLI args.

Behavioral differences from the simulator (unavoidable on real hardware):
  - Reset: interpolates smoothly to the home pose (can't teleport a real robot).
    Cube randomization is a no-op — there's no physical cube to move. The
    response returns dummy cube coordinates.
  - GetSuccessStatus: always returns goal_reached=False. Real-world success
    detection would need external tracking (vision, force sensor, manual).
  - SendCartesianDelta: same semantics — accumulates deltas on an internal
    Cartesian target, runs IK, sends joint commands via CAN.

Prerequisites:
  uv pip install -e /path/to/openarm_gripette_simu   # for proto stubs + Kinematics
  uv sync --extra kinematics                         # placo
  pip install openarm-gripette-model                 # URDF source

Usage:
  uv run python examples/openarm_gripette/grpc_server_real.py \\
      --can_port can0 --side right \\
      --camera_index 0 --arm_port 50052 --gripper_port 50051

Then from another terminal:
  uv run python examples/openarm_gripette/eval_simulator.py \\
      --checkpoint SteveNguyen/gripette_v3 \\
      --arm_addr localhost:50052 --gripper_addr localhost:50051
"""

import argparse
import logging
import threading
import time
from concurrent import futures

import cv2
import grpc
import numpy as np

# Kinematics + proto stubs from the simulator package (ensures bit-for-bit match)
from openarm_gripette_simu import Kinematics
from openarm_gripette_simu.kinematics import ARM_JOINT_NAMES as KIN_ARM_JOINT_NAMES
from openarm_gripette_simu.proto import arm_pb2, arm_pb2_grpc, gripper_pb2, gripper_pb2_grpc
from openarm_gripette_simu.rotation import rotation_6d_to_matrix, rotation_matrix_to_6d

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig

logger = logging.getLogger(__name__)

# Default safe home pose (matches simulator START_JOINTS in arm_servicer.py)
HOME_JOINTS_RAD = np.array([0.0, 0.0, 0.0, 1.57, 0.0, 0.0, 0.0])

# Reset interpolation: move from current to home over this many seconds.
# Longer = safer. Shorter = faster but more abrupt.
RESET_DURATION_S = 3.0
RESET_HZ = 50  # interpolation frequency during reset

# Gripper joint names on the real robot (match dataset names)
GRIPPER_JOINT_NAMES = ["proximal", "distal"]

# Camera stream rate (matches simulator: 50 Hz)
STREAM_HZ = 50
STREAM_INTERVAL = 1.0 / STREAM_HZ

# LeRobot's OpenArm robot uses degrees internally; the simulator uses radians.
# We convert at the CAN boundary.
DEG_TO_RAD = np.pi / 180.0
RAD_TO_DEG = 180.0 / np.pi


# ---------------------------------------------------------------------------
# Robot wrapper: unit conversion + mapping between simulator and LeRobot names
# ---------------------------------------------------------------------------


class RobotInterface:
    """Adapter between the simulator's joint-angle API (radians) and LeRobot's
    OpenArmFollower (degrees + per-motor dict commands).

    Handles:
      - Joint name mapping: simulator uses 'r_arm_pitch'..., LeRobot uses 'joint_1'...
      - Unit conversion: radians (simulator / policy) ↔ degrees (LeRobot)
      - Thread-safe access to the CAN bus
    """

    def __init__(self, robot: OpenArmFollower, arm_joint_map: dict[str, str]):
        self._robot = robot
        self._arm_joint_map = arm_joint_map  # simulator_name -> lerobot_name
        self._lerobot_arm_names = [arm_joint_map[n] for n in KIN_ARM_JOINT_NAMES]
        self._lock = threading.Lock()

    def get_arm_positions(self) -> np.ndarray:
        """Read current arm joint positions in radians, in simulator joint order."""
        with self._lock:
            obs = self._robot.get_observation()
        # LeRobot robot returns degrees under "{motor}.pos" keys
        positions_deg = np.array([obs[f"{name}.pos"] for name in self._lerobot_arm_names], dtype=np.float64)
        return positions_deg * DEG_TO_RAD

    def get_gripper_positions(self) -> np.ndarray:
        """Read current gripper joint positions in radians [proximal, distal]."""
        with self._lock:
            obs = self._robot.get_observation()
        positions_deg = np.array([obs[f"{name}.pos"] for name in GRIPPER_JOINT_NAMES], dtype=np.float64)
        return positions_deg * DEG_TO_RAD

    def get_camera_frame(self) -> np.ndarray:
        """Read the latest RGB camera frame."""
        with self._lock:
            obs = self._robot.get_observation()
        # OpenArmFollower exposes cameras under their configured names
        for _key, val in obs.items():
            if isinstance(val, np.ndarray) and val.ndim == 3 and val.shape[2] == 3:
                return val
        raise RuntimeError("No camera frame found in observation")

    def send_arm_command_rad(self, joint_angles_rad: np.ndarray):
        """Send arm joint commands in simulator order (radians)."""
        joint_angles_deg = joint_angles_rad * RAD_TO_DEG
        action = {
            f"{self._lerobot_arm_names[i]}.pos": float(joint_angles_deg[i])
            for i in range(len(joint_angles_rad))
        }
        with self._lock:
            self._robot.send_action(action)

    def send_gripper_command_rad(self, motor1_rad: float, motor2_rad: float):
        """Send gripper motor commands (radians)."""
        action = {
            f"{GRIPPER_JOINT_NAMES[0]}.pos": float(motor1_rad * RAD_TO_DEG),
            f"{GRIPPER_JOINT_NAMES[1]}.pos": float(motor2_rad * RAD_TO_DEG),
        }
        with self._lock:
            self._robot.send_action(action)


# ---------------------------------------------------------------------------
# ArmService — mirrors openarm_gripette_simu/arm_servicer.py behavior
# ---------------------------------------------------------------------------


class ArmServicer(arm_pb2_grpc.ArmServiceServicer):
    """Implements the same delta-target accumulation as the simulator.

    The "internal target" pattern: SendCartesianDelta does NOT read the current
    FK pose and add the delta. It maintains its own (_target_pos, _target_r6d)
    and accumulates deltas on that. After IK, the target is re-synced to the
    FK of the commanded joints (not the actual achieved joints — we can't read
    future positions, and tracking settles over many control cycles).
    """

    def __init__(self, robot: RobotInterface, kin: Kinematics, start_time: float):
        self._robot = robot
        self._kin = kin
        self._start_time = start_time
        self._cmd_lock = threading.Lock()  # protects IK + internal target state

        # Initialize the internal target from the current FK pose
        self._sync_target_from_robot()

    def _sync_target_from_robot(self):
        """Re-initialize internal target from the current robot state."""
        arm_joints = self._robot.get_arm_positions()
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
                # Accumulate delta on the internal target
                self._target_pos = self._target_pos + delta_pos
                self._target_r6d = self._target_r6d + delta_r6d

                # Build 4x4 target pose
                target_rot = rotation_6d_to_matrix(self._target_r6d)
                target_tf = np.eye(4)
                target_tf[:3, :3] = target_rot
                target_tf[:3, 3] = self._target_pos

                # Solve IK from current arm state
                arm_joints = self._robot.get_arm_positions()
                target_joints = self._kin.inverse(target_tf, current_joint_positions=arm_joints)

                # Send joint commands to the real robot
                self._robot.send_arm_command_rad(target_joints)

                # Re-sync the internal target to the FK of the COMMANDED joints.
                # This prevents the target from drifting out of reach while still
                # accumulating small deltas smoothly.
                achieved_tf = self._kin.forward(target_joints)
                self._target_pos = achieved_tf[:3, 3].copy()
                self._target_r6d = rotation_matrix_to_6d(achieved_tf[:3, :3]).copy()

            return arm_pb2.ArmCommandResponse(success=True)

        except Exception as e:
            logger.exception("SendCartesianDelta failed")
            return arm_pb2.ArmCommandResponse(success=False, error=str(e))

    def GetArmState(self, request, context):
        """Return the actual current arm state (from FK of measured joints)."""
        arm_joints = self._robot.get_arm_positions()
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
        """Move the arm smoothly to a home pose (no teleport on real hardware).

        Cube randomization is a no-op — there's no physical cube. Returns
        dummy cube coordinates (0, 0, 0) for API compatibility.
        """
        try:
            # Target joints: either provided or the default home pose
            if len(request.joint_positions) == 7:
                target_joints = np.array(request.joint_positions, dtype=np.float64)
            else:
                target_joints = HOME_JOINTS_RAD.copy()

            with self._cmd_lock:
                start_joints = self._robot.get_arm_positions()

                # Smooth linear interpolation from current to target
                num_steps = int(RESET_DURATION_S * RESET_HZ)
                dt = 1.0 / RESET_HZ

                logger.info(
                    f"Reset: interpolating from {start_joints.round(3).tolist()} "
                    f"to {target_joints.round(3).tolist()} over {RESET_DURATION_S}s"
                )

                for i in range(1, num_steps + 1):
                    alpha = i / num_steps
                    interp = start_joints * (1 - alpha) + target_joints * alpha
                    self._robot.send_arm_command_rad(interp)
                    time.sleep(dt)

                # Re-sync internal target after the move
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
        """No cube tracking on real robot — always returns goal_reached=False."""
        return arm_pb2.SuccessStatusResponse(
            goal_reached=False,
            cube_displacement=0.0,
        )

    def Ping(self, request, context):
        uptime = time.monotonic() - self._start_time
        return arm_pb2.ArmPingResponse(status="ok", uptime_seconds=uptime)


# ---------------------------------------------------------------------------
# GripperService — camera stream + gripper motor control
# ---------------------------------------------------------------------------


class GripperServicer(gripper_pb2_grpc.GripperServiceServicer):
    def __init__(self, robot: RobotInterface, start_time: float):
        self._robot = robot
        self._start_time = start_time

    def StreamState(self, request, context):
        logger.info("StreamState: client connected")
        sequence = 0
        next_time = time.monotonic()

        while context.is_active():
            try:
                gripper_rad = self._robot.get_gripper_positions()
                img_rgb = self._robot.get_camera_frame()
            except Exception as e:
                logger.error(f"StreamState read failed: {e}")
                time.sleep(STREAM_INTERVAL)
                continue

            # JPEG encode (simulator convention: BGR input to imencode)
            bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            _, jpeg_data = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])

            frame = gripper_pb2.GripperFrame(
                jpeg_data=jpeg_data.tobytes(),
                motor_state=gripper_pb2.MotorState(
                    motor1_position=float(gripper_rad[0]),
                    motor2_position=float(gripper_rad[1]),
                ),
                timestamp_ms=(time.monotonic() - self._start_time) * 1000.0,
                sequence=sequence,
            )
            yield frame
            sequence += 1

            next_time += STREAM_INTERVAL
            sleep_dur = next_time - time.monotonic()
            if sleep_dur > 0:
                time.sleep(sleep_dur)

        logger.info(f"StreamState: client disconnected after {sequence} frames")

    def SendMotorCommand(self, request, context):
        try:
            self._robot.send_gripper_command_rad(request.motor1_goal, request.motor2_goal)
            return gripper_pb2.MotorCommandResponse(success=True)
        except Exception as e:
            logger.exception("SendMotorCommand failed")
            return gripper_pb2.MotorCommandResponse(success=False, error=str(e))

    def ReadMotors(self, request, context):
        gripper_rad = self._robot.get_gripper_positions()
        return gripper_pb2.MotorState(
            motor1_position=float(gripper_rad[0]),
            motor2_position=float(gripper_rad[1]),
        )

    def SetTorque(self, request, context):
        # Damiao motors always have torque when the bus is enabled — no-op for API compatibility
        return gripper_pb2.TorqueResponse(success=True)

    def Ping(self, request, context):
        uptime = time.monotonic() - self._start_time
        return gripper_pb2.PingResponse(status="ok", uptime_seconds=uptime)


# ---------------------------------------------------------------------------
# CLI + server setup
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="gRPC server for real OpenArm (simulator-compatible API)")
    # Robot
    p.add_argument("--can_port", type=str, default="can0", help="CAN interface name")
    p.add_argument("--side", type=str, default="right", choices=["left", "right"])
    p.add_argument(
        "--max_relative_target",
        type=float,
        default=8.0,
        help="Max per-step joint motion in degrees (safety limit)",
    )
    # Camera
    p.add_argument("--camera_index", type=str, default="/dev/video0")
    p.add_argument("--camera_width", type=int, default=960)
    p.add_argument("--camera_height", type=int, default=720)
    p.add_argument("--camera_fps", type=int, default=50)
    # gRPC
    p.add_argument("--arm_port", type=int, default=50052)
    p.add_argument("--gripper_port", type=int, default=50051)
    # Joint naming mapping (CAN motor name ← simulator joint name)
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

    # Parse joint map
    arm_joint_map = dict(item.split("=") for item in args.arm_joint_map)
    logger.info(f"Arm joint map (sim → lerobot): {arm_joint_map}")

    # Validate that all simulator joint names are in the map
    for sim_name in KIN_ARM_JOINT_NAMES:
        if sim_name not in arm_joint_map:
            raise ValueError(
                f"Missing mapping for simulator joint '{sim_name}'. "
                f"Use --arm_joint_map to provide all 7 entries."
            )

    # ---- Camera config ----
    camera_config = OpenCVCameraConfig(
        index_or_path=args.camera_index,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
    )

    # ---- Robot setup ----
    logger.info(f"Connecting to OpenArm on {args.can_port}, side={args.side}")
    robot_config = OpenArmFollowerConfig(
        port=args.can_port,
        side=args.side,
        can_interface="socketcan",
        max_relative_target=args.max_relative_target,
        cameras={"cam0": camera_config},
    )
    robot = OpenArmFollower(robot_config)
    robot.connect()
    logger.info("Robot connected")

    robot_iface = RobotInterface(robot, arm_joint_map)

    # ---- Kinematics (same as simulator) ----
    logger.info("Loading kinematics (placo + URDF from openarm_gripette_model)")
    kin = Kinematics()
    logger.info("Kinematics loaded")

    # ---- gRPC servers ----
    start_time = time.monotonic()

    arm_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    arm_pb2_grpc.add_ArmServiceServicer_to_server(ArmServicer(robot_iface, kin, start_time), arm_server)
    arm_server.add_insecure_port(f"[::]:{args.arm_port}")

    gripper_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    gripper_pb2_grpc.add_GripperServiceServicer_to_server(
        GripperServicer(robot_iface, start_time), gripper_server
    )
    gripper_server.add_insecure_port(f"[::]:{args.gripper_port}")

    arm_server.start()
    gripper_server.start()
    logger.info(f"ArmService listening on port {args.arm_port}")
    logger.info(f"GripperService listening on port {args.gripper_port}")
    logger.info("Press Ctrl+C to stop")

    try:
        # Block forever
        arm_server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        arm_server.stop(grace=2.0)
        gripper_server.stop(grace=2.0)
        robot.disconnect()
        logger.info("Robot disconnected. Goodbye.")


if __name__ == "__main__":
    main()
