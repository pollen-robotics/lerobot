"""Move the arm end-effector along a 20cm x 20cm square in Cartesian space.

gRPC equivalent of the `cartesian_square.py` example in `openarm_gripette_simu`.
Works with the simulator OR the real robot — just change the `--arm_addr`.

The end-effector traces a square in the YZ plane while keeping orientation fixed.
Uses the `SendCartesianDelta` RPC — no local IK required (the server handles it).

Orientation is actively locked: at each step, the script reads the current EE
rotation, computes a 6D delta that would restore the initial rotation, and sends
that as the rotation delta. This counteracts the small orientation drift that
would otherwise accumulate from IK soft-constraint slack.

Usage:
  uv run python examples/openarm_gripette/cartesian_square.py \\
      --arm_addr localhost:50052

  # With camera display from the Gripette:
  uv run python examples/openarm_gripette/cartesian_square.py \\
      --arm_addr localhost:50052 \\
      --gripper_addr localhost:50051 --show_camera
"""

import argparse
import logging
import threading
import time

import grpc
import numpy as np
from openarm_gripette_simu.proto import arm_pb2, arm_pb2_grpc
from openarm_gripette_simu.rotation import rotation_6d_to_matrix

logger = logging.getLogger(__name__)


def get_ee_pose(arm_stub) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fetch the current EE position (3,), 6D rotation (6,), and joint positions (7,)."""
    state = arm_stub.GetArmState(arm_pb2.GetArmStateRequest())
    pos = np.array([state.x, state.y, state.z], dtype=np.float64)
    r6d = np.array(state.r6d, dtype=np.float64)
    joints = np.array(state.joint_positions, dtype=np.float64)
    return pos, r6d, joints


def rotation_angle_deg(r6d_a: np.ndarray, r6d_b: np.ndarray) -> float:
    """Actual rotation angle in degrees between two 6D-encoded rotations.

    Uses R_err = R_a @ R_b^T and reads the rotation angle from the trace.
    """
    R_a = rotation_6d_to_matrix(r6d_a)
    R_b = rotation_6d_to_matrix(r6d_b)
    R_err = R_a @ R_b.T
    # Rotation angle from R: acos((trace - 1) / 2), clipped for numerical safety
    cos_theta = (np.trace(R_err) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cos_theta)))


def compute_orientation_correction(
    current_r6d: np.ndarray, target_r6d: np.ndarray, gain: float = 1.0
) -> np.ndarray:
    """Compute a 6D rotation delta to restore the target orientation.

    The server's SendCartesianDelta does `target_r6d += delta_r6d`, so sending
    (target - current) * gain nudges the orientation back toward target.

    Args:
        current_r6d: current orientation (6D).
        target_r6d: desired fixed orientation (6D).
        gain: proportional gain (1.0 = full correction each step; lower = softer).

    Returns:
        6D rotation delta to send.
    """
    return (target_r6d - current_r6d) * gain


# Square geometry — kept small for a safer first test on hardware.
# Original simulator example uses 0.10 (20cm square); 0.05 (10cm) is safer to start.
SQUARE_HALF_SIZE = 0.05

# Motion parameters (conservative for real hardware: ~1mm per step, slow loop rate).
# At 20Hz with 200 steps/edge, each edge takes 10s → full square = 40s per loop.
# Per-step delta = 2 * half_size / steps_per_edge = 0.10 / 200 = 0.5mm (very safe).
STEPS_PER_EDGE = 200
COMMAND_HZ = 20

# Hard cap on per-step displacement for safety (meters).
# If the computed step exceeds this, the script raises before sending any command.
MAX_PER_STEP_MM = 2.0


def parse_args():
    p = argparse.ArgumentParser(description="Cartesian square trajectory via gRPC")
    p.add_argument("--arm_addr", type=str, default="localhost:50052", help="ArmService gRPC address")
    p.add_argument("--gripper_addr", type=str, default="localhost:50051", help="GripperService address")
    p.add_argument("--show_camera", action="store_true", help="Display the gripper camera feed")
    p.add_argument("--loops", type=int, default=0, help="Number of square loops (0 = infinite)")
    p.add_argument(
        "--half_size",
        type=float,
        default=SQUARE_HALF_SIZE,
        help="Half-edge of the square in meters (default: 0.10 = 20cm square)",
    )
    p.add_argument(
        "--steps_per_edge",
        type=int,
        default=STEPS_PER_EDGE,
        help="Number of small deltas per edge (more = smoother but slower)",
    )
    p.add_argument("--fps", type=float, default=COMMAND_HZ, help="Control loop rate in Hz")
    p.add_argument(
        "--orient_gain",
        type=float,
        default=0.3,
        help="Proportional gain for orientation correction (0 = no lock, 1 = full correction each step)",
    )
    return p.parse_args()


def build_square_deltas(half: float, steps_per_edge: int) -> list[tuple[float, float, float]]:
    """Generate per-step (dx, dy, dz) deltas that trace a square in the YZ plane.

    Square corners (relative to the starting center):
      +Y +Z   →   +Y -Z   →   -Y -Z   →   -Y +Z   →   back to start

    Each edge is divided into `steps_per_edge` small deltas.
    """
    # Edge deltas (full-edge displacement = 2 * half)
    edge_len = 2 * half
    step = edge_len / steps_per_edge

    edges = [
        (0.0, 0.0, -step),  # top-right → bottom-right: move -Z
        (0.0, -step, 0.0),  # bottom-right → bottom-left: move -Y
        (0.0, 0.0, +step),  # bottom-left → top-left: move +Z
        (0.0, +step, 0.0),  # top-left → top-right: move +Y
    ]

    deltas = []
    for edge in edges:
        for _ in range(steps_per_edge):
            deltas.append(edge)
    return deltas


def send_move_to_start(
    arm_stub, half: float, steps: int, dt: float, target_r6d: np.ndarray, orient_gain: float
):
    """Move from the current pose to the top-right corner of the square.

    Half-edge in +Y and +Z, divided into `steps` small deltas. At each step,
    computes a rotation delta to keep the orientation locked to `target_r6d`.
    """
    dy = half / steps
    dz = half / steps

    for _ in range(steps):
        _, current_r6d, _ = get_ee_pose(arm_stub)
        dr6d = compute_orientation_correction(current_r6d, target_r6d, orient_gain)
        arm_stub.SendCartesianDelta(arm_pb2.CartesianDelta(dx=0.0, dy=dy, dz=dz, dr6d=dr6d.tolist()))
        time.sleep(dt)


def camera_display_thread(gripper_addr: str, stop_event: threading.Event):
    """Show the gripper camera feed in an OpenCV window (optional)."""
    import cv2
    from openarm_gripette_simu.proto import gripper_pb2, gripper_pb2_grpc

    channel = grpc.insecure_channel(gripper_addr)
    stub = gripper_pb2_grpc.GripperServiceStub(channel)

    try:
        for frame in stub.StreamState(gripper_pb2.StreamRequest()):
            if stop_event.is_set():
                break
            img = cv2.imdecode(np.frombuffer(frame.jpeg_data, np.uint8), cv2.IMREAD_COLOR)
            cv2.imshow("Gripette camera", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_event.set()
                break
    except grpc.RpcError as e:
        if not stop_event.is_set():
            logger.warning(f"Camera stream ended: {e}")
    finally:
        channel.close()
        cv2.destroyAllWindows()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    # ---- Safety check: per-step displacement ----
    step_m = 2 * args.half_size / args.steps_per_edge
    step_mm = step_m * 1000
    if step_mm > MAX_PER_STEP_MM:
        raise ValueError(
            f"Per-step displacement {step_mm:.2f} mm exceeds safety limit "
            f"({MAX_PER_STEP_MM} mm). Increase --steps_per_edge or reduce --half_size."
        )
    logger.info(f"Motion: {step_mm:.2f} mm per step @ {args.fps:.0f} Hz = {step_mm * args.fps:.1f} mm/s")

    # ---- Connect ----
    logger.info(f"Connecting to ArmService at {args.arm_addr}")
    channel = grpc.insecure_channel(args.arm_addr)
    arm_stub = arm_pb2_grpc.ArmServiceStub(channel)
    ping = arm_stub.Ping(arm_pb2.ArmPingRequest())
    logger.info(f"Server: {ping.status} (uptime: {ping.uptime_seconds:.1f}s)")

    # ---- Optional camera display ----
    stop_event = threading.Event()
    cam_thread = None
    if args.show_camera:
        logger.info(f"Starting camera display from {args.gripper_addr}")
        cam_thread = threading.Thread(
            target=camera_display_thread, args=(args.gripper_addr, stop_event), daemon=True
        )
        cam_thread.start()

    # ---- Capture starting pose (lock orientation to this) ----
    start_pos, target_r6d, start_joints = get_ee_pose(arm_stub)
    logger.info(f"Starting EE pos: [{start_pos[0]:+.3f}, {start_pos[1]:+.3f}, {start_pos[2]:+.3f}] m")
    logger.info(f"Starting joints (rad): {start_joints.round(3).tolist()}")
    logger.info(f"Target orientation locked to: {target_r6d.round(3).tolist()}")

    dt = 1.0 / args.fps
    orient_gain = args.orient_gain

    try:
        # ---- Move to top-right corner (square start) ----
        logger.info(f"Moving to square start corner (+{args.half_size * 100:.0f}cm in Y and Z)")
        send_move_to_start(arm_stub, args.half_size, args.steps_per_edge, dt, target_r6d, orient_gain)

        # ---- Build + loop the square deltas ----
        deltas = build_square_deltas(args.half_size, args.steps_per_edge)
        total_steps = len(deltas)
        period_s = total_steps * dt
        logger.info(f"Running square loop: {total_steps} steps per loop, ~{period_s:.1f}s per loop")

        loop_idx = 0
        while not stop_event.is_set():
            for i, (dx, dy, dz) in enumerate(deltas):
                if stop_event.is_set():
                    break
                loop_start = time.perf_counter()

                # Read current state, compute rotation correction to keep orientation fixed
                current_pos, current_r6d, current_joints = get_ee_pose(arm_stub)
                dr6d = compute_orientation_correction(current_r6d, target_r6d, orient_gain)

                arm_stub.SendCartesianDelta(arm_pb2.CartesianDelta(dx=dx, dy=dy, dz=dz, dr6d=dr6d.tolist()))

                if i % (args.steps_per_edge // 4) == 0:
                    # Proper rotation angle in degrees (much more interpretable)
                    angle_deg = rotation_angle_deg(target_r6d, current_r6d)
                    # Wrist roll drift in degrees (joint 5 = r_wrist_roll)
                    wrist_roll_deg = np.rad2deg(current_joints[5] - start_joints[5])
                    logger.info(
                        f"  loop {loop_idx} step {i:>4d}/{total_steps}: "
                        f"EE [{current_pos[0]:+.3f}, {current_pos[1]:+.3f}, {current_pos[2]:+.3f}] "
                        f"orient_err={angle_deg:.2f}° wrist_roll={wrist_roll_deg:+.2f}°"
                    )
                # Keep real-time pacing
                elapsed = time.perf_counter() - loop_start
                if (remaining := dt - elapsed) > 0:
                    time.sleep(remaining)

            loop_idx += 1
            if args.loops > 0 and loop_idx >= args.loops:
                logger.info(f"Completed {args.loops} loops")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        stop_event.set()
        if cam_thread is not None:
            cam_thread.join(timeout=2.0)
        channel.close()


if __name__ == "__main__":
    main()
