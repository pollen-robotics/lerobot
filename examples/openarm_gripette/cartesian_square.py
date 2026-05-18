"""Move the EE in a square pattern using camera-LOCAL frame deltas.

This is the canonical end-to-end test of the camera-local delta convention
exposed by `arm_servicer.SendCartesianDelta` (sim) and
`grpc_server_real.SendCartesianDelta` (real). The server's integrator is:

    R_target_new   = R_target @ R_delta            (orientation)
    pos_target_new = pos_target + R_target @ Δpos  (position; LOCAL → world)

So `(dx, dy, dz)` is interpreted in the integrator's current camera frame,
NOT in world coordinates. To trace a clean shape with this script you read
the camera's local frame the same way the policy does, and send deltas
defined directly in that frame.

What the square traces (defaults):
  - Plane:     camera-local YZ plane  (image-down × optical-axis)
  - Orientation:    locked to start orientation (R_delta = identity every step)

Visual verification on the camera feed:
  - Edge 1 (camera moves -Z = backward along optical axis): scene zooms OUT.
  - Edge 2 (camera moves -Y = image-up): scene scrolls DOWN.
  - Edge 3 (camera moves +Z = forward along optical axis): scene zooms IN.
  - Edge 4 (camera moves +Y = image-down): scene scrolls UP.

If you instead see world-frame motion (e.g. always moves in the same horizontal
direction regardless of camera tilt), the integrator has reverted to world-
frame deltas — that's the regression the camera-local refactor was meant to
fix. See `feedback_action_deltas_camera_local` in memory.

Usage:
  uv run python examples/openarm_gripette/cartesian_square.py \\
      --arm_addr localhost:50052
  uv run python examples/openarm_gripette/cartesian_square.py \\
      --arm_addr <robot-ip>:50052 --gripper_addr <gripette-ip>:50051 --show_camera

  # Optional: trace the square in the camera-local XY plane (image-right ×
  # image-down) instead of the default YZ plane.
  uv run python examples/openarm_gripette/cartesian_square.py --plane xy
"""

import argparse
import logging
import threading
import time

import grpc
import numpy as np
from openarm_gripette_simu.proto import arm_pb2, arm_pb2_grpc

logger = logging.getLogger(__name__)

# Identity rotation in 6D = first two columns of I_3.
# Sending this as `dr6d` makes the server's integrator apply
# `R_target_new = R_target @ I = R_target` — i.e. orientation stays locked
# at whatever the integrator was initialised to (FK at startup or after Reset).
IDENTITY_R6D = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

# Hard cap on per-step displacement for safety (meters).
MAX_PER_STEP_MM = 2.0

# Square geometry — small and slow for a safe first test.
SQUARE_HALF_SIZE = 0.05      # 5 cm half-edge → 10 cm square
STEPS_PER_EDGE = 200         # 0.5 mm per step at default
COMMAND_HZ = 20              # 200 steps × 1/20 s = 10 s per edge


def get_ee_pose(arm_stub) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fetch FK pose (pos, r6d) and arm joints (rad) from the server.

    NB: this is the server's FK-from-measured-joints reading, not the
    integrator's internal `_target_pos/_target_r6d`. We only use it for
    logging / sanity checks. The integrator is what actually drives motion.
    """
    state = arm_stub.GetArmState(arm_pb2.GetArmStateRequest())
    return (np.array([state.x, state.y, state.z], dtype=np.float64),
            np.array(state.r6d, dtype=np.float64),
            np.array(state.joint_positions, dtype=np.float64))


def build_square_local_deltas(half: float, steps_per_edge: int, plane: str) -> list[tuple[float, float, float]]:
    """Per-step (dx, dy, dz) in the camera-LOCAL frame, tracing a closed square.

    Plane:
      yz  → camera-local YZ plane (image-down × optical-axis): forward/back × up/down
      xy  → camera-local XY plane (image-right × image-down): horizontal in the image
    """
    step = 2 * half / steps_per_edge
    if plane == "yz":
        edges = [
            (0.0, 0.0, -step),  # along -Z (optical axis, backward) → scene zooms OUT
            (0.0, -step, 0.0),  # along -Y (image-up) → scene scrolls DOWN
            (0.0, 0.0, +step),  # along +Z (forward) → scene zooms IN
            (0.0, +step, 0.0),  # along +Y (image-down) → scene scrolls UP
        ]
    elif plane == "xy":
        edges = [
            (+step, 0.0, 0.0),  # along +X (image-right) → scene scrolls LEFT
            (0.0, +step, 0.0),  # along +Y (image-down) → scene scrolls UP
            (-step, 0.0, 0.0),  # along -X (image-left) → scene scrolls RIGHT
            (0.0, -step, 0.0),  # along -Y (image-up) → scene scrolls DOWN
        ]
    else:
        raise ValueError(f"Unknown plane: {plane!r} (expected 'yz' or 'xy')")

    deltas = []
    for edge in edges:
        deltas.extend([edge] * steps_per_edge)
    return deltas


def camera_display_thread(gripper_addr: str, stop_event: threading.Event):
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


def parse_args():
    p = argparse.ArgumentParser(description="Trace a camera-LOCAL Cartesian square via gRPC")
    p.add_argument("--arm_addr", type=str, default="localhost:50052")
    p.add_argument("--gripper_addr", type=str, default="localhost:50051")
    p.add_argument("--show_camera", action="store_true")
    p.add_argument("--loops", type=int, default=0, help="0 = infinite")
    p.add_argument("--half_size", type=float, default=SQUARE_HALF_SIZE)
    p.add_argument("--steps_per_edge", type=int, default=STEPS_PER_EDGE)
    p.add_argument("--fps", type=float, default=COMMAND_HZ)
    p.add_argument("--plane", type=str, default="yz", choices=["yz", "xy"],
                   help="Camera-local plane in which to trace the square (default: yz)")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    # Safety check on per-step displacement.
    step_mm = 2 * args.half_size / args.steps_per_edge * 1000
    if step_mm > MAX_PER_STEP_MM:
        raise ValueError(
            f"Per-step displacement {step_mm:.2f} mm exceeds safety limit "
            f"{MAX_PER_STEP_MM} mm. Increase --steps_per_edge or reduce --half_size."
        )
    logger.info(
        f"Camera-local square in '{args.plane}' plane: {step_mm:.2f} mm/step @ "
        f"{args.fps:.0f} Hz → {step_mm * args.fps:.1f} mm/s, "
        f"{args.steps_per_edge * 4 / args.fps:.1f} s per loop"
    )

    channel = grpc.insecure_channel(args.arm_addr)
    arm_stub = arm_pb2_grpc.ArmServiceStub(channel)
    ping = arm_stub.Ping(arm_pb2.ArmPingRequest())
    logger.info(f"Server: {ping.status} (uptime: {ping.uptime_seconds:.1f}s)")

    stop_event = threading.Event()
    cam_thread = None
    if args.show_camera:
        cam_thread = threading.Thread(
            target=camera_display_thread,
            args=(args.gripper_addr, stop_event),
            daemon=True,
        )
        cam_thread.start()

    # Log the starting pose. We do NOT rotate deltas through it — the server
    # already applies its own integrator rotation `R_target @ Δpos`. Reading
    # the FK pose here is just for the human operator's situational awareness.
    start_pos, start_r6d, start_joints = get_ee_pose(arm_stub)
    logger.info(f"Start EE position (world): "
                f"[{start_pos[0]:+.3f}, {start_pos[1]:+.3f}, {start_pos[2]:+.3f}] m")
    logger.info(f"Start joints (rad): {start_joints.round(3).tolist()}")
    logger.info("Orientation will be locked (sending identity R_delta every step).")

    deltas = build_square_local_deltas(args.half_size, args.steps_per_edge, args.plane)
    total_steps = len(deltas)
    dt = 1.0 / args.fps

    try:
        loop_idx = 0
        while not stop_event.is_set():
            for i, (dx, dy, dz) in enumerate(deltas):
                if stop_event.is_set():
                    break
                t0 = time.perf_counter()

                arm_stub.SendCartesianDelta(arm_pb2.CartesianDelta(
                    dx=dx, dy=dy, dz=dz, dr6d=IDENTITY_R6D,
                ))

                if i % (args.steps_per_edge // 4) == 0:
                    pos, _, _ = get_ee_pose(arm_stub)
                    logger.info(
                        f"  loop {loop_idx} step {i:>4d}/{total_steps}: "
                        f"EE [{pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f}]"
                    )

                elapsed = time.perf_counter() - t0
                if elapsed < dt:
                    time.sleep(dt - elapsed)

            loop_idx += 1
            if args.loops > 0 and loop_idx >= args.loops:
                logger.info(f"Completed {args.loops} loop(s)")
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
