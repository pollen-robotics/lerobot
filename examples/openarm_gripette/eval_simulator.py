"""Run a trained Diffusion Policy on the OpenArm Gripette simulator.

Connects to the gRPC simulator and runs closed-loop inference:
  1. Get camera image + gripper joints from GripperService.
  2. Build observation: state=[proximal, distal] + camera image.
  3. Run DiffusionPolicy → 11D delta action [dx,dy,dz, dr6d_0..5, grip_1, grip_2].
  4. Send position+rotation deltas to ArmService (IK handled server-side).
  5. Send gripper goals to GripperService.

The model does NOT see absolute EE position (it's meaningless in the SLAM frame).
Actions are raw deltas — no RelativeActionsProcessorStep needed.

The simulator exposes the same gRPC API as the real robot, so this script
works with both by changing the host/port.

Prerequisites:
  - Simulator running: python -m openarm_gripette_simu [--headless]
  - Trained checkpoint from train.py
  - Generated gRPC stubs accessible (pip install openarm_gripette_simu)

Usage:
  uv run python examples/openarm_gripette/eval_simulator.py \\
      --checkpoint outputs/gripette/diffusion \\
      --duration 30

  # Custom host/ports:
  uv run python examples/openarm_gripette/eval_simulator.py \\
      --checkpoint outputs/gripette/diffusion \\
      --arm_addr localhost:50052 --gripper_addr localhost:50051
"""

import argparse
import logging
import threading
import time

import cv2
import grpc
import numpy as np
import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionPolicy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera stream reader (background thread)
# ---------------------------------------------------------------------------


class CameraStreamReader:
    """Reads the GripperService.StreamState() gRPC stream in a background thread.

    Keeps the latest camera frame and gripper motor positions available for
    the main inference loop. The stream runs at ~10 Hz from the simulator.
    """

    def __init__(self, gripper_stub):
        self._stub = gripper_stub
        self._lock = threading.Lock()
        self._latest_image: np.ndarray | None = None
        self._latest_gripper: np.ndarray | None = None
        self._frame_count = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
        # Wait for the first frame
        deadline = time.time() + 5.0
        while self._latest_image is None and time.time() < deadline:
            time.sleep(0.05)
        if self._latest_image is None:
            raise TimeoutError("No camera frame received within 5s")
        logger.info(f"Camera stream started (image shape: {self._latest_image.shape})")

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_latest(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (image_rgb, gripper_joints) from the latest stream frame.

        image_rgb: (H, W, 3) uint8 numpy array
        gripper_joints: (2,) float array [proximal, distal] in radians
        """
        with self._lock:
            if self._latest_image is None:
                raise RuntimeError("No camera frame available yet")
            return self._latest_image.copy(), self._latest_gripper.copy()

    def _stream_loop(self):
        from openarm_gripette_simu.proto import gripper_pb2

        try:
            for frame in self._stub.StreamState(gripper_pb2.StreamRequest()):
                if not self._running:
                    break
                # Decode JPEG to numpy array (BGR from cv2, convert to RGB)
                img_bgr = cv2.imdecode(np.frombuffer(frame.jpeg_data, np.uint8), cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                gripper = np.array(
                    [frame.motor_state.motor1_position, frame.motor_state.motor2_position],
                    dtype=np.float32,
                )

                with self._lock:
                    self._latest_image = img_rgb
                    self._latest_gripper = gripper
                    self._frame_count += 1
        except grpc.RpcError as e:
            if self._running:
                logger.error(f"Camera stream error: {e}")


# ---------------------------------------------------------------------------
# Debug visualization
# ---------------------------------------------------------------------------

ACTION_NAMES = [
    "dx",
    "dy",
    "dz",
    "dr6d_0",
    "dr6d_1",
    "dr6d_2",
    "dr6d_3",
    "dr6d_4",
    "dr6d_5",
    "proximal",
    "distal",
]
GRIPPER_NAMES = ["proximal", "distal"]


def debug_log(
    step: int,
    state: np.ndarray,
    action: np.ndarray,
    delta_pos: np.ndarray,
    delta_rot: np.ndarray,
    gripper_goal: np.ndarray,
    dataset_stats: dict | None = None,
):
    """Print detailed state/action info for debugging."""
    print(f"\n{'=' * 70}")
    print(f"  STEP {step}")
    print(f"{'=' * 70}")

    print("\n  Observation state (2D gripper):")
    for i, name in enumerate(GRIPPER_NAMES):
        val = state[i]
        line = f"    {name:10s}: {val:+.6f}"
        if dataset_stats and name in dataset_stats:
            mn, mx = dataset_stats[name]
            if val < mn or val > mx:
                line += f"  *** OUT OF RANGE [{mn:+.4f}, {mx:+.4f}] ***"
            else:
                frac = (val - mn) / (mx - mn + 1e-12)
                bar = "=" * int(frac * 20) + " " * (20 - int(frac * 20))
                line += f"  [{bar}] ({frac * 100:.0f}%)"
        print(line)

    print("\n  Predicted action (11D deltas + gripper):")
    for i, name in enumerate(ACTION_NAMES):
        print(f"    {name:10s}: {action[i]:+.6f}")

    print(f"\n  Position delta: [{delta_pos[0]:+.6f}, {delta_pos[1]:+.6f}, {delta_pos[2]:+.6f}]")
    print(
        f"  Rotation delta: [{delta_rot[0]:+.4f}, {delta_rot[1]:+.4f}, {delta_rot[2]:+.4f}, "
        f"{delta_rot[3]:+.4f}, {delta_rot[4]:+.4f}, {delta_rot[5]:+.4f}]"
    )
    print(f"  Gripper goal:   [{gripper_goal[0]:+.4f}, {gripper_goal[1]:+.4f}]")

    pos_magnitude = np.linalg.norm(delta_pos)
    print(f"  Delta pos magnitude: {pos_magnitude * 1000:.2f} mm")


def debug_show_image(image_rgb: np.ndarray, step: int):
    """Show the camera image that the policy sees (with step overlay)."""
    img_display = image_rgb.copy()
    # Add step counter overlay
    cv2.putText(img_display, f"Step {step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    # Show in BGR for cv2
    cv2.imshow("Policy Camera Input", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def load_dataset_stats_for_debug(checkpoint_path: str) -> dict:
    """Load state min/max from the preprocessor stats for range checking."""
    from pathlib import Path

    from safetensors.torch import load_file

    stats_file = Path(checkpoint_path) / "policy_preprocessor_step_4_normalizer_processor.safetensors"
    if not stats_file.exists():
        return {}

    stats = load_file(stats_file)
    result = {}
    state_min = stats.get("observation.state/min")
    state_max = stats.get("observation.state/max")
    if state_min is not None and state_max is not None:
        for i, name in enumerate(GRIPPER_NAMES):
            if i < len(state_min):
                result[name] = (state_min[i].item(), state_max[i].item())
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Run Gripette policy on simulator")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    p.add_argument("--arm_addr", type=str, default="localhost:50052", help="ArmService gRPC address")
    p.add_argument("--gripper_addr", type=str, default="localhost:50051", help="GripperService gRPC address")
    p.add_argument("--device", type=str, default="cuda", help="Compute device")
    p.add_argument("--duration", type=float, default=30.0, help="Run duration in seconds")
    p.add_argument(
        "--fps", type=float, default=10.0, help="Control loop frequency (limited by camera stream)"
    )
    p.add_argument("--debug", action="store_true", help="Show camera feed + log detailed state/action info")
    p.add_argument("--no_send", action="store_true", help="Debug only: do NOT send commands to the robot")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device)

    # ---- Load policy and processors ----
    logger.info(f"Loading policy from {args.checkpoint}")
    policy = DiffusionPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=args.checkpoint)

    action_dim = policy.config.action_feature.shape[0]
    logger.info(
        f"Policy: action_dim={action_dim}, "
        f"n_action_steps={policy.config.n_action_steps}, "
        f"relative_actions={policy.config.use_relative_actions}"
    )

    # ---- Connect to simulator gRPC services ----
    # Requires: pip install openarm-gripette-simu (or uv pip install -e <path>)
    # The simulator package provides the gRPC proto stubs.
    from openarm_gripette_simu.proto import arm_pb2, arm_pb2_grpc, gripper_pb2, gripper_pb2_grpc

    logger.info(f"Connecting to ArmService at {args.arm_addr}")
    arm_channel = grpc.insecure_channel(args.arm_addr)
    arm_stub = arm_pb2_grpc.ArmServiceStub(arm_channel)

    logger.info(f"Connecting to GripperService at {args.gripper_addr}")
    gripper_channel = grpc.insecure_channel(args.gripper_addr)
    gripper_stub = gripper_pb2_grpc.GripperServiceStub(gripper_channel)

    # Ping both services to verify connection
    arm_ping = arm_stub.Ping(arm_pb2.ArmPingRequest())
    logger.info(f"ArmService: {arm_ping.status} (uptime: {arm_ping.uptime_seconds:.1f}s)")

    gripper_ping = gripper_stub.Ping(gripper_pb2.PingRequest())
    logger.info(f"GripperService: {gripper_ping.status} (uptime: {gripper_ping.uptime_seconds:.1f}s)")

    # ---- Start camera stream reader ----
    camera_reader = CameraStreamReader(gripper_stub)
    camera_reader.start()

    # ---- Control loop ----
    dt = 1.0 / args.fps
    start_time = time.time()
    step_count = 0
    policy.reset()

    # Load dataset stats for range-checking in debug mode
    dataset_stats = load_dataset_stats_for_debug(args.checkpoint) if args.debug else {}
    if args.debug and dataset_stats:
        logger.info(f"Debug mode: loaded state range stats ({len(dataset_stats)} dims)")
    if args.no_send:
        logger.info("NO_SEND mode: commands will NOT be sent to the robot")

    logger.info(f"Running for {args.duration}s at {args.fps} Hz")

    try:
        while (time.time() - start_time) < args.duration:
            loop_start = time.perf_counter()

            # --- 1. Get current state ---
            # Gripper + camera from the streaming thread
            camera_image, gripper_joints = camera_reader.get_latest()

            # --- 2. Build observation ---
            # observation.state = gripper only (2D) — no absolute position
            state = gripper_joints  # [proximal, distal]

            # --- 3. Build policy input tensors ---
            state_tensor = torch.from_numpy(state).float()
            image_tensor = torch.from_numpy(camera_image).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1).contiguous()  # (H,W,C) -> (C,H,W)

            batch = {
                "observation.state": state_tensor.unsqueeze(0).to(device),
                "observation.images.cam0": image_tensor.unsqueeze(0).to(device),
            }

            # --- 4. Preprocess -> Policy -> Postprocess ---
            # The model outputs raw deltas (pre-computed in dataset, no RelativeActionsProcessorStep)
            batch = preprocessor(batch)
            with torch.no_grad():
                action = policy.select_action(batch)
            action = postprocessor(action)

            # --- 5. Extract deltas ---
            action_np = action.squeeze(0).cpu().numpy()

            # Action is already deltas: [dx, dy, dz, dr6d_0..5, proximal, distal]
            delta_pos = action_np[:3]
            delta_rot_6d = action_np[3:9]
            gripper_goal = action_np[9:]  # absolute gripper targets

            # --- 6. Debug inspection ---
            if args.debug:
                debug_log(step_count, state, action_np, delta_pos, delta_rot_6d, gripper_goal, dataset_stats)
                debug_show_image(camera_image, step_count)

            # --- 7. Send commands (unless --no_send) ---
            if not args.no_send:
                arm_response = arm_stub.SendCartesianDelta(
                    arm_pb2.CartesianDelta(
                        dx=float(delta_pos[0]),
                        dy=float(delta_pos[1]),
                        dz=float(delta_pos[2]),
                        dr6d=delta_rot_6d.tolist(),
                    )
                )

                if not arm_response.success:
                    logger.warning(f"Arm command failed: {arm_response.error}")

                gripper_stub.SendMotorCommand(
                    gripper_pb2.MotorCommand(
                        motor1_goal=float(gripper_goal[0]),
                        motor2_goal=float(gripper_goal[1]) if len(gripper_goal) > 1 else 0.0,
                    )
                )

            step_count += 1

            # --- Timing ---
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Periodic status log (non-debug mode)
            if not args.debug and step_count % 10 == 0:
                actual_fps = 1.0 / max(time.perf_counter() - loop_start, 1e-6)
                delta_mm = np.linalg.norm(delta_pos) * 1000
                logger.info(
                    f"Step {step_count:>5d} | "
                    f"FPS: {actual_fps:5.1f} | "
                    f"delta: [{delta_pos[0]:+.4f}, {delta_pos[1]:+.4f}, {delta_pos[2]:+.4f}] "
                    f"({delta_mm:.1f}mm)"
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        camera_reader.stop()
        arm_channel.close()
        gripper_channel.close()
        if args.debug:
            cv2.destroyAllWindows()
        logger.info(f"Done. Executed {step_count} steps in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
