"""Run a trained Diffusion Policy on the OpenArm Gripette simulator.

Connects to the gRPC simulator and runs closed-loop inference.
Auto-detects the observation.state mode from the checkpoint:
  - 2D state (gripper only): just reads gripper joints from the camera stream.
  - 11D state (relative proprioception): also reads EE pose from ArmService,
    computes position + rotation relative to the episode start.

Usage:
  uv run python examples/openarm_gripette/eval_simulator.py \\
      --checkpoint outputs/gripette/diffusion --duration 30

  # Debug mode (shows camera feed + detailed state/action log):
  uv run python examples/openarm_gripette/eval_simulator.py \\
      --checkpoint outputs/gripette/diffusion --debug --no_send
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
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_IMAGES
from lerobot.utils.rotation import (
    rotation_6d_to_rotation_matrix_numpy,
    rotation_matrix_to_rotation_6d_numpy,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Camera stream reader (background thread)
# ---------------------------------------------------------------------------


class CameraStreamReader:
    """Reads GripperService.StreamState() in a background thread."""

    def __init__(self, gripper_stub, gripper_pb2):
        self._stub = gripper_stub
        self._gripper_pb2 = gripper_pb2
        self._lock = threading.Lock()
        self._latest_image: np.ndarray | None = None
        self._latest_gripper: np.ndarray | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._stream_loop, daemon=True)
        self._thread.start()
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
        """Returns (image_rgb, gripper_joints)."""
        with self._lock:
            if self._latest_image is None:
                raise RuntimeError("No camera frame available yet")
            return self._latest_image.copy(), self._latest_gripper.copy()

    def _stream_loop(self):
        try:
            for frame in self._stub.StreamState(self._gripper_pb2.StreamRequest()):
                if not self._running:
                    break
                img_bgr = cv2.imdecode(np.frombuffer(frame.jpeg_data, np.uint8), cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                gripper = np.array(
                    [frame.motor_state.motor1_position, frame.motor_state.motor2_position],
                    dtype=np.float32,
                )
                with self._lock:
                    self._latest_image = img_rgb
                    self._latest_gripper = gripper
        except grpc.RpcError as e:
            if self._running:
                logger.error(f"Camera stream error: {e}")


# ---------------------------------------------------------------------------
# Relative proprioception helpers
# ---------------------------------------------------------------------------


def compute_relative_state(
    arm_state, gripper_joints: np.ndarray, start_pos: np.ndarray, start_rot_matrix: np.ndarray
) -> np.ndarray:
    """Compute 11D state: position + rotation relative to episode start, plus gripper.

    Args:
        arm_state: gRPC ArmState with x, y, z, r6d fields.
        gripper_joints: (2,) array [proximal, distal].
        start_pos: (3,) position at episode start.
        start_rot_matrix: (3, 3) rotation matrix at episode start.

    Returns:
        (11,) array: [dx_start, dy_start, dz_start, r6d_rel_0..5, proximal, distal]
    """
    pos = np.array([arm_state.x, arm_state.y, arm_state.z], dtype=np.float32)
    rot_6d = np.array(list(arm_state.r6d), dtype=np.float32)

    # Position relative to start
    rel_pos = pos - start_pos

    # Rotation relative to start: R_rel = R_current @ R_start^{-1}
    r_current = rotation_6d_to_rotation_matrix_numpy(rot_6d.reshape(1, 6))[0]
    r_relative = r_current @ start_rot_matrix.T
    rel_rot_6d = rotation_matrix_to_rotation_6d_numpy(r_relative.reshape(1, 3, 3))[0]

    return np.concatenate([rel_pos, rel_rot_6d, gripper_joints])


def capture_start_pose(arm_stub, arm_pb2):
    """Capture the EE pose at the start of an episode for relative computation."""
    arm_state = arm_stub.GetArmState(arm_pb2.GetArmStateRequest())
    start_pos = np.array([arm_state.x, arm_state.y, arm_state.z], dtype=np.float32)
    start_r6d = np.array(list(arm_state.r6d), dtype=np.float32)
    start_rot_matrix = rotation_6d_to_rotation_matrix_numpy(start_r6d.reshape(1, 6))[0]
    return start_pos, start_rot_matrix


# ---------------------------------------------------------------------------
# Temporal ensembling (ACT-style)
# ---------------------------------------------------------------------------


class TemporalEnsembleBuffer:
    """Rolling buffer of recent predicted action chunks + recency-weighted aggregation.

    At each inference step t, push the full predicted chunk (shape [horizon, action_dim]).
    aggregate() returns a single action for the current timestep computed as the
    exponentially-recency-weighted average of every stored chunk's prediction for t.

    Weight for a chunk of age i (0 = pushed this step) is exp(-k * i). Smaller k
    = more uniform smoothing (more lag); larger k = weight only recent predictions
    (less smoothing).
    """

    def __init__(self, horizon: int, k: float = 0.1):
        self.horizon = horizon
        self.k = k
        # chunks[i] holds the chunk pushed at global step (self.step - 1 - i)
        self.chunks: list[np.ndarray] = []

    def push(self, chunk: np.ndarray) -> None:
        """chunk: (horizon, action_dim) — prediction produced this step."""
        self.chunks.insert(0, chunk.copy())
        # Anything older than horizon has no prediction for the current step.
        if len(self.chunks) > self.horizon:
            self.chunks = self.chunks[: self.horizon]

    def aggregate(self) -> np.ndarray:
        """Return weighted action for the *most recent* timestep (age-0 position)."""
        preds = np.stack([c[i] for i, c in enumerate(self.chunks)])  # (N, action_dim)
        weights = np.exp(-self.k * np.arange(len(self.chunks)))
        weights /= weights.sum()
        return (preds * weights[:, None]).sum(axis=0)


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


def debug_log(step: int, state: np.ndarray, state_names: list, action: np.ndarray):
    """Print detailed state/action info for debugging."""
    print(f"\n{'=' * 60}")
    print(f"  STEP {step}")
    print(f"{'=' * 60}")

    print(f"\n  Observation state ({len(state)}D):")
    for i, name in enumerate(state_names):
        print(f"    {name:12s}: {state[i]:+.6f}")

    print(f"\n  Predicted action ({len(action)}D):")
    for i, name in enumerate(ACTION_NAMES):
        print(f"    {name:10s}: {action[i]:+.6f}")

    delta_mm = np.linalg.norm(action[:3]) * 1000
    print(f"\n  Delta pos magnitude: {delta_mm:.2f} mm")


def debug_show_image(image_rgb: np.ndarray, step: int):
    img_display = image_rgb.copy()
    cv2.putText(img_display, f"Step {step}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.imshow("Policy Camera Input", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


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
    p.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help="Override policy.config.n_action_steps at inference time. "
        "Smaller values re-infer more often and smooth chunk-boundary jerks "
        "(e.g. 1 = re-infer every step). If omitted, use the checkpoint's value.",
    )
    p.add_argument(
        "--temporal_ensemble",
        action="store_true",
        help="Enable ACT-style temporal ensembling: aggregate every step's full predicted "
        "chunk via a recency-weighted average, producing a smoothed command trajectory. "
        "Bypasses select_action and runs inference every control step.",
    )
    p.add_argument(
        "--temporal_ensemble_k",
        type=float,
        default=0.1,
        help="Weight decay for temporal ensembling: w_i = exp(-k * age). "
        "Smaller k = more uniform averaging (heavier smoothing, more lag). "
        "Typical range 0.01 (very smooth) -> 0.5 (light smoothing).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    # force=True: some lerobot submodules configure the root logger during import,
    # which makes a plain basicConfig() a no-op and silences our logs.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    device = torch.device(args.device)

    # ---- Load policy and processors ----
    logger.info(f"Loading policy from {args.checkpoint}")
    policy = DiffusionPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=args.checkpoint)

    # Optional override of n_action_steps at inference time (no retraining needed).
    # Smaller values re-infer more often, eliminating chunk-boundary jerks at the
    # cost of more GPU calls per second.
    if args.n_action_steps is not None:
        logger.info(f"Overriding n_action_steps: {policy.config.n_action_steps} -> {args.n_action_steps}")
        policy.config.n_action_steps = args.n_action_steps

    # Auto-detect state mode from the checkpoint
    state_dim = policy.config.robot_state_feature.shape[0]
    use_relative_proprio = state_dim > 2
    state_names = (
        [
            "dx_start",
            "dy_start",
            "dz_start",
            "r6d_rel_0",
            "r6d_rel_1",
            "r6d_rel_2",
            "r6d_rel_3",
            "r6d_rel_4",
            "r6d_rel_5",
            "proximal",
            "distal",
        ]
        if use_relative_proprio
        else ["proximal", "distal"]
    )

    logger.info(
        f"Policy: action_dim={policy.config.action_feature.shape[0]}, "
        f"state_dim={state_dim} ({'relative proprio' if use_relative_proprio else 'gripper only'}), "
        f"n_action_steps={policy.config.n_action_steps}"
    )

    # ---- Connect to simulator gRPC services ----
    from openarm_gripette_simu.proto import arm_pb2, arm_pb2_grpc, gripper_pb2, gripper_pb2_grpc

    arm_channel = grpc.insecure_channel(args.arm_addr)
    arm_stub = arm_pb2_grpc.ArmServiceStub(arm_channel)
    gripper_channel = grpc.insecure_channel(args.gripper_addr)
    gripper_stub = gripper_pb2_grpc.GripperServiceStub(gripper_channel)

    arm_stub.Ping(arm_pb2.ArmPingRequest())
    gripper_stub.Ping(gripper_pb2.PingRequest())
    logger.info("Connected to simulator")

    # ---- Start camera stream ----
    camera_reader = CameraStreamReader(gripper_stub, gripper_pb2)
    camera_reader.start()

    # ---- Capture start pose (for relative proprioception) ----
    start_pos, start_rot_matrix = None, None
    if use_relative_proprio:
        start_pos, start_rot_matrix = capture_start_pose(arm_stub, arm_pb2)
        logger.info(f"Captured start pose: pos=[{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")

    if args.no_send:
        logger.info("NO_SEND mode: commands will NOT be sent to the robot")

    # ---- Control loop ----
    dt = 1.0 / args.fps
    start_time = time.time()
    step_count = 0
    policy.reset()

    # ---- Temporal ensembling setup ----
    ensemble: TemporalEnsembleBuffer | None = None
    if args.temporal_ensemble:
        # Each predicted chunk used for ensembling has length
        # `horizon - n_obs_steps + 1` (generate_actions slices from index n_obs_steps-1).
        effective_horizon = policy.config.horizon - policy.config.n_obs_steps + 1
        ensemble = TemporalEnsembleBuffer(horizon=effective_horizon, k=args.temporal_ensemble_k)
        logger.info(
            f"Temporal ensembling ON (effective_horizon={effective_horizon}, "
            f"k={args.temporal_ensemble_k}). "
            f"Bypassing select_action; running inference every control step."
        )

    logger.info(f"Running for {args.duration}s at {args.fps} Hz")

    try:
        while (time.time() - start_time) < args.duration:
            loop_start = time.perf_counter()

            # --- 1. Get observations ---
            camera_image, gripper_joints = camera_reader.get_latest()

            if use_relative_proprio:
                arm_state = arm_stub.GetArmState(arm_pb2.GetArmStateRequest())
                state = compute_relative_state(arm_state, gripper_joints, start_pos, start_rot_matrix)
            else:
                state = gripper_joints  # 2D

            # --- 2. Build policy input ---
            state_tensor = torch.from_numpy(state).float()
            image_tensor = torch.from_numpy(camera_image).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1).contiguous()

            batch = {
                "observation.state": state_tensor.unsqueeze(0).to(device),
                "observation.images.cam0": image_tensor.unsqueeze(0).to(device),
            }

            # --- 3. Preprocess -> Policy -> Postprocess ---
            batch = preprocessor(batch)
            if ensemble is None:
                # Standard path: policy manages its own internal chunk queue.
                with torch.no_grad():
                    action = policy.select_action(batch)
            else:
                # Temporal-ensembling path: generate a full chunk every step,
                # push into the ensemble buffer, and use its recency-weighted
                # aggregate as the current action. Unnormalization is linear,
                # so we can ensemble in normalized space and postprocess after.
                batch_for_policy = dict(batch)
                # The preprocessor writes a normalized `action` placeholder
                # (Normalizer covers both input and output features); select_action
                # pops it before queue population, and we do the same here to
                # avoid stuffing a None/placeholder into the ACTION deque.
                batch_for_policy.pop(ACTION, None)
                if policy.config.image_features:
                    batch_for_policy[OBS_IMAGES] = torch.stack(
                        [batch_for_policy[k] for k in policy.config.image_features], dim=-4
                    )
                policy._queues = populate_queues(policy._queues, batch_for_policy)
                # generate_actions slices to n_action_steps; override temporarily
                # so predict_action_chunk returns the remaining horizon for the ensemble.
                saved_n_action_steps = policy.config.n_action_steps
                policy.config.n_action_steps = policy.config.horizon - policy.config.n_obs_steps + 1
                try:
                    with torch.no_grad():
                        chunk = policy.predict_action_chunk(batch_for_policy)  # (1, T, A)
                finally:
                    policy.config.n_action_steps = saved_n_action_steps
                ensemble.push(chunk.squeeze(0).cpu().numpy())
                agg_np = ensemble.aggregate()
                action = torch.from_numpy(agg_np).float().unsqueeze(0).to(device)
            action = postprocessor(action)

            # --- 4. Extract deltas ---
            action_np = action.squeeze(0).cpu().numpy()
            delta_pos = action_np[:3]
            delta_rot_6d = action_np[3:9]
            gripper_goal = action_np[9:]

            # --- 5. Debug ---
            if args.debug:
                debug_log(step_count, state, state_names, action_np)
                debug_show_image(camera_image, step_count)

            # --- 6. Send commands ---
            if not args.no_send:
                arm_stub.SendCartesianDelta(
                    arm_pb2.CartesianDelta(
                        dx=float(delta_pos[0]),
                        dy=float(delta_pos[1]),
                        dz=float(delta_pos[2]),
                        dr6d=delta_rot_6d.tolist(),
                    )
                )
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

            if not args.debug and step_count % 10 == 0:
                actual_fps = 1.0 / max(time.perf_counter() - loop_start, 1e-6)
                delta_mm = np.linalg.norm(delta_pos) * 1000
                logger.info(f"Step {step_count:>5d} | FPS: {actual_fps:5.1f} | delta: {delta_mm:.1f}mm")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        camera_reader.stop()
        if args.debug:
            cv2.destroyAllWindows()
        arm_channel.close()
        gripper_channel.close()
        logger.info(f"Done. Executed {step_count} steps in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
