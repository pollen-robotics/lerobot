"""Evaluate a trained Diffusion Policy on the Gripette simulator over multiple episodes.

Runs repeated episodes with environment reset and randomization.
Auto-detects the observation.state mode (2D gripper-only or 11D relative proprioception)
from the checkpoint.

Usage:
  uv run python examples/openarm_gripette/evaluate.py \\
      --checkpoint outputs/gripette/diffusion \\
      --num_episodes 20

  # With debug visualization:
  uv run python examples/openarm_gripette/evaluate.py \\
      --checkpoint outputs/gripette/diffusion \\
      --num_episodes 5 --debug
"""

import argparse
import logging
import time

import cv2
import grpc
import numpy as np
import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.utils.rotation import (
    rotation_6d_to_rotation_matrix_numpy,
    rotation_matrix_to_rotation_6d_numpy,
)

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Gripette policy on simulator")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    p.add_argument("--arm_addr", type=str, default="localhost:50052", help="ArmService gRPC address")
    p.add_argument("--gripper_addr", type=str, default="localhost:50051", help="GripperService gRPC address")
    p.add_argument("--device", type=str, default="cuda", help="Compute device")
    p.add_argument("--num_episodes", type=int, default=20, help="Number of evaluation episodes")
    p.add_argument("--max_steps", type=int, default=300, help="Max steps per episode")
    p.add_argument("--fps", type=float, default=10.0, help="Control loop frequency")
    p.add_argument("--success_check_freq", type=int, default=10, help="Check success every N steps")
    p.add_argument("--debug", action="store_true", help="Show camera feed during evaluation")
    return p.parse_args()


def get_camera_frame(gripper_stub, gripper_pb2):
    """Get latest camera frame and gripper state from the streaming service."""
    for frame in gripper_stub.StreamState(gripper_pb2.StreamRequest()):
        img_bgr = cv2.imdecode(np.frombuffer(frame.jpeg_data, np.uint8), cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gripper = np.array(
            [frame.motor_state.motor1_position, frame.motor_state.motor2_position],
            dtype=np.float32,
        )
        return img_rgb, gripper
    raise RuntimeError("No frame received from camera stream")


def capture_start_pose(arm_stub, arm_pb2):
    """Capture the EE pose at episode start for relative proprioception."""
    arm_state = arm_stub.GetArmState(arm_pb2.GetArmStateRequest())
    start_pos = np.array([arm_state.x, arm_state.y, arm_state.z], dtype=np.float32)
    start_r6d = np.array(list(arm_state.r6d), dtype=np.float32)
    start_rot = rotation_6d_to_rotation_matrix_numpy(start_r6d.reshape(1, 6))[0]
    return start_pos, start_rot


def compute_relative_state(arm_state, gripper_joints, start_pos, start_rot):
    """Compute 11D relative state: [pos_rel(3), rot_rel_6d(6), gripper(2)]."""
    pos = np.array([arm_state.x, arm_state.y, arm_state.z], dtype=np.float32)
    rot_6d = np.array(list(arm_state.r6d), dtype=np.float32)

    rel_pos = pos - start_pos

    r_current = rotation_6d_to_rotation_matrix_numpy(rot_6d.reshape(1, 6))[0]
    r_relative = r_current @ start_rot.T
    rel_rot_6d = rotation_matrix_to_rotation_6d_numpy(r_relative.reshape(1, 3, 3))[0]

    return np.concatenate([rel_pos, rel_rot_6d, gripper_joints])


def build_observation(
    arm_stub,
    arm_pb2,
    gripper_stub,
    gripper_pb2,
    use_relative_proprio,
    start_pos,
    start_rot,
):
    """Build the full observation (camera image + state) for one step."""
    camera_image, gripper_joints = get_camera_frame(gripper_stub, gripper_pb2)

    if use_relative_proprio:
        arm_state = arm_stub.GetArmState(arm_pb2.GetArmStateRequest())
        state = compute_relative_state(arm_state, gripper_joints, start_pos, start_rot)
    else:
        state = gripper_joints

    return camera_image, state


def run_episode(
    policy,
    preprocessor,
    postprocessor,
    arm_stub,
    gripper_stub,
    arm_pb2,
    gripper_pb2,
    device,
    max_steps,
    fps,
    success_check_freq,
    debug,
    use_relative_proprio,
    start_pos,
    start_rot,
) -> dict:
    """Run a single evaluation episode. Returns dict with stats."""
    dt = 1.0 / fps
    episode_start = time.perf_counter()

    for step in range(max_steps):
        loop_start = time.perf_counter()

        # --- Observe ---
        camera_image, state = build_observation(
            arm_stub,
            arm_pb2,
            gripper_stub,
            gripper_pb2,
            use_relative_proprio,
            start_pos,
            start_rot,
        )

        state_tensor = torch.from_numpy(state).float()
        image_tensor = torch.from_numpy(camera_image).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).contiguous()

        batch = {
            "observation.state": state_tensor.unsqueeze(0).to(device),
            "observation.images.cam0": image_tensor.unsqueeze(0).to(device),
        }

        # --- Inference ---
        batch = preprocessor(batch)
        with torch.no_grad():
            action = policy.select_action(batch)
        action = postprocessor(action)

        action_np = action.squeeze(0).cpu().numpy()
        delta_pos = action_np[:3]
        delta_rot_6d = action_np[3:9]
        gripper_goal = action_np[9:]

        # --- Send commands ---
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

        # --- Debug display ---
        if debug:
            img_display = camera_image.copy()
            delta_mm = np.linalg.norm(delta_pos) * 1000
            cv2.putText(
                img_display,
                f"Step {step} | delta {delta_mm:.1f}mm",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Evaluation", cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        # --- Check success ---
        if step > 0 and step % success_check_freq == 0:
            status = arm_stub.GetSuccessStatus(arm_pb2.SuccessStatusRequest())
            if status.goal_reached:
                return {
                    "success": True,
                    "steps": step + 1,
                    "displacement_mm": status.cube_displacement * 1000,
                    "duration_s": time.perf_counter() - episode_start,
                }

        # --- Timing ---
        elapsed = time.perf_counter() - loop_start
        if (remaining := dt - elapsed) > 0:
            time.sleep(remaining)

    # Episode ended without success
    status = arm_stub.GetSuccessStatus(arm_pb2.SuccessStatusRequest())
    return {
        "success": status.goal_reached,
        "steps": max_steps,
        "displacement_mm": status.cube_displacement * 1000,
        "duration_s": time.perf_counter() - episode_start,
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device)

    # ---- Load policy ----
    logger.info(f"Loading policy from {args.checkpoint}")
    policy = DiffusionPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=args.checkpoint)

    # Auto-detect state mode
    state_dim = policy.config.robot_state_feature.shape[0]
    use_relative_proprio = state_dim > 2
    logger.info(
        f"Policy: state_dim={state_dim} ({'relative proprio' if use_relative_proprio else 'gripper only'}), "
        f"action_dim={policy.config.action_feature.shape[0]}, "
        f"n_action_steps={policy.config.n_action_steps}"
    )

    # ---- Connect to simulator ----
    from openarm_gripette_simu.proto import arm_pb2, arm_pb2_grpc, gripper_pb2, gripper_pb2_grpc

    arm_channel = grpc.insecure_channel(args.arm_addr)
    arm_stub = arm_pb2_grpc.ArmServiceStub(arm_channel)
    gripper_channel = grpc.insecure_channel(args.gripper_addr)
    gripper_stub = gripper_pb2_grpc.GripperServiceStub(gripper_channel)

    arm_stub.Ping(arm_pb2.ArmPingRequest())
    gripper_stub.Ping(gripper_pb2.PingRequest())
    logger.info("Connected to simulator")

    # ---- Evaluation loop ----
    results = []
    logger.info(
        f"\nStarting evaluation: {args.num_episodes} episodes, "
        f"max {args.max_steps} steps/episode at {args.fps} Hz\n"
    )

    for ep in range(args.num_episodes):
        # Reset environment with randomization
        reset_resp = arm_stub.Reset(arm_pb2.ResetRequest())
        if not reset_resp.success:
            logger.error(f"Reset failed: {reset_resp.error}")
            continue

        # Reset policy action queue
        policy.reset()

        # Small delay for physics to settle
        time.sleep(0.5)

        # Capture start pose for relative proprioception (after reset)
        start_pos, start_rot = None, None
        if use_relative_proprio:
            start_pos, start_rot = capture_start_pose(arm_stub, arm_pb2)

        logger.info(
            f"Episode {ep + 1}/{args.num_episodes} — "
            f"cube at ({reset_resp.cube_x:.3f}, {reset_resp.cube_y:.3f}, {reset_resp.cube_z:.3f})"
        )

        result = run_episode(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            arm_stub=arm_stub,
            gripper_stub=gripper_stub,
            arm_pb2=arm_pb2,
            gripper_pb2=gripper_pb2,
            device=device,
            max_steps=args.max_steps,
            fps=args.fps,
            success_check_freq=args.success_check_freq,
            debug=args.debug,
            use_relative_proprio=use_relative_proprio,
            start_pos=start_pos,
            start_rot=start_rot,
        )
        results.append(result)

        status_str = "SUCCESS" if result["success"] else "FAIL"
        logger.info(
            f"  -> {status_str} | steps: {result['steps']:>3d} | "
            f"displacement: {result['displacement_mm']:.1f}mm | "
            f"time: {result['duration_s']:.1f}s"
        )

    # ---- Summary ----
    num_success = sum(r["success"] for r in results)
    num_total = len(results)
    success_rate = num_success / num_total * 100 if num_total > 0 else 0
    avg_displacement = np.mean([r["displacement_mm"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])
    success_results = [r for r in results if r["success"]]
    avg_success_steps = np.mean([r["steps"] for r in success_results]) if success_results else 0

    print(f"\n{'=' * 60}")
    print("  EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  State mode:       {'relative proprio (11D)' if use_relative_proprio else 'gripper only (2D)'}")
    print(f"  Episodes:         {num_total}")
    print(f"  Success rate:     {num_success}/{num_total} ({success_rate:.1f}%)")
    print(f"  Avg displacement: {avg_displacement:.1f} mm")
    print(f"  Avg steps (all):  {avg_steps:.0f}")
    if success_results:
        print(f"  Avg steps (success): {avg_success_steps:.0f}")
    print(f"{'=' * 60}")

    if args.debug:
        cv2.destroyAllWindows()
    arm_channel.close()
    gripper_channel.close()


if __name__ == "__main__":
    main()
