"""Deploy a trained Diffusion Policy on an OpenArm robot.

Runs the Gripette policy in closed-loop on a real OpenArm. The policy operates in
11D state/action space with 6D continuous rotation:
  [x, y, z, r6d_0..r6d_5, proximal, distal]

Pipeline per control step:
  1. Read joint encoders + gripper + camera from the robot.
  2. FK: joint angles -> Cartesian EE pose (4x4 matrix).
  3. Convert rotation matrix -> 6D representation.
  4. Build 11D state vector + camera image.
  5. Preprocess (normalize, relative actions).
  6. DiffusionPolicy.select_action() -> delta 11D action (from action chunk queue).
  7. Postprocess (unnormalize, deltas -> absolute).
  8. Convert 6D rotation back to rotation matrix for IK.
  9. IK: Cartesian target -> joint angles.
  10. Send joint commands to the robot.

See README.md in this directory for the full setup guide.

Prerequisites:
  - Trained checkpoint from train.py (model + processors).
  - OpenArm URDF file with correct EE frame.
  - placo library: uv sync --extra kinematics

Usage:
  uv run python examples/openarm_gripette/eval_on_robot.py \\
      --checkpoint outputs/gripette/diffusion \\
      --urdf path/to/openarm.urdf \\
      --can_port can0 --side right --duration 30
"""

import argparse
import logging
import time

import numpy as np
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.model import RobotKinematics
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
from lerobot.utils.rotation import (
    rotation_6d_to_rotation_matrix_numpy,
    rotation_matrix_to_rotation_6d_numpy,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware configuration -- adjust these to match your OpenArm setup
# ---------------------------------------------------------------------------

# End-effector frame name in the URDF (the FK target frame).
# TODO: Set this to the actual frame name once the URDF is finalized.
EE_FRAME_NAME = "gripper_frame_link"

# Arm joint names used for FK/IK (7 DOF, no gripper).
ARM_JOINT_NAMES = [
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "joint_7",
]

# Gripper joint names (passed through directly, outside of IK).
# TODO: Adjust to match the 2-DOF gripper naming in the dataset and robot.
GRIPPER_JOINT_NAMES = ["gripper"]

# Camera config for the gripper-mounted camera.
GRIPPER_CAMERA_CONFIG = OpenCVCameraConfig(
    index_or_path=0,
    width=960,
    height=720,
    fps=50,
)

# Optional rotation from robot FK frame to SLAM recording frame.
# Both are Z-up, gravity-aligned. If the horizontal axes differ (e.g. X/Y swapped
# or rotated), set this to the appropriate 3x3 rotation matrix.
# Identity means no correction needed.
R_FK_TO_SLAM = np.eye(3)


# ---------------------------------------------------------------------------
# Kinematics helpers (11D with 6D rotation)
# ---------------------------------------------------------------------------


def fk_to_state(
    kin: RobotKinematics,
    joint_angles_deg: np.ndarray,
    gripper_joints: np.ndarray,
) -> np.ndarray:
    """Compute the 11D observation state from joint angles via forward kinematics.

    Converts the FK rotation matrix to 6D continuous representation.

    Args:
        kin: RobotKinematics instance (placo + URDF).
        joint_angles_deg: Arm joint positions in degrees (7D).
        gripper_joints: Gripper joint positions in degrees (2D).

    Returns:
        11D state vector: [x, y, z, r6d_0..r6d_5, proximal, distal]
        Position in meters, orientation as 6D rotation, gripper in degrees.
    """
    tf_matrix = kin.forward_kinematics(joint_angles_deg)

    # Apply optional frame rotation to bring FK output into the SLAM frame.
    pos = R_FK_TO_SLAM @ tf_matrix[:3, 3]
    rot_matrix = R_FK_TO_SLAM @ tf_matrix[:3, :3]

    # Convert 3x3 rotation matrix to 6D representation (first two columns)
    rot_6d = rotation_matrix_to_rotation_6d_numpy(rot_matrix.reshape(1, 3, 3)).squeeze(0)

    return np.concatenate([pos, rot_6d, gripper_joints])


def state_to_ik_target(cart_state: np.ndarray) -> np.ndarray:
    """Build a 4x4 SE(3) matrix from the first 9 dims of the 11D state.

    Converts the 6D rotation back to a rotation matrix and applies the inverse
    of the FK-to-SLAM rotation so the IK solver works in the robot's native frame.

    Args:
        cart_state: [x, y, z, r6d_0..r6d_5] (9D) in the SLAM frame.

    Returns:
        4x4 homogeneous transformation matrix in the FK frame.
    """
    pos_slam = cart_state[:3]
    rot_6d = cart_state[3:9]

    # Recover 3x3 rotation matrix from 6D via Gram-Schmidt
    rot_matrix_slam = rotation_6d_to_rotation_matrix_numpy(rot_6d.reshape(1, 6)).squeeze(0)

    # Rotate from SLAM frame back to robot FK frame
    pos_fk = R_FK_TO_SLAM.T @ pos_slam
    rot_matrix_fk = R_FK_TO_SLAM.T @ rot_matrix_slam

    tf_target = np.eye(4)
    tf_target[:3, :3] = rot_matrix_fk
    tf_target[:3, 3] = pos_fk
    return tf_target


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Deploy Gripette policy on OpenArm")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint directory")
    p.add_argument("--urdf", type=str, required=True, help="Path to OpenArm URDF file")
    p.add_argument("--can_port", type=str, default="can0", help="CAN interface for the arm")
    p.add_argument("--side", type=str, default="right", help="Arm side: left or right")
    p.add_argument("--device", type=str, default="cuda", help="Compute device (cuda / cpu)")
    p.add_argument("--duration", type=float, default=30.0, help="Run duration in seconds")
    p.add_argument("--fps", type=float, default=50.0, help="Control loop frequency in Hz")
    p.add_argument(
        "--orientation_weight",
        type=float,
        default=0.01,
        help="IK orientation weight (lower = prioritize position over orientation)",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device)

    # ---- Load trained policy and processors ----
    logger.info(f"Loading policy from {args.checkpoint}")
    policy = DiffusionPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()

    # The processors contain the normalization stats and relative action config,
    # loaded from the checkpoint directory. No dataset stats needed here.
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=args.checkpoint)

    logger.info(
        f"Policy: action_dim={policy.config.action_feature.shape[0]}, "
        f"n_action_steps={policy.config.n_action_steps}, "
        f"relative_actions={policy.config.use_relative_actions}"
    )

    # ---- FK/IK setup ----
    logger.info(f"Loading kinematics from {args.urdf}")
    kin = RobotKinematics(
        urdf_path=args.urdf,
        target_frame_name=EE_FRAME_NAME,
        joint_names=ARM_JOINT_NAMES,
    )

    # ---- Robot setup ----
    robot_config = OpenArmFollowerConfig(
        port=args.can_port,
        side=args.side,
        cameras={"gripper": GRIPPER_CAMERA_CONFIG},
    )
    robot = OpenArmFollower(robot_config)
    robot.connect()

    logger.info(f"Robot connected. Running for {args.duration}s at {args.fps} Hz")

    # ---- Control loop ----
    dt = 1.0 / args.fps
    start_time = time.time()
    step_count = 0

    try:
        while (time.time() - start_time) < args.duration:
            loop_start = time.perf_counter()

            # --- 1. Read robot sensors ---
            obs = robot.get_observation()

            arm_joints = np.array([obs[f"{j}.pos"] for j in ARM_JOINT_NAMES], dtype=np.float64)
            gripper_joints = np.array([obs[f"{j}.pos"] for j in GRIPPER_JOINT_NAMES], dtype=np.float64)
            camera_image = obs["gripper"]  # (H, W, C) uint8 numpy array

            # --- 2. FK: joint angles -> 11D Cartesian state (with 6D rotation) ---
            state = fk_to_state(kin, arm_joints, gripper_joints)

            # --- 3. Build policy input tensors ---
            state_tensor = torch.from_numpy(state).float()
            image_tensor = torch.from_numpy(camera_image).float() / 255.0
            image_tensor = image_tensor.permute(2, 0, 1).contiguous()  # (H,W,C) -> (C,H,W)

            batch = {
                "observation.state": state_tensor.unsqueeze(0).to(device),
                "observation.images.cam0": image_tensor.unsqueeze(0).to(device),
            }

            # --- 4. Preprocess -> Policy -> Postprocess ---
            # Preprocessor: normalizes, converts to relative actions.
            # select_action: generates/pops from action chunk queue.
            # Postprocessor: unnormalizes, converts deltas back to absolute.
            batch = preprocessor(batch)
            action = policy.select_action(batch)
            action = postprocessor(action)

            # --- 5. Extract Cartesian + gripper targets (11D) ---
            action_np = action.squeeze(0).cpu().numpy()
            cart_target = action_np[:9]  # [x, y, z, r6d_0..r6d_5] absolute
            gripper_target = action_np[9:]  # [proximal, distal] absolute

            # --- 6. IK: Cartesian target (with 6D rotation) -> joint angles ---
            ee_target = state_to_ik_target(cart_target)
            joint_targets_deg = kin.inverse_kinematics(
                current_joint_pos=arm_joints,
                desired_ee_pose=ee_target,
                position_weight=1.0,
                orientation_weight=args.orientation_weight,
            )

            # --- 7. Send motor commands ---
            action_dict = {}
            for i, name in enumerate(ARM_JOINT_NAMES):
                action_dict[f"{name}.pos"] = float(joint_targets_deg[i])
            for i, name in enumerate(GRIPPER_JOINT_NAMES):
                action_dict[f"{name}.pos"] = float(gripper_target[i])

            robot.send_action(action_dict)
            step_count += 1

            # --- Timing ---
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Periodic status log
            if step_count % 50 == 0:
                actual_fps = 1.0 / max(time.perf_counter() - loop_start, 1e-6)
                logger.info(
                    f"Step {step_count:>5d} | "
                    f"FPS: {actual_fps:5.1f} | "
                    f"EE: [{cart_target[0]:+.3f}, {cart_target[1]:+.3f}, {cart_target[2]:+.3f}] m"
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        robot.disconnect()
        logger.info(f"Done. Executed {step_count} steps in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
