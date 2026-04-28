"""Compare policy output on a DATASET frame vs a LIVE simulator frame.

Why this script exists
----------------------
We've already established that:
  * Training converged (train_loss 0.012, val_loss 0.014).
  * OOD loss on the held-out cube_y strip is 1.16x val_loss — the policy
    predicts ground-truth actions accurately on dataset frames.
  * At deployment the policy outputs consistent template-like motion
    independent of the cube position.

Together these say: the model works on DATASET inputs but breaks on LIVE
inputs. So either the *image* or the *state* the deployed policy receives
differs from what training saw, or the deployment-time queue/chunk
handling is doing something the dataset path doesn't.

This script feeds the policy:
  (a) one frame straight from the dataset (the same path eval_ood_loss uses),
  (b) one frame captured live from the simulator (the same path
      eval_simulator uses),
and prints both outputs side by side. If they match closely, the policy
is consistent and the bug is in chunk/queue handling. If they differ, we
have a concrete pixel-level mismatch to chase.

Both frames are saved to PNG for visual comparison (`debug_dataset.png`,
`debug_live.png`).

Usage:
  # 1. Start the sim server in another terminal:
  #    cd .../openarm_gripette_simu
  #    uv run python -m openarm_gripette_simu --scene scenes/table_grasp.xml \\
  #        --initial-joints 1.0 0.0 0.0 0.5 0.0 0.0 1.5
  #
  # 2. Run this script:
  uv run python examples/openarm_gripette/debug_policy_input.py \\
      --checkpoint outputs/gripette/no_proprio3/best/ \\
      --dataset_repo_id SteveNguyen/sim_grasp_eval_v1
"""

from __future__ import annotations

import argparse
import time

import cv2
import grpc
import numpy as np
import torch

from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionPolicy


def parse_args():
    p = argparse.ArgumentParser(description="Compare policy output: dataset frame vs live sim frame")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--dataset_repo_id", type=str, required=True)
    p.add_argument("--dataset_frame", type=int, default=0,
                   help="Frame index from the dataset to use as reference.")
    p.add_argument("--arm_addr", type=str, default="localhost:50052")
    p.add_argument("--gripper_addr", type=str, default="localhost:50051")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def fetch_live_frame(gripper_stub, gripper_pb2):
    """Fetch one frame from the running simulator's StreamState."""
    for frame in gripper_stub.StreamState(gripper_pb2.StreamRequest()):
        bgr = cv2.imdecode(np.frombuffer(frame.jpeg_data, np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gripper = np.array(
            [frame.motor_state.motor1_position, frame.motor_state.motor2_position],
            dtype=np.float32,
        )
        return rgb, gripper
    raise RuntimeError("No frame from camera stream")


def predict_from_inputs(policy, preprocessor, postprocessor, image_chw_float, state_2d, device,
                        n_obs_steps: int):
    """Build a batch the same way eval_simulator does, run select_action, return action np.

    Calls the policy `n_obs_steps` times so the internal observation queue
    is filled with this same observation; this matches what the deployed
    eval_simulator does over its first n_obs_steps loop iterations.
    """
    policy.reset()
    policy.eval()
    state_t = torch.from_numpy(state_2d).float().unsqueeze(0).to(device)
    img_t = image_chw_float.unsqueeze(0).to(device)
    batch = {
        "observation.state": state_t,
        "observation.images.cam0": img_t,
    }
    batch = preprocessor(batch)
    actions = []
    with torch.no_grad():
        for _ in range(n_obs_steps + 1):
            action = policy.select_action(batch)
            actions.append(postprocessor(action).squeeze(0).cpu().numpy())
    return actions  # one per call; we'll print all of them


def to_chw_float(rgb_hwc_u8):
    """RGB HWC uint8 -> RGB CHW float32 [0,1] (matches eval_simulator)."""
    t = torch.from_numpy(rgb_hwc_u8).float() / 255.0
    return t.permute(2, 0, 1).contiguous()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # --- Load policy ---
    print(f"Loading policy from {args.checkpoint}")
    policy = DiffusionPolicy.from_pretrained(args.checkpoint).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=args.checkpoint,
    )
    n_obs_steps = policy.config.n_obs_steps
    print(f"  state_dim     : {policy.config.robot_state_feature.shape[0]}")
    print(f"  action_dim    : {policy.config.action_feature.shape[0]}")
    print(f"  n_obs_steps   : {n_obs_steps}")
    print(f"  n_action_steps: {policy.config.n_action_steps}")

    # --- (a) Dataset frame ---
    print(f"\n=== DATASET path: {args.dataset_repo_id} frame {args.dataset_frame} ===")
    ds = LeRobotDataset(args.dataset_repo_id)  # no delta_timestamps -> single-frame samples
    sample = ds[args.dataset_frame]
    ds_state = sample["observation.state"].numpy().astype(np.float32)  # (2,)
    ds_image = sample["observation.images.cam0"]                       # (3,H,W) float [0,1]
    if ds_image.dtype != torch.float32:
        ds_image = ds_image.float() / 255.0
    ds_image_hwc_u8 = (ds_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    print(f"  state (open ~ 0)  : proximal={ds_state[0]:+.4f}, distal={ds_state[1]:+.4f}")
    print(f"  image shape       : {tuple(ds_image.shape)} (CHW float)")
    print(f"  image dtype       : {ds_image.dtype}, min={ds_image.min():.3f}, max={ds_image.max():.3f}")
    cv2.imwrite("debug_dataset.png", cv2.cvtColor(ds_image_hwc_u8, cv2.COLOR_RGB2BGR))
    print(f"  saved -> debug_dataset.png")

    actions_ds = predict_from_inputs(
        policy, preprocessor, postprocessor, ds_image, ds_state, device, n_obs_steps,
    )
    print(f"  policy output (last call, after {n_obs_steps + 1} warm-up calls):")
    a = actions_ds[-1]
    print(f"    delta_pos (mm) = {a[0]*1000:+8.3f}, {a[1]*1000:+8.3f}, {a[2]*1000:+8.3f}")
    print(f"    delta_r6d      = " + ", ".join(f"{x:+.4f}" for x in a[3:9]))
    print(f"    gripper        = proximal={a[9]:+.4f}, distal={a[10]:+.4f}")

    # --- (b) Live sim frame ---
    print(f"\n=== LIVE path: {args.arm_addr} / {args.gripper_addr} ===")
    from openarm_gripette_simu.proto import gripper_pb2, gripper_pb2_grpc
    gripper_channel = grpc.insecure_channel(args.gripper_addr)
    gripper_stub = gripper_pb2_grpc.GripperServiceStub(gripper_channel)
    rgb, gripper_state = fetch_live_frame(gripper_stub, gripper_pb2)

    print(f"  state             : proximal={gripper_state[0]:+.4f}, distal={gripper_state[1]:+.4f}")
    print(f"  image shape       : {rgb.shape} (HWC uint8 RGB)")
    print(f"  image dtype       : {rgb.dtype}, min={rgb.min()}, max={rgb.max()}")
    cv2.imwrite("debug_live.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"  saved -> debug_live.png")

    live_image = to_chw_float(rgb)
    actions_live = predict_from_inputs(
        policy, preprocessor, postprocessor, live_image, gripper_state, device, n_obs_steps,
    )
    print(f"  policy output (last call, after {n_obs_steps + 1} warm-up calls):")
    a = actions_live[-1]
    print(f"    delta_pos (mm) = {a[0]*1000:+8.3f}, {a[1]*1000:+8.3f}, {a[2]*1000:+8.3f}")
    print(f"    delta_r6d      = " + ", ".join(f"{x:+.4f}" for x in a[3:9]))
    print(f"    gripper        = proximal={a[9]:+.4f}, distal={a[10]:+.4f}")

    # --- Image stats comparison ---
    print(f"\n=== Image stats (raw uint8 RGB, full frame) ===")
    print(f"  dataset: shape={ds_image_hwc_u8.shape}, mean={ds_image_hwc_u8.mean():.1f}, std={ds_image_hwc_u8.std():.1f}")
    print(f"  live   : shape={rgb.shape}, mean={rgb.mean():.1f}, std={rgb.std():.1f}")
    if ds_image_hwc_u8.shape != rgb.shape:
        print(f"  ⚠️  shape mismatch — image preprocessing diverges between train and eval")
    else:
        # Pixel-wise error if shapes match.
        diff = np.abs(ds_image_hwc_u8.astype(np.int16) - rgb.astype(np.int16))
        print(f"  pixel diff: mean={diff.mean():.2f}, p95={np.percentile(diff, 95):.2f}, max={diff.max()}")
        print("  NB: comparing different scenes (the dataset frame's cube position is not")
        print("  the live cube position) — pixel diff is informational only.")

    # --- Action delta diff ---
    print(f"\n=== Action diff (live - dataset, last warm-up call) ===")
    diff = actions_live[-1] - actions_ds[-1]
    pos_diff_mm = float(np.linalg.norm(diff[:3])) * 1000
    print(f"  |Δ delta_pos|  : {pos_diff_mm:.2f} mm")
    print(f"  |Δ gripper|    : prox={abs(diff[9]):.4f}, dist={abs(diff[10]):.4f}")
    if pos_diff_mm < 0.5 and abs(diff[9]) < 0.05:
        print("  Verdict: live and dataset outputs are CLOSE. Policy is responding consistently;")
        print("  the deployment failure is likely in chunk/queue handling or in the arm-server side.")
    else:
        print("  Verdict: live and dataset outputs DIVERGE. Inspect debug_live.png vs debug_dataset.png")
        print("  (likely lighting, fisheye distortion, JPEG-vs-AV1 compression, or shape mismatch).")


if __name__ == "__main__":
    main()
