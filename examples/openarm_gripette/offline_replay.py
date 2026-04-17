"""Offline replay: feed recorded observations through the policy and compare predictions vs ground truth.

This script loads the training dataset, runs the trained policy on recorded observations,
and compares predicted actions against the actual recorded actions. This diagnoses whether
the model learned the correct mapping without the simulator in the loop.

Outputs per episode:
  - Per-step comparison: predicted vs ground truth delta (position + rotation + gripper)
  - Mean absolute error (MAE) per action dimension
  - Trajectory plot: predicted vs ground truth position deltas over time

Usage:
  uv run python examples/openarm_gripette/offline_replay.py \\
      --checkpoint outputs/gripette/relative_proprio \\
      --dataset_repo_id SteveNguyen/simu_reach_redcube500 \\
      --num_episodes 5
"""

import argparse
import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features

matplotlib.use("Agg")  # non-interactive backend for saving plots

logger = logging.getLogger(__name__)


def make_delta_timestamps(delta_indices, fps):
    return [i / fps for i in delta_indices] if delta_indices else [0]


def parse_args():
    p = argparse.ArgumentParser(description="Offline replay evaluation")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    p.add_argument("--dataset_repo_id", type=str, required=True, help="Dataset repo ID")
    p.add_argument("--device", type=str, default="cuda", help="Compute device")
    p.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to replay")
    p.add_argument("--output_dir", type=str, default="outputs/gripette/replay", help="Where to save plots")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device(args.device)

    from pathlib import Path

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load policy ----
    logger.info(f"Loading policy from {args.checkpoint}")
    policy = DiffusionPolicy.from_pretrained(args.checkpoint)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=args.checkpoint)

    state_dim = policy.config.robot_state_feature.shape[0]
    action_dim = policy.config.action_feature.shape[0]
    logger.info(f"Policy: state_dim={state_dim}, action_dim={action_dim}")

    # ---- Load dataset (without temporal stacking — single frames for inference) ----
    meta = LeRobotDatasetMetadata(args.dataset_repo_id)
    dataset = LeRobotDataset(args.dataset_repo_id)

    # Determine which camera to use
    features = dataset_to_policy_features(meta.features)
    camera_keys = [k for k, ft in features.items() if ft.type is FeatureType.VISUAL]
    camera_key = camera_keys[0] if camera_keys else None
    logger.info(f"Camera: {camera_key}")

    action_names = meta.features["action"]["names"]
    logger.info(f"Action names: {action_names}")
    logger.info(f"FPS: {meta.fps}")

    # ---- Get episode list ----
    all_episodes = sorted(dataset.meta.episodes["episode_index"])
    episodes_to_replay = all_episodes[: args.num_episodes]
    logger.info(f"Replaying {len(episodes_to_replay)} episodes: {episodes_to_replay}")

    # ---- Replay each episode ----
    all_maes = []

    for ep_idx in episodes_to_replay:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Episode {ep_idx}")
        logger.info(f"{'=' * 60}")

        # Load single-frame dataset for this episode
        ep_dataset = LeRobotDataset(args.dataset_repo_id, episodes=[ep_idx])
        num_frames = len(ep_dataset)
        logger.info(f"  Frames: {num_frames}")

        # Collect predictions and ground truth
        gt_actions = []
        pred_actions = []

        policy.reset()

        for frame_idx in range(num_frames):
            sample = ep_dataset[frame_idx]

            # Build single-frame observation (same as eval_simulator does)
            obs = {
                "observation.state": sample["observation.state"].unsqueeze(0).to(device),
            }
            if camera_key:
                obs[camera_key] = sample[camera_key].unsqueeze(0).to(device)

            # Run policy
            obs = preprocessor(obs)
            with torch.no_grad():
                action = policy.select_action(obs)
            action = postprocessor(action)

            pred = action.squeeze(0).cpu().numpy()
            gt = sample["action"].numpy()

            pred_actions.append(pred)
            gt_actions.append(gt)

        pred_actions = np.array(pred_actions)  # (T, action_dim)
        gt_actions = np.array(gt_actions)  # (T, action_dim)

        # ---- Compute MAE per dimension ----
        mae = np.abs(pred_actions - gt_actions).mean(axis=0)
        all_maes.append(mae)

        logger.info("\n  MAE per action dimension:")
        for i, name in enumerate(action_names):
            logger.info(f"    {name:10s}: {mae[i]:.6f}")

        # Position MAE in mm
        pos_mae_mm = mae[:3].mean() * 1000
        logger.info(f"\n  Position MAE: {pos_mae_mm:.2f} mm")

        # ---- Plot: predicted vs ground truth for position deltas ----
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        time_s = np.arange(num_frames) / meta.fps

        for i, (ax, name) in enumerate(zip(axes, ["dx", "dy", "dz"], strict=True)):
            ax.plot(time_s, gt_actions[:, i] * 1000, label="Ground truth", alpha=0.8, linewidth=1)
            ax.plot(time_s, pred_actions[:, i] * 1000, label="Predicted", alpha=0.8, linewidth=1)
            ax.set_ylabel(f"{name} (mm)")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"Episode {ep_idx} — Position deltas (pred vs GT)")
        fig.tight_layout()
        plot_path = output_dir / f"episode_{ep_idx:03d}_position.png"
        fig.savefig(plot_path, dpi=100)
        plt.close(fig)
        logger.info(f"  Saved plot: {plot_path}")

        # ---- Plot: gripper predicted vs ground truth ----
        if action_dim > 9:
            fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
            grip_names = action_names[9:]
            for i, (ax, name) in enumerate(zip(axes, grip_names, strict=True)):
                ax.plot(time_s, gt_actions[:, 9 + i], label="Ground truth", alpha=0.8)
                ax.plot(time_s, pred_actions[:, 9 + i], label="Predicted", alpha=0.8)
                ax.set_ylabel(name)
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel("Time (s)")
            fig.suptitle(f"Episode {ep_idx} — Gripper (pred vs GT)")
            fig.tight_layout()
            plot_path = output_dir / f"episode_{ep_idx:03d}_gripper.png"
            fig.savefig(plot_path, dpi=100)
            plt.close(fig)

        # ---- Plot: rotation deltas ----
        fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex=True)
        rot_names = action_names[3:9]
        for i, name in enumerate(rot_names):
            ax = axes[i % 3, i // 3]
            ax.plot(time_s, gt_actions[:, 3 + i], label="GT", alpha=0.8, linewidth=1)
            ax.plot(time_s, pred_actions[:, 3 + i], label="Pred", alpha=0.8, linewidth=1)
            ax.set_ylabel(name, fontsize=8)
            ax.legend(loc="upper right", fontsize=7)
            ax.grid(True, alpha=0.3)

        axes[-1, 0].set_xlabel("Time (s)")
        axes[-1, 1].set_xlabel("Time (s)")
        fig.suptitle(f"Episode {ep_idx} — Rotation deltas (pred vs GT)")
        fig.tight_layout()
        plot_path = output_dir / f"episode_{ep_idx:03d}_rotation.png"
        fig.savefig(plot_path, dpi=100)
        plt.close(fig)

    # ---- Overall summary ----
    all_maes = np.array(all_maes)
    mean_mae = all_maes.mean(axis=0)

    print(f"\n{'=' * 60}")
    print("  OFFLINE REPLAY SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Episodes replayed: {len(episodes_to_replay)}")
    print(f"  State dim: {state_dim}, Action dim: {action_dim}")
    print("\n  Mean MAE per action dimension (across all episodes):")
    for i, name in enumerate(action_names):
        print(f"    {name:10s}: {mean_mae[i]:.6f}")

    pos_mae_mm = mean_mae[:3].mean() * 1000
    rot_mae = mean_mae[3:9].mean()
    grip_mae = mean_mae[9:].mean() if action_dim > 9 else 0
    print(f"\n  Position MAE:  {pos_mae_mm:.2f} mm")
    print(f"  Rotation MAE:  {rot_mae:.6f}")
    print(f"  Gripper MAE:   {grip_mae:.6f}")
    print(f"\n  Plots saved to: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
