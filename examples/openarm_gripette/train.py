"""Train a Diffusion Policy with relative actions for the Gripette project.

This script trains a DiffusionPolicy on a dataset recorded with a hand-mounted SLAM
device. The dataset contains absolute Cartesian poses + gripper joints + camera images.
During training, Cartesian dims are automatically converted to deltas by the processor
pipeline while gripper joints stay absolute.

See README.md in this directory for the full setup guide.

Prerequisites:
  - Dataset exists locally or on HuggingFace Hub.
  - Dataset stats recomputed for relative actions:
      uv run lerobot-edit-dataset \\
          --repo-id <DATASET_REPO_ID> \\
          --operation.type recompute_stats \\
          --operation.relative_action true \\
          --operation.relative_exclude_joints "['grip_1', 'grip_2']" \\
          --operation.chunk_size 16

Usage:
  uv run python examples/openarm_gripette/train.py
  uv run python examples/openarm_gripette/train.py --dataset_repo_id pollen/gripette_demo --batch_size 64
"""

import argparse
from pathlib import Path

import torch

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    """Convert frame-offset indices to seconds for the dataset's delta_timestamps."""
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion Policy for Gripette")
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        default="pollen/gripette_demo",
        help="LeRobot dataset repo ID",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/gripette/diffusion",
        help="Directory for checkpoints",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--training_steps", type=int, default=200_000, help="Total training steps")
    parser.add_argument("--log_freq", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save_freq", type=int, default=10_000, help="Save checkpoint every N steps")
    parser.add_argument(
        "--gripper_joints",
        type=str,
        nargs="+",
        default=["grip_1", "grip_2"],
        help="Gripper joint names to exclude from relative action conversion",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ---- Dataset metadata ----
    # Load metadata without downloading the full dataset. This gives us the feature
    # definitions and statistics needed to configure the policy.
    dataset_metadata = LeRobotDatasetMetadata(args.dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    # The action feature names are needed by RelativeActionsProcessorStep to build
    # the exclude_joints mask (it matches names like "grip_1" against this list).
    action_feature_names = dataset_metadata.features.get("action", {}).get("names")

    print(f"Dataset:          {args.dataset_repo_id}")
    print(f"FPS:              {dataset_metadata.fps}")
    print(f"Input features:   {list(input_features.keys())}")
    print(f"Output features:  {list(output_features.keys())}")
    print(f"Action names:     {action_feature_names}")
    print(f"Gripper excluded: {args.gripper_joints}")

    # ---- Policy configuration ----
    # The config defines both the model architecture and the processor pipeline settings.
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        # -- Temporal structure --
        # n_obs_steps=2: condition on the current and previous observation.
        # horizon=16: predict 16 future action steps.
        # n_action_steps=8: execute 8 before re-planning (receding horizon).
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        # -- Vision encoder --
        # ResNet18 with GroupNorm (required when not using pretrained weights).
        # SpatialSoftmax extracts 32 keypoints from the feature maps.
        vision_backbone="resnet18",
        resize_shape=(240, 320),
        crop_ratio=0.9,
        crop_is_random=True,
        pretrained_backbone_weights=None,
        use_group_norm=True,
        spatial_softmax_num_keypoints=32,
        # -- U-Net --
        # 3-stage 1D convolutional U-Net with FiLM conditioning.
        # (256, 512, 1024) is sufficient for 8D actions. The default (512, 1024, 2048)
        # is designed for higher-dimensional action spaces (e.g., bimanual).
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        diffusion_step_embed_dim=128,
        use_film_scale_modulation=True,
        # -- Diffusion scheduler --
        # DDPM with cosine beta schedule and epsilon prediction.
        # 100 denoising steps at training; same at inference unless overridden.
        noise_scheduler_type="DDPM",
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
        clip_sample_range=1.0,
        # -- Normalization --
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        },
        # -- Relative actions --
        # Converts Cartesian + orientation dims to deltas (action -= state).
        # Gripper joints are excluded and stay as absolute targets.
        use_relative_actions=True,
        relative_exclude_joints=args.gripper_joints,
        action_feature_names=list(action_feature_names) if action_feature_names else None,
        # -- Optimizer --
        optimizer_lr=1e-4,
        optimizer_betas=(0.95, 0.999),
        optimizer_weight_decay=1e-6,
        scheduler_name="cosine",
        scheduler_warmup_steps=500,
    )

    # ---- Instantiate policy ----
    policy = DiffusionPolicy(cfg)
    policy.train()
    policy.to(device)

    param_count = sum(p.numel() for p in policy.parameters())
    print(f"\nModel parameters: {param_count:,}")

    # ---- Pre/post processors ----
    # The preprocessor converts raw data to model input:
    #   rename -> add batch dim -> move to device -> relative actions -> normalize
    # The postprocessor reverses the action transforms:
    #   unnormalize -> absolute actions -> move to CPU
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    # ---- Dataset with temporal windowing ----
    # delta_timestamps tells the dataset to stack multiple frames per sample.
    # For observations: current frame + 1 previous frame (n_obs_steps=2).
    # For actions: a window of `horizon` frames starting from the previous observation.
    delta_timestamps = {
        "observation.state": make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps),
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    # Image features use the same observation timestamps as the state.
    delta_timestamps |= {
        k: make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        for k in cfg.image_features
    }

    dataset = LeRobotDataset(args.dataset_repo_id, delta_timestamps=delta_timestamps)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
        num_workers=4,
    )

    # ---- Optimizer ----
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())

    # ---- Training loop ----
    print(f"\nStarting training for {args.training_steps} steps on {device}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Dataset frames:   {len(dataset)}")
    print(f"  Relative actions: enabled (excluding {args.gripper_joints})")
    print(f"  Checkpoints:      {output_dir}\n")

    step = 0
    done = False
    while not done:
        for batch in dataloader:
            # Forward pass: preprocessor normalizes and converts to deltas,
            # then the diffusion model computes the denoising loss.
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % args.log_freq == 0:
                print(f"step: {step:>7d} / {args.training_steps}  loss: {loss.item():.4f}")

            # Periodic checkpoint
            if step > 0 and step % args.save_freq == 0:
                ckpt_dir = output_dir / f"checkpoint_{step:06d}"
                policy.save_pretrained(ckpt_dir)
                preprocessor.save_pretrained(ckpt_dir)
                postprocessor.save_pretrained(ckpt_dir)
                print(f"  -> saved checkpoint to {ckpt_dir}")

            step += 1
            if step >= args.training_steps:
                done = True
                break

    # ---- Save final checkpoint ----
    # This saves the model weights, config, and processor pipelines.
    # The processors include the RelativeActionsProcessorStep config and normalization
    # stats, so they are self-contained for deployment.
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    print(f"\nTraining complete. Model saved to {output_dir}")


if __name__ == "__main__":
    main()
