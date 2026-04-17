"""Train a Diffusion Policy for the Gripette project.

This script trains a DiffusionPolicy on a dataset prepared by convert_dataset.py:
  - observation.state = [proximal, distal]  (2D gripper joints — no absolute position)
  - action = [dx, dy, dz, dr6d_0..5, proximal, distal]  (11D: deltas + gripper)
  - observation.images.cam0 = camera image

The model sees camera + gripper state as input, and predicts delta actions.
No absolute position is fed to the model (it's meaningless in the SLAM reference frame).
Delta actions are pre-computed in the dataset (following the UMI approach).

See README.md in this directory for the full setup guide.

Prerequisites:
  - Dataset converted with convert_dataset.py:
      uv run python examples/openarm_gripette/convert_dataset.py

Usage:
  uv run python examples/openarm_gripette/train.py
  uv run python examples/openarm_gripette/train.py \\
      --dataset_repo_id SteveNguyen/Grabette_redcube_quest --batch_size 64
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


@torch.no_grad()
def compute_val_loss(policy, preprocessor, val_dataloader, device, max_batches=50):
    """Compute average loss on the validation set."""
    policy.eval()
    total_loss = 0.0
    num_batches = 0
    for batch in val_dataloader:
        batch = preprocessor(batch)
        loss, _ = policy.forward(batch)
        total_loss += loss.item()
        num_batches += 1
        if num_batches >= max_batches:
            break
    policy.train()
    return total_loss / max(num_batches, 1)


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
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=8,
        help="Actions executed before re-planning (4=reactive, 8=default, 16=smooth)",
    )
    parser.add_argument("--log_freq", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save_freq", type=int, default=10_000, help="Save checkpoint every N steps")
    parser.add_argument("--eval_freq", type=int, default=200, help="Evaluate on validation set every N steps")
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Fraction of episodes used for validation"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None, help="Wandb project name (None = disabled)"
    )
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=["observation.images.cam0"],
        help="Camera feature keys to use as input (others are excluded)",
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
    # Filter input features: keep only selected cameras + non-visual features.
    # This excludes cameras not listed in --cameras (e.g., cam1 when only cam0 is used).
    input_features = {
        key: ft
        for key, ft in features.items()
        if key not in output_features and (ft.type is not FeatureType.VISUAL or key in args.cameras)
    }

    # The action feature names are needed by RelativeActionsProcessorStep to build
    # the exclude_joints mask (it matches names like "grip_1" against this list).
    action_feature_names = dataset_metadata.features.get("action", {}).get("names")

    print(f"Dataset:          {args.dataset_repo_id}")
    print(f"FPS:              {dataset_metadata.fps}")
    print(f"Input features:   {list(input_features.keys())}")
    print(f"Output features:  {list(output_features.keys())}")
    print(f"Action names:     {action_feature_names}")

    # ---- Policy configuration ----
    # Parameters are aligned with the UMI (Universal Manipulation Interface) project,
    # which is a known-working diffusion policy for SLAM-recorded Cartesian datasets.
    # See docs/umi_analysis.md for the full comparison.
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        # -- Temporal structure (same as UMI) --
        n_obs_steps=2,
        horizon=16,
        n_action_steps=args.n_action_steps,
        # -- Vision encoder --
        # ResNet18 with GroupNorm + SpatialSoftmax (32 keypoints).
        # UMI uses ViT-base (CLIP pretrained), but ResNet18 is lighter and faster.
        # Images resized to 224x224 (standard for pretrained vision models).
        # No cropping — resize only to preserve full field of view.
        vision_backbone="resnet18",
        resize_shape=(224, 224),
        crop_ratio=1.0,  # 1.0 = no crop, resize only
        pretrained_backbone_weights=None,
        use_group_norm=True,
        spatial_softmax_num_keypoints=32,
        # -- U-Net (same as UMI) --
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        diffusion_step_embed_dim=128,
        use_film_scale_modulation=True,
        # -- Diffusion scheduler --
        # DDIM with 50 training steps and 16 inference steps (matching UMI).
        # DDIM is ~6x faster than DDPM at inference with minimal quality loss.
        noise_scheduler_type="DDIM",
        num_train_timesteps=50,
        num_inference_steps=16,
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
        # -- No RelativeActionsProcessorStep needed --
        # Delta actions are pre-computed in the dataset by convert_dataset.py.
        # observation.state = gripper only (2D), no absolute position.
        use_relative_actions=False,
        # -- Optimizer (matching UMI: lr=3e-4, warmup=2000) --
        optimizer_lr=3e-4,
        optimizer_betas=(0.95, 0.999),
        optimizer_weight_decay=1e-6,
        scheduler_name="cosine",
        scheduler_warmup_steps=2000,
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

    # ---- Train/val split by episodes ----
    all_episodes = sorted(dataset_metadata.episodes["episode_index"])
    num_val = max(1, int(len(all_episodes) * args.val_ratio))
    # Use last episodes as validation (deterministic split, no randomness)
    val_episodes = all_episodes[-num_val:]
    train_episodes = all_episodes[:-num_val]

    train_dataset = LeRobotDataset(
        args.dataset_repo_id, delta_timestamps=delta_timestamps, episodes=train_episodes
    )
    val_dataset = LeRobotDataset(
        args.dataset_repo_id, delta_timestamps=delta_timestamps, episodes=val_episodes
    )

    print(f"  Train episodes:   {len(train_episodes)} ({len(train_dataset)} frames)")
    print(f"  Val episodes:     {len(val_episodes)} ({len(val_dataset)} frames)")

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
        num_workers=4,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type != "cpu",
        drop_last=False,
        num_workers=2,
    )

    # ---- Optimizer ----
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())

    # ---- Wandb ----
    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "dataset": args.dataset_repo_id,
                "batch_size": args.batch_size,
                "training_steps": args.training_steps,
                "lr": cfg.optimizer_lr,
                "warmup_steps": cfg.scheduler_warmup_steps,
                "n_obs_steps": cfg.n_obs_steps,
                "horizon": cfg.horizon,
                "n_action_steps": cfg.n_action_steps,
                "noise_scheduler": cfg.noise_scheduler_type,
                "num_train_timesteps": cfg.num_train_timesteps,
                "num_inference_steps": cfg.num_inference_steps,
                "vision_backbone": cfg.vision_backbone,
                "resize_shape": cfg.resize_shape,
                "down_dims": cfg.down_dims,
                "use_relative_actions": cfg.use_relative_actions,
                "relative_exclude_joints": cfg.relative_exclude_joints,
                "action_dim": cfg.action_feature.shape[0],
                "state_dim": cfg.robot_state_feature.shape[0],
                "cameras": args.cameras,
                "model_params": param_count,
            },
        )

    # ---- Training loop ----
    print(f"\nStarting training for {args.training_steps} steps on {device}")
    print(f"  Batch size:       {args.batch_size}")
    print("  Actions:          pre-computed deltas (11D)")
    print(f"  Eval every:       {args.eval_freq} steps")
    print(f"  Checkpoints:      {output_dir}")
    if use_wandb:
        print(f"  Wandb:            {args.wandb_project}")
    print()

    best_val_loss = float("inf")
    step = 0
    done = False
    while not done:
        for batch in train_dataloader:
            # Forward pass
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss = loss.item()

            # ---- Validation ----
            val_loss = None
            if step > 0 and step % args.eval_freq == 0:
                val_loss = compute_val_loss(policy, preprocessor, val_dataloader, device)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    # Save best checkpoint
                    best_dir = output_dir / "best"
                    policy.save_pretrained(best_dir)
                    preprocessor.save_pretrained(best_dir)
                    postprocessor.save_pretrained(best_dir)

                print(
                    f"step: {step:>7d} / {args.training_steps}  "
                    f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
                    f"best_val: {best_val_loss:.4f}" + (" *" if val_loss <= best_val_loss else "")
                )

            if use_wandb:
                log_dict = {"train_loss": train_loss, "step": step}
                if val_loss is not None:
                    log_dict["val_loss"] = val_loss
                    log_dict["best_val_loss"] = best_val_loss
                wandb.log(log_dict)

            if step % args.log_freq == 0 and val_loss is None:
                print(f"step: {step:>7d} / {args.training_steps}  train_loss: {train_loss:.4f}")

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

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
