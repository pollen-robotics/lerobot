"""Train an ACT (Action Chunking Transformer) policy for the Gripette project.

This is the ACT counterpart to train.py (Diffusion Policy). It trains on the
same dataset produced by convert_dataset.py:
  - observation.state = [proximal, distal]  (2D gripper) or 11D if --proprioception relative
  - action = [dx, dy, dz, dr6d_0..5, proximal, distal]  (11D: deltas + gripper)
  - observation.images.cam0 = camera image

ACT (Zhao et al., RSS 2023) is a CVAE + Transformer that predicts action chunks.
Compared to diffusion, it:
  - Trains faster (single forward pass vs. iterative denoising)
  - Runs faster at inference (no denoising loop)
  - Is often more sample-efficient
  - Can use temporal ensemble at inference: averages overlapping chunk predictions
    with exponential weights. Set --temporal_ensemble to enable; this forces
    n_action_steps=1 so the policy runs every step.

Note on relative actions: the dataset already contains pre-computed deltas, so
we set no runtime RelativeActionsProcessorStep — same as train.py.

Prerequisites:
  - Dataset converted with convert_dataset.py:
      uv run python examples/openarm_gripette/convert_dataset.py

Usage:
  uv run python examples/openarm_gripette/train_act.py \\
      --dataset_repo_id SteveNguyen/Grabette_redcube_quest

  # With temporal ensemble (recommended for best smoothness):
  uv run python examples/openarm_gripette/train_act.py \\
      --dataset_repo_id SteveNguyen/Grabette_redcube_quest \\
      --temporal_ensemble
"""

import argparse
from pathlib import Path

import torch
import torchvision.transforms as T

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features


def apply_color_jitter(batch: dict, image_keys: list[str], jitter: T.ColorJitter) -> dict:
    """Apply color jitter to image tensors in the batch (training-only augmentation)."""
    for key in image_keys:
        if key not in batch:
            continue
        img = batch[key]
        if img.ndim == 5:  # (B, T, C, H, W)
            b, t = img.shape[:2]
            img = jitter(img.reshape(b * t, *img.shape[2:]))
            batch[key] = img.reshape(b, t, *img.shape[1:])
        else:
            batch[key] = jitter(img)
    return batch


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    """Convert frame-offset indices to seconds for the dataset's delta_timestamps."""
    if delta_indices is None:
        return [0]
    return [i / fps for i in delta_indices]


@torch.no_grad()
def compute_val_loss(policy, preprocessor, val_dataloader, device, max_batches=50):
    """Compute average loss on the validation set.

    We intentionally keep the policy in training mode. ACT's VAE encoder only
    runs when `self.training` is True (modeling_act.py:398); in eval mode, `mu`
    and `log_sigma_x2` are set to None, so the KL term in ACTPolicy.forward
    crashes with `1 + None`. Dropout is therefore active during validation —
    this makes the loss slightly noisier but is still a valid relative signal
    for early stopping / best-checkpoint selection.
    """
    total_loss = 0.0
    num_batches = 0
    for batch in val_dataloader:
        batch = preprocessor(batch)
        loss, _ = policy.forward(batch)
        total_loss += loss.item()
        num_batches += 1
        if num_batches >= max_batches:
            break
    return total_loss / max(num_batches, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ACT Policy for Gripette")
    parser.add_argument(
        "--dataset_repo_id",
        type=str,
        default="pollen/gripette_demo",
        help="LeRobot dataset repo ID",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/gripette/act",
        help="Directory for checkpoints",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Compute device")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    parser.add_argument("--training_steps", type=int, default=200_000, help="Total training steps")
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=32,
        help="Length of predicted action chunks (in frames). At 50 FPS: 16=0.32s, 32=0.64s, 50=1s",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=16,
        help="Actions executed before re-planning. Ignored when --temporal_ensemble is set (forced to 1).",
    )
    parser.add_argument(
        "--temporal_ensemble",
        action="store_true",
        help="Enable ACT temporal ensemble at inference (paper default coeff=0.01). "
        "Forces n_action_steps=1 — policy is called every step and actions are averaged.",
    )
    parser.add_argument(
        "--temporal_ensemble_coeff",
        type=float,
        default=0.01,
        help="Exponential weighting coefficient for temporal ensemble (paper value: 0.01).",
    )
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=10.0,
        help="KL divergence weight for the VAE loss (ACT paper default: 10).",
    )
    parser.add_argument(
        "--no_vae",
        action="store_true",
        help="Disable the CVAE style encoder (deterministic ACT).",
    )
    parser.add_argument(
        "--optimizer_lr", type=float, default=1e-5, help="Learning rate (ACT paper default: 1e-5)"
    )
    parser.add_argument(
        "--optimizer_lr_backbone",
        type=float,
        default=1e-5,
        help="Learning rate for the vision backbone",
    )
    parser.add_argument("--log_freq", type=int, default=100, help="Log every N steps")
    parser.add_argument("--save_freq", type=int, default=10_000, help="Save checkpoint every N steps")
    parser.add_argument("--eval_freq", type=int, default=200, help="Evaluate on validation set every N steps")
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Fraction of episodes used for validation"
    )
    parser.add_argument(
        "--exclude_episodes", type=int, nargs="+", default=None,
        help="Episode indices to drop before the train/val split (e.g. the IK-flip "
             "episodes reported by convert_to_jointspace.py). Pass the SAME list to "
             "the cartesian and joint-space runs to keep the A/B on one episode set.",
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
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="HuggingFace Hub repo ID to push final + best checkpoints to (e.g. 'user/gripette_act_v1')",
    )
    parser.add_argument(
        "--hub_private",
        action="store_true",
        help="Make the HuggingFace Hub repo private (default: public)",
    )
    parser.add_argument(
        "--color_jitter",
        action="store_true",
        help="Enable color jitter augmentation during training (UMI values)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ---- Dataset metadata ----
    dataset_metadata = LeRobotDatasetMetadata(args.dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {
        key: ft
        for key, ft in features.items()
        if key not in output_features and (ft.type is not FeatureType.VISUAL or key in args.cameras)
    }

    action_feature_names = dataset_metadata.features.get("action", {}).get("names")

    print(f"Dataset:          {args.dataset_repo_id}")
    print(f"FPS:              {dataset_metadata.fps}")
    print(f"Input features:   {list(input_features.keys())}")
    print(f"Output features:  {list(output_features.keys())}")
    print(f"Action names:     {action_feature_names}")

    # ---- Temporal ensemble constraint ----
    # ACT requires n_action_steps=1 when temporal ensembling is enabled (policy must run
    # every step to build the running average).
    n_action_steps = 1 if args.temporal_ensemble else args.n_action_steps
    temporal_ensemble_coeff = args.temporal_ensemble_coeff if args.temporal_ensemble else None
    if n_action_steps > args.chunk_size:
        raise ValueError(
            f"n_action_steps ({n_action_steps}) must be <= chunk_size ({args.chunk_size})"
        )

    # ---- Policy configuration ----
    # ACT paper defaults: chunk_size=100, lr=1e-5, kl_weight=10.
    # n_obs_steps must be 1 (ACT limitation — see configuration_act.py:148-151).
    cfg = ACTConfig(
        input_features=input_features,
        output_features=output_features,
        # -- Temporal structure --
        n_obs_steps=1,
        chunk_size=args.chunk_size,
        n_action_steps=n_action_steps,
        # -- Vision backbone --
        # ImageNet-pretrained ResNet18 is the ACT paper default and usually gives
        # better sample efficiency than training from scratch.
        vision_backbone="resnet18",
        pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
        replace_final_stride_with_dilation=False,
        # -- Transformer --
        pre_norm=False,
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        feedforward_activation="relu",
        n_encoder_layers=4,
        n_decoder_layers=1,  # ACT codebase bug: only first layer used. Match upstream.
        # -- CVAE --
        use_vae=not args.no_vae,
        latent_dim=32,
        n_vae_encoder_layers=4,
        # -- Inference --
        temporal_ensemble_coeff=temporal_ensemble_coeff,
        # -- Training --
        dropout=0.1,
        kl_weight=args.kl_weight,
        # -- Normalization --
        # ACT paper uses MEAN_STD for everything. Delta actions from convert_dataset.py
        # are zero-centered so MEAN_STD is appropriate (MIN_MAX from the diffusion
        # script also works but tends to be more sensitive to outliers in delta space).
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        # -- Optimizer --
        optimizer_lr=args.optimizer_lr,
        optimizer_lr_backbone=args.optimizer_lr_backbone,
        optimizer_weight_decay=1e-4,
    )

    # ---- Instantiate policy ----
    policy = ACTPolicy(cfg)
    policy.train()
    policy.to(device)

    param_count = sum(p.numel() for p in policy.parameters())
    print(f"\nModel parameters: {param_count:,}")

    # ---- Pre/post processors ----
    # Same pipeline as diffusion (rename -> batch dim -> device -> normalize).
    # No RelativeActionsProcessorStep: deltas are pre-computed in the dataset.
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    # ---- Dataset with temporal windowing ----
    # For ACT: observation_delta_indices is None (n_obs_steps=1), so observation
    # keys are NOT added to delta_timestamps — otherwise the dataset would stack a
    # phantom time dim of size 1 and ACT's forward would see 4D tensors where it
    # expects 3D. action_delta_indices is always [0, 1, ..., chunk_size-1].
    # This matches upstream behavior in `lerobot.datasets.factory.resolve_delta_timestamps`.
    delta_timestamps = {
        "action": make_delta_timestamps(cfg.action_delta_indices, dataset_metadata.fps),
    }
    if cfg.observation_delta_indices is not None:
        obs_ts = make_delta_timestamps(cfg.observation_delta_indices, dataset_metadata.fps)
        delta_timestamps["observation.state"] = obs_ts
        delta_timestamps |= {k: obs_ts for k in cfg.image_features}

    # ---- Train/val split by episodes ----
    all_episodes = sorted(dataset_metadata.episodes["episode_index"])
    if args.exclude_episodes:
        # Drop flagged episodes (e.g. IK-reconfiguration "flip" episodes from
        # convert_to_jointspace.py). Pass the SAME list to the cartesian and
        # joint-space trainings so the A/B stays on an identical episode set.
        excl = set(args.exclude_episodes)
        all_episodes = [e for e in all_episodes if e not in excl]
        print(f"  Excluding {len(excl)} episodes; {len(all_episodes)} remain.")
    num_val = max(1, int(len(all_episodes) * args.val_ratio))
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
    # ACT's preset is plain AdamW (no scheduler). If you want a warmup or cosine
    # schedule, wrap this with torch.optim.lr_scheduler manually.
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())

    # ---- Color jitter augmentation ----
    color_jitter = None
    image_keys: list[str] = []
    if args.color_jitter:
        color_jitter = T.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08)
        image_keys = list(cfg.image_features.keys())
        print(f"  Color jitter:     enabled on {image_keys}")

    # ---- Wandb ----
    use_wandb = args.wandb_project is not None
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "policy": "act",
                "dataset": args.dataset_repo_id,
                "batch_size": args.batch_size,
                "training_steps": args.training_steps,
                "lr": cfg.optimizer_lr,
                "lr_backbone": cfg.optimizer_lr_backbone,
                "n_obs_steps": cfg.n_obs_steps,
                "chunk_size": cfg.chunk_size,
                "n_action_steps": cfg.n_action_steps,
                "temporal_ensemble_coeff": cfg.temporal_ensemble_coeff,
                "vision_backbone": cfg.vision_backbone,
                "pretrained_backbone": cfg.pretrained_backbone_weights,
                "dim_model": cfg.dim_model,
                "n_heads": cfg.n_heads,
                "dim_feedforward": cfg.dim_feedforward,
                "n_encoder_layers": cfg.n_encoder_layers,
                "n_decoder_layers": cfg.n_decoder_layers,
                "use_vae": cfg.use_vae,
                "kl_weight": cfg.kl_weight,
                "dropout": cfg.dropout,
                "color_jitter": args.color_jitter,
                "action_dim": cfg.action_feature.shape[0],
                "state_dim": cfg.robot_state_feature.shape[0]
                if cfg.robot_state_feature is not None
                else 0,
                "cameras": args.cameras,
                "model_params": param_count,
            },
        )

    # ---- Training loop ----
    print(f"\nStarting training for {args.training_steps} steps on {device}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Chunk size:       {cfg.chunk_size}")
    print(f"  n_action_steps:   {cfg.n_action_steps}")
    print(
        f"  Temporal ensemble: {'on (coeff=' + str(cfg.temporal_ensemble_coeff) + ')' if cfg.temporal_ensemble_coeff else 'off'}"
    )
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
            if color_jitter is not None:
                batch = apply_color_jitter(batch, image_keys, color_jitter)

            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)

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
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)
    print(f"\nTraining complete. Model saved to {output_dir}")

    # ---- Push to HuggingFace Hub ----
    if args.push_to_hub is not None:
        final_repo = args.push_to_hub
        best_repo = f"{args.push_to_hub}-best"
        print("\nPushing to HuggingFace Hub:")
        print(f"  final checkpoint  -> {final_repo}")
        print(f"  best checkpoint   -> {best_repo}")

        policy.push_to_hub(final_repo, private=args.hub_private)
        preprocessor.push_to_hub(final_repo, private=args.hub_private)
        postprocessor.push_to_hub(final_repo, private=args.hub_private)

        best_dir = output_dir / "best"
        if best_dir.exists():
            best_policy = ACTPolicy.from_pretrained(best_dir)
            best_pre, best_post = make_pre_post_processors(best_policy.config, pretrained_path=best_dir)
            best_policy.push_to_hub(best_repo, private=args.hub_private)
            best_pre.push_to_hub(best_repo, private=args.hub_private)
            best_post.push_to_hub(best_repo, private=args.hub_private)
            print(f"  best_val_loss:    {best_val_loss:.4f}")

        print("\nTo use on another machine:")
        print(f"  --checkpoint {best_repo}   (recommended, lowest val_loss)")
        print(f"  --checkpoint {final_repo}  (final step)")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
