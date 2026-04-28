"""Score a trained Diffusion Policy on a held-out (OOD) dataset.

Why this script exists
----------------------
The standard train/val split in `train.py` is a random 10 % of the *same*
training distribution — same cube area, same motion templates. That tells
us "the policy is internally consistent on this distribution"; it does NOT
tell us "the policy generalises to new cube positions". With scripted
demos that all share the same motion shape, a model can fit train+val
perfectly by memorising the motion template and ignoring the camera.

A meaningful test is to score the policy on a *different* slice of the
cube workspace — exactly what the `--split eval` flag of the dataset
collector produces (a 2 cm strip on the +y edge of the cube area, never
shown at training time). If the loss on that OOD set is close to the
in-distribution val_loss, the policy generalises. If it's much higher,
the policy is memorising.

Usage:
  uv run python examples/openarm_gripette/eval_ood_loss.py \\
      --checkpoint outputs/gripette/diffusion/best \\
      --dataset_repo_id SteveNguyen/sim_grasp_eval_v1
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionPolicy


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    """Mirror train.py: convert frame-offset indices to seconds."""
    if delta_indices is None:
        return None
    return [i / fps for i in delta_indices]


def parse_args():
    p = argparse.ArgumentParser(
        description="Score a trained Diffusion Policy on an OOD held-out dataset."
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint directory (e.g. outputs/.../best).")
    p.add_argument("--dataset_repo_id", type=str, required=True,
                   help="Held-out dataset repo id, e.g. SteveNguyen/sim_grasp_eval_v1.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers. 0 surfaces real errors directly; "
                        "higher is faster once the dataset is known clean.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--bf16", action="store_true",
                   help="Use bf16 autocast (matches train --bf16).")
    p.add_argument("--max_batches", type=int, default=0,
                   help="0 = score the whole dataset; >0 = early-stop after N batches.")
    p.add_argument("--skip_decode_errors", action="store_true",
                   help="Skip batches that hit a video-decoder error (count them "
                        "and continue) instead of aborting. Useful when only a "
                        "few frames are unreadable.")
    p.add_argument("--in_dist_val_loss", type=float, default=None,
                   help="Optional reference loss from the training run, for the diagnostic line.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # ---- Load policy + processor (same procedure as eval_simulator.py) ----
    print(f"Loading checkpoint from {args.checkpoint}")
    policy = DiffusionPolicy.from_pretrained(args.checkpoint).to(device).eval()
    preprocessor, _ = make_pre_post_processors(
        policy.config, pretrained_path=args.checkpoint,
    )

    # ---- Build the dataset with the same temporal windowing as training ----
    # delta_timestamps come from the policy config (mirrors train.py L364-L372).
    metadata = LeRobotDatasetMetadata(args.dataset_repo_id)
    fps = metadata.fps

    delta_timestamps = {
        "observation.state": make_delta_timestamps(
            policy.config.observation_delta_indices, fps),
        "action": make_delta_timestamps(
            policy.config.action_delta_indices, fps),
    }
    delta_timestamps |= {
        k: make_delta_timestamps(policy.config.observation_delta_indices, fps)
        for k in policy.config.image_features
    }

    ds = LeRobotDataset(
        args.dataset_repo_id, delta_timestamps=delta_timestamps,
    )
    print(f"Dataset:          {args.dataset_repo_id}")
    print(f"  total frames:   {len(ds)}")
    print(f"  episodes:       {ds.meta.total_episodes}")
    print(f"  fps:            {fps}")
    print(f"  state shape:    {ds[0]['observation.state'].shape}")
    print(f"  action shape:   {ds[0]['action'].shape}")
    print(f"  image keys:     {list(policy.config.image_features.keys())}")

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # ---- Score ----
    print(f"\nScoring on {len(ds)} frames in {len(loader)} batches...")
    losses: list[float] = []
    n_decode_errors = 0
    t_start = time.perf_counter()
    use_amp = args.bf16 and device.type == "cuda"

    loader_iter = iter(loader)
    i = 0
    with torch.no_grad():
        while True:
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            except RuntimeError as e:
                # torchcodec / libavcodec failures bubble up as RuntimeError.
                if args.skip_decode_errors and "decoder" in str(e).lower():
                    n_decode_errors += 1
                    continue
                raise
            batch = preprocessor(batch)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss, _ = policy.forward(batch)
            else:
                loss, _ = policy.forward(batch)
            losses.append(float(loss.item()))
            i += 1
            if args.max_batches and i >= args.max_batches:
                break
    dt = time.perf_counter() - t_start
    if n_decode_errors:
        print(f"WARNING: skipped {n_decode_errors} batch(es) due to video decoder errors.")

    losses = np.array(losses)
    mean_loss = float(losses.mean())
    median_loss = float(np.median(losses))
    print(f"\nDone in {dt:.1f}s.")
    print(f"OOD loss (mean):    {mean_loss:.5f}")
    print(f"OOD loss (median):  {median_loss:.5f}")
    print(f"OOD loss (std):     {losses.std():.5f}")
    print(f"OOD loss (min/max): {losses.min():.5f} / {losses.max():.5f}")

    if args.in_dist_val_loss is not None:
        ratio = mean_loss / args.in_dist_val_loss
        print(f"\nIn-distribution val_loss reference: {args.in_dist_val_loss:.5f}")
        print(f"OOD/in-dist ratio:                  {ratio:.2f}x")
        if ratio < 1.5:
            print("Verdict: OOD loss close to in-distribution val loss — policy is "
                  "generalising across cube positions, not just memorising the template.")
        elif ratio < 3.0:
            print("Verdict: OOD loss notably higher than val loss. Some generalisation, "
                  "but the policy is partially overfit to the training cube area.")
        else:
            print("Verdict: OOD loss is much higher than val loss. The policy has "
                  "memorised the motion template and does not handle held-out cube "
                  "positions. More visual diversity (DR / distractors / more demos) "
                  "needed before re-training.")


if __name__ == "__main__":
    main()
