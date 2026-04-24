"""Probe a DiffusionPolicy checkpoint for input-sensitivity bugs.

Runs the policy on a fixed battery of synthetic observations and reports how
different the outputs are. If wildly different inputs produce nearly-identical
outputs, the model has either collapsed or is ignoring its inputs — both
explain the symptom "every trained model behaves the same on the real robot".

Can compare multiple checkpoints on the exact same inputs in one run, so you
can verify two trained models actually differ.

Usage:
  # Single checkpoint, default battery of inputs.
  uv run python examples/openarm_gripette/probe_model.py \\
      --checkpoint outputs/gripette/run_001/best

  # Compare several checkpoints on the same inputs.
  uv run python examples/openarm_gripette/probe_model.py \\
      --checkpoint outputs/gripette/run_001/best \\
      --checkpoint outputs/gripette/run_002/best \\
      --checkpoint outputs/gripette/run_003/best

Interpretation guide:
  - Within a single checkpoint: if max pairwise action L2 across inputs is
    < ~0.01, the model is output-constant regardless of input. Likely bugs:
    collapsed to mean prediction, broken image encoder, or wrong weights.
  - Across checkpoints on the SAME input: if max pairwise L2 is near zero,
    the "different" checkpoints are producing the same actions. Either
    they're loading the same weights, or they've all collapsed to the same
    degenerate solution.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_IMAGES

logger = logging.getLogger(__name__)


def build_test_inputs(state_dim: int, image_shape: tuple[int, int, int]) -> dict[str, dict]:
    """Return a labelled set of (image, state) pairs covering wildly varied inputs.

    image_shape is (C, H, W). State is zeroed in all cases so we isolate the
    model's sensitivity to *visual* input — if everything is the same, the
    image path is broken.
    """
    zero_state = np.zeros(state_dim, dtype=np.float32)
    rng_a = np.random.RandomState(42)
    rng_b = np.random.RandomState(7)

    inputs: dict[str, dict] = {
        "black": {
            "image": np.zeros(image_shape, dtype=np.float32),
            "state": zero_state,
        },
        "white": {
            "image": np.ones(image_shape, dtype=np.float32),
            "state": zero_state,
        },
        "gray_0_5": {
            "image": np.full(image_shape, 0.5, dtype=np.float32),
            "state": zero_state,
        },
        "random_A": {
            "image": rng_a.uniform(0.0, 1.0, size=image_shape).astype(np.float32),
            "state": zero_state,
        },
        "random_B": {
            "image": rng_b.uniform(0.0, 1.0, size=image_shape).astype(np.float32),
            "state": zero_state,
        },
    }

    # Also test state-only sensitivity: same (gray) image, different states.
    if state_dim >= 3:
        state_x = np.zeros(state_dim, dtype=np.float32)
        state_x[0] = 0.1  # 10cm offset on dx_start
        state_y = np.zeros(state_dim, dtype=np.float32)
        state_y[1] = 0.1
        inputs["gray_state_x"] = {"image": np.full(image_shape, 0.5, dtype=np.float32), "state": state_x}
        inputs["gray_state_y"] = {"image": np.full(image_shape, 0.5, dtype=np.float32), "state": state_y}

    return inputs


def run_one(policy, preprocessor, postprocessor, device, image_np, state_np) -> np.ndarray:
    """Single forward pass through the full inference stack. Returns (horizon, action_dim)."""
    image_tensor = torch.from_numpy(image_np).float().unsqueeze(0).to(device)  # (1, C, H, W)
    state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(device)  # (1, state_dim)

    batch = {
        "observation.images.cam0": image_tensor,
        "observation.state": state_tensor,
    }
    batch = preprocessor(batch)

    # Replicate the manual path from eval_simulator.py so we get the full
    # predicted chunk, not just a single pop from the internal queue.
    batch.pop(ACTION, None)
    if policy.config.image_features:
        batch[OBS_IMAGES] = torch.stack([batch[k] for k in policy.config.image_features], dim=-4)
    # Fresh queues per call: we don't want cross-call contamination.
    policy.reset()
    policy._queues = populate_queues(policy._queues, batch)

    # Request the full remaining horizon.
    saved_n = policy.config.n_action_steps
    policy.config.n_action_steps = policy.config.horizon - policy.config.n_obs_steps + 1
    try:
        with torch.no_grad():
            chunk = policy.predict_action_chunk(batch)  # (1, T, action_dim)
    finally:
        policy.config.n_action_steps = saved_n

    # Postprocess per-timestep (unnormalize): apply once to the first timestep
    # to keep this minimal; we only need to compare numerical scale.
    chunk = chunk.squeeze(0)  # (T, action_dim)
    out = []
    for t in range(chunk.shape[0]):
        a = postprocessor(chunk[t : t + 1])
        out.append(a.squeeze(0).cpu().numpy())
    return np.stack(out)


def summarize(name: str, actions_by_input: dict[str, np.ndarray]) -> None:
    """Print action magnitudes + pairwise L2 distances over timestep 0."""
    logger.info(f"\n===== Checkpoint: {name} =====")
    logger.info("Action chunks shape (horizon, action_dim), showing timestep 0 only:")
    t0_actions = {k: v[0] for k, v in actions_by_input.items()}
    for k, a in t0_actions.items():
        pos_mm = float(np.linalg.norm(a[:3]) * 1000)
        grip = a[9:] if len(a) >= 11 else np.array([np.nan, np.nan])
        logger.info(
            f"  {k:16s}  dpos |{pos_mm:5.2f}mm|  "
            f"dpos={np.round(a[:3] * 1000, 2).tolist()}  "
            f"grip={np.round(grip, 3).tolist()}"
        )

    keys = list(t0_actions.keys())
    logger.info("Pairwise L2 distances (full action vector, timestep 0):")
    max_dist = 0.0
    min_dist = float("inf")
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1 :]:
            d = float(np.linalg.norm(t0_actions[k1] - t0_actions[k2]))
            max_dist = max(max_dist, d)
            min_dist = min(min_dist, d)
            logger.info(f"  {k1:16s}  vs  {k2:16s}  L2 = {d:.6f}")
    logger.info(f"  max pairwise L2 = {max_dist:.6f}")
    logger.info(f"  min pairwise L2 = {min_dist:.6f}")

    # Simple verdict.
    if max_dist < 0.01:
        logger.warning(
            "  VERDICT: model appears INPUT-INSENSITIVE. Max pairwise L2 across wildly "
            "different inputs is < 0.01 — likely collapsed or broken image path."
        )
    elif max_dist < 0.1:
        logger.warning(
            "  VERDICT: weak input sensitivity (max L2 < 0.1). Policy may be underfit "
            "or overly smoothed. Worth investigating."
        )
    else:
        logger.info("  VERDICT: model is input-sensitive (max L2 >= 0.1). Input pipeline is likely healthy.")


def cross_checkpoint_diff(by_ckpt: dict[str, dict[str, np.ndarray]]) -> None:
    """For each input, compare outputs across all checkpoints."""
    if len(by_ckpt) < 2:
        return
    logger.info("\n===== Cross-checkpoint comparison =====")
    ckpts = list(by_ckpt.keys())
    inputs = list(next(iter(by_ckpt.values())).keys())
    for inp in inputs:
        logger.info(f"\n  Input: {inp}")
        max_dist = 0.0
        for i, c1 in enumerate(ckpts):
            for c2 in ckpts[i + 1 :]:
                a1 = by_ckpt[c1][inp][0]
                a2 = by_ckpt[c2][inp][0]
                d = float(np.linalg.norm(a1 - a2))
                max_dist = max(max_dist, d)
                logger.info(f"    {c1}  vs  {c2}  L2 = {d:.6f}")
        if max_dist < 0.01:
            logger.warning(
                f"    BAD: on input {inp!r}, all checkpoints produce nearly identical "
                "outputs — they may be the same weights, or all have collapsed."
            )


def sample_dataset_inputs(
    dataset_repo_id: str, state_dim: int, num_samples: int, image_key: str
) -> dict[str, dict]:
    """Pull N varied frames from the training dataset for a realistic probe.

    Frames are drawn at spaced intervals across the dataset so they cover
    different phases / scenes / gripper states. State is replaced with zero
    to isolate the image channel (pass zero_state separately if you want to
    test state too).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    logger.info(f"Loading dataset {dataset_repo_id} for real-image probe")
    ds = LeRobotDataset(dataset_repo_id)
    n = len(ds)
    if n == 0:
        raise SystemExit(f"Dataset {dataset_repo_id} has no frames")

    indices = np.linspace(0, n - 1, num_samples, dtype=int)
    zero_state = np.zeros(state_dim, dtype=np.float32)

    out: dict[str, dict] = {}
    for i, idx in enumerate(indices):
        sample = ds[int(idx)]
        if image_key not in sample:
            raise SystemExit(f"Dataset sample missing key {image_key!r}. Available: {list(sample.keys())}")
        img = sample[image_key]
        # Dataset may return multi-step (T, C, H, W) or single-step (C, H, W).
        img_np = img.cpu().numpy() if hasattr(img, "cpu") else np.asarray(img)
        if img_np.ndim == 4:
            img_np = img_np[-1]  # most recent frame of the stacked observation

        # Also grab the recorded state for a "real state" counterpart.
        real_state = sample.get("observation.state")
        if real_state is not None:
            real_state_np = real_state.cpu().numpy() if hasattr(real_state, "cpu") else np.asarray(real_state)
            if real_state_np.ndim == 2:
                real_state_np = real_state_np[-1]
        else:
            real_state_np = zero_state

        out[f"ds_frame_{idx}_zero_state"] = {"image": img_np.astype(np.float32), "state": zero_state}
        out[f"ds_frame_{idx}_real_state"] = {
            "image": img_np.astype(np.float32),
            "state": real_state_np.astype(np.float32),
        }
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Probe DiffusionPolicy checkpoint(s) for input-sensitivity bugs")
    p.add_argument(
        "--checkpoint",
        type=str,
        action="append",
        required=True,
        help="Checkpoint path or HF Hub repo id. Repeat to compare multiple.",
    )
    p.add_argument(
        "--dataset_repo_id",
        type=str,
        default=None,
        help="If set, also probe the model with real training images drawn from this dataset. "
        "Much more informative than synthetic images — tests whether the policy discriminates "
        "between actual scenes.",
    )
    p.add_argument(
        "--num_dataset_samples",
        type=int,
        default=4,
        help="Number of frames to sample from the dataset (spaced evenly across all frames).",
    )
    p.add_argument(
        "--image_key",
        type=str,
        default="observation.images.cam0",
        help="Image feature key in the dataset.",
    )
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    device = torch.device(args.device)

    by_ckpt: dict[str, dict[str, np.ndarray]] = {}

    for ckpt_path in args.checkpoint:
        logger.info(f"\nLoading {ckpt_path}")
        policy = DiffusionPolicy.from_pretrained(ckpt_path)
        policy.to(device)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=ckpt_path)

        state_dim = policy.config.robot_state_feature.shape[0]
        first_img_feature = next(iter(policy.config.image_features.values()))
        image_shape: tuple[int, int, int] = tuple(first_img_feature.shape)  # (C, H, W)

        logger.info(f"  state_dim={state_dim}, image_shape={image_shape}")
        logger.info(f"  horizon={policy.config.horizon}, action_dim={policy.config.action_feature.shape[0]}")

        inputs = build_test_inputs(state_dim, image_shape)
        if args.dataset_repo_id is not None:
            inputs.update(
                sample_dataset_inputs(
                    args.dataset_repo_id, state_dim, args.num_dataset_samples, args.image_key
                )
            )
        actions_by_input: dict[str, np.ndarray] = {}
        for name, obs in inputs.items():
            actions_by_input[name] = run_one(
                policy, preprocessor, postprocessor, device, obs["image"], obs["state"]
            )

        label = Path(ckpt_path).name if "/" in ckpt_path else ckpt_path
        summarize(label, actions_by_input)
        by_ckpt[label] = actions_by_input

    cross_checkpoint_diff(by_ckpt)


if __name__ == "__main__":
    main()
