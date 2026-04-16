"""Convert Grabette dataset for diffusion policy training.

Transforms the dataset to match the UMI approach:
  1. Converts rotation from axis-angle (3D) to 6D continuous representation.
  2. Computes delta actions: action[t] = pose[t+1] - pose[t] for position/rotation,
     gripper stays absolute.
  3. Sets observation.state to gripper joints only (2D) — the model should NOT see
     absolute position (it's in an arbitrary SLAM reference frame).

Before:
  observation.state = [x, y, z, ax, ay, az, proximal, distal]  (8D absolute)
  action            = [x, y, z, ax, ay, az, proximal, distal]  (8D absolute)

After:
  observation.state = [proximal, distal]                                           (2D)
  action            = [dx, dy, dz, dr6d_0..5, proximal, distal]                  (11D)
                       ^^^^^^^^^^^^^^^^^^^^^^^^                  ^^^^^^^^^^^^^^^
                       deltas (pose[t+1] - pose[t])             absolute gripper

Why:
  - The SLAM reference frame has an arbitrary origin — absolute position is meaningless.
  - The model gets spatial info from the camera image, not from position numbers.
  - Delta actions are origin-invariant.
  - 6D rotation is continuous (no singularities unlike axis-angle).

Usage:
  uv run python examples/openarm_gripette/convert_dataset.py \\
      --repo_id SteveNguyen/Grabette_redcube_quest
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from lerobot.datasets import LeRobotDataset
from lerobot.utils.rotation import rotvec_to_rotation_6d

logger = logging.getLogger(__name__)

# Feature names after conversion
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
STATE_NAMES = ["proximal", "distal"]


def pose_8d_to_11d(data_8d: np.ndarray) -> np.ndarray:
    """Convert 8D (pos + axis-angle + gripper) to 11D (pos + rot6d + gripper).

    Args:
        data_8d: (N, 8) — [x, y, z, ax, ay, az, proximal, distal]

    Returns:
        (N, 11) — [x, y, z, r6d_0..5, proximal, distal]
    """
    pos = data_8d[:, :3]
    rotvec = data_8d[:, 3:6]
    gripper = data_8d[:, 6:]

    rotvec_t = torch.from_numpy(rotvec).float()
    rot6d = rotvec_to_rotation_6d(rotvec_t).numpy()

    return np.concatenate([pos, rot6d, gripper], axis=1)


def compute_delta_actions(poses_11d: np.ndarray, episode_indices: np.ndarray) -> np.ndarray:
    """Compute per-frame delta actions from absolute poses.

    For position + rotation dims: delta[t] = pose[t+1] - pose[t]
    For gripper dims: kept absolute (not delta).
    At the last frame of each episode: delta = 0 (no next frame).

    Args:
        poses_11d: (N, 11) absolute poses [x, y, z, r6d_0..5, proximal, distal]
        episode_indices: (N,) episode index per frame

    Returns:
        (N, 11) delta actions [dx, dy, dz, dr6d_0..5, proximal, distal]
    """
    n = len(poses_11d)
    actions = np.zeros((n, 11), dtype=np.float32)

    # Position + rotation deltas (dims 0-8)
    # Shift by 1: delta[t] = pose[t+1] - pose[t]
    actions[:-1, :9] = poses_11d[1:, :9] - poses_11d[:-1, :9]

    # Zero out deltas at episode boundaries (last frame of each episode)
    ep_change = np.where(episode_indices[1:] != episode_indices[:-1])[0]
    actions[ep_change, :9] = 0.0
    # Last frame of the whole dataset
    actions[-1, :9] = 0.0

    # Gripper: absolute values (dims 9-10), taken from current frame
    actions[:, 9:] = poses_11d[:, 9:]

    return actions


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Grabette dataset for training")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="SteveNguyen/Grabette_redcube_quest",
        help="LeRobot dataset repo ID",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    ds = LeRobotDataset(args.repo_id)
    root = Path(ds.root)
    logger.info(f"Dataset root: {root}")
    logger.info(f"Frames: {len(ds)}, Episodes: {ds.meta.total_episodes}")

    # --- 1. Convert parquet data ---
    data_dir = root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    logger.info(f"Processing {len(parquet_files)} parquet file(s)...")

    for pf in parquet_files:
        table = pq.read_table(pf)

        # Read the original 8D action column (= absolute pose at each frame)
        action_col = table.column("action")
        poses_8d = np.array(action_col.to_pylist(), dtype=np.float32)

        if poses_8d.shape[1] == 11:
            logger.info(f"  {pf.name}: already 11D, checking if deltas are computed...")
            # Check if this looks like deltas (mean ~0 for position dims) or absolute
            pos_mean = np.abs(poses_8d[:, :3].mean(axis=0))
            if np.all(pos_mean < 0.01):
                logger.info(f"  {pf.name}: looks like deltas already, skipping")
                continue
            poses_11d = poses_8d
        elif poses_8d.shape[1] == 8:
            # Convert axis-angle to 6D rotation
            poses_11d = pose_8d_to_11d(poses_8d)
            logger.info(f"  {pf.name}: converted rotation 8D -> 11D")
        else:
            raise ValueError(f"Unexpected action dim: {poses_8d.shape[1]}")

        # Episode indices for boundary detection
        episode_indices = np.array(table.column("episode_index").to_pylist())

        # Compute delta actions
        delta_actions = compute_delta_actions(poses_11d, episode_indices)
        logger.info(
            f"  {pf.name}: computed delta actions "
            f"(pos delta mean magnitude: {np.linalg.norm(delta_actions[:, :3], axis=1).mean() * 1000:.2f} mm)"
        )

        # Observation state = gripper only (dims 9-10 of the 11D pose)
        gripper_state = poses_11d[:, 9:]  # (N, 2) — proximal, distal

        # Rebuild table
        df_dict = {}
        for col in table.column_names:
            if col == "action":
                df_dict[col] = pa.array(delta_actions.tolist(), type=pa.list_(pa.float32()))
            elif col == "observation.state":
                df_dict[col] = pa.array(gripper_state.tolist(), type=pa.list_(pa.float32()))
            else:
                df_dict[col] = table.column(col)

        # Add observation.state if it didn't exist
        if "observation.state" not in table.column_names:
            df_dict["observation.state"] = pa.array(gripper_state.tolist(), type=pa.list_(pa.float32()))

        new_table = pa.table(df_dict)
        pq.write_table(new_table, pf)
        logger.info(f"  {pf.name}: written ({new_table.num_rows} rows)")

    # --- 2. Update info.json ---
    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    info["features"]["action"] = {
        "dtype": "float32",
        "shape": [11],
        "names": ACTION_NAMES,
    }
    info["features"]["observation.state"] = {
        "dtype": "float32",
        "shape": [2],
        "names": STATE_NAMES,
    }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    logger.info("Updated info.json: action=11D (deltas), observation.state=2D (gripper)")

    # --- 3. Recompute stats (no relative_action flag — deltas are pre-computed) ---
    logger.info("Recomputing stats...")
    from lerobot.datasets.dataset_tools import recompute_stats

    ds_updated = LeRobotDataset(args.repo_id)
    recompute_stats(ds_updated, skip_image_video=True)

    # --- 4. Verify ---
    logger.info("\n=== Verification ===")
    ds_final = LeRobotDataset(args.repo_id, episodes=[0])

    logger.info(f"observation.state shape: {ds_final.meta.features['observation.state']['shape']}")
    logger.info(f"observation.state names: {ds_final.meta.features['observation.state']['names']}")
    logger.info(f"action shape: {ds_final.meta.features['action']['shape']}")
    logger.info(f"action names: {ds_final.meta.features['action']['names']}")

    sample = ds_final[50]
    logger.info("\nSample frame 50:")
    logger.info(f"  observation.state (gripper): {sample['observation.state'].tolist()}")

    action = sample["action"].tolist()
    logger.info("  action (11D deltas + gripper):")
    for n, v in zip(ACTION_NAMES, action, strict=True):
        logger.info(f"    {n:8s}: {v:+.6f}")

    # Sanity: position deltas should be small (mm scale at 50fps)
    pos_delta = np.array(action[:3])
    logger.info(
        f"\n  Position delta magnitude: {np.linalg.norm(pos_delta) * 1000:.2f} mm (should be ~1-5 mm)"
    )

    stats_path = root / "meta" / "stats.json"
    with open(stats_path) as f:
        stats = json.load(f)
    logger.info("\n  Action stats (should all be delta-scale):")
    for i, n in enumerate(ACTION_NAMES):
        mn = stats["action"]["min"][i]
        mx = stats["action"]["max"][i]
        logger.info(f"    {n:8s}: min={mn:+.6f}, max={mx:+.6f}")

    logger.info("\nConversion complete!")


if __name__ == "__main__":
    main()
