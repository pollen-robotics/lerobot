"""Convert Grabette dataset rotation from axis-angle (3D) to 6D continuous representation.

Transforms both observation.state and action columns from 8D to 11D:
  Before: [x, y, z, ax, ay, az, proximal, distal]             (8D)
  After:  [x, y, z, r6d_0, r6d_1, r6d_2, r6d_3, r6d_4, r6d_5, proximal, distal]  (11D)

The 6D rotation representation (Zhou et al., CVPR 2019) encodes orientation as the
first two columns of the rotation matrix. It is continuous (no singularities) and has
better gradient properties than axis-angle or quaternions for neural network training.

Usage:
  uv run python examples/openarm_gripette/convert_rotation_6d.py \\
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

# New feature names after conversion (11D)
NEW_NAMES = ["x", "y", "z", "r6d_0", "r6d_1", "r6d_2", "r6d_3", "r6d_4", "r6d_5", "proximal", "distal"]


def convert_rotvec_to_6d(data_8d: np.ndarray) -> np.ndarray:
    """Convert 8D (pos + axis-angle + gripper) to 11D (pos + rot6d + gripper).

    Args:
        data_8d: Array of shape (N, 8) — [x, y, z, ax, ay, az, proximal, distal]

    Returns:
        Array of shape (N, 11) — [x, y, z, r6d_0..r6d_5, proximal, distal]
    """
    pos = data_8d[:, :3]  # (N, 3)
    rotvec = data_8d[:, 3:6]  # (N, 3) axis-angle
    gripper = data_8d[:, 6:]  # (N, 2)

    # Convert axis-angle to 6D rotation via torch (handles batches efficiently)
    rotvec_t = torch.from_numpy(rotvec).float()
    rot6d_t = rotvec_to_rotation_6d(rotvec_t)
    rot6d = rot6d_t.numpy()  # (N, 6)

    return np.concatenate([pos, rot6d, gripper], axis=1)  # (N, 11)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Grabette dataset rotation to 6D")
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

    # Load dataset to get its local path
    ds = LeRobotDataset(args.repo_id)
    root = Path(ds.root)
    logger.info(f"Dataset root: {root}")
    logger.info(f"Frames: {len(ds)}, Episodes: {ds.meta.total_episodes}")

    # --- 1. Convert parquet data ---
    data_dir = root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    logger.info(f"Converting {len(parquet_files)} parquet file(s)...")

    for pf in parquet_files:
        table = pq.read_table(pf)
        columns_to_convert = []

        for col_name in ["action", "observation.state"]:
            if col_name not in table.column_names:
                logger.warning(f"  {pf.name}: missing column '{col_name}', skipping")
                continue
            columns_to_convert.append(col_name)

        if not columns_to_convert:
            continue

        # Read all rows into a dict for modification
        df_dict = {}
        for col in table.column_names:
            df_dict[col] = table.column(col)

        for col_name in columns_to_convert:
            # Extract as numpy: each element is a fixed-size list of 8 floats
            col = table.column(col_name)
            data_8d = np.array(col.to_pylist(), dtype=np.float32)

            if data_8d.shape[1] == 11:
                logger.info(f"  {pf.name}: {col_name} already 11D, skipping")
                continue

            if data_8d.shape[1] != 8:
                raise ValueError(f"Expected 8D data, got {data_8d.shape[1]}D in {col_name}")

            # Convert to 11D
            data_11d = convert_rotvec_to_6d(data_8d)

            # Replace the column: store as list of lists for Arrow
            new_col = pa.array(data_11d.tolist(), type=pa.list_(pa.float32()))
            df_dict[col_name] = new_col

        # Rebuild the table
        new_table = pa.table(df_dict)
        pq.write_table(new_table, pf)
        logger.info(f"  {pf.name}: converted {columns_to_convert} from 8D to 11D ({new_table.num_rows} rows)")

    # --- 2. Update info.json ---
    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    for feat_name in ["action", "observation.state"]:
        if feat_name in info["features"]:
            info["features"][feat_name]["shape"] = [11]
            info["features"][feat_name]["names"] = NEW_NAMES
            logger.info(f"Updated info.json: {feat_name} -> shape=[11], names={NEW_NAMES}")

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)

    # --- 3. Recompute stats ---
    logger.info("Recomputing stats with relative actions (excluding proximal, distal)...")
    # Reload dataset with updated schema
    ds_updated = LeRobotDataset(args.repo_id)

    from lerobot.datasets.dataset_tools import recompute_stats

    recompute_stats(
        ds_updated,
        skip_image_video=True,
        relative_action=True,
        relative_exclude_joints=["proximal", "distal"],
        chunk_size=16,
        num_workers=0,
    )

    # --- 4. Verify ---
    logger.info("\n=== Verification ===")
    ds_final = LeRobotDataset(args.repo_id, episodes=[0])
    sample = ds_final[50]

    logger.info(f"action shape: {sample['action'].shape}")
    logger.info(f"state shape:  {sample['observation.state'].shape}")
    logger.info(f"action names: {ds_final.meta.features['action']['names']}")

    action = sample["action"].tolist()
    names = NEW_NAMES
    logger.info("Sample action values:")
    for n, v in zip(names, action, strict=True):
        logger.info(f"  {n:8s}: {v:+.6f}")

    # Sanity: rot6d columns should have magnitude ~1 (columns of rotation matrix)
    rot6d = sample["action"][3:9]
    col1_norm = rot6d[:3].norm().item()
    col2_norm = rot6d[3:].norm().item()
    logger.info(f"\nrot6d column norms: col1={col1_norm:.4f}, col2={col2_norm:.4f} (should be ~1.0)")

    logger.info("\nConversion complete!")


if __name__ == "__main__":
    main()
