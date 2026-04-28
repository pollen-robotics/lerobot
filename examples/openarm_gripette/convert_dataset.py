"""Convert Grabette dataset for diffusion policy training.

Transforms the dataset:
  1. Converts rotation from axis-angle (3D) to 6D continuous representation.
  2. Computes delta actions: action[t] = pose[t+1] - pose[t] for position/rotation,
     gripper stays absolute.
  3. Builds observation.state depending on --proprioception mode:

     --proprioception none (default):
       observation.state = [proximal, distal]  (2D)
       Model sees camera + gripper only. Simplest approach.

     --proprioception relative:
       observation.state = [dx_start, dy_start, dz_start, r6d_rel_0..5, proximal, distal]  (11D)
       Includes position and rotation relative to episode start (UMI approach).
       Frame-independent proprioception — the model knows how far it moved/rotated.

Usage:
  # Gripper-only state (2D):
  uv run python examples/openarm_gripette/convert_dataset.py \\
      --repo_id SteveNguyen/Grabette_redcube_quest

  # With relative proprioception (11D, UMI-style):
  uv run python examples/openarm_gripette/convert_dataset.py \\
      --repo_id SteveNguyen/Grabette_redcube_quest --proprioception relative
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
from lerobot.utils.rotation import (
    rotation_6d_to_rotation_matrix_numpy,
    rotation_matrix_to_rotation_6d_numpy,
    rotvec_to_rotation_6d,
)

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
STATE_NAMES_NONE = ["proximal", "distal"]
STATE_NAMES_RELATIVE = [
    "dx_start",
    "dy_start",
    "dz_start",
    "r6d_rel_0",
    "r6d_rel_1",
    "r6d_rel_2",
    "r6d_rel_3",
    "r6d_rel_4",
    "r6d_rel_5",
    "proximal",
    "distal",
]


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
    actions[:-1, :9] = poses_11d[1:, :9] - poses_11d[:-1, :9]

    # Zero out deltas at episode boundaries
    ep_change = np.where(episode_indices[1:] != episode_indices[:-1])[0]
    actions[ep_change, :9] = 0.0
    actions[-1, :9] = 0.0

    # Gripper: absolute values (dims 9-10)
    actions[:, 9:] = poses_11d[:, 9:]

    return actions


def compute_relative_to_start_state(poses_11d: np.ndarray, episode_indices: np.ndarray) -> np.ndarray:
    """Compute position and rotation relative to episode start.

    For each frame, computes:
      - Position: pos[t] - pos[episode_start]
      - Rotation: R[t] @ R[episode_start]^{-1}, encoded as 6D

    Args:
        poses_11d: (N, 11) absolute poses [x, y, z, r6d_0..5, proximal, distal]
        episode_indices: (N,) episode index per frame

    Returns:
        (N, 11) relative state [dx_start, dy_start, dz_start, r6d_rel_0..5, proximal, distal]
    """
    n = len(poses_11d)
    relative_state = np.zeros((n, 11), dtype=np.float32)

    # Find the start index of each episode
    unique_eps = np.unique(episode_indices)
    ep_start_idx = {}
    for ep in unique_eps:
        ep_start_idx[ep] = np.where(episode_indices == ep)[0][0]

    for i in range(n):
        ep = episode_indices[i]
        start_i = ep_start_idx[ep]

        # Position relative to episode start
        relative_state[i, :3] = poses_11d[i, :3] - poses_11d[start_i, :3]

        # Rotation relative to episode start: R_rel = R_current @ R_start^{-1}
        r6d_current = poses_11d[i, 3:9]
        r6d_start = poses_11d[start_i, 3:9]

        r_current = rotation_6d_to_rotation_matrix_numpy(r6d_current.reshape(1, 6))[0]
        r_start = rotation_6d_to_rotation_matrix_numpy(r6d_start.reshape(1, 6))[0]
        r_relative = r_current @ r_start.T  # R_current @ R_start^{-1}

        relative_state[i, 3:9] = rotation_matrix_to_rotation_6d_numpy(r_relative.reshape(1, 3, 3))[0]

    # Gripper: absolute values
    relative_state[:, 9:] = poses_11d[:, 9:]

    return relative_state


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Grabette dataset for training")
    parser.add_argument(
        "--repo_id",
        type=str,
        default="SteveNguyen/Grabette_redcube_quest",
        help="LeRobot dataset repo ID",
    )
    parser.add_argument(
        "--proprioception",
        type=str,
        choices=["none", "relative"],
        default="none",
        help="State mode: 'none' = gripper only (2D), 'relative' = pose relative to episode start (11D)",
    )
    parser.add_argument(
        "--push_to_hub",
        type=str,
        default=None,
        help="If set, push the converted dataset to this Hub repo id "
             "(e.g. 'SteveNguyen/sim_grasp_train_v2'). If the local repo "
             "id does not match, the push retargets to this id.",
    )
    parser.add_argument(
        "--hub_private",
        action="store_true",
        help="Make the Hub repo private (default: public).",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    use_relative = args.proprioception == "relative"
    state_names = STATE_NAMES_RELATIVE if use_relative else STATE_NAMES_NONE
    state_dim = len(state_names)

    logger.info(f"Proprioception mode: {args.proprioception} ({state_dim}D state)")

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

        # Read the original action column (= absolute pose at each frame)
        action_col = table.column("action")
        poses_raw = np.array(action_col.to_pylist(), dtype=np.float32)

        # Convert to 11D if still 8D
        if poses_raw.shape[1] == 8:
            poses_11d = pose_8d_to_11d(poses_raw)
            logger.info(f"  {pf.name}: converted rotation 8D -> 11D")
        elif poses_raw.shape[1] == 11:
            # Check if this is already deltas or still absolute
            pos_mean = np.abs(poses_raw[:, :3].mean(axis=0))
            if np.all(pos_mean < 0.01):
                logger.warning(f"  {pf.name}: appears to already be deltas, re-run on original data")
                continue
            poses_11d = poses_raw
        else:
            raise ValueError(f"Unexpected action dim: {poses_raw.shape[1]}")

        # Episode indices
        episode_indices = np.array(table.column("episode_index").to_pylist())

        # Compute delta actions
        delta_actions = compute_delta_actions(poses_11d, episode_indices)
        logger.info(
            f"  {pf.name}: delta actions "
            f"(mean pos delta: {np.linalg.norm(delta_actions[:, :3], axis=1).mean() * 1000:.2f} mm)"
        )

        # Compute observation state
        if use_relative:
            obs_state = compute_relative_to_start_state(poses_11d, episode_indices)
            logger.info(f"  {pf.name}: computed relative-to-start state (11D)")
        else:
            obs_state = poses_11d[:, 9:]  # gripper only (2D)

        # Rebuild table
        df_dict = {}
        for col in table.column_names:
            if col == "action":
                df_dict[col] = pa.array(delta_actions.tolist(), type=pa.list_(pa.float32()))
            elif col == "observation.state":
                df_dict[col] = pa.array(obs_state.tolist(), type=pa.list_(pa.float32()))
            else:
                df_dict[col] = table.column(col)

        if "observation.state" not in table.column_names:
            df_dict["observation.state"] = pa.array(obs_state.tolist(), type=pa.list_(pa.float32()))

        new_table = pa.table(df_dict)
        pq.write_table(new_table, pf)
        logger.info(f"  {pf.name}: written ({new_table.num_rows} rows)")

    # --- 2. Update info.json ---
    info_path = root / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    info["features"]["action"] = {"dtype": "float32", "shape": [11], "names": ACTION_NAMES}
    info["features"]["observation.state"] = {
        "dtype": "float32",
        "shape": [state_dim],
        "names": state_names,
    }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    logger.info(
        f"Updated info.json: action=11D (deltas), observation.state={state_dim}D ({args.proprioception})"
    )

    # --- 3. Recompute stats ---
    logger.info("Recomputing stats...")
    from lerobot.datasets.dataset_tools import recompute_stats

    ds_updated = LeRobotDataset(args.repo_id)
    recompute_stats(ds_updated, skip_image_video=True)

    # --- 4. Verify ---
    logger.info("\n=== Verification ===")
    ds_final = LeRobotDataset(args.repo_id, episodes=[0])

    logger.info(
        f"observation.state: shape={ds_final.meta.features['observation.state']['shape']}, "
        f"names={ds_final.meta.features['observation.state']['names']}"
    )
    logger.info(
        f"action: shape={ds_final.meta.features['action']['shape']}, "
        f"names={ds_final.meta.features['action']['names']}"
    )

    sample = ds_final[50]
    state = sample["observation.state"].tolist()
    logger.info("\nSample frame 50:")
    logger.info(f"  observation.state ({state_dim}D):")
    for n, v in zip(state_names, state, strict=True):
        logger.info(f"    {n:12s}: {v:+.6f}")

    action = sample["action"].tolist()
    logger.info("  action (11D):")
    for n, v in zip(ACTION_NAMES, action, strict=True):
        logger.info(f"    {n:8s}: {v:+.6f}")

    if use_relative:
        # Frame 0 of episode should have zero relative pose
        sample0 = ds_final[0]
        state0 = sample0["observation.state"].tolist()
        logger.info("\n  Frame 0 state (should be ~0 for pose dims, nonzero for gripper):")
        for n, v in zip(state_names, state0, strict=True):
            logger.info(f"    {n:12s}: {v:+.6f}")

    pos_delta = np.array(action[:3])
    logger.info(f"\n  Position delta magnitude: {np.linalg.norm(pos_delta) * 1000:.2f} mm")

    logger.info("\nConversion complete!")

    if args.push_to_hub:
        target = args.push_to_hub
        logger.info(f"\nPushing to Hub repo: {target} (private={args.hub_private})")
        ds_push = LeRobotDataset(args.repo_id)
        if target != args.repo_id:
            # Retarget — same approach as the standalone push helper.
            ds_push.repo_id = target
            ds_push.meta.repo_id = target
        ds_push.push_to_hub(private=args.hub_private, push_videos=True)
        logger.info(f"Pushed: {target}")


if __name__ == "__main__":
    main()
