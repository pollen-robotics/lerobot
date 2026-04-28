"""Inspect mean per-axis action at key trajectory frames across all episodes.

Used to confirm the trajectory's average behavior matches what we expect.
For our pipeline:
  * frame  0 (initial settle): mean action should be ~0 in all dims.
  * frame  5 (start of approach): small Δpos toward cube, mostly slow.
  * frame 25 (mid-approach): the LARGEST per-frame Δpos. Should be
    cube-directional (negative dz if straight descend; positive dz if
    arc_lift > 0 — the latter is what we're trying to remove).
  * frame 60 (descend): clearly -dz (descending onto cube).
  * frame 120 (lift): clearly +dz (lifting off).

Usage:
    uv run python examples/openarm_gripette/check_action_means.py --repo_id sim_grasp_check_v3
"""

from __future__ import annotations

import argparse

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", type=str, required=True)
    p.add_argument("--frames", type=int, nargs="+",
                   default=[0, 5, 25, 60, 120, 200],
                   help="Frame offsets within each episode to inspect.")
    args = p.parse_args()

    ds = LeRobotDataset(args.repo_id)
    print(f"frames={len(ds)}, episodes={ds.meta.total_episodes}, fps={ds.meta.fps}")
    print()

    # Find the first frame index of each episode
    ep_starts: list[int] = []
    prev = -1
    for i, e in enumerate(ds.hf_dataset["episode_index"]):
        if int(e) != prev:
            ep_starts.append(i)
            prev = int(e)

    print(f"{'frame':>6}  {'mean dx':>9}  {'mean dy':>9}  {'mean dz':>9}  "
          f"{'mean prox':>10}  {'mean dist':>10}  {'n':>4}")
    for f in args.frames:
        acts = []
        for s in ep_starts:
            if s + f < len(ds):
                acts.append(ds[s + f]["action"].numpy())
        if not acts:
            continue
        arr = np.stack(acts)
        print(f"{f:>6}  {arr[:, 0].mean():+9.5f}  {arr[:, 1].mean():+9.5f}  "
              f"{arr[:, 2].mean():+9.5f}  {arr[:, 9].mean():+10.4f}  "
              f"{arr[:, 10].mean():+10.4f}  {len(acts):>4}")


if __name__ == "__main__":
    main()
