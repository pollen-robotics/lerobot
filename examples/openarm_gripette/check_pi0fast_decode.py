"""Quick offline decode check for a Pi0Fast checkpoint (any step).

Loads a checkpoint, runs one held-out sample through the full
preprocess -> select_action -> postprocess path, and compares the decoded
action against ground truth. Use it to confirm the model's FAST decode
produces sane physical actions BEFORE committing to a sim/real eval — e.g.
on a mid-training checkpoint to verify the custom-tokenizer retrain is on
track without waiting for the full run.

Healthy result: RAW ~ small (±a few), FINAL ~ ground truth (few-mm Δpos,
gripper near 0 or the closed values). Degenerate result (the v1 bug): RAW
near-constant ~±90, FINAL blown up to ~0.1 m / ~-90 gripper.

Runs on a free GPU or CPU so it doesn't disturb a training job on the main
GPU. Examples:
  # mid-training checkpoint on a second GPU
  CUDA_VISIBLE_DEVICES=1 uv run python examples/openarm_gripette/check_pi0fast_decode.py \
      --checkpoint outputs/gripette/pi0fast_sim_v2/checkpoints/005000/pretrained_model \
      --dataset SteveNguyen/sim_grasp_eval_v8

  # or on CPU (slow but won't touch the training GPU)
  uv run python examples/openarm_gripette/check_pi0fast_decode.py \
      --checkpoint outputs/gripette/pi0fast_sim_v2/checkpoints/005000/pretrained_model \
      --dataset SteveNguyen/sim_grasp_eval_v8 --device cpu
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.factory import get_policy_class


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Local dir or Hub repo id")
    p.add_argument("--dataset", required=True, help="Dataset to sample a held-out frame from")
    p.add_argument("--device", default="cuda", help="cuda / cuda:N / cpu")
    p.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help="float32 (default; correct for ACT/Diffusion) or bfloat16 "
        "(use to fit the 2.3B Pi0Fast on a small GPU).",
    )
    p.add_argument("--frames", type=int, nargs="+", default=[0, 80, 160], help="Frame indices to test")
    p.add_argument("--no_kv_cache", action="store_true",
                   help="Pi0Fast diagnostic: force the non-KV-cache decode path "
                        "(isolates a transformers-v5 KV-cache generation bug).")
    p.add_argument("--task", default="grasp and lift cube", help="Task string for VLA conditioning")
    return p.parse_args()


def _policy_type(ckpt: str) -> str:
    cfg = Path(ckpt) / "config.json"
    if cfg.is_file():
        return json.loads(cfg.read_text())["type"]
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(ckpt, "config.json")).read_text())["type"]


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    policy = get_policy_class(_policy_type(args.checkpoint)).from_pretrained(args.checkpoint)
    if args.no_kv_cache and hasattr(policy.config, "use_kv_cache"):
        # Diagnostic: force the non-KV-cache autoregressive decode path.
        # The KV-cache path is the most transformers-v5-fragile (attention
        # masking + cache semantics changed in v5); if generation is
        # degenerate only with the cache, this isolates it.
        policy.config.use_kv_cache = False
        print("use_kv_cache forced to False")
    policy = policy.to(device=device, dtype=dtype).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.checkpoint)

    ds = LeRobotDataset(args.dataset)
    for i in args.frames:
        s = ds[i]
        gt = s["action"].numpy()
        batch = {
            "observation.state": s["observation.state"].unsqueeze(0).to(device),
            "observation.images.cam0": s["observation.images.cam0"].unsqueeze(0).to(device),
            "task": s.get("task", args.task),
        }
        policy.reset()
        with torch.no_grad():
            raw = policy.select_action(pre(batch))
        fin = post(raw).squeeze(0).cpu().float().numpy()
        raw = raw.squeeze(0).cpu().float().numpy()
        print(f"\nframe {i}:")
        print(f"  RAW   pos {np.round(raw[:3], 3)}  grip {np.round(raw[9:], 3)}")
        print(f"  FINAL pos(mm) {np.round(fin[:3] * 1000, 1)}  grip {np.round(fin[9:], 3)}")
        print(f"  GT    pos(mm) {np.round(gt[:3] * 1000, 1)}  grip {np.round(gt[9:], 3)}")


if __name__ == "__main__":
    main()
