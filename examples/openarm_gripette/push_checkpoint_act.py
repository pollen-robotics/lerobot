"""Push a trained ACT checkpoint (model + processors) to HuggingFace Hub.

ACT counterpart to push_checkpoint.py — identical except for the policy class.

Prerequisites:
  - Logged in to HuggingFace: `huggingface-cli login`
  - Or `HF_TOKEN` environment variable set.

Usage:
  # Push the best checkpoint
  uv run python examples/openarm_gripette/push_checkpoint_act.py \\
      --checkpoint /data1/simsim/trained-policies/gripette_act_v1/best \\
      --repo_id SteveNguyen/gripette_act_v1

  # Push a specific step checkpoint
  uv run python examples/openarm_gripette/push_checkpoint_act.py \\
      --checkpoint /data1/simsim/trained-policies/gripette_act_v1/checkpoint_020000 \\
      --repo_id SteveNguyen/gripette_act_v1_step20k

  # Private repo
  uv run python examples/openarm_gripette/push_checkpoint_act.py \\
      --checkpoint /data1/simsim/trained-policies/gripette_act_v1/best \\
      --repo_id SteveNguyen/gripette_act_v1 --private
"""

import argparse
from pathlib import Path

from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTPolicy


def parse_args():
    p = argparse.ArgumentParser(description="Push a trained ACT checkpoint to HuggingFace Hub")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to local checkpoint directory")
    p.add_argument("--repo_id", type=str, required=True, help="HuggingFace Hub repo ID (e.g. 'user/name')")
    p.add_argument("--private", action="store_true", help="Make the repo private (default: public)")
    return p.parse_args()


def main():
    args = parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_path}")

    print(f"Loading checkpoint from {ckpt_path}")
    policy = ACTPolicy.from_pretrained(ckpt_path)
    preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=ckpt_path)

    print(f"Pushing to HuggingFace Hub: {args.repo_id} (private={args.private})")
    policy.push_to_hub(args.repo_id, private=args.private)
    preprocessor.push_to_hub(args.repo_id, private=args.private)
    postprocessor.push_to_hub(args.repo_id, private=args.private)

    print(f"\nDone. View at: https://huggingface.co/{args.repo_id}")
    print("\nTo use on another machine:")
    print(f"  --checkpoint {args.repo_id}")


if __name__ == "__main__":
    main()
