"""Push a trained checkpoint (model + processors) to HuggingFace Hub.

Use this after training completes if you forgot `--push_to_hub`, or to
re-push a specific checkpoint (e.g. `checkpoint_002000/` instead of `best/`).

Prerequisites:
  - Logged in to HuggingFace: `huggingface-cli login`
  - Or `HF_TOKEN` environment variable set.

Usage:
  # Push the best checkpoint
  uv run python examples/openarm_gripette/push_checkpoint.py \\
      --checkpoint outputs/gripette/test_antoine3/best \\
      --repo_id SteveNguyen/gripette_v3

  # Push a specific step checkpoint
  uv run python examples/openarm_gripette/push_checkpoint.py \\
      --checkpoint outputs/gripette/test_antoine3/checkpoint_002000 \\
      --repo_id SteveNguyen/gripette_v3_step2k

  # Private repo
  uv run python examples/openarm_gripette/push_checkpoint.py \\
      --checkpoint outputs/gripette/test_antoine3/best \\
      --repo_id SteveNguyen/gripette_v3 --private
"""

import argparse
from pathlib import Path

from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionPolicy


def parse_args():
    p = argparse.ArgumentParser(description="Push a trained checkpoint to HuggingFace Hub")
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
    policy = DiffusionPolicy.from_pretrained(ckpt_path)
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
