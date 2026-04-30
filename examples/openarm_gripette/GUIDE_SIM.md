# Gripette: Simulation Pipeline Guide

End-to-end workflow for **generating a grasp dataset in MuJoCo, training a
diffusion policy, and evaluating it against the simulator**. This is the
pipeline-certification path: any future real-robot failure that doesn't
also occur here points at sim-to-real or hardware, not at the data
pipeline.

For real-hardware deployment, see [`GUIDE_REAL.md`](GUIDE_REAL.md).

For design rationale, see [`README.md`](README.md).

---

## Prerequisites

Two repositories side by side:

- `lerobot/` (this repo) — training, conversion, evaluation.
- `openarm_gripette_simu/` — MuJoCo sim, gRPC server, dataset collector.

```bash
# Clone both
git clone https://github.com/huggingface/lerobot.git
git clone <internal>/openarm_gripette_simu.git

# Install lerobot with the gripette extra (bundles diffusion, training,
# grpcio==1.73.1, placo, mujoco, PyQt5, matplotlib).
cd lerobot
uv sync --locked --extra gripette

# Editable installs for the out-of-tree packages.
# --no-deps required: they declare opencv-python (GUI) which conflicts
# with lerobot's opencv-python-headless.
uv pip install -e ../openarm_gripette_model --no-deps
uv pip install -e ../openarm_gripette_simu  --no-deps

# HF login for dataset/model push.
uv run huggingface-cli login
```

After every `uv sync`, re-run the two `uv pip install -e ... --no-deps` lines.

---

## 1. Generate a dataset

The collector samples episodes from a training distribution (cube position,
home pose, grasp orientation), filters them through the OpenArm's IK
feasibility, and records each successful run as a LeRobotDataset entry.

The wrapper script handles cache wipe, generation, conversion, and optional
push to Hub. **Recommended path**:

```bash
/path/to/openarm_gripette_simu/scripts/gen_grasp_dataset.sh v1 --push <YOUR_HF_USER>
```

This produces and pushes:

- `<USER>/sim_grasp_train_v1` — 200 episodes
- `<USER>/sim_grasp_eval_v1` — 30 episodes from a held-out cube_y strip

Default mix: 70% normal grasps, 15% release (gripper starts closed, opens
during initial settle), 15% hover (approach but don't grasp). The negative
examples are critical — without them the policy learns "see cube → close"
and gets stuck once it commits.

Useful variants:

```bash
# Local-only (no push)
.../scripts/gen_grasp_dataset.sh v1

# Quick smoke test (10 episodes + action-mean diagnostic)
.../scripts/smoke_test_dataset.sh

# Larger dataset
.../scripts/gen_grasp_dataset.sh v1 --train-episodes 500 --push <USER>

# Just the eval split
.../scripts/gen_grasp_dataset.sh v1 --eval-only --push <USER>
```

What the script does internally:

1. `collect_grasp_dataset.py --episodes N --split train` — samples plans,
   filters via `IKFeasibilityChecker`, runs MuJoCo physics, records
   per-frame camera + state + action.
2. `convert_dataset.py --proprioception none` — 8D axis-angle → 11D delta
   actions, builds the 2D `[proximal, distal]` state column.
3. (Optional) Push to Hub via `convert_dataset.py --push_to_hub`.

Sanity-check the dataset before training:

```bash
uv run python examples/openarm_gripette/check_action_means.py --repo_id sim_grasp_train_v1
```

What you want to see:

- Frame 0 mean dz ≈ 0 (initial settle, no motion).
- Frame 25 mean dz ≤ 0 (descending toward cube).
- Frame 200 mean (proximal, distal) ≈ (-1.5, -2.1) for normal episodes,
  blended toward 0 if hover episodes are mixed in.

---

## 2. Train

Working command — verified to converge to a 70%+ grasp success rate at 50k
steps on a v4-style dataset:

```bash
uv run python examples/openarm_gripette/train.py --dataset_repo_id <USER>/sim_grasp_train_v1 --output_dir outputs/gripette/run1 --training_steps 50000 --batch_size 64 --bf16 --num_workers 2 --eval_freq 500 --save_freq 5000 --wandb_project gripette --wandb_run_name run1 --push_to_hub <USER>/gripette_run1 --color_jitter --state_noise_std 0.01
```

Validated knobs:

- `--training_steps 50000` — task usually converges by 30-50k. Don't trust
  early loss plateau; val_loss can keep dropping after train_loss flattens.
- `--batch_size 64 --bf16` — the proven combo. Halve batch to 32 if you
  hit OOM.
- `--num_workers 2` — higher worker counts can OOM the dataloader on
  smaller GPUs because of video decoder buffers. Bump if you have headroom.
- `--color_jitter` — UMI defaults (brightness 0.3, contrast 0.4,
  saturation 0.5, hue 0.08). Helps the visual encoder generalize.
- `--state_noise_std 0.01` — small Gaussian noise on `observation.state`
  during training. Discourages state-only shortcuts.

If a run dies (OOM, signal, etc.), resume from the last checkpoint:

```bash
uv run python examples/openarm_gripette/train.py [...same args...] --resume_from outputs/gripette/run1/checkpoint_010000 --wandb_resume_id <ORIGINAL_WANDB_RUN_ID>
```

Without `--wandb_resume_id` the resume produces a fresh wandb run; with it,
the loss curves continue in the same run.

---

## 3. Offline verification

Before sim eval, sanity-check the model's quantitative fit on the held-out
eval set:

```bash
uv run python examples/openarm_gripette/eval_ood_loss.py --checkpoint outputs/gripette/run1/best --dataset_repo_id <USER>/sim_grasp_eval_v1 --in_dist_val_loss <VAL_LOSS_FROM_WANDB>
```

Verdict thresholds:

- **OOD loss / val_loss < 1.5x** — model generalizes. Proceed to sim eval.
- **1.5-3x** — partial memorization. Worth more demos or stronger DR.
- **> 3x** — memorization confirmed. Don't bother with sim eval; address
  the dataset.

---

## 4. Sim closed-loop eval

Two terminals.

### Terminal 1 — sim server

```bash
cd /path/to/openarm_gripette_simu && uv run python -m openarm_gripette_simu --scene scenes/table_grasp.xml
```

Look for `Gripper initialized to OPEN` (gripper-init fix is live) and
`Viewer launched. Reset shortcuts: R=both, 1=arm only, 2=cube only.`

In-viewer keyboard shortcuts:

- **R** — reset cube + arm to a fresh training-distribution config.
- **1** — reset arm only (around the current cube position).
- **2** — reset cube only.

### Terminal 2 — single-episode debug eval

```bash
cd /path/to/lerobot && uv run python examples/openarm_gripette/eval_simulator.py --checkpoint outputs/gripette/run1/best --duration 30 --debug
```

`--debug` opens the camera window the policy sees and prints the per-step
action vector. Use it to diagnose:

- Δpos magnitudes ≈ 1-3 mm during approach, larger during descent. Tiny
  Δpos with closed gripper at step 0 = policy thinks it's mid-trajectory
  (often a state-shortcut or OOD initial config).
- Predicted gripper drifts only when expected (close phase, lift). Drift
  in approach phase = compounding artifact, usually a dataset issue.

### Terminal 2 — multi-episode quantitative eval

```bash
uv run python examples/openarm_gripette/evaluate.py --checkpoint outputs/gripette/run1/best --num_episodes 30
```

Each episode auto-resets via `arm_stub.Reset()` which now samples from the
training distribution (same as `gen_grasp_dataset.sh`). Output is a
success-rate summary; aim for ≥50% on the OOD eval distribution to
consider the pipeline certified.

---

## 5. Iteration loop

Once the first run works, common adjustments:

```bash
# More episodes / different seed
.../scripts/gen_grasp_dataset.sh v2 --train-episodes 500 --push <USER>

# Different release/hover mix (more negative examples)
.../scripts/gen_grasp_dataset.sh v2 --release-fraction 0.20 --hover-fraction 0.20 --push <USER>

# Continue training a model that didn't fully converge
uv run python examples/openarm_gripette/train.py [...] --resume_from outputs/gripette/run1
```

If you're tracking down a regression, the action-means script and the
debug eval are your two main tools. `check_action_means.py` answers "what
does the dataset say should happen?"; `eval_simulator.py --debug --no_send`
answers "what does the policy think should happen at this step?".

---

## Common failure modes (and what we already learned)

| Symptom | Cause | Fix |
| --- | --- | --- |
| Policy commands closed gripper at step 0 from open state | Compounding action bias from the dataset | Verify `check_action_means.py` shows frame 0 dz/dy/dx all ≈ 0 |
| Arm goes UP then closes regardless of cube position | Dataset's mean approach action is +z | Reduce or remove `arc_lift` in `mid_approach_pose`; raise home_z bound |
| Policy never reopens after a missed grasp | Training data has no closed→open transitions | Add release-type episodes (`--release_fraction 0.15`) |
| Closes too early / above the cube | Visual encoder lacks depth cue from sparse data | Add hover-type episodes (`--hover_fraction 0.15`); more episodes; varying cube colors |
| Sim-eval reset puts arm at OOD pose | `arm_servicer.Reset` using legacy random | Pass `server` ref into `ArmServicer` (now done by default) |
| Gripper stuck closed after manual sim reset (R) | Policy queue carries stale closed-phase context | `--gripper-hold-open-duration 1.5` on sim server (default 0; v4 model doesn't need it) |

---

## What's certified by passing this guide

A successful run end-to-end (≥50% on `evaluate.py`) certifies:

- The 8D → 11D conversion in `convert_dataset.py` is correct.
- The training pipeline (`train.py`) converges to a usable policy on this
  data format.
- The deployment interface (gRPC `ArmService` / `GripperService`) is
  correctly consumed by `eval_simulator.py` / `evaluate.py`.
- The dataset format is what the policy expects.

Future failures on **real hardware** must be explained by sim-to-real
divergence (lighting, dynamics, calibration) or hardware issues — not by a
pipeline bug.
