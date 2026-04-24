# Gripette: Quickstart Guide

Step-by-step walkthrough to **train** a Diffusion Policy on a Gripette dataset,
**verify** it offline, and **deploy** it on a real OpenArm + Gripette.

Assumes you've already recorded a dataset with the SLAM-mounted Gripette and
pushed it to the HuggingFace Hub. If you need to record first, skip to the
[`lerobot` recording documentation](https://huggingface.co/docs/lerobot/).

For the design rationale (why 6D rotations, why delta actions, etc.),
see [`README.md`](README.md).

---

## 1. One-time setup

On any machine you'll use for training, inference, or robot control:

```bash
# Clone the repo (skip if already done)
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Install — the `gripette` extra bundles everything:
# diffusion, training, dataset, grpcio==1.73.1, placo, mujoco, PyQt5, matplotlib.
uv sync --locked --extra gripette

# Editable installs for the two out-of-tree packages (not in the lockfile).
# --no-deps is required — they declare opencv-python (GUI) which conflicts
# with LeRobot's opencv-python-headless.
uv pip install -e /path/to/openarm_gripette_model --no-deps
uv pip install -e /path/to/openarm_gripette_simu  --no-deps

# HF login (once, needed for dataset pull and model push).
uv run huggingface-cli login
```

**After every `uv sync`**, re-run the two `uv pip install -e ... --no-deps`
lines. `uv sync --locked` wipes editable installs that aren't in the lockfile.

---

## 2. Prepare the dataset

Convert the raw 8D axis-angle dataset to the 11D format the policy trains on:
axis-angle → 6D rotation, pre-computed delta actions, 2D gripper-only
`observation.state`.

```bash
# Wipe any stale local cache for this dataset (safer).
rm -rf ~/.cache/huggingface/lerobot/hub/datasets--<USER>--<DATASET_NAME>

uv run python examples/openarm_gripette/convert_dataset.py \
    --repo_id <USER>/<DATASET_NAME> \
    --proprioception none
```

Always use `--proprioception none` (2D `observation.state` = gripper only).
The 11D relative-proprio mode was a dead end in our experiments — the policy
learns to ignore the image and replay trajectories by state alone.

Verify the conversion worked:

```bash
uv run python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('<USER>/<DATASET_NAME>')
print('Action names:', ds.features['action'].get('names'))
print('State names:',  ds.features['observation.state'].get('names'))
print('FPS:', ds.meta.fps, '  Episodes:', ds.meta.total_episodes)
"
```

You want: 11 action names starting with `dx, dy, dz, dr6d_0..5`, 2 state names
(`proximal, distal`).

---

## 3. Train

Stable, proven command — works on most machines:

```bash
uv run python examples/openarm_gripette/train.py \
    --dataset_repo_id <USER>/<DATASET_NAME> \
    --output_dir outputs/gripette/<RUN_NAME> \
    --training_steps 200000 \
    --batch_size 32 \
    --num_workers 4 \
    --eval_freq 500 --save_freq 5000 \
    --wandb_project gripette --wandb_run_name "<RUN_NAME>" \
    --push_to_hub <USER>/<MODEL_NAME>
```

Notes:

- **~150 episodes of demonstrations is a working minimum.** Fewer than ~50 will
  struggle regardless of training time.
- **200 k steps × batch 32 ≈ 6.4 M sample-steps.** On a typical dataset
  (~15 k frames), that's roughly 400 epochs — enough for convergence on a
  single-task policy.
- **Wandb is recommended.** Watch `val_loss` — training is done when it
  plateaus or starts ticking up. The `--push_to_hub` flag auto-pushes
  `/final/` and `/best/` at the end.
- **Do not enable `--bf16` / `--compile` / `--color_jitter` until you've
  confirmed the base command works** on this machine. They're power-user
  flags that can silently OOM or fail.

---

## 4. Verify the model offline

Before touching the robot, confirm the model has actually learned the training
data:

```bash
uv run python examples/openarm_gripette/offline_replay.py \
    --checkpoint outputs/gripette/<RUN_NAME>/best \
    --dataset_repo_id <USER>/<DATASET_NAME> \
    --num_episodes 5
```

Target numbers:

| Metric       | Healthy | Undertrained | Collapsed |
| ------------ | ------- | ------------ | --------- |
| Position MAE | < 1 mm  | 2-10 mm      | > 10 mm   |
| Gripper MAE  | < 0.01  | 0.02-0.05    | > 0.1     |

If the numbers are bad, train longer or investigate the dataset. If they're
good, also run `probe_model.py` for a sanity check on input-sensitivity:

```bash
uv run python examples/openarm_gripette/probe_model.py \
    --checkpoint outputs/gripette/<RUN_NAME>/best \
    --dataset_repo_id <USER>/<DATASET_NAME>
```

Look for: `max pairwise L2 > 0.1` (model is input-sensitive) and the
cross-image L2 should be substantially higher than gray+zero baseline.

---

## 5. Deploy on the real robot

The deployment has three sides, each on their own machine (or any combination
co-located):

```
 ┌──────────────────┐    ┌─────────────────────────┐   ┌──────────────────┐
 │  Inference PC    │    │   Robot controller PC   │   │  Gripette (Pi)    │
 │  (GPU)           │    │                         │   │                   │
 │                  │    │  grpc_server_real.py    │   │  GripperService   │
 │  eval_simulator  ├───►│  ArmService :50052      │   │  :50051           │
 │      .py         │    │  CAN bus → OpenArm      │   │  camera + motors  │
 │                  │    │                         │   │                   │
 │                  ├──────────────────────────────────►                   │
 └──────────────────┘    └─────────────────────────┘   └──────────────────┘
```

### 5.1 — On the robot controller PC: start the arm server

Check the CAN bus:

```bash
uv run lerobot-setup-can --mode test
```

Start the server. The built-in 50 Hz joint interpolator handles low-level
smoothing — no need to tune unless motion feels jerky.

```bash
uv run python examples/openarm_gripette/grpc_server_real.py \
    --can_port can0 --side right
```

Expected log: `Joint interpolator ON: 50 Hz, alpha=0.30 ...`

### 5.2 — Start the Gripette service (if not running already)

The Gripette ships its own `GripperService` that exposes camera stream +
motor goals on port 50051. No lerobot-side script needed — just ensure it's
running and reachable.

Verify from the inference PC:

```bash
uv run python examples/openarm_gripette/view_camera.py \
    --gripper_addr <GRIPETTE_IP>:50051
```

A Qt window should open showing the live camera feed.

### 5.3 — Home the arm (carefully)

If there's a table or obstacle in front of the arm, define safe waypoints:

```bash
uv run python examples/openarm_gripette/reset_arm.py \
    --arm_addr <ARM_PC_IP>:50052 \
    --waypoint_deg -10 0 0 0 0 0 0 \
    --waypoint_deg -10 0 0 100 0 0 0 \
    --waypoint_deg 0 0 0 90 0 0 0
```

Otherwise, a single-pose reset via `set_arm_pose.py` is enough:

```bash
uv run python examples/openarm_gripette/set_arm_pose.py \
    --arm_addr <ARM_PC_IP>:50052 --joints_deg 0 0 0 90 0 0 0
```

### 5.4 — First policy run (slow, safe defaults)

```bash
uv run python examples/openarm_gripette/eval_simulator.py \
    --checkpoint <USER>/<MODEL_NAME> \
    --arm_addr <ARM_PC_IP>:50052 \
    --gripper_addr <GRIPETTE_IP>:50051 \
    --device cuda \
    --gripper_async \
    --action_scale 0.5 \
    --duration 30
```

- `--gripper_async` — required. Keeps the control loop at 10 Hz even though
  the Gripette's RPC is slow.
- `--action_scale 0.5` — halves commanded Cartesian speed. Start here, ramp
  up once you trust the behavior.
- `--duration 30` — 30 s is enough for a full grasp attempt.

Watch the log:

- `total` per step should stay < 100 ms (10 Hz).
- `grip` should be 0 ms (async sender).
- `cart` should be 20-50 ms. Sustained 100 ms+ → CAN contention on the robot PC.
- `delta` column shows commanded motion magnitude per step.

### 5.5 — If motion looks wrong

| Symptom                       | Most likely cause                    | Fix                                                    |
| ----------------------------- | ------------------------------------ | ------------------------------------------------------ |
| Arm moves too fast, scary     | Action scale too high                | Drop `--action_scale 0.5 → 0.3`                        |
| Jerky at 10 Hz buzz frequency | Motor interpolator tuning            | Server: `--interp_alpha 0.2` (more smoothing)          |
| Arm drifts / ignores the cube | State-shortcut or bad camera framing | Verify `probe_model.py`, check camera matches training |
| Control loop runs at < 5 Hz   | Gripper RPC blocking                 | Make sure `--gripper_async` is set                     |
| `SendCartesianDelta` timeouts | CAN bus saturation / packet drops    | Lower `--interp_hz 25` on the server                   |

Deeper tuning reference: see `README.md` → "Tuning for Real Hardware".

---

## 6. Clean repeatable loop

Once everything works once, the day-to-day loop is just:

```bash
# 1. (if you changed the dataset) reconvert + retrain
uv run python examples/openarm_gripette/convert_dataset.py --repo_id ... --proprioception none
uv run python examples/openarm_gripette/train.py --dataset_repo_id ... --push_to_hub ...

# 2. verify
uv run python examples/openarm_gripette/offline_replay.py --checkpoint ...

# 3. on the robot
uv run python examples/openarm_gripette/grpc_server_real.py --can_port can0 --side right &
uv run python examples/openarm_gripette/reset_arm.py --arm_addr ...
uv run python examples/openarm_gripette/eval_simulator.py --checkpoint ... --gripper_async --action_scale 0.5
```
