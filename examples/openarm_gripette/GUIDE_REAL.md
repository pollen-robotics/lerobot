# Gripette: Real-Hardware Pipeline Guide

End-to-end workflow for **recording demonstrations on the real Grabette,
training a diffusion policy, and deploying it on the real OpenArm**.

For the simulation pipeline (used for end-to-end pipeline certification
without hardware), see [`GUIDE_SIM.md`](GUIDE_SIM.md).

For design rationale, see [`README.md`](README.md).

---

## Prerequisites

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
uv sync --locked --extra gripette

# Editable installs for the out-of-tree packages.
uv pip install -e /path/to/openarm_gripette_model --no-deps
uv pip install -e /path/to/openarm_gripette_simu  --no-deps

uv run huggingface-cli login
```

Re-run the two `uv pip install -e ... --no-deps` lines after every
`uv sync`.

---

## 1. Record demonstrations

This is the new step relative to sim — the dataset comes from physical
recordings, not from a scripted collector. Quality of recording dominates
final policy quality at this scale.

### Demo recipe

**Quantity**: 200-500 demos. Start with 200, train, evaluate, add more if
needed. UMI-style tasks at this scale converge well; more becomes
diminishing returns past 500.

**Episode-type mix** — same structure as the sim certification dataset:

- ~70% **normal**: approach → grasp → lift.
- ~15% **release**: gripper *starts closed*; you open it during the first
  ~0.3 s, then proceed normally. Teaches the model closed→open transitions.
- ~15% **hover**: approach the object to ~grasp distance but **don't close**;
  hover for ~1 s, then retract with the gripper open. Critical negative
  example. Without it, the policy commits to closing as soon as the object
  is roughly centered in view.

### What to vary across demos

- **Object position**: 8-10 distinct spots across the working area
  (~15 × 20 cm). Don't cluster.
- **Object orientation**: rotate ±20° between demos. Forces the visual
  encoder to learn shape, not orientation-locked silhouette.
- **Starting Grabette pose**: vary by ~10 cm in each axis. Don't always
  start from the same hand position. The frame-0 visual anchors the
  policy's initial action.
- **Approach trajectory shape**: mix straight descents, diagonal
  approaches, slight arcs. Smooth motions only.
- **Lighting** (if your space allows): record across different times of
  day or with lights on/off occasionally. Real lighting variation makes
  artificial DR mostly redundant.

### Motion quality

- **Smooth, deliberate motion**. Don't jerk the Grabette around — the
  model learns from your trajectory and replays the style.
- **Consistent grasp timing**. Close the gripper at a clear visual moment
  (e.g., when the V-pocket is around the object). Don't pre-empt; don't lag.
- **Don't recover from failed attempts within a single demo**. If you
  miss, stop and start a new demo. Recovery actions in a single recording
  confuse the action distribution.
- **5-7 s per demo at 50 fps**. Faster = fewer frames; slower = the model
  commands smaller-than-expected deltas at deployment.

### Reachability constraint

The biggest sim-to-real gap is the OpenArm's limited workspace. Hand-held
demos can go anywhere; the arm can't.

- Define a **physical demo box** that fits the arm's reach (roughly:
  object area within ~30 cm of base in front, gripper homes within ~20 cm
  of object). Tape on the table to mark the region; only place the object
  there, only let the gripper enter that volume.
- After recording, run the IK feasibility filter
  (`openarm_gripette_simu.ik_feasibility`) against the recorded camera
  poses. Drop episodes with > 10% unreachable frames before training.

### Pose convention

Recorded poses must be **camera-site SE(3) in a Z-up gravity-aligned world
frame** — what the iPhone SLAM gives you. Confirm before bulk recording.
Capture 5 demos and verify:

- `action[:, 2]` (dz) trends negative during approach phase.
- `action[:, 9:11]` (gripper) goes from open (~0) to closed (~-1.5, -2.1)
  exactly once per normal episode.
- No huge spikes (would indicate SLAM tracking loss).

### Validation cadence

Don't record all 500 in one go. Iterate:

1. Record 30 demos.
2. Convert + train a small run (5k steps).
3. Inspect predictions on a held-out demo.
4. If the model behaves reasonably, scale to 200.
5. Eval on the real arm. If it works, push to 500.

---

## 2. Convert the dataset

Same `convert_dataset.py` as the sim path. Produces 11D delta actions and
2D gripper-only state column.

```bash
uv run python examples/openarm_gripette/convert_dataset.py --repo_id <USER>/<DATASET_NAME> --proprioception none --push_to_hub <USER>/<DATASET_NAME> --hub_private
```

Drop `--push_to_hub` for local-only.

Always use `--proprioception none`. The 11D relative-proprio mode was
explored and abandoned — the policy learned to ignore the image and replay
trajectories from state alone.

Verify:

```bash
uv run python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; ds = LeRobotDataset('<USER>/<DATASET_NAME>'); print('action:', ds[0]['action'].shape, 'state:', ds[0]['observation.state'].shape, 'episodes:', ds.meta.total_episodes, 'fps:', ds.meta.fps)"
```

Expect `action: torch.Size([11])`, `state: torch.Size([2])`.

---

## 3. Train

Same command as sim, swap dataset id:

```bash
uv run python examples/openarm_gripette/train.py --dataset_repo_id <USER>/<DATASET_NAME> --output_dir outputs/gripette/<RUN_NAME> --training_steps 50000 --batch_size 64 --bf16 --num_workers 2 --eval_freq 500 --save_freq 5000 --wandb_project gripette --wandb_run_name <RUN_NAME> --push_to_hub <USER>/<MODEL_NAME> --color_jitter --state_noise_std 0.01
```

Notes:

- **50k steps × batch 64 × bf16** is the validated sweet spot. Trust the
  val_loss curve; task success can keep improving past apparent plateau.
- **`--bf16`** is safe and gives ~1.5-2x speedup on Ampere+ / Blackwell
  (RTX 30xx/40xx/50xx). Same numbers as fp32.
- **`--color_jitter`** is recommended even with real recordings — the
  jitter reaches feature regions the recordings don't visit.
- **Resume support**: if a run dies, restart with
  `--resume_from outputs/gripette/<RUN_NAME>/checkpoint_010000`. Add
  `--wandb_resume_id <ID>` to keep wandb curves continuous.

---

## 4. Verify the model offline

Before touching the robot, confirm the model has actually learned the data:

```bash
uv run python examples/openarm_gripette/offline_replay.py --checkpoint outputs/gripette/<RUN_NAME>/best --dataset_repo_id <USER>/<DATASET_NAME> --num_episodes 5
```

Target numbers:

| Metric       | Healthy | Undertrained | Collapsed |
| ------------ | ------- | ------------ | --------- |
| Position MAE | < 1 mm  | 2-10 mm      | > 10 mm   |
| Gripper MAE  | < 0.01  | 0.02-0.05    | > 0.1     |

If the numbers are bad, train longer or investigate the dataset. If they're
good, also run `probe_model.py` for a sanity check on input-sensitivity:

```bash
uv run python examples/openarm_gripette/probe_model.py --checkpoint outputs/gripette/<RUN_NAME>/best --dataset_repo_id <USER>/<DATASET_NAME>
```

Look for: `max pairwise L2 > 0.1` (model is input-sensitive) and the
cross-image L2 substantially higher than gray+zero baseline.

You can also run the OOD-loss check if you have a held-out eval split:

```bash
uv run python examples/openarm_gripette/eval_ood_loss.py --checkpoint outputs/gripette/<RUN_NAME>/best --dataset_repo_id <USER>/<EVAL_DATASET> --in_dist_val_loss <VAL_LOSS>
```

---

## 5. Deploy on the real robot

Three sides, each on their own machine (or co-located):

```
 ┌──────────────────┐    ┌─────────────────────────┐   ┌──────────────────┐
 │  Inference PC    │    │   Robot controller PC   │   │  Gripette (Pi)   │
 │  (GPU)           │    │                         │   │                  │
 │                  │    │  grpc_server_real.py    │   │  GripperService  │
 │  eval_simulator  ├───►│  ArmService :50052      │   │  :50051          │
 │      .py         │    │  CAN bus → OpenArm      │   │  camera + motors │
 │                  │    │                         │   │                  │
 │                  ├──────────────────────────────────►                  │
 └──────────────────┘    └─────────────────────────┘   └──────────────────┘
```

### 5.1 — On the robot controller PC: start the arm server

Check the CAN bus:

```bash
uv run lerobot-setup-can --mode test
```

Start the server. The 50 Hz joint interpolator handles low-level smoothing
— no need to tune unless motion feels jerky.

```bash
uv run python examples/openarm_gripette/grpc_server_real.py --can_port can0 --side right
```

Expected log: `Joint interpolator ON: 50 Hz, alpha=0.30 ...`

### 5.2 — Start the Gripette service (if not running already)

The Gripette ships its own `GripperService` exposing camera stream + motor
goals on port 50051. No lerobot-side script needed.

Verify from the inference PC:

```bash
uv run python examples/openarm_gripette/view_camera.py --gripper_addr <GRIPETTE_IP>:50051
```

A Qt window should open showing the live camera feed.

### 5.3 — Home the arm (carefully)

If a table or obstacle is in front of the arm, define safe waypoints:

```bash
uv run python examples/openarm_gripette/reset_arm.py --arm_addr <ARM_PC_IP>:50052 --waypoint_deg -10 0 0 0 0 0 0 --waypoint_deg -10 0 0 100 0 0 0 --waypoint_deg 0 0 0 90 0 0 0
```

Otherwise, a single-pose reset is enough:

```bash
uv run python examples/openarm_gripette/set_arm_pose.py --arm_addr <ARM_PC_IP>:50052 --joints_deg 0 0 0 90 0 0 0
```

### 5.4 — First policy run (slow, safe defaults)

```bash
uv run python examples/openarm_gripette/eval_simulator.py --checkpoint <USER>/<MODEL_NAME> --arm_addr <ARM_PC_IP>:50052 --gripper_addr <GRIPETTE_IP>:50051 --device cuda --gripper_async --action_scale 0.5 --duration 30
```

- `--gripper_async` — required. Keeps the control loop at 10 Hz despite
  the slow Gripette RPC.
- `--action_scale 0.5` — halves commanded Cartesian speed. Start here,
  ramp up once you trust the behavior.
- `--duration 30` — 30 s is enough for a full grasp attempt.

Watch the log:

- `total` per step should stay < 100 ms (10 Hz).
- `grip` should be 0 ms (async sender).
- `cart` should be 20-50 ms. Sustained 100+ ms = CAN contention.
- `delta` column shows commanded motion magnitude per step.

### 5.5 — If motion looks wrong

| Symptom | Most likely cause | Fix |
| --- | --- | --- |
| Arm moves too fast, scary | Action scale too high | Drop `--action_scale 0.5 → 0.3` |
| Jerky at 10 Hz buzz | Motor interpolator tuning | Server: `--interp_alpha 0.2` (more smoothing) |
| Arm drifts / ignores object | State-shortcut or bad camera framing | Verify `probe_model.py`; check camera matches training |
| Control loop runs at < 5 Hz | Gripper RPC blocking | Make sure `--gripper_async` is set |
| `SendCartesianDelta` timeouts | CAN bus saturation | Lower `--interp_hz 25` on the server |
| Closes gripper way above the object | Depth-perception failure (sparse data, no negative examples) | Re-record with more **hover** episodes; larger dataset |
| Doesn't reopen after a missed grasp | Training data lacks closed→open transitions | Re-record with **release** episodes (~15%) |

Deeper tuning reference: see `README.md` → "Tuning for Real Hardware".

---

## 6. Clean repeatable loop

Once everything works once, the day-to-day loop:

```bash
# 1. (if you changed the dataset) reconvert + retrain
uv run python examples/openarm_gripette/convert_dataset.py --repo_id ... --proprioception none --push_to_hub <USER>/<DATASET_NAME>
uv run python examples/openarm_gripette/train.py --dataset_repo_id ... --push_to_hub ...

# 2. verify offline
uv run python examples/openarm_gripette/offline_replay.py --checkpoint ...

# 3. on the robot
uv run python examples/openarm_gripette/grpc_server_real.py --can_port can0 --side right &
uv run python examples/openarm_gripette/reset_arm.py --arm_addr ...
uv run python examples/openarm_gripette/eval_simulator.py --checkpoint ... --gripper_async --action_scale 0.5
```
