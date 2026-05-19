# OpenArm Gripette: Diffusion Policy with SLAM-Recorded Demonstrations

> **Want to just train and deploy a model?** See [**GUIDE.md**](GUIDE.md) — a
> linear, copy-paste walkthrough from setup to real-robot evaluation.
> This README is the reference for the _how_ and _why_ (design decisions,
> architecture, tuning levers); GUIDE.md is the _what to run_.

Train a **Diffusion Policy** on demonstrations recorded with a hand-held SLAM device,
and deploy it on a **Pollen OpenArm** robot (or its MuJoCo simulator).

A human records demonstrations by moving a hand-held gripper equipped with a SLAM
tracker, a camera, and a 2-DOF gripper mechanism. No robot is involved during data
collection. The trained policy is then deployed via a gRPC interface that handles
FK/IK internally — the same interface works for both the simulator and the real robot.

### Why "Gripette"?

"Grip" + "-ette" = small gripper — the hand-held device used for data collection.

## How It Works

### Diffusion Policy

Diffusion Policy (Chi et al., RSS 2023) models the distribution of action trajectories
using a denoising diffusion probabilistic model. Instead of predicting a single action,
it generates a full trajectory of future actions and executes a subset before re-planning
(action chunking + receding horizon control).

```
Camera image (224x224) ──► ResNet18 ──► SpatialSoftmax ──► Linear ──┐
                                                                      ├── concat ──► global_cond
Robot state (2D or 11D) ──────────────────────────────────────────────┘               │
                                                                                      ▼
                                                             1D U-Net (FiLM-conditioned)
                                                             noise ──► denoised action chunk
                                                             (batch, horizon=16, action_dim=11)
```

### Data Collection: SLAM Device (No Robot)

The recording setup captures:

| Signal                   | Source            | Format                                                    |
| ------------------------ | ----------------- | --------------------------------------------------------- |
| End-effector position    | SLAM tracker      | `(x, y, z)` in meters, Z-up, gravity-aligned              |
| End-effector orientation | SLAM tracker      | Axis-angle rotation vector (converted to 6D for training) |
| Gripper camera           | Camera on gripper | RGB frames at 50 FPS                                      |
| Gripper joints           | Gripper mechanism | 2 joint angles (proximal, distal) in radians              |

The SLAM reference frame has an arbitrary origin but is gravity-aligned. Since the
policy uses **delta actions**, the unknown origin doesn't matter — only the direction
and magnitude of movements are learned.

> **Recorded pose = camera SE(3) pose**, not Quest controller pose. The Grabette
> hardware records the Quest controller's pose; an offline calibration step
> (`grabette-data/scripts/batch_transform_quest.py` using
> `config/quest_to_camera_calibration.json`) applies the rigid controller→camera
> transform so the dataset's `observation.pose` is the *camera* frame. Skipping
> that step is the most common cause of a model that approaches the cube but
> misses by a consistent rotation/offset. See `GUIDE_REAL.md` step 1.

### Frame Convention (the most important section)

Every position/rotation delta in the entire pipeline — dataset, training,
sim eval, real eval, the cartesian smoke tests — lives in the **camera's local
frame at time t**, never in the world frame.

Concretely, given a per-frame camera pose `(pos[t], R[t])` in the recording's
Z-up world frame, the dataset's 11D action is built as:

```python
delta_pos_world = pos[t+1] - pos[t]
action[t, :3]   = R[t].T @ delta_pos_world          # camera-LOCAL position delta
R_delta         = R[t].T @ R[t+1]                   # camera-LOCAL rotation delta
action[t, 3:9]  = rotation_matrix_to_rotation_6d_numpy(R_delta)
action[t, 9:]   = gripper_joints[t+1]               # gripper is absolute
```

At deployment, the gRPC `SendCartesianDelta` integrator on the server side
(`arm_servicer.py` in sim, `grpc_server_real.py` on real) inverts this with
the **integrator's current target** `(_target_pos, _target_r6d)`:

```python
R_target = rotation_6d_to_matrix(self._target_r6d)
self._target_pos = self._target_pos + R_target @ delta_pos_local   # back to world
self._target_r6d = rotation_matrix_to_6d(R_target @ R_delta_local)
# IK to the new (target_pos, target_r6d) and command the arm joints.
```

Why this matters: the SLAM world frame has an arbitrary horizontal yaw at each
recording session (and the robot's world frame has yet another orientation).
World-frame deltas point in different physical directions across sessions, so
a model trained on world deltas appears to work in sim (single fixed yaw) but
sends the real arm "the wrong way". Camera-local deltas are session-invariant:
they describe motion *relative to what the camera sees*, which is consistent
across sessions by construction.

**Failure signature** if this convention is violated anywhere in the chain:
the arm moves in a consistent but visibly wrong direction on real hardware
while passing sim eval at the same metrics. Use `cartesian_square.py` (the
gRPC-client version in this directory) as the canonical end-to-end smoke
test — its docstring lists what scene motion you should see on the camera
feed for each edge.

> **Two scripts with the same name**. There is also a
> `examples/cartesian_square.py` inside `openarm_gripette_simu/` which spawns
> its **own** standalone MuJoCo simulation and does NOT connect to any gRPC
> server. Don't confuse it with the one here — only the version in *this*
> directory tests the deployment delta convention.

### State and Action Space

**Actions** are 11-dimensional (pre-computed as deltas in the dataset):

```
Index:  0   1   2   3       4       5       6       7       8       9         10
Name:   dx  dy  dz  dr6d_0  dr6d_1  dr6d_2  dr6d_3  dr6d_4  dr6d_5  proximal  distal
        |──────── position + rotation deltas ────────|      |── absolute gripper ──|
```

**Observation state** is configurable via `--proprioception`:

- **`none`** (default, 2D): `[proximal, distal]` — gripper joints only. The model
  sees only the camera image for spatial information. Simplest approach.
- **`relative`** (11D, UMI-style): `[dx_start, dy_start, dz_start, r6d_rel_0..5, proximal, distal]` —
  position and rotation relative to the episode start. The model knows how far it has
  moved from the beginning of the episode. Frame-independent proprioception.

> **Important**: Absolute position is NEVER fed to the model. It's meaningless in the
> SLAM reference frame. Spatial awareness comes from the camera and (optionally) from
> relative-to-start proprioception.

#### Why 6D rotation?

The 6D representation (Zhou et al., CVPR 2019) encodes a rotation matrix as its first
two columns. The third column is recovered via Gram-Schmidt orthogonalization. Benefits
over axis-angle or quaternions:

- **Continuous**: No singularities (axis-angle wraps near π, quaternions have double cover).
- **Better gradients**: Smoother loss landscape for neural network training.
- **Bounded values**: Rotation matrix columns have unit norm — normalization stats
  stay in ~[-1, 1] instead of spanning [-π, +π] for axis-angle.

Conversion utilities are in `lerobot.utils.rotation`:
`rotation_matrix_to_rotation_6d_numpy`, `rotation_6d_to_rotation_matrix_numpy`, etc.

### Deployment via gRPC

At deployment, the client (our scripts) talks to a gRPC server (simulator or real robot).
The server handles FK/IK internally — the client only sends/receives Cartesian deltas.

```
          Our inference scripts                    gRPC Server (sim or robot)
          ─────────────────────                    ──────────────────────────
                                                   ┌───────────────────────────┐
  Policy predicts:                                 │ Read joint encoders       │
   [dx, dy, dz, dr6d_0..5,           GetArmState   │ FK ──► Cartesian EE pose  │
    proximal, distal]               ◄────────────  │                           │
           │                                       │                           │
           ▼                                       │                           │
  SendCartesianDelta(                              │ IK ──► target joint angles│
      dx, dy, dz, dr6d)   ────────────────────────►│ Send to motors            │
  SendMotorCommand(                                │                           │
      proximal, distal)   ────────────────────────►│ Set gripper motors        │
                                                   └───────────────────────────┘
```

This same API works for:

- **Simulator**: [openarm_gripette_simu](https://github.com/...) — MuJoCo-based, used
  for fast iteration.
- **Real robot**: Pollen OpenArm with Damiao motors over CAN bus — same protocol,
  just a different server implementation.

## Directory Contents

```
examples/openarm_gripette/
    README.md              # This file

    # --- Dataset & training ---
    convert_dataset.py     # Transform dataset: axis-angle → 6D, compute deltas, choose proprioception
    convert_rotation_6d.py # Utility: rotation representation conversion
    train.py               # Training script (validation split + wandb + HF Hub push)
    push_checkpoint.py     # Push an already-trained checkpoint to HuggingFace Hub

    # --- Diagnostics (no policy / no GPU required) ---
    offline_replay.py      # Replay recorded episodes through the policy, compare predictions vs GT
    read_arm_state.py      # Print the live arm state (joints + EE pose) from ArmService
    view_camera.py         # Display the Gripette camera feed in an OpenCV window
    cartesian_square.py    # gRPC-client smoke test — square in the CAMERA-LOCAL frame
                           # (canonical test of the deployment delta convention).
                           # Flags: --plane {yz,xy}, --tiny, --half_size, --fps,
                           # --log_gripper_frame (adds gripper-tip FK + camera roll).
    cartesian_sinusoid.py  # Clean sinusoidal Cartesian motion — isolates pipeline vs policy jerk
    set_arm_pose.py        # Smoothly move the arm to a specified single joint configuration
    set_gripper_pose.py    # Send goal positions to the Gripette's 2-DOF gripper motors;
                           # supports --open / --close presets, rad or deg, --torque on/off
    reset_arm.py           # Multi-waypoint safe reset (visits joint configs in sequence)

    # --- One-time-per-arm calibration ---
    calibrate_arm_no_gripper.py  # OpenArm motor-zero calibration WITHOUT the gripper motor.
                                 # Run with `/usr/bin/python3.14` (the system Python that has
                                 # `openarm_can`), NOT via `uv run`.

    # --- Closed-loop evaluation (simulator or real robot) ---
    eval_simulator.py      # Continuous inference via gRPC — sim defaults
    evaluate.py            # Episode-based evaluation with reset + success tracking
    eval_on_robot.py       # Continuous inference via gRPC — real-robot defaults
                           # (slim wrapper that imports helpers from eval_simulator.py;
                           # identical control path, stricter safety defaults)

    # --- Real robot deployment ---
    grpc_server_real.py    # gRPC server driving a real OpenArm via CAN — same API as simulator.
                           # Server-side safety: max_relative_target (per-step joint clamp),
                           # IK-jump watchdog (rejects singularity-driven branch flips),
                           # interpolator alpha (motor-command smoothing). See GUIDE_REAL.md
                           # §5.5b for the full safety-knob table.
```

## Prerequisites

### Software

Two commands. `uv sync --locked --extra gripette` bundles everything LeRobot
needs for this workflow (diffusion, training, dataset, grpcio==1.73.1,
kinematics/placo, openarms/CAN, mujoco). The two editable installs add the
out-of-tree simulator and model packages, which are not in the lockfile.

```bash
# 1. LeRobot side (includes wandb, matplotlib, grpcio, placo, mujoco, etc.)
uv sync --locked --extra gripette

# 2. Out-of-tree editable installs. --no-deps keeps the simulator's
#    `opencv-python` (GUI variant) from clobbering LeRobot's
#    `opencv-python-headless`. All its other deps are already in the
#    `gripette` extra at pinned versions, so --no-deps leaves nothing missing.
uv pip install -e /path/to/openarm_gripette_model --no-deps
uv pip install -e /path/to/openarm_gripette_simu  --no-deps
```

**Repeat steps 2 and 3 after every `uv sync --locked`** — uv enforces the
lockfile exactly and wipes editable installs not declared in it. The commands
are idempotent, so re-running them after a sync takes a second.

**If you see a grpcio version mismatch** when running any eval script, the
simulator's gRPC stubs were generated against a different grpcio version.
Regenerate them:

```bash
cd /path/to/openarm_gripette_simu
uv run --with "grpcio-tools==1.73.1" python -m grpc_tools.protoc \
    -I proto --python_out=openarm_gripette_simu/proto \
    --grpc_python_out=openarm_gripette_simu/proto \
    proto/arm.proto proto/gripper.proto
sed -i 's/^import arm_pb2 as/from . import arm_pb2 as/' openarm_gripette_simu/proto/arm_pb2_grpc.py
sed -i 's/^import gripper_pb2 as/from . import gripper_pb2 as/' openarm_gripette_simu/proto/gripper_pb2_grpc.py
```

### Hardware (for real robot only)

- Pollen OpenArm with Damiao motors, CAN interface
- Gripper-mounted USB camera
- OpenArm URDF file

### Dataset

A LeRobot dataset in the expected format (before conversion):

- `observation.images.cam0`: video frames, e.g. `(3, 720, 960)`
- `action`: 8D raw poses `[x, y, z, ax, ay, az, proximal, distal]` (axis-angle rotation)

After conversion by `convert_dataset.py`:

- `observation.images.cam0`: unchanged
- `observation.state`: 2D `[proximal, distal]` OR 11D `[dx_start, ..., proximal, distal]`
- `action`: 11D `[dx, dy, dz, dr6d_0..5, proximal, distal]` (pre-computed deltas)

## Step-by-Step Guide

### Step 1: Convert the Dataset

Choose one of two proprioception modes:

**Simple mode** (default): observation.state = gripper only, the model relies entirely
on the camera for spatial info.

```bash
uv run python examples/openarm_gripette/convert_dataset.py \
    --repo_id <YOUR_DATASET_REPO_ID> \
    --proprioception none
```

**Relative proprioception** (UMI-style): observation.state includes position and
rotation relative to the episode start.

```bash
uv run python examples/openarm_gripette/convert_dataset.py \
    --repo_id <YOUR_DATASET_REPO_ID> \
    --proprioception relative
```

Both modes:

- Convert rotation from axis-angle (3D) to 6D continuous representation
- Compute delta actions (pose\[t+1\] - pose\[t\] for position and rotation, absolute for gripper)
- Add `observation.state` column (if missing)
- Recompute normalization statistics

> **Note**: The conversion modifies the dataset in-place (in your local HF cache). To
> switch between modes, delete the cached dataset first:
>
> ```bash
> rm -rf ~/.cache/huggingface/lerobot/hub/datasets--<user>--<name>
> ```

### Step 2: Train the Policy

```bash
uv run python examples/openarm_gripette/train.py \
    --dataset_repo_id <YOUR_DATASET_REPO_ID> \
    --output_dir outputs/gripette/run_001 \
    --training_steps 5000 \
    --batch_size 32 \
    --eval_freq 200 \
    --save_freq 500 \
    --wandb_project gripette \
    --wandb_run_name "baseline"
```

Features:

- **Train/val split by episodes**: last 10% held out (`--val_ratio 0.1`). Validation
  loss computed every `--eval_freq` steps.
- **Best checkpoint**: automatically saved to `<output_dir>/best/` when val_loss improves.
- **Periodic checkpoints**: saved every `--save_freq` steps to `checkpoint_NNNNNN/`.
- **Wandb logging**: train_loss, val_loss, best_val_loss curves. Enabled with
  `--wandb_project`.
- **Auto-detects proprioception mode** from the converted dataset.

Key CLI arguments:

| Argument           | Default | Meaning                                                                 |
| ------------------ | ------- | ----------------------------------------------------------------------- |
| `--training_steps` | 200000  | Total training steps                                                    |
| `--batch_size`     | 64      | Batch size                                                              |
| `--n_action_steps` | 8       | Actions executed before re-planning (try 4 for reactive, 16 for smooth) |
| `--eval_freq`      | 200     | Validation eval every N steps                                           |
| `--save_freq`      | 10000   | Checkpoint save every N steps                                           |
| `--val_ratio`      | 0.1     | Fraction of episodes held out for validation                            |
| `--cameras`        | cam0    | Which cameras to use (others filtered out)                              |
| `--color_jitter`   | off     | Enable color jitter augmentation (UMI values) — training-only           |
| `--wandb_project`  | None    | Wandb project name (None = disabled)                                    |
| `--push_to_hub`    | None    | HF Hub repo ID — auto-pushes final + best checkpoints after training    |
| `--hub_private`    | off     | Make the pushed repo private                                            |

Data augmentation (always on):

- **Random crop** (95% ratio): images are resized to 236x236 and randomly cropped to
  224x224 during training; center-cropped at inference. Matches UMI.

Opt-in via `--color_jitter`:

- **Color jitter**: random brightness / contrast / saturation / hue perturbations on
  training images (UMI values: 0.3/0.4/0.5/0.08). Not applied at inference.

Training output:

```
step:       0 / 5000  train_loss: 1.1920
step:     100 / 5000  train_loss: 0.3142
step:     200 / 5000  train_loss: 0.0843  val_loss: 0.0912  best_val: 0.0912 *
step:     400 / 5000  train_loss: 0.0195  val_loss: 0.0523  best_val: 0.0523 *
step:     600 / 5000  train_loss: 0.0098  val_loss: 0.0687  best_val: 0.0523
                                                              ^ val going up = overfitting
```

When val_loss stops improving, use `--checkpoint <output_dir>/best` for deployment.

### Step 3: Diagnostic — Offline Replay

Before touching any robot, verify the model learned the training distribution:

```bash
uv run python examples/openarm_gripette/offline_replay.py \
    --checkpoint outputs/gripette/run_001/best \
    --dataset_repo_id <YOUR_DATASET_REPO_ID> \
    --num_episodes 5
```

This feeds recorded observations through the policy and compares predicted actions
against ground truth. Outputs:

- Per-dimension MAE
- Plots saved to `outputs/gripette/replay/`: position / rotation / gripper,
  predicted vs ground truth over time

Interpretation:

- **Good tracking** → the model learned the training data. Test closed-loop next.
- **Poor tracking** → underfitting (more training) or data issue (check dataset).
- **Perfect tracking (MAE << 1mm)** → overfitting. Use an earlier checkpoint.

### Step 4: Closed-Loop Evaluation (Simulator)

Launch the simulator in another terminal:

```bash
python -m openarm_gripette_simu
```

Then run evaluation. The eval scripts auto-detect the proprioception mode from the
checkpoint — no flag needed.

**Episode-based evaluation** with environment reset and success detection:

```bash
uv run python examples/openarm_gripette/evaluate.py \
    --checkpoint outputs/gripette/run_001/best \
    --num_episodes 20 \
    --max_steps 300
```

Per episode: simulator resets (randomized arm pose + cube position), the policy runs
for up to `--max_steps` steps, and success is detected when the cube moves more than
3mm. Prints a summary with success rate.

**Continuous inference** (single long run, no reset):

```bash
uv run python examples/openarm_gripette/eval_simulator.py \
    --checkpoint outputs/gripette/run_001/best \
    --duration 30 \
    --debug
```

The `--debug` flag shows the camera feed and logs detailed state/action values. The
`--no_send` flag lets you inspect predictions without moving the robot.

### Step 5: Deploy on Real Robot

There are two ways to run the policy on the real robot:

#### Option A (recommended): two gRPC servers — arm + Gripette

The Gripette has **its own gRPC service** exposing the exact same API as
`gripper.proto` (camera stream + gripper motors). We complement it with a small
gRPC server that controls the arm via CAN and exposes the same `ArmService` API
as the simulator. The eval client connects to both endpoints.

```
┌──────────────────┐         ┌─────────────────────────────┐
│  Inference PC    │         │       Robot controller PC    │
│  (GPU)           │         │                              │
│                  │         │  grpc_server_real.py         │
│  eval_simulator  │─ArmSvc─►│  (ArmService on port 50052) │
│      .py         │         │       │                      │
│                  │         │       ▼ CAN bus              │
│                  │         │  [ OpenArm 7-DOF arm ]       │
│                  │         │                              │
│                  │         │  (Gripette controller)       │
│                  │─Grip───►│  GripperService on port X    │
│                  │         │  Camera + 2-DOF gripper      │
└──────────────────┘         └─────────────────────────────┘
```

```bash
# On the robot controller machine: start the arm-only gRPC server.
# The server has a built-in 50 Hz joint-space setpoint interpolator
# (see "Tuning for Real Hardware" below).
uv run python examples/openarm_gripette/grpc_server_real.py \
    --can_port can0 --side right --arm_port 50052

# The Gripette's gRPC service is already running (on <gripette-ip>:<gripette-port>)
# — it ships with the Gripette.

# On the inference machine, home the arm first. Use reset_arm.py with a
# multi-waypoint preset if a direct move would collide with the table.
uv run python examples/openarm_gripette/reset_arm.py \
    --arm_addr <robot-ip>:50052 --preset home_right_over_table

# Or, for a single-hop reset to a known-safe pose:
uv run python examples/openarm_gripette/set_arm_pose.py \
    --arm_addr <robot-ip>:50052 \
    --joints_deg 0 0 0 90 0 0 0

# Run the policy. For a first trial: slow it down and use the async gripper
# sender (see "Tuning" for what these do).
uv run python examples/openarm_gripette/eval_simulator.py \
    --checkpoint outputs/gripette/run_001/best \
    --arm_addr <robot-ip>:50052 \
    --gripper_addr <gripette-ip>:<gripette-port> \
    --device cuda --gripper_async --action_scale 0.5 \
    --duration 30
```

Why this is clean:

- Same client code for sim and real — no risk of "it worked in sim but the real-robot
  script has a subtle bug".
- Arm server uses the same `Kinematics` class and URDF as the simulator for bit-for-bit
  FK/IK compatibility.
- The Gripette already speaks the exact gripper API — no translation needed.
- Decouples GPU (for inference) from the robot controller.
- `set_arm_pose.py` uses the `Reset` RPC (smooth interpolation) to home the arm before runs.

#### Option B: `eval_on_robot.py` — same gRPC architecture, real-robot defaults

`eval_on_robot.py` is the canonical entry point for real-robot evaluation. It
is structurally identical to `eval_simulator.py` (sibling import of the camera
reader, async gripper sender, and `compute_relative_state` helpers) and uses
the exact same `SendCartesianDelta` → server integrator path. The only
differences are:

- `--action_scale 0.5`, `--ood_delta_mm 8.0`, `--duration 20.0` as defaults.
- `--arm_addr` and `--gripper_addr` are required (no `localhost` default).
- `--gripper_async` is on by default.
- A safety banner + 1 s delay before the loop starts.

```bash
uv run python examples/openarm_gripette/eval_on_robot.py \
    --checkpoint outputs/gripette/run_001/best \
    --arm_addr <robot-ip>:50052 \
    --gripper_addr <gripette-ip>:50051 \
    --duration 20
```

> Historical note: an earlier version of this script ran local `placo` FK/IK
> and treated the policy's 9D Cartesian output as an *absolute* SE(3) target.
> That bypassed the integrator and was the root cause of poor real-robot
> grasps even when sim eval looked correct (the dataset's actions are
> *camera-local deltas*, not absolute poses — see "Frame Convention" above).
> The current script is delta-based and goes through `grpc_server_real.py`
> exactly like `eval_simulator.py`.

### Sharing Models

To share a trained model across machines, push it to the HuggingFace Hub:

```bash
# Automatic push after training
uv run python examples/openarm_gripette/train.py \
    --dataset_repo_id <DATASET> \
    --push_to_hub SteveNguyen/gripette_v1 \
    ...

# Or push an already-trained checkpoint
uv run python examples/openarm_gripette/push_checkpoint.py \
    --checkpoint outputs/gripette/run_001/best \
    --repo_id SteveNguyen/gripette_v1
```

On the target machine, use the Hub repo ID as the checkpoint:

```bash
uv run python examples/openarm_gripette/eval_simulator.py \
    --checkpoint SteveNguyen/gripette_v1 \
    --arm_addr <robot-ip>:50052 ...
```

`DiffusionPolicy.from_pretrained()` handles Hub paths transparently — no code changes
needed. Requires `huggingface-cli login` once on the target machine (or `HF_TOKEN`
environment variable).

## Configuration Reference

### Current DiffusionConfig

| Parameter              | Value                | Notes                                                 |
| ---------------------- | -------------------- | ----------------------------------------------------- |
| `n_obs_steps`          | 2                    | Past observation frames used as input                 |
| `horizon`              | 16                   | Total action trajectory length predicted              |
| `n_action_steps`       | 8 (CLI configurable) | Actions executed before re-planning                   |
| `vision_backbone`      | `resnet18`           | With GroupNorm (not BatchNorm)                        |
| `resize_shape`         | `(236, 236)`         | Resize before random crop (UMI pattern)               |
| `crop_ratio`           | 0.95                 | Random crop to 224x224 during training, centered eval |
| `down_dims`            | `(256, 512, 1024)`   | U-Net channels per stage (matches UMI)                |
| `noise_scheduler_type` | `DDIM`               | Fast inference                                        |
| `num_train_timesteps`  | 50                   | Training denoising steps                              |
| `num_inference_steps`  | 16                   | Inference denoising steps                             |
| `optimizer_lr`         | `3e-4`               | Learning rate (matches UMI)                           |
| `scheduler_warmup`     | 2000                 | Cosine warmup steps (matches UMI)                     |

See `docs/umi_analysis.md` for a detailed comparison with the UMI reference
implementation.

## Tuning for Real Hardware

Closed-loop policy execution on the real arm introduces several loop-timing and
smoothness issues that don't exist in simulation. The scripts ship with a set of
CLI knobs for addressing each one independently.

### Server-side: IK-jump watchdog (`grpc_server_real.py`)

Catches singularity-driven branch flips in Placo's IK that the other safety
layers can't. Compares each new IK solution to the previous accepted one;
if any joint would change by more than `--max_ik_jump_deg` in a single step,
rejects the Cartesian command, rolls back the integrator, and (after
`--max_ik_jump_violations` consecutive rejections) disables the interpolator
so the arm freezes. This was added after a wrist-explosion event during
model eval where the model commanded the arm near a wrist singularity.

```
--max_ik_jump_deg 10 --max_ik_jump_violations 1   # tight (first-run safety)
--max_ik_jump_deg 20 --max_ik_jump_violations 2   # cruise (trusted models)
--max_ik_jump_deg 0                                # disabled (not recommended)
```

Trip log line: `IK-jump watchdog tripped: joint '<name>' would change by X° in one step`.
On a latch (after N consecutive violations), the integrator re-syncs to the
current FK pose and motors stop. Restart the server and re-home before
proceeding.

### Server-side: joint-space setpoint interpolator (`grpc_server_real.py`)

Policy commands arrive at ~10 Hz but MIT motor gains are stiff — without
smoothing, the motors see a 10 Hz staircase and the arm feels jerky. The server
runs a background thread that drives the motors at `--interp_hz` (default 50 Hz)
with an exponential approach toward the latest IK solution:

```
# Defaults are usually fine; lower alpha for more smoothing + more lag.
uv run python examples/openarm_gripette/grpc_server_real.py \
    --can_port can0 --side right \
    --interp_hz 50 --interp_alpha 0.3
```

| Flag             | Default | Effect                                                                                                                         |
| ---------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `--interp_hz`    | 50      | Interpolator tick rate. Higher = smoother motor input.                                                                         |
| `--interp_alpha` | 0.3     | Per-tick exponential approach rate. Lag ≈ 1 / (`alpha` · `interp_hz`) ≈ 67 ms at defaults. 0.1 = heavy smoothing, 0.5 = light. |
| `--kp_scale`     | 1.0     | Multiplier on MIT position_kp. `0.5` halves all kp (softer tracking). Usually unnecessary once the interpolator is in use.     |
| `--kd_scale`     | 1.0     | Multiplier on MIT position_kd. Scale ~`sqrt(kp_scale)` to preserve damping.                                                    |

Use `cartesian_sinusoid.py` to tune the interpolator independently of the policy:
it sends a mathematically smooth sinusoid through the pipeline, so any remaining
jerk is pure server/motor/IK. See its `--help` for options.

### Client-side: speed control (`eval_simulator.py` / `evaluate.py`)

```
# Safe-first defaults for the real robot.
... --gripper_async --action_scale 0.5
```

| Flag             | Default | Effect                                                                                                                             |
| ---------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `--action_scale` | 1.0 (sim) / 0.5 (real) | Multiplies Cartesian Δ-position. **Caveat:** also multiplies the 6D rotation Δ, but Gram-Schmidt normalizes that server-side, so the rotation Δ is *not* actually slowed. Use the server-side `--max_ik_jump_deg` watchdog for rotation safety. Gripper unchanged. |
| `--fps`          | 10      | Control loop rate. Lower = slower motion + slower observation rate. Prefer `--action_scale` for position-speed control.            |

### Client-side: decoupling the Gripette RPC (`eval_simulator.py`)

The Gripette's `SendMotorCommand` RPC can take 100 ms – several seconds on the
Pi Zero 2W under load (camera streaming contends for CPU). Sending it
synchronously in the control loop stalls the arm:

```
... --gripper_async
```

A background thread takes the latest goal and fires the RPC whenever the server
can accept one; intermediate goals are dropped (last-wins). The control loop
runs at 10 Hz regardless of Gripette latency.

### Client-side: policy output smoothing (`eval_simulator.py`)

The server-side interpolator smooths the _pipeline_ jerk. These client-side
knobs smooth the _policy's own output noise_ before it reaches the server —
often not needed once the interpolator is tuned, but available:

| Flag                    | Default   | Effect                                                                                                                         |
| ----------------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `--temporal_ensemble`   | off       | ACT-style: predict every step, weighted-average across overlapping chunks for each timestep. Bypasses `select_action`'s queue. |
| `--temporal_ensemble_k` | 0.1       | Weight decay for temporal ensembling (`w_i = exp(-k · age)`). `0.02` = near-uniform averaging (very smooth); `0.5` = light.    |
| `--delta_ema_alpha`     | off       | EMA low-pass on Cartesian deltas. `0.3` = moderate, `0.15` = heavy. Does _not_ touch the gripper (absolute, not a delta).      |
| `--n_action_steps`      | from ckpt | Override the checkpoint's value at inference time. `1` = re-infer every step (eliminates chunk-boundary jumps).                |

### Diagnostic workflow

When motion looks wrong, in this order:

1. **`cartesian_sinusoid.py`** — clean input, isolates the pipeline. If jerky
   here, it's server/motors. If smooth here, the pipeline is fine.
2. **`read_arm_state.py`** and **`view_camera.py`** — sanity-check the
   observation side before accusing the policy.
3. **`offline_replay.py`** — compare policy predictions vs dataset ground
   truth. Rules out a model-quality issue before tuning loops.
4. Only then: tune the policy run (`--action_scale`, `--temporal_ensemble`,
   `--delta_ema_alpha`).

## Troubleshooting

### "grpcio version mismatch" when running eval_simulator.py or evaluate.py

The simulator's generated gRPC stubs check the grpcio version. LeRobot pins grpcio
1.73.1. Regenerate the stubs with the matching version — see the installation section
above.

### Simulator state is out of the dataset range

Since we no longer feed absolute position to the model, this shouldn't happen. If you
see out-of-range warnings in `eval_simulator.py --debug`, it's only for the gripper
joints or the relative-proprioception state — both should be consistent between
training and deployment.

### Loss decreases quickly but simulator behavior is poor

Most likely causes:

1. **Overfitting** — try an earlier checkpoint (e.g., `checkpoint_000500/` instead of the final one).
2. **Data is not diverse enough** — 500 similar episodes may not cover the state space.
   Try varying starting positions, approach angles, and lighting.
3. **FPS mismatch** — the training data and the eval loop must run at the same
   effective FPS, otherwise delta magnitudes are wrong by a constant factor.

### Policy produces very slow motion

Likely a **FPS scaling issue**: if data was recorded at 50 FPS but eval runs at 10 Hz,
each delta covers 100ms instead of 20ms — effectively 5x slower. Check that
`eval_simulator.py --fps` matches the dataset's recording FPS.

### Policy moves too fast on the real arm (scary)

Use `--action_scale 0.5` on the client to halve commanded Cartesian velocity while
keeping the observation rate at the trained FPS. Start at `0.3-0.5` for initial
safety trials. See "Tuning for Real Hardware" → Client-side: speed control.

### Real-robot control loop runs at 1-4 Hz (should be 10 Hz) with SLOW tags everywhere

Enable `--debug`'s per-step breakdown in `eval_simulator.py` and check which phase
is the bottleneck (`cam`, `getstate`, `infer`, `cart`, `grip`). The usual culprit
is `grip` — the Gripette's `SendMotorCommand` can block the loop for hundreds of
ms. Fix: `--gripper_async`. See "Tuning for Real Hardware" → Client-side: decoupling
the Gripette RPC.

### Arm motion feels jerky / buzzy at the policy rate

First run `cartesian_sinusoid.py` through the same server to confirm the jerk
is or isn't in the pipeline:

```bash
uv run python examples/openarm_gripette/cartesian_sinusoid.py \
    --arm_addr <robot-ip>:50052
```

If the sinusoid is jerky, it's a server-side issue. The fix is the built-in
joint-space setpoint interpolator — already on by default. Tune with
`--interp_alpha` on `grpc_server_real.py` (lower = smoother). See "Tuning for
Real Hardware".

If the sinusoid is smooth but the policy is jerky, it's policy-output noise.
Try `--delta_ema_alpha 0.3` or `--temporal_ensemble --temporal_ensemble_k 0.05`
on the client.

### Reset to home collides with workspace (e.g., table)

The server's single-hop `Reset` does a straight-line joint interpolation from
the current pose to the target, which can cross obstacles. Use
`reset_arm.py` instead to visit a sequence of safe intermediate waypoints:

```bash
uv run python examples/openarm_gripette/reset_arm.py \
    --arm_addr <robot-ip>:50052 --preset home_right_over_table
```

Edit the `PRESETS` dict in the script to match your actual table geometry.

### Mode averaging (policy moves toward a "preferred" position regardless of target)

- **Symptom**: same predicted trajectory regardless of cube position.
- **Cause**: the camera signal isn't discriminative enough, or the model is
  underfitting.
- **Try**: more training steps, more diverse data, or add relative proprioception
  (`--proprioception relative`).

## Design Decisions

### Why pre-compute deltas in the dataset (no RelativeActionsProcessorStep)?

- Simpler pipeline: no processor state to cache.
- No absolute position ever reaches the model — the dataset's state column IS the model's input.
- Matches UMI's approach.
- The alternative (RelativeActionsProcessorStep at training time) doesn't work well
  with multi-step observations because the processor caches the full state internally.

### Why gripper absolute while Cartesian is delta?

Gripper represents a physical open/close state, not a trajectory in space. "Open more"
or "close more" (delta) is less intuitive than "go to this opening" (absolute). The
dataset conversion keeps gripper absolute.

## References

- **Diffusion Policy**: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via
  Action Diffusion", RSS 2023. [Paper](https://diffusion-policy.cs.columbia.edu/)
- **UMI**: Chi et al., "Universal Manipulation Interface: In-The-Wild Robot Teaching
  Without In-The-Wild Robots", CoRL 2024.
- **6D Rotation**: Zhou et al., "On the Continuity of Rotation Representations in
  Neural Networks", CVPR 2019.
- **LeRobot**: [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
- **OpenArm**: [docs.openarm.dev](https://docs.openarm.dev/)
