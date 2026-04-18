# OpenArm Gripette: Diffusion Policy with SLAM-Recorded Demonstrations

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
    convert_dataset.py     # Transform dataset: axis-angle → 6D, compute deltas, choose proprioception
    train.py               # Training script (with validation split + wandb)
    evaluate.py            # Episode-based evaluation on simulator (reset between episodes)
    eval_simulator.py      # Continuous inference on simulator (no reset)
    eval_on_robot.py       # Deployment on real OpenArm (uses placo FK/IK directly)
    offline_replay.py      # Diagnostic: run policy on recorded data, compare to ground truth
```

## Prerequisites

### Software

```bash
# Base LeRobot + diffusion + training extras (includes wandb, matplotlib)
uv sync --locked --extra diffusion --extra training --extra dataset

# For real robot deployment (placo FK/IK)
uv sync --locked --extra kinematics

# For simulator inference: install the simulator package as editable
uv pip install -e /path/to/openarm_gripette_simu

# IMPORTANT: the simulator's gRPC stubs must be generated with a grpcio version
# compatible with LeRobot's pinned version (1.73.1). If you see a version mismatch
# error at runtime, regenerate the stubs:
#   cd /path/to/openarm_gripette_simu
#   uv run --with "grpcio-tools==1.73.1" python -m grpc_tools.protoc \
#       -I proto --python_out=openarm_gripette_simu/proto \
#       --grpc_python_out=openarm_gripette_simu/proto \
#       proto/arm.proto proto/gripper.proto
#   # Then fix relative imports in the generated _grpc.py files:
#   sed -i 's/^import arm_pb2 as/from . import arm_pb2 as/' openarm_gripette_simu/proto/arm_pb2_grpc.py
#   sed -i 's/^import gripper_pb2 as/from . import gripper_pb2 as/' openarm_gripette_simu/proto/gripper_pb2_grpc.py
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
| `--wandb_project`  | None    | Wandb project name (None = disabled)                                    |

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

### Step 3: Test on the Simulator

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

### Step 4: Diagnostic — Offline Replay

To verify the model learned the training distribution without involving the simulator:

```bash
uv run python examples/openarm_gripette/offline_replay.py \
    --checkpoint outputs/gripette/run_001/best \
    --dataset_repo_id <YOUR_DATASET_REPO_ID> \
    --num_episodes 5
```

This feeds recorded observations through the policy and compares predicted actions
against ground truth. Outputs:

- Per-dimension MAE
- Plots saved to `outputs/gripette/replay/`:
  - Position deltas (dx, dy, dz) predicted vs ground truth
  - Rotation deltas (dr6d_0..5) predicted vs ground truth
  - Gripper predicted vs ground truth

If the offline replay shows poor tracking, the model hasn't learned the training data —
more training steps, or review the dataset. If it shows good tracking but simulator
behavior is poor, the issue is distribution shift (visual or state-space gap between
training data and simulator).

### Step 5: Deploy on Real Robot

```bash
uv run python examples/openarm_gripette/eval_on_robot.py \
    --checkpoint outputs/gripette/run_001/best \
    --urdf path/to/openarm.urdf \
    --can_port can0 \
    --side right \
    --duration 30
```

The real-robot script uses `placo` for FK/IK directly (no gRPC). It reads joint
encoders, computes FK, runs the policy, runs IK, and sends joint commands via CAN bus.

**Note**: If the simulator uses the same gRPC protocol as the real robot, you can use
`eval_simulator.py` with the real-robot gRPC endpoint instead — it's the same API.

## Configuration Reference

### Current DiffusionConfig

| Parameter              | Value                | Notes                                      |
| ---------------------- | -------------------- | ------------------------------------------ |
| `n_obs_steps`          | 2                    | Past observation frames used as input      |
| `horizon`              | 16                   | Total action trajectory length predicted   |
| `n_action_steps`       | 8 (CLI configurable) | Actions executed before re-planning        |
| `vision_backbone`      | `resnet18`           | With GroupNorm (not BatchNorm)             |
| `resize_shape`         | `(224, 224)`         | Standard size for pretrained vision models |
| `crop_ratio`           | 1.0                  | No crop — resize only                      |
| `down_dims`            | `(256, 512, 1024)`   | U-Net channels per stage (matches UMI)     |
| `noise_scheduler_type` | `DDIM`               | Fast inference                             |
| `num_train_timesteps`  | 50                   | Training denoising steps                   |
| `num_inference_steps`  | 16                   | Inference denoising steps                  |
| `optimizer_lr`         | `3e-4`               | Learning rate (matches UMI)                |
| `scheduler_warmup`     | 2000                 | Cosine warmup steps (matches UMI)          |

See `docs/umi_analysis.md` for a detailed comparison with the UMI reference
implementation.

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
