# OpenArm Gripette: Diffusion Policy with SLAM-Recorded Demonstrations

## Overview

This example shows how to train and deploy a **Diffusion Policy** on a **Pollen OpenArm**
robot using demonstrations recorded with a hand-mounted SLAM device (no robot in the loop
during data collection).

The key idea: a human demonstrates tasks by moving a hand-held device equipped with a
SLAM tracker, a gripper camera, and a gripper mechanism. The SLAM device records Cartesian
end-effector poses. A Diffusion Policy is then trained on these demonstrations and deployed
on the OpenArm robot, which uses forward/inverse kinematics to bridge between Cartesian
policy outputs and joint-space motor commands.

### Why "Gripette"?

The project name refers to the hand-held gripper device used for data collection
("grip" + "-ette" = small gripper).

## How It Works

### The Diffusion Policy

Diffusion Policy (Chi et al., 2023) learns a visuomotor policy by modeling the
distribution of action trajectories using a denoising diffusion probabilistic model
(DDPM). Instead of predicting a single action, it generates a full trajectory of future
actions, then executes a subset before re-planning.

Key properties:

- **Multimodal**: Can represent multiple valid strategies for the same observation.
- **Action chunking**: Predicts 16 future steps, executes 8, providing temporal
  consistency.
- **Receding horizon**: Re-plans every 8 steps with fresh observations.

Architecture:

```
Camera image -----> ResNet18 --> SpatialSoftmax(32 keypoints) --> Linear --+
                                                                          |-- concat --> global_cond
Robot state (8D) ---------------------------------------------------------+
                                                                          |
                                                                          v
                                                    1D U-Net (FiLM-conditioned)
                                                    noise --> denoised action trajectory
                                                    (batch, horizon=16, action_dim=8)
```

### Data Collection: SLAM Device (No Robot)

Data is recorded using a hand-mounted SLAM device. The recording setup captures:

| Signal                   | Source                | Format                                                 |
| ------------------------ | --------------------- | ------------------------------------------------------ |
| End-effector position    | SLAM tracker          | `(x, y, z)` in meters, Z-up, gravity-aligned           |
| End-effector orientation | SLAM tracker          | Quaternion (converted to rotation vector for training) |
| Gripper camera           | Camera on the gripper | RGB frames at 50 FPS                                   |
| Gripper joints           | Gripper mechanism     | 2 joint angles in degrees                              |

The SLAM reference frame has an arbitrary origin but is gravity-aligned with Z pointing
up. Since the policy uses **relative/delta actions**, the unknown origin does not matter
-- only the direction and magnitude of movements are learned.

### State and Action Space

The policy operates in an 8-dimensional state/action space:

```
Index:   0    1    2    3    4    5    6       7
Name:    x    y    z    rx   ry   rz   grip_1  grip_2
         |-------- relative --------|  |-- absolute --|
```

- **Dims 0-2**: Cartesian position (meters). Converted to deltas during training.
- **Dims 3-5**: Orientation as rotation vector (converted from quaternion). Converted to
  deltas during training.
- **Dims 6-7**: Gripper joint angles (degrees). Stay absolute -- they represent physical
  open/close state, not a trajectory in space.

### Relative (Delta) Actions

Actions in the dataset are stored as **absolute positions**. At training time, the
`RelativeActionsProcessorStep` converts them to deltas:

```
delta_action[t] = action[t] - state[t]    (for Cartesian + orientation dims)
```

Gripper joints are excluded from this conversion via the `relative_exclude_joints`
config. This is handled automatically by the processor pipeline and reversed at inference
time by the `AbsoluteActionsProcessorStep`.

Why deltas?

- The SLAM device records in an arbitrary reference frame with unknown origin.
- Delta actions are **origin-invariant** -- only direction and magnitude matter.
- The Diffusion Policy paper found that delta actions improve generalization.

### Deployment: FK/IK on the OpenArm

At inference time, the robot needs to translate between joint space (what the motors
understand) and Cartesian space (what the policy expects). This uses LeRobot's built-in
kinematics infrastructure:

```
                              Deployment Pipeline

     OpenArm joints                                          OpenArm joints
          |                                                       ^
          v                                                       |
    FK (placo + URDF)                                   IK (placo + URDF)
          |                                                       ^
          v                                                       |
    Cartesian EE pose                                    Cartesian EE target
    + gripper joints                                     + gripper target
    + camera image                                                ^
          |                                                       |
          v                                                       |
     Preprocessor                                          Postprocessor
     (normalize,                                           (unnormalize,
      delta actions)                                        absolute actions)
          |                                                       ^
          v                                                       |
     DiffusionPolicy.select_action()  --->  delta actions --------+
```

The kinematics use the `placo` library with the OpenArm URDF model. The
`RobotKinematics` wrapper in `lerobot.model.kinematics` provides:

- `forward_kinematics(joint_angles_deg)` -- returns 4x4 SE(3) matrix
- `inverse_kinematics(current_joints, target_pose)` -- returns joint angles

## Directory Contents

```
examples/openarm_gripette/
    README.md                  # This file
    train.py                   # Training script
    eval_on_robot.py           # Deployment script (inference on real OpenArm)
```

## Prerequisites

### Software

```bash
# Base LeRobot installation
uv sync --locked

# Required extras for diffusion policy + kinematics
uv sync --locked --extra diffusion --extra kinematics

# For training with GPU
uv sync --locked --extra training
```

### Hardware (for deployment only)

- Pollen OpenArm robot with Damiao motors
- CAN interface (socketcan on Linux)
- Gripper-mounted USB camera
- OpenArm URDF file (with correct EE frame name)

### Dataset

A LeRobot dataset recorded with the SLAM device, containing:

- `observation.state`: shape `(8,)`, names `["x", "y", "z", "rx", "ry", "rz", "grip_1", "grip_2"]`
- `observation.images.gripper`: video frames `(720, 960, 3)`
- `action`: shape `(8,)`, same names as state

> **Note**: Adjust the feature names throughout the scripts if your dataset uses different
> names. The critical requirement is that `relative_exclude_joints` matches the exact
> gripper joint names in your dataset.

## Step-by-Step Guide

### Step 1: Recompute Dataset Statistics for Relative Actions

The dataset stores absolute positions, but the model trains on deltas. The normalization
statistics must be computed in the delta space so that MIN_MAX normalization is correct.

```bash
uv run lerobot-edit-dataset \
    --repo-id <YOUR_DATASET_REPO_ID> \
    --operation.type recompute_stats \
    --operation.relative_action true \
    --operation.relative_exclude_joints "['grip_1', 'grip_2']" \
    --operation.chunk_size 16
```

**Sanity check**: After recomputing, inspect `stats.json`:

- Action min/max for dims 0-5 should be small values (millimeter-scale per-frame deltas
  at 50 FPS).
- Action min/max for dims 6-7 (gripper) should span the full gripper range (absolute).
- State min/max should span the full workspace (absolute positions).

### Step 2: Train the Policy

```bash
uv run python examples/openarm_gripette/train.py
```

Or with custom parameters:

```bash
uv run python examples/openarm_gripette/train.py \
    --dataset_repo_id pollen/gripette_pickplace \
    --output_dir outputs/gripette/diffusion_v1 \
    --batch_size 64 \
    --training_steps 200000 \
    --device cuda
```

The script:

1. Loads the dataset and configures temporal windowing for action chunking.
2. Creates a `DiffusionConfig` with `use_relative_actions=True`.
3. Builds the pre/post processor pipeline (including `RelativeActionsProcessorStep`).
4. Trains the policy with cosine LR schedule and periodic checkpointing.
5. Saves the final checkpoint (model + processors).

**What to expect**:

- Loss should decrease steadily over the first few thousand steps.
- With 50-100 demos, expect ~200K steps to converge.
- Training takes ~2-4 hours on a single GPU (RTX 3090 / A100).

You can also use the main LeRobot training CLI directly:

```bash
uv run lerobot-train \
    --dataset.repo_id=<YOUR_DATASET_REPO_ID> \
    --policy.type=diffusion \
    --policy.use_relative_actions=true \
    --policy.relative_exclude_joints="['grip_1', 'grip_2']" \
    --policy.vision_backbone=resnet18 \
    --policy.resize_shape="[240, 320]" \
    --policy.down_dims="[256, 512, 1024]" \
    --batch_size=64 \
    --steps=200000
```

### Step 3: Deploy on the Robot

```bash
uv run python examples/openarm_gripette/eval_on_robot.py \
    --checkpoint outputs/gripette/diffusion_v1 \
    --urdf path/to/openarm.urdf \
    --can_port can0 \
    --side right \
    --duration 30
```

The script:

1. Loads the trained policy and processors from the checkpoint.
2. Initializes `RobotKinematics` with the OpenArm URDF for FK/IK.
3. Connects to the robot and camera.
4. Runs the closed-loop control at 50 FPS:
   - FK: joint angles --> Cartesian EE pose
   - Policy: image + state --> delta actions (via action queue)
   - Postprocess: deltas --> absolute Cartesian target
   - IK: Cartesian target --> joint angles
   - Send to motors

## Configuration Reference

### DiffusionConfig Key Parameters

| Parameter                 | Default                | Description                                 |
| ------------------------- | ---------------------- | ------------------------------------------- |
| `n_obs_steps`             | 2                      | Number of past observations to condition on |
| `horizon`                 | 16                     | Total action trajectory length predicted    |
| `n_action_steps`          | 8                      | Actions executed before re-planning         |
| `vision_backbone`         | `resnet18`             | Image encoder backbone                      |
| `resize_shape`            | `(240, 320)`           | Resize camera frames before encoding        |
| `crop_ratio`              | 0.9                    | Random crop ratio (training augmentation)   |
| `down_dims`               | `(256, 512, 1024)`     | U-Net channel dimensions per stage          |
| `num_train_timesteps`     | 100                    | Diffusion denoising steps                   |
| `noise_scheduler_type`    | `DDPM`                 | Scheduler type (`DDPM` or `DDIM`)           |
| `use_relative_actions`    | `True`                 | Convert Cartesian actions to deltas         |
| `relative_exclude_joints` | `["grip_1", "grip_2"]` | Joints to keep absolute                     |
| `optimizer_lr`            | `1e-4`                 | Learning rate                               |

### Inference Speed

| Setting                | Inference time per chunk | Notes                                                       |
| ---------------------- | ------------------------ | ----------------------------------------------------------- |
| DDPM, 100 steps, GPU   | ~100-200 ms              | Default, highest quality                                    |
| DDIM, 10 steps, GPU    | ~15-30 ms                | Set `noise_scheduler_type="DDIM"`, `num_inference_steps=10` |
| + `compile_model=True` | ~10-20 ms                | PyTorch 2.0+ torch.compile                                  |

At 50 FPS with `n_action_steps=8`, the policy runs every 160 ms. DDPM fits within this
budget on GPU. For CPU inference, use DDIM with fewer steps.

## Troubleshooting

### "Action stats look wrong after recomputing"

Make sure `relative_exclude_joints` names match exactly what's in your dataset's
`action.names` field. Check with:

```bash
uv run lerobot-edit-dataset --repo-id <DATASET> --operation.type info --operation.show_features true
```

### "Robot moves in the wrong direction"

The SLAM frame and robot FK frame may have different horizontal axis conventions
(both are Z-up, but X/Y might be swapped or rotated). Set `R_FK_TO_SLAM` in
`eval_on_robot.py` to the appropriate rotation matrix. Verify by:

1. Moving the robot along its +X axis.
2. Checking if the FK delta matches what SLAM would report as +X.

### "IK fails or produces wild joint configurations"

- Reduce `orientation_weight` (default 0.01) to prioritize position over orientation.
- Check that the Cartesian target is within the robot's reachable workspace.
- Ensure the URDF joint limits match the physical robot.

### "Policy outputs constant/zero actions"

- Verify dataset stats were recomputed with `--relative_action true`.
- Check that the loss converged during training (should be < 0.01 after 100K steps).
- Try more demonstrations (50-100 is a minimum for simple tasks).

## References

- **Diffusion Policy**: Chi et al., "Diffusion Policy: Visuomotor Policy Learning via
  Action Diffusion", RSS 2023. [Paper](https://diffusion-policy.cs.columbia.edu/)
- **LeRobot**: [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
- **OpenArm**: [docs.openarm.dev](https://docs.openarm.dev/)
