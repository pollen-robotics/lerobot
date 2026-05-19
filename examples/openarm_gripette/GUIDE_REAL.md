# Gripette: Real-Hardware Pipeline Guide

End-to-end workflow for **recording demonstrations on the real Grabette,
training a diffusion policy, and deploying it on the real OpenArm**.

For the simulation pipeline (used for end-to-end pipeline certification
without hardware), see [`GUIDE_SIM.md`](GUIDE_SIM.md).

For design rationale, see [`README.md`](README.md).

> **Frame convention (read once, internalize).** Every position/rotation
> delta in this pipeline — in the dataset, in the policy output, and on the
> wire via `SendCartesianDelta` — is in the **camera-local frame at time t**,
> never in the world frame. The dataset's `observation.pose` must be the
> *camera* SE(3) (Z-up world). The Grabette device tracks the **Quest
> controller**, not the camera, so `grabette-data` must apply the
> controller→camera calibration (`config/quest_to_camera_calibration.json`,
> applied by `batch_transform_quest.py` / `transform_quest_trajectory.py`) to
> produce `camera_trajectory.csv`. Skipping that step is the most common
> cause of a model that *almost* works on the real arm — see
> `README.md` → "Frame Convention". After deploying, the one-minute sanity
> check is step **5.4** below.

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
frame**. There are two recording stacks:

- **Camera SLAM** (iPhone, RPi+IMU, OAK-D, …): the SLAM trajectory IS the
  camera-site pose. Use directly.
- **Quest controller** (Grabette quest-branch device): the Quest tracks the
  **controller**, not the camera. The mounting of the controller on the
  Grabette body is a fixed but non-trivial rigid transform. You must apply
  the controller→camera calibration before downstream conversion:
  ```bash
  # In grabette-data/, on the quest branch:
  uv run python scripts/batch_transform_quest.py \
      -i ~/data/dataset \
      -c config/quest_to_camera_calibration.json
  ```
  This writes `camera_trajectory.csv` per episode. If `camera_trajectory.csv`
  does not exist after this step, downstream conversion will silently use
  `r_hand_traj.json` directly and your dataset will be in the *controller*
  frame — that's the regression behind almost-but-not-quite-working real
  deployments.

Confirm before bulk recording. Capture 5 demos and verify:

- `action[:, 2]` (dz) trends negative during approach phase.
- `action[:, 9:11]` (gripper) goes from open (~0) to closed (~-1.5, -2.1)
  exactly once per normal episode.
- No huge spikes (would indicate SLAM tracking loss).
- After conversion: `R[t][:, 2]` (third column of the per-frame rotation
  matrix in the dataset) should point in the direction the camera is
  *looking* at frame t (the optical axis, OpenCV convention). If it points
  somewhere else, the controller→camera calibration was missed or wrong.

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
 │ eval_on_robot.py ├───►│  ArmService :50052      │   │  :50051          │
 │                  │    │  CAN bus → OpenArm      │   │  camera + motors │
 │                  │    │                         │   │                  │
 │                  ├──────────────────────────────────►                  │
 └──────────────────┘    └─────────────────────────┘   └──────────────────┘
```

The arm-side server (`grpc_server_real.py`) runs an integrator that interprets
every incoming `(dx, dy, dz, dr6d)` as a **camera-local** delta, exactly like
the simulator's `arm_servicer.py`. The same policy checkpoint runs against
both — see `README.md` → "Frame Convention" for the math.

### 5.0 — Per-arm prerequisites (once per physical arm)

These are one-time checks per physical OpenArm. Skip them only if you have
done them on this specific arm before. They are **not** needed every session.

**(a) URDF axis signs match physical motors.** The auto-generated URDF/MJCF
historically has wrong-sign axes on some joints (see commit history in
`openarm_gripette_model`). Verify each joint individually against the sim
viewer:

```bash
# Open the sim viewer in one window. In another:
uv run python examples/openarm_gripette/set_arm_pose.py --arm_addr <ARM_PC_IP>:50052 --joints_deg 0 0 0 90 0 0 0
# For each joint i in 1..7, command 30° on that joint alone and compare physical
# vs sim direction. Example for j1:
uv run python examples/openarm_gripette/set_arm_pose.py --arm_addr <ARM_PC_IP>:50052 --joints_deg 30 0 0 90 0 0 0
# Real and sim should rotate the same physical direction.
```

If any joint physically goes the opposite direction from sim, see the same
fix pattern as `9c15ea0` / `f9b0c44` in `openarm_gripette_model` (flip
`axis="0 0 1"` → `axis="0 0 -1"` plus swap-negate the limits, in both URDF
and MJCF). The fix is per-joint; restart the server after each fix.

**(b) Motor zeros are calibrated.** Use the gripper-less calibration script
(the upstream OpenArm tool hangs on the Gripette config):

```bash
sudo ip link set can0 up type can bitrate 1000000
/usr/bin/python3.14 examples/openarm_gripette/calibrate_arm_no_gripper.py \
    --canport can0 --arm-side right_arm
```

After it completes, `set_arm_pose ... --joints_deg 0 0 0 90 0 0 0` should
place the arm at its canonical home pose.

**(c) Gripper sanity check.** Verify the Gripette's motor direction matches
the dataset's open/close convention (this is the one mismatch that *cannot*
be silently absorbed by the FK loop — the gripper goal is absolute, not a
delta):

```bash
uv run python examples/openarm_gripette/set_gripper_pose.py --gripper_addr <GRIPETTE_IP>:50051 --open
# should open the V-pocket fully
uv run python examples/openarm_gripette/set_gripper_pose.py --gripper_addr <GRIPETTE_IP>:50051 --close
# should close the V-pocket on whatever is in it
```

If `--open` closes and `--close` opens, the Gripette firmware (or wiring)
has the gripper motors sign-flipped relative to the dataset. Fix on the
Gripette side before any policy run.

### 5.1 — On the robot controller PC: start the arm server

Check the CAN bus:

```bash
uv run lerobot-setup-can --mode test
```

Start the server. Default safety: 50 Hz interpolator + `max_relative_target=8°/step`
+ IK-jump watchdog at 15°. For a *first* run on an untrusted policy, tighten
these (see "Safety layers" below):

```bash
# Default (well-tested, ramp-up motion):
uv run python examples/openarm_gripette/grpc_server_real.py --can_port can0 --side right

# First run on an untrusted policy — tighter safety:
uv run python examples/openarm_gripette/grpc_server_real.py \
    --can_port can0 --side right \
    --max_ik_jump_deg 10 --max_ik_jump_violations 1 \
    --interp_alpha 0.15
```

Expected log: `Joint interpolator ON: 50 Hz, alpha=0.30 ...`. If the IK-jump
watchdog trips during operation, you'll see:
```
IK-jump watchdog tripped: joint 'r_wrist_yaw' would change by +35.2° in one step
```
That's the server refusing a Cartesian command because Placo's IK output
jumped (typically a singularity branch flip). The integrator is rolled back
and the arm freezes. Re-home before any further commands.

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

### 5.4 — Cartesian-square smoke test (do this before any policy run)

Before sending policy actions to a real arm, verify the deployment delta
convention end-to-end. `cartesian_square.py` traces a 10 cm square *in the
camera-local frame* with the orientation locked (it sends identity `R_delta`
every step), so what you observe on the camera feed unambiguously tells you
whether the integrator is consuming the deltas correctly.

```bash
# From the inference machine, with grpc_server_real.py already running on
# the robot side (5.1) and the Gripette serving its camera (5.2):
uv run python examples/openarm_gripette/cartesian_square.py \
    --arm_addr <ARM_PC_IP>:50052 \
    --gripper_addr <GRIPETTE_IP>:50051 \
    --show_camera \
    --plane yz \
    --loops 1 \
    --log_gripper_frame
```

`--log_gripper_frame` adds per-step columns for the gripper-tip world position
(computed client-side via FK) and the camera roll angle around the optical
axis. The script also prints a per-edge "position error" (cumulative target
vs measured FK), so the closed-loop tracking is visible directly:

```
loop 0 step  200/800: EE [+0.287, -0.242, +0.526] expect [+0.288, -0.244, +0.524]
  err=  3.1mm gripper [+0.355, -0.238, +0.452]
  optical [+1.00, -0.02, -0.08] img_right [-0.02, -1.00, -0.03] roll=  +0.6°
  joints (deg): j1=-1.3 j2=+25.3 j3=+2.9 j4=+83.0 j5=+26.5 j6=+0.2 j7=+4.2
```

A healthy run: peak `err` under 5 mm, peak `|roll|` under 2°, optical axis
constant, no joints flagged near a limit. A short summary line at the end:

```
Position-error summary: peak 4.7 mm, final 2.7 mm.
Camera-roll summary:    peak |roll| = 1.0° around the optical axis.
```

There are also size presets if you want to be conservative on a new arm:

```bash
# A 2 cm square (extremely safe) — for first-ever motion check.
uv run python examples/openarm_gripette/cartesian_square.py \
    --arm_addr ... --plane xy --tiny --loops 1

# A 10 cm square (the "real" test). Trips workspace limits and singular
# regions if your URDF / home pose is off.
uv run python examples/openarm_gripette/cartesian_square.py \
    --arm_addr ... --plane xy --half_size 0.05 --fps 50 --loops 1
```

What you should see on the camera feed, in order (defaults, `--plane yz`):

| Edge | Local delta | Camera motion | Scene motion in image |
| --- | --- | --- | --- |
| 1 | `Δ = (0, 0, -step)` | backwards along the optical axis | zoom **OUT** |
| 2 | `Δ = (0, -step, 0)` | along image-up (camera goes up) | scroll **DOWN** |
| 3 | `Δ = (0, 0, +step)` | forwards along the optical axis | zoom **IN** |
| 4 | `Δ = (0, +step, 0)` | along image-down (camera goes down) | scroll **UP** |

If you see this, the camera-local integrator is wired correctly and the policy
will deploy with the same frame the dataset was trained on. If you see
world-frame motion instead (e.g. the arm always slides in the same horizontal
direction regardless of where it is pointing), the integrator has reverted to
world-frame deltas or the URDF's camera-site definition is wrong — fix it
before running a policy.

> Don't confuse this script with `openarm_gripette_simu/examples/cartesian_square.py`
> in the simulator repo, which spawns its own MuJoCo simulation and ignores
> any gRPC server.

### 5.5 — First policy run (slow, safe defaults)

```bash
# Pair with the tight-safety server from 5.1.
uv run python examples/openarm_gripette/eval_on_robot.py \
    --checkpoint <USER>/<MODEL_NAME> \
    --arm_addr <ARM_PC_IP>:50052 \
    --gripper_addr <GRIPETTE_IP>:50051 \
    --device cuda \
    --action_scale 0.5 \
    --ood_delta_mm 5.0 --ood_halt_count 2 \
    --duration 20
```

`eval_on_robot.py` is the real-robot entry point. It is structurally
identical to `eval_simulator.py` (same `SendCartesianDelta` path, same
`CameraStreamReader` and async gripper sender, imported as a sibling module)
with stricter defaults for safety. The same checkpoint works in both.

- `--action_scale 0.5` — halves commanded Cartesian *position* speed.
  Caveat: it does **not** slow the rotation delta. The 6D rotation gets
  Gram-Schmidt-normalized server-side, so multiplying its 6D representation
  by a positive scalar is a no-op for the resulting rotation matrix. The
  IK-jump watchdog on the server is the actual guard against fast rotation.
- `--ood_delta_mm 5.0 --ood_halt_count 2` — refuse Cartesian Δ above 5 mm,
  halt the client loop after 2 consecutive OOD steps.
- `--gripper_async` (default ON) — keeps the loop at the target FPS despite
  slow Gripette RPCs.
- `--duration 20` — long enough for a full grasp attempt.

Once that run completes without incident, relax progressively:

```bash
# Trusted-model defaults, faster:
... --action_scale 0.8 --ood_delta_mm 10.0 --ood_halt_count 3

# Server side: looser IK-jump threshold once you're confident the model
# doesn't drive into singularities.
... --max_ik_jump_deg 20 --max_ik_jump_violations 2 --interp_alpha 0.3
```

### 5.5b — Safety layers (full table)

| Layer | Where | Default (tight) | Default (cruise) | What it does |
|---|---|---|---|---|
| `--action_scale` | client `eval_on_robot.py` | `0.5` | `0.8` | Multiply Cartesian Δ-position. (Note: 6D rotation Δ is *not* slowed; see above.) |
| `--ood_delta_mm` | client | `5.0` | `10.0` | Zero out Cartesian Δ when `\|Δpos\|` exceeds this in mm |
| `--ood_halt_count` | client | `2` | `3` | Halt the client loop after N consecutive OOD steps |
| Camera-local integrator | server | always on | always on | Cumulative target bounded to a smoothly-reachable region |
| `--max_ik_jump_deg` | server | `10` | `20` | Refuse the new IK solution if any joint would change by more than N° in one step (catches singularity branch flips) |
| `--max_ik_jump_violations` | server | `1` | `2` | Disable the interpolator after N consecutive IK-jump rejections |
| `max_relative_target` | driver (`OpenArm7Follower.send_action`) | `8°/step` | `8°/step` | Per-joint per-step motor-command clamp |
| Driver `joint_limits` | `OpenArm7FollowerConfig` | per-joint | per-joint | Hard-clip each joint to a safe range (e.g. j6 ∈ [-40°, 40°]) |
| `--interp_alpha` | server interpolator | `0.15` | `0.3` | Exponential approach rate on motor commands. Lower = smoother + more lag |
| `--kp_scale` / `--kd_scale` | server (MIT gains) | `1.0` | `1.0` | Multiplier on Damiao motor PID gains. Drop to `0.5` for softer tracking |
| Damiao firmware velocity/torque | motor firmware | per-motor | per-motor | Hard upper bound, set in firmware. Not in software-reach. |

If any of these trip during a run, the gRPC server will log a clear message
saying which layer fired and why. The IK-jump watchdog is the only one that
*latches* — once it disables the interpolator, the arm freezes and stays
frozen until you restart the server. The others are advisory / per-step.

#### What none of these catch

- **Slow drift** into a workspace edge with no abrupt joint changes — neither
  watchdog will trip; you have to physically halt.
- **Self-collisions.** The URDF doesn't model self-collision avoidance.
  Placo will happily solve to a config where the elbow strikes the chassis.
  Keep the e-stop in reach for the first runs.
- **Mechanical play / loose couplings.** Encoders + FK + the integrator can
  all be internally consistent while the physical link is at a different
  position. Test by hand-twisting each joint with the server stopped — there
  shouldn't be more than degree-or-two of slop on any joint.

### 5.6 — If motion looks wrong

| Symptom | Most likely cause | Fix |
| --- | --- | --- |
| Arm moves too fast (position) | Action scale too high | Drop `--action_scale 0.5 → 0.3` |
| Arm rotates too fast (despite low action_scale) | `action_scale` doesn't slow 6D-rotation Δ; relies on IK-jump watchdog | Lower `--max_ik_jump_deg` on the server |
| **Wrist "explosion" / sudden large motion** | **Singularity branch flip in Placo's IK** | The IK-jump watchdog catches this. If it didn't trip, lower `--max_ik_jump_deg` (e.g. 10). Re-home far from the singular region before retrying. |
| Server logs `IK-jump watchdog tripped` | Singularity / unreachable target / bad model output | Re-home arm with `set_arm_pose`; consider running cartesian_square smoke test first |
| Jerky at 10 Hz buzz | Motor interpolator tuning | Server: `--interp_alpha 0.15` (more smoothing) |
| Arm drifts / ignores object | State-shortcut or bad camera framing | Verify `probe_model.py`; check camera matches training |
| Control loop runs at < 5 Hz | Gripper RPC blocking | `--gripper_async` is on by default in `eval_on_robot.py`; verify it isn't disabled |
| Arm motion looks "approximately right" but consistently misses | Frame-convention break (controller-frame deltas leaked into deployment), URDF axis mismatch, or motor zero off | (1) Run **5.4 cartesian_square smoke test**; (2) compare per-joint motion in real vs sim (step **5.0a**); (3) recalibrate motor zeros (step **5.0b**) |
| `cartesian_square` first edge "misses" | URDF axis sign wrong on a shoulder joint | Step **5.0a** per-joint comparison; pattern fix in `openarm_gripette_model` |
| `SendCartesianDelta` timeouts | CAN bus saturation | Lower `--interp_hz 25` on the server |
| Gripper opens when closing (or vice versa) | Gripette motor sign-flipped relative to dataset convention | Step **5.0c** sanity check; fix in Gripette firmware |
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

# 3. on the robot — start servers with cruise safety settings
uv run python examples/openarm_gripette/grpc_server_real.py \
    --can_port can0 --side right \
    --max_ik_jump_deg 20 --max_ik_jump_violations 2 &
uv run python examples/openarm_gripette/reset_arm.py --arm_addr <ARM_IP>:50052

# 4. (one-time per session, optional) verify gripper open/close direction
uv run python examples/openarm_gripette/set_gripper_pose.py --gripper_addr <GRIPETTE_IP>:50051 --open
uv run python examples/openarm_gripette/set_gripper_pose.py --gripper_addr <GRIPETTE_IP>:50051 --close

# 5. smoke-test the integrator (camera-local square) — only on first session per day,
#    or after URDF / driver / model changes
uv run python examples/openarm_gripette/cartesian_square.py \
    --arm_addr <ARM_IP>:50052 --gripper_addr <GRIPETTE_IP>:50051 \
    --show_camera --plane xy --half_size 0.05 --fps 50 --loops 1

# 6. run the policy
uv run python examples/openarm_gripette/eval_on_robot.py \
    --checkpoint <USER>/<MODEL> \
    --arm_addr <ARM_IP>:50052 --gripper_addr <GRIPETTE_IP>:50051 \
    --action_scale 0.5 --ood_delta_mm 5.0 --ood_halt_count 2
```

For an **untrusted model** (just trained, never run on real before), use the
tight-safety server flags from step 5.1 instead of cruise:

```bash
# Tight first-run safety
uv run python examples/openarm_gripette/grpc_server_real.py \
    --can_port can0 --side right \
    --max_ik_jump_deg 10 --max_ik_jump_violations 1 --interp_alpha 0.15
```
