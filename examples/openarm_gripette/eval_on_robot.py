"""Deploy a trained Diffusion Policy on a real OpenArm + Gripette via gRPC.

Architecture (identical to `eval_simulator.py` — only the gRPC endpoints differ):

    ┌────────────────────┐         SendCartesianDelta(dx,dy,dz,dr6d)
    │  this script       │ ──────────────────────────────────► ArmService
    │  (policy + loop)   │ ◄────────────────────────────────── (grpc_server_real.py)
    └────────────────────┘         GetArmState  →  ArmState                 │
              │                                                             ▼
              │                                                       real OpenArm via CAN
              │
              │              StreamState → camera frame + gripper joints
              └──────────────────────────────────────────►  GripperService
                                                            (Gripette device)
                             SendMotorCommand(motor1, motor2)

The model outputs camera-LOCAL frame deltas (`R_t^T @ Δpos_world` and
`R_t^T @ R_{t+1}` packed as 6D). The ArmService server (`grpc_server_real.py`)
integrates them with the same math as the simulator's `arm_servicer.py`:

    R_target_new = R_target @ R_delta
    pos_target_new = pos_target + R_target @ delta_pos

so the deployment delta convention matches the training delta convention exactly.

NOTE: this script replaces the previous direct-FK/IK version, which treated
the policy's delta output as an absolute Cartesian target and ran IK locally.
That bypassed the integrator and was the root cause of poor real-robot grasps
even when sim-eval looked correct.

Usage:
  # Terminal 1 — on the robot machine (or wherever CAN is attached):
  uv run python examples/openarm_gripette/grpc_server_real.py \\
      --can_port can0 --side right --arm_port 50052

  # Terminal 2 — Gripette device (start its own GripperService).

  # Terminal 3 — inference machine:
  uv run python examples/openarm_gripette/eval_on_robot.py \\
      --checkpoint outputs/gripette/diffusion \\
      --arm_addr <robot-ip>:50052 \\
      --gripper_addr <gripette-ip>:50051 \\
      --duration 30 --action_scale 0.5 --ood_delta_mm 8.0
"""

import argparse
import logging
import sys
import time

import grpc
import numpy as np
import torch

from lerobot.policies import make_pre_post_processors
from lerobot.policies.diffusion import DiffusionPolicy
from lerobot.utils.rotation import rotation_6d_to_rotation_matrix_numpy

# Reuse the helpers from eval_simulator.py — sim and real share the exact same
# gRPC contract, so there is no reason to duplicate the camera reader, async
# gripper sender, or relative-proprio logic. Any improvement to the simulator
# eval loop automatically benefits the real-robot eval too.
#
# Sibling-module import: when invoked as `python eval_on_robot.py`, the script's
# directory is on sys.path, so a bare `import eval_simulator` resolves to the
# sibling file (there is no `examples/openarm_gripette` Python package).
from eval_simulator import (  # noqa: E402
    AsyncGripperSender,
    CameraStreamReader,
    capture_start_pose,
    compute_relative_state,
)

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Deploy Gripette policy on real OpenArm via gRPC")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint")
    p.add_argument("--arm_addr", type=str, required=True,
                   help="ArmService address (grpc_server_real.py), e.g. 192.168.1.42:50052")
    p.add_argument("--gripper_addr", type=str, required=True,
                   help="GripperService address (Gripette device), e.g. 192.168.1.51:50051")
    p.add_argument("--device", type=str, default="cuda", help="Compute device")
    p.add_argument("--duration", type=float, default=20.0, help="Run duration in seconds")
    p.add_argument("--fps", type=float, default=10.0, help="Control loop frequency")
    # Real-robot safety defaults are stricter than sim.
    p.add_argument(
        "--action_scale", type=float, default=0.5,
        help="Multiplier on Cartesian deltas (pos + 6D rotation). Gripper unchanged. "
        "0.5 (default for real) halves the commanded speed; use 1.0 to match training.",
    )
    p.add_argument(
        "--ood_delta_mm", type=float, default=8.0,
        help="OOD safety watchdog: predictions with |delta_pos| > this (mm) get zeroed. "
        "After --ood_halt_count consecutive OOD steps the loop halts. Disable with 0.",
    )
    p.add_argument("--ood_halt_count", type=int, default=3,
                   help="Consecutive OOD steps before halting.")
    p.add_argument("--gripper_async", action="store_true", default=True,
                   help="Send gripper commands from a background thread (last-goal-wins).")
    p.add_argument("--no_send_arm", action="store_true",
                   help="Dry run: skip SendCartesianDelta calls (still runs inference).")
    p.add_argument("--no_send_gripper", action="store_true",
                   help="Dry run: skip SendMotorCommand calls.")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    device = torch.device(args.device)

    # ---- Policy ----
    logger.info(f"Loading policy from {args.checkpoint}")
    policy = DiffusionPolicy.from_pretrained(args.checkpoint).to(device).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, pretrained_path=args.checkpoint
    )

    state_dim = policy.config.robot_state_feature.shape[0]
    use_relative_proprio = state_dim > 2
    logger.info(
        f"Policy: action_dim={policy.config.action_feature.shape[0]}, "
        f"state_dim={state_dim} ({'relative proprio' if use_relative_proprio else 'gripper only'}), "
        f"n_action_steps={policy.config.n_action_steps}"
    )

    # ---- gRPC connections (sim package supplies the .proto stubs; the real
    #      ArmService server uses the same proto) ----
    from openarm_gripette_simu.proto import arm_pb2, arm_pb2_grpc, gripper_pb2, gripper_pb2_grpc

    arm_channel = grpc.insecure_channel(args.arm_addr)
    arm_stub = arm_pb2_grpc.ArmServiceStub(arm_channel)
    gripper_channel = grpc.insecure_channel(args.gripper_addr)
    gripper_stub = gripper_pb2_grpc.GripperServiceStub(gripper_channel)

    try:
        arm_ping = arm_stub.Ping(arm_pb2.ArmPingRequest(), timeout=5.0)
        gripper_ping = gripper_stub.Ping(gripper_pb2.PingRequest(), timeout=5.0)
    except grpc.RpcError as e:
        logger.error(f"gRPC ping failed: {e}")
        sys.exit(1)
    logger.info(f"ArmService: {arm_ping.status} (uptime {arm_ping.uptime_seconds:.1f}s)")
    logger.info(f"GripperService: {gripper_ping.status} (uptime {gripper_ping.uptime_seconds:.1f}s)")

    # ---- Camera stream + async gripper ----
    camera_reader = CameraStreamReader(gripper_stub, gripper_pb2)
    camera_reader.start()

    gripper_sender: AsyncGripperSender | None = None
    if args.gripper_async and not args.no_send_gripper:
        gripper_sender = AsyncGripperSender(gripper_stub, gripper_pb2)
        gripper_sender.start()

    # ---- Relative-proprio start pose ----
    start_pos, start_rot_matrix = (None, None)
    if use_relative_proprio:
        start_pos, start_rot_matrix = capture_start_pose(arm_stub, arm_pb2)
        logger.info(f"Start pose: [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}] m")

    # ---- Safety banner ----
    logger.warning("REAL ROBOT MODE — action_scale=%.2f, ood_delta_mm=%s",
                   args.action_scale, args.ood_delta_mm or "off")
    logger.warning("Keep the e-stop within reach. Ctrl+C halts the loop.")
    time.sleep(1.0)

    # ---- Control loop ----
    dt = 1.0 / args.fps
    policy.reset()
    start_time = time.time()
    step_count = 0
    ood_consecutive = 0

    try:
        while (time.time() - start_time) < args.duration:
            loop_start = time.perf_counter()

            # 1. Observation
            camera_image, gripper_joints = camera_reader.get_latest()
            if use_relative_proprio:
                arm_state = arm_stub.GetArmState(arm_pb2.GetArmStateRequest())
                state = compute_relative_state(
                    arm_state, gripper_joints, start_pos, start_rot_matrix
                )
            else:
                state = gripper_joints

            # 2. Policy
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            image_tensor = (
                torch.from_numpy(camera_image).float().div_(255.0).permute(2, 0, 1).contiguous()
                .unsqueeze(0).to(device)
            )
            batch = preprocessor({
                "observation.state": state_tensor,
                "observation.images.cam0": image_tensor,
            })
            with torch.no_grad():
                action = policy.select_action(batch)
            action = postprocessor(action)
            action_np = action.squeeze(0).cpu().numpy()

            # 3. Extract camera-local deltas + absolute gripper goal
            delta_pos = action_np[:3].copy() * args.action_scale
            delta_rot_6d = action_np[3:9].copy() * args.action_scale
            gripper_goal = action_np[9:]

            # 4. OOD safety
            commanded_mm = float(np.linalg.norm(delta_pos) * 1000)
            ood_active = args.ood_delta_mm and args.ood_delta_mm > 0
            if ood_active and commanded_mm > args.ood_delta_mm:
                ood_consecutive += 1
                logger.warning(
                    f"OOD |Δpos|={commanded_mm:.1f}mm > {args.ood_delta_mm}mm "
                    f"({ood_consecutive}/{args.ood_halt_count}) — zeroing Cartesian delta"
                )
                delta_pos = np.zeros(3)
                delta_rot_6d = np.zeros(6)
                if ood_consecutive >= args.ood_halt_count:
                    logger.error("OOD halt threshold reached — exiting.")
                    break
            else:
                ood_consecutive = 0

            # 5. Send commands
            if not args.no_send_arm:
                arm_stub.SendCartesianDelta(arm_pb2.CartesianDelta(
                    dx=float(delta_pos[0]),
                    dy=float(delta_pos[1]),
                    dz=float(delta_pos[2]),
                    dr6d=delta_rot_6d.tolist(),
                ))
            if not args.no_send_gripper:
                m1 = float(gripper_goal[0])
                m2 = float(gripper_goal[1]) if len(gripper_goal) > 1 else 0.0
                if gripper_sender is not None:
                    gripper_sender.send(m1, m2)
                else:
                    gripper_stub.SendMotorCommand(
                        gripper_pb2.MotorCommand(motor1_goal=m1, motor2_goal=m2)
                    )

            step_count += 1
            if step_count % 20 == 0:
                logger.info(
                    f"Step {step_count:>5d} | Δpos={commanded_mm:5.1f}mm | "
                    f"gripper=[{gripper_goal[0]:+.3f}, "
                    f"{gripper_goal[1] if len(gripper_goal) > 1 else 0.0:+.3f}]"
                )

            elapsed = time.perf_counter() - loop_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if gripper_sender is not None:
            gripper_sender.stop()
        camera_reader.stop()
        arm_channel.close()
        gripper_channel.close()
        logger.info(f"Done. {step_count} steps in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
