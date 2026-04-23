"""Display the live camera feed from the Gripette's GripperService.

Usage:
  uv run python examples/openarm_gripette/view_camera.py \\
      --gripper_addr localhost:50051

Press 'q' or Ctrl+C in the terminal to quit.
"""

import argparse
import time

import cv2
import grpc
import numpy as np
from openarm_gripette_simu.proto import gripper_pb2, gripper_pb2_grpc


def parse_args():
    p = argparse.ArgumentParser(description="View the Gripette camera feed")
    p.add_argument("--gripper_addr", type=str, default="localhost:50051", help="GripperService gRPC address")
    p.add_argument(
        "--show_gripper_state",
        action="store_true",
        help="Overlay motor1/motor2 positions on the frame",
    )
    return p.parse_args()


def main():
    args = parse_args()
    channel = grpc.insecure_channel(args.gripper_addr)
    stub = gripper_pb2_grpc.GripperServiceStub(channel)

    ping = stub.Ping(gripper_pb2.PingRequest())
    print(f"Connected to {args.gripper_addr} (server uptime: {ping.uptime_seconds:.1f}s)")
    print("Press 'q' in the window (or Ctrl+C) to quit.\n")

    last_tick = time.perf_counter()
    fps_ema = 0.0
    try:
        for frame in stub.StreamState(gripper_pb2.StreamRequest()):
            img = cv2.imdecode(np.frombuffer(frame.jpeg_data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            now = time.perf_counter()
            dt = now - last_tick
            last_tick = now
            fps_ema = 0.9 * fps_ema + 0.1 * (1.0 / dt) if dt > 0 else fps_ema

            overlay = img.copy()
            cv2.putText(
                overlay,
                f"{fps_ema:5.1f} fps  {img.shape[1]}x{img.shape[0]}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            if args.show_gripper_state:
                m = frame.motor_state
                cv2.putText(
                    overlay,
                    f"m1={m.motor1_position:+.3f}  m2={m.motor2_position:+.3f}",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            cv2.imshow("Gripette camera", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    except grpc.RpcError as e:
        print(f"gRPC error: {e}")
    finally:
        cv2.destroyAllWindows()
        channel.close()


if __name__ == "__main__":
    main()
