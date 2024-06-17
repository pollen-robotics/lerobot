import time

import FramesViewer.utils as fv_utils
import numpy as np
from pollen_vision.camera_wrappers.depthai import SDKWrapper
from pollen_vision.camera_wrappers.depthai.utils import get_config_file_path
from reachy2_sdk import ReachySDK

from lerobot.pollen.policy_wrapper import PolicyWrapper

right_start_pose = np.array(
    [
        [0.0, -0.0, -1.0, 0.2],
        [0.0, 1.0, -0.0, -0.24],
        [1.0, 0.0, 0.0, -0.3],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
left_start_pose = np.array(
    [
        [0.0, -0.0, -1.0, 0.2],
        [0.0, 1.0, -0.0, 0.24],
        [1.0, 0.0, 0.0, -0.3],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

left_drop_pose = np.array(
    [
        [0.0, -0.0, -1.0, 0.3],
        [0.0, 1.0, -0.0, 0.05],
        [1.0, 0.0, 0.0, -0.35],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

left_drop_pose = fv_utils.rotateInSelf(left_drop_pose, [-60, 0, 0], degrees=True)

FPS = 30

cam = SDKWrapper(get_config_file_path("CONFIG_SR"), fps=FPS)

reachy = ReachySDK("192.168.1.42")
# reachy = ReachySDK("localhost")
reachy.turn_on()
time.sleep(1)


pw = PolicyWrapper(
    pretrained_policy_name_or_path="pollen-robotics/grasp_mug2_with_time_80K",
    cam=cam,
    reachy=reachy,
)

STATE = "GRASP"
dropped_timeout = 0
prev_action = None
while True:
    if STATE == "GRASP":
        start = time.time()
        predicted_step, action = pw.infer()
        print("predicted step", predicted_step)
        took = time.time() - start
        time.sleep(max(0, 1 / FPS - took))
        if predicted_step > 0.95 and dropped_timeout <= 0:
            STATE = "DROP"
        prev_action = action

    elif STATE == "DROP":
        reachy.l_arm.goto_from_matrix(left_drop_pose, duration=2.0)
        time.sleep(2)
        reachy.l_arm.gripper.open()
        time.sleep(0.2)
        pw.goto_action(prev_action, duration=2.0)
        # reachy.l_arm.goto_from_matrix(left_start_pose, duration=2.0)
        # time.sleep(2)
        STATE = "GRASP"
        dropped_timeout = 1

    if dropped_timeout > 0:
        dropped_timeout = max(0, dropped_timeout - 0.01)

    # print("STATE", STATE)
    # print("dropped timeout", dropped_timeout)
