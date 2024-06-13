import gymnasium as gym
import numpy as np
from gymnasium import spaces
from pollen_vision.camera_wrappers import CameraWrapper
from reachy2_sdk import ReachySDK

ACTIONS = [
    "l_arm_shoulder_pitch",
    "l_arm_shoulder_roll",
    "l_arm_elbow_yaw",
    "l_arm_elbow_pitch",
    "l_arm_wrist_roll",
    "l_arm_wrist_pitch",
    "l_arm_wrist_yaw",
    "l_gripper",
    "r_arm_shoulder_pitch",
    "r_arm_shoulder_roll",
    "r_arm_elbow_yaw",
    "r_arm_elbow_pitch",
    "r_arm_wrist_roll",
    "r_arm_wrist_pitch",
    "r_arm_wrist_yaw",
    "r_gripper",
    "mobile_base_vx",
    "mobile_base_vy",
    "mobile_base_vtheta",
    "head_roll",
    "head_pitch",
    "head_yaw",
]


JOINTS = [
    "l_arm_shoulder_pitch",
    "l_arm_shoulder_roll",
    "l_arm_elbow_yaw",
    "l_arm_elbow_pitch",
    "l_arm_wrist_roll",
    "l_arm_wrist_pitch",
    "l_arm_wrist_yaw",
    "l_gripper",
    "r_arm_shoulder_pitch",
    "r_arm_shoulder_roll",
    "r_arm_elbow_yaw",
    "r_arm_elbow_pitch",
    "r_arm_wrist_roll",
    "r_arm_wrist_pitch",
    "r_arm_wrist_yaw",
    "r_gripper",
    "mobile_base_vx",
    "mobile_base_vy",
    "mobile_base_vtheta",
    "head_roll",
    "head_pitch",
    "head_yaw",
]

CAMERAS = {
    "cam_trunk": (800, 1280, 3),
}


class Reachy2Env(gym.Env):
    def __init__(self, fps: int, cam: CameraWrapper, reachy: ReachySDK):
        self.fps = fps
        self.cam = cam
        self.reachy = reachy

        observation_space = {}

        observation_space["agent_pos"] = spaces.Box(
            low=-1000.0,
            high=1000.0,
            shape=(len(JOINTS),),
            dtype=np.float64,
        )

        pixels_space = {}
        for camera, hwc_shape in CAMERAS.items():
            # Assumes images are unsigned int8 in [0,255]
            pixels_space[camera] = spaces.Box(
                low=0,
                high=255,
                shape=hwc_shape,
                dtype=np.uint8,
            )
        observation_space["pixels"] = spaces.Dict(pixels_space)

        self.observation_space = spaces.Dict(observation_space)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(len(ACTIONS),), dtype=np.float32
        )

        self._observation = {"pixels": {}, "agent_pos": None}

    def reset(self):
        self._get_obs()
        return self._observation, {}

    def goto_action(self, action):
        l_joints = action[0:7]
        for i in range(len(l_joints)):
            l_joints[i] = np.rad2deg(l_joints[i])
        r_joints = action[8:15]
        for i in range(len(r_joints)):
            r_joints[i] = np.rad2deg(r_joints[i])
        self.reachy.r_arm.goto_joints(r_joints, duration=1 / self.fps)
        self.reachy.l_arm.goto_joints(l_joints, duration=1 / self.fps)

    def step(self, action):

        # self.goto_action(action.copy())

        self.reachy.l_arm.shoulder.pitch.goal_position = np.rad2deg(action[0])
        self.reachy.l_arm.shoulder.roll.goal_position = np.rad2deg(action[1])
        self.reachy.l_arm.elbow.yaw.goal_position = np.rad2deg(action[2])
        self.reachy.l_arm.elbow.pitch.goal_position = np.rad2deg(action[3])
        self.reachy.l_arm.wrist.roll.goal_position = np.rad2deg(action[4])
        self.reachy.l_arm.wrist.pitch.goal_position = np.rad2deg(action[5])
        self.reachy.l_arm.wrist.yaw.goal_position = np.rad2deg(action[6])
        # self.reachy.l_arm.gripper.set_opening(
        #     min(100, max(0, action[7] / 2.26 * 100))
        # )  # replay true action value
        self.reachy.l_arm.gripper.set_opening(
            0 if action[7] < 2.0 else 100
        )  # trick to force the gripper to close fully

        self.reachy.r_arm.shoulder.pitch.goal_position = np.rad2deg(action[8])
        self.reachy.r_arm.shoulder.roll.goal_position = np.rad2deg(action[9])
        self.reachy.r_arm.elbow.yaw.goal_position = np.rad2deg(action[10])
        self.reachy.r_arm.elbow.pitch.goal_position = np.rad2deg(action[11])
        self.reachy.r_arm.wrist.roll.goal_position = np.rad2deg(action[12])
        self.reachy.r_arm.wrist.pitch.goal_position = np.rad2deg(action[13])
        self.reachy.r_arm.wrist.yaw.goal_position = np.rad2deg(action[14])
        # self.reachy.r_arm.gripper.set_opening(
        #     min(100, max(0, action[15] / 2.26 * 100))
        # )  # replay true action value
        self.reachy.r_arm.gripper.set_opening(
            0 if action[15] < 2.0 else 100
        )  # trick to force the gripper to close fully

        # self.reachy.mobile_base.set_speed(
        #     action[16], action[17], np.rad2deg(action[18])
        # )

        self.reachy.head.neck.roll.goal_position = np.rad2deg(action[19])
        self.reachy.head.neck.pitch.goal_position = np.rad2deg(action[20])
        self.reachy.head.neck.yaw.goal_position = np.rad2deg(action[21])

        self._get_obs()
        return self._observation, 0, False, False, {}

    def _get_obs(self):
        data, _, _ = self.cam.get_data()
        left = data["left"]

        # mobile_base_pos = self.reachy.mobile_base.odometry
        mobile_base_pos = {"vx": 0, "vy": 0, "vtheta": 0}
        qpos = {
            "l_arm_shoulder_pitch": np.deg2rad(
                self.reachy.l_arm.shoulder.pitch.present_position
            ),
            "l_arm_shoulder_roll": np.deg2rad(
                self.reachy.l_arm.shoulder.roll.present_position
            ),
            "l_arm_elbow_yaw": np.deg2rad(self.reachy.l_arm.elbow.yaw.present_position),
            "l_arm_elbow_pitch": np.deg2rad(
                self.reachy.l_arm.elbow.pitch.present_position
            ),
            "l_arm_wrist_roll": np.deg2rad(
                self.reachy.l_arm.wrist.roll.present_position
            ),
            "l_arm_wrist_pitch": np.deg2rad(
                self.reachy.l_arm.wrist.pitch.present_position
            ),
            "l_arm_wrist_yaw": np.deg2rad(self.reachy.l_arm.wrist.yaw.present_position),
            "l_gripper": self.reachy.l_arm.gripper._present_position,
            "r_arm_shoulder_pitch": np.deg2rad(
                self.reachy.r_arm.shoulder.pitch.present_position
            ),
            "r_arm_shoulder_roll": np.deg2rad(
                self.reachy.r_arm.shoulder.roll.present_position
            ),
            "r_arm_elbow_yaw": np.deg2rad(self.reachy.r_arm.elbow.yaw.present_position),
            "r_arm_elbow_pitch": np.deg2rad(
                self.reachy.r_arm.elbow.pitch.present_position
            ),
            "r_arm_wrist_roll": np.deg2rad(
                self.reachy.r_arm.wrist.roll.present_position
            ),
            "r_arm_wrist_pitch": np.deg2rad(
                self.reachy.r_arm.wrist.pitch.present_position
            ),
            "r_arm_wrist_yaw": np.deg2rad(self.reachy.r_arm.wrist.yaw.present_position),
            "r_gripper": self.reachy.r_arm.gripper._present_position,
            "mobile_base_vx": mobile_base_pos["vx"],
            "mobile_base_vy": mobile_base_pos["vy"],
            "mobile_base_vtheta": np.deg2rad(mobile_base_pos["vtheta"]),
            "head_roll": np.deg2rad(self.reachy.head.neck.roll.present_position),
            "head_pitch": np.deg2rad(self.reachy.head.neck.pitch.present_position),
            "head_yaw": np.deg2rad(self.reachy.head.neck.yaw.present_position),
        }

        self._observation["agent_pos"] = np.array(list(qpos.values()))
        self._observation["pixels"]["cam_trunk"] = left
