import argparse
import time

import cv2
import numpy as np
from pollen_vision.camera_wrappers import CameraWrapper
from reachy2_sdk import ReachySDK

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.pollen.reachy2_env import Reachy2Env

parser = argparse.ArgumentParser()
parser.add_argument("--robot_ip", type=str, default="localhost")
parser.add_argument("--repo_id", type=str, required=True)
parser.add_argument("--episode_id", type=str, required=True, default=0)
args = parser.parse_args()


class ReplayCameraWrapper(CameraWrapper):
    def __init__(self, dataset: LeRobotDataset, from_index: int, to_index: int):
        self.dataset = dataset
        self.from_index = from_index
        self.to_index = to_index
        self.index = 0

    def get_data(self):
        image = dataset[from_index.item() + self.index]["observation.images.cam_trunk"]
        image = image.permute((1, 2, 0)).numpy() * 255
        image = image.astype(np.uint8)

        data = {"left": image}

        self.index += 1

        return data, None, None

    def get_K(self):
        pass


reachy = ReachySDK(args.robot_ip)
reachy.turn_on()
time.sleep(1)
dataset = LeRobotDataset(args.repo_id)
from_index = dataset.episode_data_index["from"][int(args.episode_id)]
to_index = dataset.episode_data_index["to"][int(args.episode_id)]

actions = dataset.hf_dataset["action"][from_index:to_index]

env = Reachy2Env(30, ReplayCameraWrapper(dataset, from_index, to_index), reachy)

for action in actions:
    env.step(action)
    obs = env._observation
    image = obs["pixels"]["cam_trunk"]
    cv2.imshow("image", image)
    cv2.waitKey(1)
