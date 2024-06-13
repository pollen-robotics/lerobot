import pickle
import time
from contextlib import nullcontext
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.envs.registration import register
from huggingface_hub import snapshot_download
from huggingface_hub.utils._errors import RepositoryNotFoundError
from huggingface_hub.utils._validators import HFValidationError
from pollen_vision.camera_wrappers import CameraWrapper
from pollen_vision.camera_wrappers.depthai import SDKWrapper
from pollen_vision.camera_wrappers.depthai.utils import get_config_file_path

# from pollen_vision.perception import Perception
from reachy2_sdk import ReachySDK

from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import init_hydra_config
from lerobot.pollen.reachy2_env import Reachy2Env

FPS = 30


class PolicyWrapper:
    def __init__(
        self, pretrained_policy_name_or_path: str, cam: CameraWrapper, reachy: ReachySDK
    ) -> None:
        config_overrides = [
            "eval.n_episodes=1",
            "eval.batch_size=1",
            "env.episode_length=20000",
            "policy.n_action_steps=100",
        ]

        self.env = Reachy2Env
        self.cam = cam
        self.reachy = reachy
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_counter = 0

        try:
            self.pretrained_policy_path = Path(
                snapshot_download(pretrained_policy_name_or_path)
            )
        except (HFValidationError, RepositoryNotFoundError) as e:
            if isinstance(e, HFValidationError):
                error_message = "The provided pretrained_policy_name_or_path is not a valid Hugging Face Hub repo ID."
            else:
                error_message = "The provided pretrained_policy_name_or_path was not found on the Hugging Face Hub."
            print(f"{error_message} Treating it as a local directory.")
            self.pretrained_policy_path = Path(pretrained_policy_name_or_path)
        if (
            not self.pretrained_policy_path.is_dir()
            or not self.pretrained_policy_path.exists()
        ):
            raise ValueError(
                "The provided pretrained_policy_name_or_path is not a valid/existing Hugging Face Hub "
                "repo ID, nor is it an existing local directory."
            )

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

        self.hydra_cfg = init_hydra_config(
            str(self.pretrained_policy_path / "config.yaml"), config_overrides
        )

        self._make_env()
        print("making policy...")
        self.policy = make_policy(
            hydra_cfg=self.hydra_cfg,
            pretrained_policy_name_or_path=str(self.pretrained_policy_path),
        )
        self.policy.eval()
        self.observation, _ = self.env.reset()
        print("Turning on robot ...")
        self.reachy.turn_on()
        time.sleep(1)
        print("PolicyWrapper ready")

    def goto_action(self, action, duration=0.3):
        l_joints = action[0:7]
        for i in range(len(l_joints)):
            l_joints[i] = np.rad2deg(l_joints[i])
        r_joints = action[8:15]
        for i in range(len(r_joints)):
            r_joints[i] = np.rad2deg(r_joints[i])
        self.reachy.r_arm.goto_joints(r_joints, duration=duration)
        self.reachy.l_arm.goto_joints(l_joints, duration=duration)

        time.sleep(duration)

    def infer(self):
        """
        Returns the predicted_step action and the raw action array
        """

        with torch.no_grad(), nullcontext():
            self.observation = preprocess_observation(self.observation)
            observation = {
                key: self.observation[key].to(self.device, non_blocking=True)
                for key in self.observation
            }
            with torch.inference_mode():
                action = self.policy.select_action(observation)
            action = action.to("cpu").numpy()

            if (
                self.action_counter == 100
            ):  # for smooth transition between action sequences. This number is policy.n_action_steps=100 in config_overrides
                self.action_counter = 0
                self.goto_action(action[0].copy())  # .copy() is very important here :)
            elif self.action_counter == 0:
                self.goto_action(
                    action[0].copy(), duration=4
                )  # .copy() is very important here :)

            self.action_counter += 1

            self.observation, _, _, _, _ = self.env.step(action)

            return np.round(action[0][-1], 2), action[0]

    def _make_env(self):

        register(
            id="reachy2_env",
            entry_point="lerobot.pollen.reachy2_env:Reachy2Env",
        )
        env_cls = gym.vector.SyncVectorEnv
        gym_kwgs = {"fps": FPS, "cam": self.cam, "reachy": self.reachy}
        self.env = env_cls(
            [
                lambda: gym.make("reachy2_env", disable_env_checker=True, **gym_kwgs)
                for _ in range(1)
            ]
        )


if __name__ == "__main__":
    cam = SDKWrapper(get_config_file_path("CONFIG_SR"), fps=FPS, compute_depth=True)
    reachy = ReachySDK("192.168.1.42")
    # reachy = ReachySDK("localhost")
    pw = PolicyWrapper(
        pretrained_policy_name_or_path="pollen-robotics/grasp_mug_with_time_80K",
        cam=cam,
        reachy=reachy,
    )
    actions = []
    try:
        while True:
            start = time.time()
            _, action = pw.infer()
            took = time.time() - start
            actions.append(action)

            time.sleep(max(0, 1 / FPS - took))
    except KeyboardInterrupt:
        print("Saving actions")
        with open("actions.pkl", "wb") as f:
            pickle.dump(actions, f)
        exit()
