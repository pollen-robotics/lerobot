import h5py
import torch
import os
import argparse
from glob import glob
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import cv2


parser = argparse.ArgumentParser(description='Convert hdf5 dataset (raw) to LeRobot dataset')
parser.add_argument('--hdf5_path', type=str, help='Path to the hdf5 dataset directory')
# parser.add_argument('--repo_id', type=str, help='Repo ID')
parser.add_argument('--out_dir', type=str, help='Path to the output directory', default='out_lerobot')
parser.add_argument('--push_to_hub', action = 'store_true', help='Push to hub')
parser.add_argument('--fps', type=int, help='FPS', default=30)
args = parser.parse_args()

hdf5_paths = glob(os.path.join(args.hdf5_path, 'episode_*.hdf5'))

features = {
    "observation.state": {
        "dtype": "float32",
        "shape": (22,),
        "names": None
    },
    "action": {
        "dtype": "float32",
        "shape": (22,),
        "names": None
    },
    "observation.images.cam_teleop": {
        "dtype": "video",
        "shape": (3, 720, 960),
        "names": [
            "channel",
            "height",
            "width",
        ],
    },
    "observation.images.cam_trunk": {
        "dtype": "video",
        "shape": (3, 720, 1280),
        "names": [
            "channel",
            "height",
            "width",
        ],
    },
}


dataset = LeRobotDataset.create(
    repo_id="pollen-robotics/TMP_TESTTTT",
    fps=args.fps,
    robot_type="reachy2",
    features=features,
    image_writer_threads=4,
)

def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    return frames

episodes = range(len(hdf5_paths))
for hdf5_path in hdf5_paths:
    with h5py.File(hdf5_path, "r") as data:

        print("===", hdf5_path, "===")
        action = data["/action"][:]
        state = data["/observations/qpos"][:]
        image_idx = {}
        image_idx["cam_trunk"] = data["/observations/images_ids/cam_trunk"][:]
        image_idx["cam_teleop"] = data["/observations/images_ids/cam_teleop"][:]

        cam_trunk_video_path = hdf5_path.replace(".hdf5", "_cam_trunk.mp4")
        cam_teleop_video_path = hdf5_path.replace(".hdf5", "_cam_teleop.mp4")

        trunk_video_frames = get_video_frames(cam_trunk_video_path)
        teleop_video_frames = get_video_frames(cam_teleop_video_path)

        for i in range(len(action)):
            frame = {
                "action": torch.from_numpy(action[i]),
                "observation.state": torch.from_numpy(state[i]),
                "observation.images.cam_trunk": torch.from_numpy(trunk_video_frames[image_idx["cam_trunk"][i]]),
                "observation.images.cam_teleop": torch.from_numpy(teleop_video_frames[image_idx["cam_teleop"][i]]),
            }
            dataset.add_frame(frame)
    dataset.save_episode(task="TMP_TESTTTT")

dataset.consolidate()

if args.push_to_hub:
    dataset.push_to_hub()
