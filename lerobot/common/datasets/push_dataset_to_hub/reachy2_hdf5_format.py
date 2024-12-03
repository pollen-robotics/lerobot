#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Contains utilities to process raw data format of HDF5 files like in: https://github.com/tonyzhaozh/act
"""

import gc
import re
import shutil
from pathlib import Path

import h5py
import torch
import tqdm
from datasets import Dataset, Features, Image, Sequence, Value

from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame


def get_cameras(hdf5_data):
    # ignore depth channel, not currently handled
    # TODO(rcadene): add depth
    rgb_cameras = [key for key in hdf5_data["/observations/images_ids"].keys() if "depth" not in key]  # noqa: SIM118
    return rgb_cameras


def check_format(raw_dir) -> bool:
    hdf5_paths = list(raw_dir.glob("episode_*.hdf5"))
    assert len(hdf5_paths) != 0
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, "r") as data:
            assert "/action" in data
            assert "/observations/qpos" in data

            assert data["/action"].ndim == 2
            assert data["/observations/qpos"].ndim == 2

            num_frames = data["/action"].shape[0]
            assert num_frames == data["/observations/qpos"].shape[0]

            for camera in get_cameras(data):
                assert num_frames == data[f"/observations/images_ids/{camera}"].shape[0]
                assert (raw_dir / hdf5_path.name.replace(".hdf5", f"_{camera}.mp4")).exists()

                # assert data[f"/observations/images_ids/{camera}"].ndim == 4
                # b, h, w, c = data[f"/observations/images_ids/{camera}"].shape
                # assert c < h and c < w, f"Expect (h,w,c) image format but ({h=},{w=},{c=}) provided."


# def load_from_raw(raw_dir, out_dir, fps, video, debug):
def load_from_raw(
    raw_dir: Path,
    videos_dir: Path,
    fps: int,
    video: bool,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    hdf5_files = list(raw_dir.glob("*.hdf5"))
    ep_dicts = []
    episode_data_index = {"from": [], "to": []}

    id_from = 0
    for ep_idx, ep_path in tqdm.tqdm(enumerate(hdf5_files), total=len(hdf5_files)):
        match = re.search(r"_(\d+).hdf5", ep_path.name)
        if not match:
            raise ValueError(ep_path.name)
        raw_ep_idx = int(match.group(1))

        with h5py.File(ep_path, "r") as ep:
            num_frames = ep["/action"].shape[0]

            # last step of demonstration is considered done
            done = torch.zeros(num_frames, dtype=torch.bool)
            done[-1] = True

            state = torch.from_numpy(ep["/observations/qpos"][:])
            action = torch.from_numpy(ep["/action"][:])
            if "/observations/qvel" in ep:
                velocity = torch.from_numpy(ep["/observations/qvel"][:])
            if "/observations/effort" in ep:
                effort = torch.from_numpy(ep["/observations/effort"][:])

            ep_dict = {}

            videos_dir = videos_dir / "videos"
            videos_dir.mkdir(parents=True, exist_ok=True)

            for camera in get_cameras(ep):
                img_key = f"observation.images.{camera}"

                raw_fname = f"episode_{raw_ep_idx}_{camera}.mp4"
                new_fname = f"{img_key}_episode_{ep_idx:06d}.mp4"
                shutil.copy(str(raw_dir / raw_fname), str(videos_dir / new_fname))

                # store the reference to the video frame
                ep_dict[img_key] = [
                    {"path": f"videos/{new_fname}", "timestamp": i / fps} for i in range(num_frames)
                ]

            ep_dict["observation.state"] = state
            if "/observations/velocity" in ep:
                ep_dict["observation.velocity"] = velocity
            if "/observations/effort" in ep:
                ep_dict["observation.effort"] = effort
            ep_dict["action"] = action
            ep_dict["episode_index"] = torch.tensor([ep_idx] * num_frames)
            ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
            ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
            ep_dict["next.done"] = done
            # TODO(rcadene): add reward and success by computing them in sim

            assert isinstance(ep_idx, int)
            ep_dicts.append(ep_dict)

            episode_data_index["from"].append(id_from)
            episode_data_index["to"].append(id_from + num_frames)

        id_from += num_frames

        gc.collect()

        # # process first episode only
        # if debug:
        #     break

    data_dict = concatenate_episodes(ep_dicts)
    return data_dict, episode_data_index


def to_hf_dataset(data_dict, video) -> Dataset:
    features = {}

    keys = [key for key in data_dict if "observation.images." in key]
    for key in keys:
        if video:
            features[key] = VideoFrame()
        else:
            features[key] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    if "observation.velocity" in data_dict:
        features["observation.velocity"] = Sequence(
            length=data_dict["observation.velocity"].shape[1], feature=Value(dtype="float32", id=None)
        )
    if "observation.effort" in data_dict:
        features["observation.effort"] = Sequence(
            length=data_dict["observation.effort"].shape[1], feature=Value(dtype="float32", id=None)
        )
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(raw_dir: Path, videos_dir: Path, fps: int | None = None, video: bool = True, episodes: list[int] | None = None, encoding: dict | None = None):
    # sanity check
    check_format(raw_dir)

    if fps is None:
        fps = 30

    data_dir, episode_data_index = load_from_raw(raw_dir, videos_dir, fps, video, episodes, encoding)
    hf_dataset = to_hf_dataset(data_dir, video)

    info = {
        "fps": fps,
        "video": video,
    }
    return hf_dataset, episode_data_index, info
