from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    "pollen-robotics/apple_storage_2_modified",
    root="data2/",
    local_files_only=True,
)

dataset.consolidate(run_compute_stats=True)
