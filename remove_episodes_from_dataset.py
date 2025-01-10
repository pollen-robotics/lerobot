"""
Removes the specified episodes.

example  :

python3 rm_rename.py -e 1 2 3 -d data_path


Note : didn't do the full implementation, just what would have been very tedious to do by hand, since HF will do it natively in lerobot at some point.

- Supprimer les fichiers parquet/videos correspondant à cet épisode
- Renomer les numéros des épisodes après celui-ci (pour que la numérotation des épisodes reste contigue)
- Actualiser les métadata:
  - info.json: nb d'épisodes et de frames
  - episodes.jsonl: virer l'épisode et renuméroter
  - stats.json: recalculer les stats. Mais ça aussi c'est une galère actuellement parce qu'il faut recalculer toutes les stats. On va refactorer cette partie pour avoir des stats par épisode (et donc ce sera bcp plus facile d'en ajouter/enlever)
  - tasks.jsonl: Enlever la tâche de l'épisode si elle n'apparaissait que dans celui-ci

"""

import argparse
import os
from glob import glob

DRY_RUN = False

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--episodes_to_remove", nargs="+", type=int)
parser.add_argument("-d", "--dataset_path", type=str)
args = parser.parse_args()

if not DRY_RUN:
    print("====")
    print("====")
    print("====")
    print("MAKE A BACKUP FIRST")
    print("====")
    print("====")
    print("====")
    print("")
    print(
        "Will remove the following episodes :" + str(args.episodes_to_remove),
        "from",
        args.dataset_path,
    )
    res = input("Do you want to proceed? (y/N) ")
    if res != "y":
        print("Cancelled")
        exit()
else:
    print("DRY RUN")


episodes_to_remove = sorted(args.episodes_to_remove, reverse=True)
print("episodes_to_remove", episodes_to_remove)


def compute_episode_prefix(episode: int):
    prefix = ""
    for _ in range(6 - len(str(episode))):
        prefix += "0"
    return prefix


def get_episode_index(episode_path: str):
    file_name = episode_path.split("/")[-1].split(".")[0]
    return int(file_name.split("_")[-1])


# Remove .parquet files and rename the files after them
def remove_and_rename_files(dataset_path: str, episode_index: int, parquet=True):
    if parquet:
        dir_path = os.path.join(dataset_path, "data", "chunk-000")
        ext = ".parquet"
    else:
        dir_path = os.path.join(dataset_path, "videos","chunk-000", "observation.images.head_left")
        ext = ".mp4"

    prefix = compute_episode_prefix(episode_index)

    episode_name = f"episode_{prefix}{episode_index}{ext}"
    episode_path = os.path.join(dir_path, episode_name)
    print(f"removing {episode_path}")

    if not DRY_RUN:
        os.remove(episode_path)

    # rename the files after the removed one
    all_files = glob(dir_path + f"/*{ext}")

    all_files_after_episode = []
    for file in all_files:
        if get_episode_index(file) > episode_index:
            all_files_after_episode.append(file)

    # sort by episode index
    all_files_after_episode = sorted(all_files_after_episode, key=lambda x: get_episode_index(x))

    for file in all_files_after_episode:
        new_episode_index = get_episode_index(file) - 1
        new_episode_name = f"episode_{compute_episode_prefix(new_episode_index)}{new_episode_index}{ext}"
        new_episode_path = os.path.join(dir_path, new_episode_name)
        print(f"renaming {file} to {new_episode_path}")
        if not DRY_RUN:
            os.rename(file, new_episode_path)

for episode in episodes_to_remove:
    remove_and_rename_files(args.dataset_path, int(episode), parquet=True)
    print("---")
    remove_and_rename_files(args.dataset_path, int(episode), parquet=False)
    print("===")

print("DO THE REST MANUALLY")
