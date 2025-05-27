"""Quite slow (~1 file/second)"""

from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_tree
from tqdm import tqdm

DATASET_NAME = "allenai/s2-naip"

WORLDCOVER_DIR = Path("./worldcover")
WORLDCOVER_DIR.mkdir(exist_ok=True)

def download_worldcover():
    files = list_repo_tree(repo_id=DATASET_NAME, repo_type="dataset")

    worldcover_dirs = [f.path for f in files if f.path.startswith("worldcover_")]
    print("Downloading directories:", worldcover_dirs)

    for folder in tqdm(worldcover_dirs, desc="Folders"):
        sub_files = [f.path for f in list_repo_tree(repo_id=DATASET_NAME, repo_type="dataset", path_in_repo=folder)]

        for file_path in tqdm(sub_files, desc=f"Files in {folder}"):
            print(f"Downloading: {file_path}")
            hf_hub_download(
                repo_id=DATASET_NAME,
                repo_type="dataset",
                filename=file_path,
                local_dir=WORLDCOVER_DIR,
            )

if __name__ == '__main__':
    download_worldcover()