"""Took roughly 1 hour to complete"""

from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_tree
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

DATASET_NAME = "allenai/s2-naip"
WORLDCOVER_DIR = Path("./worldcover")
WORLDCOVER_DIR.mkdir(exist_ok=True)

MAX_WORKERS = 32

def download_file(file_path: str):
    try:
        hf_hub_download(
            repo_id=DATASET_NAME,
            repo_type="dataset",
            filename=file_path,
            local_dir=WORLDCOVER_DIR,
        )
        return file_path, None
    except Exception as e:
        return file_path, e

def download_worldcover():
    print("Listing top-level repo tree...")
    files = list_repo_tree(repo_id=DATASET_NAME, repo_type="dataset")

    worldcover_dirs = [f.path for f in files if f.path.startswith("worldcover_")]
    print("Found directories:", worldcover_dirs)

    all_files_to_download = []

    for folder in tqdm(worldcover_dirs, desc="Fetching folder contents"):
        sub_files = list_repo_tree(repo_id=DATASET_NAME, repo_type="dataset", path_in_repo=folder)
        for f in sub_files:
            all_files_to_download.append(f.path)

    print(f"Total files to download: {len(all_files_to_download)}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_file, path): path for path in all_files_to_download}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading files"):
            file_path, error = future.result()
            if error:
                print(f"[ERROR] Failed to download {file_path}: {error}")

if __name__ == '__main__':
    download_worldcover()
