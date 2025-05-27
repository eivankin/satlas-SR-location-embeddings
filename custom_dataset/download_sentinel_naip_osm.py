"""
Using the group utm, row & col from split CSVs, download Sentinel-2, NAIP & OSM from the HF dataset in the same manner as WorldCover data.
Recap: dataset structure is following - "{satellite}_{utm_zone}/{utm_zone}_{row}_{col}.tar/{satellite}/{utm_zone}_{image_row}_{image_col}.{png|tif|geojson}",
where {satellite} is either "sentinel2", "naip" or "openstreetmap", format is "tif" for Sentinel-2, "png" for NAIP and "geojson" for OSM.
Sentinel files also have suffixes before the extension indicating channel resolutions: "_8", "_16", "_32" (for example, sentinel2_32610/32610_898_-6475.tar/sentinel2/32610_898_-6475_8.tif).
We are interested only in files with suffix "_8".

After downloading the .tar file for a group, extract only the needed files indicated with image_row and image_col from the CSV.
"""
import random
import time
from collections import defaultdict

import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download
import tarfile
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import numpy as np

SPLIT_DIR = Path("./splits")
DATASET_NAME = "allenai/s2-naip"
OUTPUT_DIR = Path("./downloaded")
OUTPUT_DIR.mkdir(exist_ok=True)
SATELLITES = ["sentinel2", "naip", "openstreetmap"]
MAX_WORKERS = 16
DOWNLOAD_RETRIES = 20

# Helper to get all required (group, image_row, image_col) from split CSVs
def get_required_images(split_files):
    required = set()
    for split_file in split_files:
        df = pd.read_csv(split_file)
        for _, row in df.iterrows():
            required.add((int(row['utm_zone']), int(row['row']), int(row['col']), int(row['image_row']), int(row['image_col'])))
    return required

def get_expected_filename(sat, utm_zone, image_row, image_col):
    if sat == "sentinel2":
        return f"{sat}/{utm_zone}_{image_row}_{image_col}_8.tif"
    elif sat == "naip":
        return f"{sat}/{utm_zone}_{image_row}_{image_col}.png"
    elif sat == "openstreetmap":
        return f"{sat}/{utm_zone}_{image_row}_{image_col}.geojson"
    else:
        return None

def extract_member(tar, member_name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / os.path.basename(member_name)
    if not out_path.exists():
        member = tar.getmember(member_name)
        with tar.extractfile(member) as src, open(out_path, 'wb') as dst:
            dst.write(src.read())

def is_valid_naip_tile(tar, fname):
    try:
        with tar.extractfile(fname) as f:
            img = Image.open(f)
            arr = np.array(img)
            if arr.ndim == 3:
                arr = arr[..., 0]  # Use first channel if multi-channel
            h, w = arr.shape
            if h < 256 or w < 256:
                return False
            # Sliding window check for 256x256 all nonzero
            for i in range(h - 256 + 1):
                for j in range(w - 256 + 1):
                    window = arr[i:i+256, j:j+256]
                    if np.all(window != 0):
                        return True
            return False
    except Exception:
        return False

def download_and_extract_group(utm_zone, row, col, images_for_group, add_random_images=True, updated_split_records=None, add_extra=3, add_extra_if_less=5):
    tar_paths = {}
    members = {}
    available_pairs = None
    for sat in SATELLITES:
        tar_name = f"{utm_zone}_{row}_{col}.tar"
        repo_path = f"{sat}_{utm_zone}/{tar_name}"
        local_tar_dir = OUTPUT_DIR
        local_tar_path = local_tar_dir / repo_path
        if not local_tar_path.exists():
            for retry in range(DOWNLOAD_RETRIES):
                try:
                    hf_hub_download(
                        repo_id=DATASET_NAME,
                        repo_type="dataset",
                        filename=repo_path,
                        local_dir=local_tar_dir,
                    )
                    break
                except Exception as e:
                    print(f"Retry #{retry}:", e)
                    time.sleep(1)
                    continue

        tar_paths[sat] = local_tar_path
        with tarfile.open(local_tar_path, 'r') as tar:
            sat_members = set()
            for m in tar.getmembers():
                if sat == "sentinel2" and not m.name.endswith("_8.tif"):
                    continue
                fname = m.name.split("/")[-1]
                parts = fname.split("_")
                if len(parts) < 3:
                    continue
                try:
                    image_row = int(parts[1])
                    image_col = int(parts[2].split(".")[0])
                except Exception:
                    continue
                sat_members.add((image_row, image_col))
            members[sat] = sat_members
            if available_pairs is None:
                available_pairs = sat_members.copy()
            else:
                available_pairs &= sat_members
    # Only keep pairs that are present in all sources
    available_pairs = set(available_pairs)
    required_pairs = set((img_row, img_col) for _, _, _, img_row, img_col in images_for_group)
    present_pairs = required_pairs & available_pairs
    n_needed = len(required_pairs) + (add_extra if len(required_pairs) < add_extra_if_less else 0)
    # If not enough, sample more from available_pairs
    if add_random_images and len(present_pairs) < n_needed:
        n_missing = n_needed - len(present_pairs)
        random.seed(f"{utm_zone}_{row}_{col}")
        extra_pairs = list(available_pairs - present_pairs)
        random.shuffle(extra_pairs)
        present_pairs = set(list(present_pairs) + extra_pairs[:n_missing])
    # Now, check NAIP validity only for present_pairs (from split or sampled)
    valid_pairs = set()
    naip_tar_path = tar_paths["naip"]
    with tarfile.open(naip_tar_path, 'r') as naip_tar:
        for image_row, image_col in present_pairs:
            naip_fname = get_expected_filename("naip", utm_zone, image_row, image_col)
            if is_valid_naip_tile(naip_tar, naip_fname):
                valid_pairs.add((image_row, image_col))
    # If still not enough, sample more from available_pairs not already checked
    if add_random_images and len(valid_pairs) < n_needed:
        already_checked = present_pairs
        candidates = list(available_pairs - already_checked)
        random.shuffle(candidates)
        with tarfile.open(naip_tar_path, 'r') as naip_tar:
            for image_row, image_col in candidates:
                if len(valid_pairs) >= n_needed:
                    break
                naip_fname = get_expected_filename("naip", utm_zone, image_row, image_col)
                if is_valid_naip_tile(naip_tar, naip_fname):
                    valid_pairs.add((image_row, image_col))
    # Save updated split records
    if updated_split_records is not None:
        for image_row, image_col in valid_pairs:
            urban_coverage = None
            for tup in images_for_group:
                if tup[3] == image_row and tup[4] == image_col:
                    if len(tup) > 5:
                        urban_coverage = tup[5]
                    break
            updated_split_records.append({
                'utm_zone': utm_zone,
                'row': row,
                'col': col,
                'image_row': image_row,
                'image_col': image_col,
                'urban_coverage': urban_coverage
            })
    # Extract for all 3 sources
    for sat in SATELLITES:
        tar_path = tar_paths[sat]
        with tarfile.open(tar_path, 'r') as tar:
            for image_row, image_col in valid_pairs:
                fname = get_expected_filename(sat, utm_zone, image_row, image_col)
                try:
                    member = tar.getmember(fname)
                except KeyError:
                    continue
                out_dir = OUTPUT_DIR / f"{sat}_{utm_zone}" / f"{utm_zone}_{row}_{col}"
                extract_member(tar, fname, out_dir)

def download_file_for_group_with_split(args, split_csvs, split_lookup):
    utm_zone, row, col, images_for_group = args
    split_records = {split: [] for split in split_csvs}
    def record_split(image_row, image_col, urban_coverage=None):
        key = (utm_zone, row, col, image_row, image_col)
        split = split_lookup.get(key, None)
        if split is None:
            split = "train.csv"
        split_records[split].append({
            'utm_zone': utm_zone,
            'row': row,
            'col': col,
            'image_row': image_row,
            'image_col': image_col,
            'urban_coverage': urban_coverage
        })
    local_updated = []
    download_and_extract_group(utm_zone, row, col, images_for_group, updated_split_records=local_updated)
    for rec in local_updated:
        record_split(rec['image_row'], rec['image_col'], rec.get('urban_coverage'))
    # Cleanup downloaded .tar files for this group
    # for sat in SATELLITES:
    #     tar_name = f"{utm_zone}_{row}_{col}.tar"
    #     repo_path = Path(f"{sat}_{utm_zone}/{tar_name}")
    #     local_tar_path = OUTPUT_DIR / repo_path
    #     if local_tar_path.exists():
    #         try:
    #             local_tar_path.unlink()
    #         except Exception as e:
    #             print(f"[WARN] Could not delete {local_tar_path}: {e}")
    return (utm_zone, row, col), split_records, None

def process_all_splits():
    split_csvs = ["val.csv", "test.csv", "train.csv"]
    split_files = [SPLIT_DIR / f for f in split_csvs if (SPLIT_DIR / f).exists()]
    required = get_required_images(split_files)
    group_dict = defaultdict(list)
    for tup in required:
        group = tup[:3]
        group_dict[group].append(tup)
    print(f"Need to download {len(group_dict)} groups, {len(required)} images")
    group_items = list(group_dict.items())
    random.seed(42)
    updated_split_records = {split: [] for split in split_csvs}
    split_lookup = {}
    for split, split_file in zip(split_csvs, split_files):
        df = pd.read_csv(split_file)
        for _, row in df.iterrows():
            key = (int(row['utm_zone']), int(row['row']), int(row['col']), int(row['image_row']), int(row['image_col']))
            split_lookup[key] = split
    def task(args):
        return download_file_for_group_with_split(args, split_csvs, split_lookup)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(task, (utm_zone, row, col, images)): (utm_zone, row, col) for (utm_zone, row, col), images in group_items}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading groups"):
            group, split_records, error = future.result()
            if error:
                print(f"[ERROR] Failed to download group {group}: {error}")
            else:
                for split, records in split_records.items():
                    updated_split_records[split].extend(records)
    for split in split_csvs:
        updated_df = pd.DataFrame(updated_split_records[split])
        out_path = SPLIT_DIR / split.replace('.csv', '_updated.csv')
        updated_df.to_csv(out_path, index=False)
        print(f"Updated split saved to {out_path} with {len(updated_df)} images.")

if __name__ == "__main__":
    process_all_splits()

