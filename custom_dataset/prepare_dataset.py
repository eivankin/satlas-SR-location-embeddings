"""
1. Downscale NAIP data from 512x512 to 256x256
2. Scale S2 data (divide by 10 & clip to (0, 255) to cast to 8 bit)
3. Select k non-cloudy (by pixels values) S2 scenes for each tile so they are in first 4*K channels
4. Merge S2 scenes within tile using median & quantiles (.18?). Add them to the last 4 channels after the cloudless scenes.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd
import rasterio
from tqdm import tqdm
import shutil

# Settings
SPLIT_DIR = Path("./splits")
DATA_DIR = Path("./downloaded")
OUT_DIR = Path("./prepared")
OUT_DIR.mkdir(exist_ok=True)
K = 8  # Number of non-cloudy S2 scenes to select per tile
CLOUD_THRESHOLD = 220  # Pixel value above which is considered cloudy for S2
STATS_TO_MERGE = ["q_50", "q_18"]  # "q_50" is median

# Helper: Crop zero rows/cols from NAIP and downscale to 256x256
def process_naip(img_path):
    img = Image.open(img_path)
    # arr = np.array(img)
    # # Crop zero rows/cols
    # mask = np.any(arr != 0, axis=-1) if arr.ndim == 3 else (arr != 0)
    # rows = np.where(mask.any(axis=1))[0]
    # cols = np.where(mask.any(axis=0))[0]
    # if len(rows) == 0 or len(cols) == 0:
    #     return None
    # arr_cropped = arr[rows[0]:rows[-1]+1, cols[0]:cols[-1]+1]
    # # Downscale to 256x256 or less
    # img_cropped = Image.fromarray(arr_cropped)
    # cropped_width, cropped_height = img_cropped.size
    # if cropped_width != 512 or cropped_height != 512:
    #     print(cropped_width, cropped_height)
    # img_resized = img_cropped.resize((cropped_width // 2, cropped_height // 2), Image.BILINEAR)
    img_resized = img.resize((256, 256), Image.BILINEAR)
    return np.array(img_resized)

# Helper: Scale S2 (divide by 10, clip to 0-255, cast to uint8)
def process_s2(img_path):
    with rasterio.open(img_path) as src:
        arr = src.read()  # shape: (4*N, H, W)
    arr = arr.astype(np.float32) / 10.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    # Split into scenes: each scene is 4 bands
    n_scenes = arr.shape[0] // 4
    scenes = [arr[i*4:(i+1)*4] for i in range(n_scenes)]
    return scenes  # list of (4, H, W)

# Helper: Calculate cloud fraction
def cloud_fraction(arr):
    return np.mean(arr > CLOUD_THRESHOLD)

# Helper: Save S2 as GeoTIFF
def save_s2_tif(s2_stack, out_path, transform, crs):
    # s2_stack: (C, H, W), C = 4*K+16
    count, height, width = s2_stack.shape
    dtype = s2_stack.dtype
    assert dtype == np.uint8
    with rasterio.open(
        out_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform
    ) as dst:
        for i in range(count):
            dst.write(s2_stack[i], i+1)

# Main processing loop
def process_split(split_csv):
    df = pd.read_csv(SPLIT_DIR / split_csv).drop(columns="urban_coverage")
    split_out_dir = OUT_DIR / split_csv.replace("_merged.csv", "")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_csv}"):
        utm_zone, row_id, col_id, image_row, image_col = row['utm_zone'], row['row'], row['col'], row['image_row'], row['image_col']
        group_id = f"{utm_zone}_{row_id}_{col_id}"
        # NAIP
        naip_path = DATA_DIR / f"naip_{utm_zone}" / group_id / f"{utm_zone}_{image_row}_{image_col}.png"
        if not naip_path.exists():
            raise ValueError(f"NAIP file not found: {naip_path}")
        naip = process_naip(naip_path)
        if naip is None:
            raise ValueError(f"NAIP file is empty: {naip_path}")
        # S2: single .tif for this tile
        s2_path = DATA_DIR / f"sentinel2_{utm_zone}" / group_id / f"{utm_zone}_{image_row}_{image_col}_8.tif"
        if not s2_path.exists():
            raise ValueError(f"S2 file not found: {s2_path}")
        s2_scenes = process_s2(str(s2_path))  # list of (4, H, W)
        if not s2_scenes:
            raise ValueError(f"S2 file is empty: {s2_path}")
        # Select K least cloudy (by checking all 4 bands)
        cloud_fracs = [cloud_fraction(arr) for arr in s2_scenes]
        sorted_indices = np.argsort(cloud_fracs)
        selected = [s2_scenes[i] for i in sorted_indices[:K]]
        if len(selected) < K:
            raise ValueError(f"Not enough non-cloudy S2 scenes for ({image_row}, {image_col}): {len(selected)} < {K}")
        # Merge S2 scenes (all, not just non-cloudy)
        all_s2 = np.stack(s2_scenes, axis=0)  # (N, 4, H, W)
        stats = []
        for stat in STATS_TO_MERGE:
            quantile = int(stat.split("_")[-1]) / 100
            stats.append(np.quantile(all_s2, quantile, axis=0).astype(np.uint8))
        if not stats:
            continue
        merged_stats = np.stack(stats, axis=0)  # (len(STATS_TO_MERGE), 4, H, W)
        merged_stats = merged_stats.reshape(-1, merged_stats.shape[2], merged_stats.shape[3])  # (4*len(STATS_TO_MERGE), H, W)
        # Stack: first 4*K channels are least cloudy S2, last 4*len(STATS_TO_MERGE) are merged stats
        s2_stack = np.concatenate([np.concatenate(selected, axis=0), merged_stats], axis=0)
        # Save NAIP and S2 for this tile only, preserving folder structure and georeference
        out_dir = split_out_dir / f"sentinel2_{utm_zone}" / group_id
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save georeference from original S2 tif
        with rasterio.open(s2_path) as src:
            transform = src.transform
            crs = src.crs
        # Save S2 as GeoTIFF
        s2_tif_path = out_dir / f"{utm_zone}_{image_row}_{image_col}.tif"
        save_s2_tif(s2_stack, s2_tif_path, transform, crs)
        # Save NAIP as PNG in the same structure
        naip_out_dir = split_out_dir / f"naip_{utm_zone}" / group_id
        naip_out_dir.mkdir(parents=True, exist_ok=True)
        naip_png_path = naip_out_dir / f"{utm_zone}_{image_row}_{image_col}.png"
        Image.fromarray(naip).save(naip_png_path)
        # Copy OSM geojson if exists
        osm_src = DATA_DIR / f"openstreetmap_{utm_zone}" / group_id / f"{utm_zone}_{image_row}_{image_col}.geojson"
        if osm_src.exists():
            osm_out_dir = split_out_dir / f"openstreetmap_{utm_zone}" / group_id
            osm_out_dir.mkdir(parents=True, exist_ok=True)
            osm_dst = osm_out_dir / f"{utm_zone}_{image_row}_{image_col}.geojson"
            shutil.copy2(osm_src, osm_dst)

if __name__ == "__main__":
    for split in ["val_merged.csv", "test_merged.csv", "train_merged.csv"]:
        process_split(split)

