"""
1. For each tile in each split, extract OSM buildings polygons
2. Convert these polygons into mask spanning the tile, if mask is empty (less than K buildings pixels) - skip it.
3. Save the masks as images, keeping the splits & the same folder structure as other dataset parts (use "osm_masks" as id)
"""

import os
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import rasterio.features
from shapely import affinity
from shapely.geometry import shape, mapping
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Settings
SPLIT_DIR = Path("./splits")
SPLIT_FILES = ["train_merged.csv", "val_merged.csv", "test_merged.csv"]
OSM_ROOT = Path("./prepared")
MASK_ROOT = Path("./prepared")
TILE_SIZE = 256
MIN_PIXELS = 10
VISUALIZE = False  # Set to True to enable visualization
VIS_DIR = Path("./mask_debug")
VIS_DIR.mkdir(exist_ok=True)

def get_osm_geojson_path(utm_zone, group_id, chip_name, split):
    return OSM_ROOT / split / f"openstreetmap" / group_id / f"{chip_name}.geojson"

def get_mask_save_path(utm_zone, group_id, chip_name, split):
    out_dir = MASK_ROOT / split / f"osm_masks" / group_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{chip_name}.png"

def extract_building_polygons(geojson_path, downscale=0.5):
    if not geojson_path.exists():
        raise ValueError(f"GeoJSON file not found: {geojson_path}")
    with open(geojson_path) as f:
        data = json.load(f)
    polygons = []
    if not data:
        return polygons
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        if props.get("category", "") == "building":
            geom = feature.get("geometry", None)
            if geom is not None:
                try:
                    poly = shape(geom)
                    if poly.is_valid:
                        # Downscale coordinates
                        poly = affinity.scale(poly, xfact=downscale, yfact=downscale, origin=(0, 0))
                        polygons.append(poly)
                except Exception:
                    continue
    return polygons

def visualize_polygons_on_naip(naip_path, polygons, out_path):
    img = np.array(Image.open(naip_path))
    if img.shape[-1] == 4:  # If RGBA, drop alpha
        img = img[..., :3]
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img)
    for poly in polygons:
        if poly.is_empty:
            continue
        x, y = poly.exterior.xy
        ax.plot(x, y, color='blue', linewidth=2)
    ax.set_title("Polygons on NAIP")
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    plt.savefig(out_path)
    plt.close(fig)

def visualize_mask(mask, out_path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(mask, cmap='gray')
    ax.set_title("Rasterized Mask")
    plt.savefig(out_path)
    plt.close(fig)

def rasterize_polygons(polygons, out_shape=(TILE_SIZE, TILE_SIZE)):
    if not polygons:
        return np.zeros(out_shape, dtype=np.uint8)
    mask = rasterio.features.rasterize(
        [(mapping(poly), 1) for poly in polygons],
        out_shape=out_shape,
        fill=0,
        dtype=np.uint8
    )
    return mask

def main():
    for split_file in SPLIT_FILES:
        split = split_file.replace("_merged.csv", "")
        split_path = SPLIT_DIR / split_file
        if not split_path.exists():
            continue
        df = pd.read_csv(split_path).drop(columns="urban_coverage")
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_file}"):
            utm_zone = int(row['utm_zone'])
            group_id = f"{utm_zone}_{row['row']}_{row['col']}"
            chip_name = f"{utm_zone}_{row['image_row']}_{row['image_col']}"
            geojson_path = get_osm_geojson_path(utm_zone, group_id, chip_name, split)
            polygons = extract_building_polygons(geojson_path)
            naip_path = OSM_ROOT / split / "naip" / group_id / f"{chip_name}.png"
            if VISUALIZE and polygons and naip_path.exists():
                vis_naip_poly_path = VIS_DIR / f"{chip_name}_naip_polygons.png"
                visualize_polygons_on_naip(naip_path, polygons, vis_naip_poly_path)
            mask = rasterize_polygons(polygons)
            if VISUALIZE and polygons:
                vis_mask_path = VIS_DIR / f"{chip_name}_mask.png"
                visualize_mask(mask, vis_mask_path)
            # break
            if np.sum(mask) < MIN_PIXELS:
                continue
            mask_path = get_mask_save_path(utm_zone, group_id, chip_name, split)
            Image.fromarray(mask * 255).save(mask_path)

if __name__ == "__main__":
    main()