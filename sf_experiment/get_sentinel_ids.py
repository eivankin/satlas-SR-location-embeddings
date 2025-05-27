"""
So there is a 38 GB CSV (be careful not to overflow memory with it) that maps UTM zone + row + column + sequence number in tile to Sentinel-2 image ID.
In script 'prepare_s2.py' I tried to take just first 8 images from each tile,
but it is not guaranteed that i-th temporal image from tile is from the same
timestamp as i-th temporal image from another tile, which resulted in visible inconsistencies.

In this script I extract all Sentinel-2 image IDs from the CSV for given tiles,
then find scene IDs that are present for all tiles, select the first 8 common scenes (sorted in ascending order),
map them to sequence numbers for each tile, and save the mapping for later use in 'prepare_s2.py'.

FIXME: this approach is not working - there is no common scenes for ROI.
"""

from pathlib import Path
import csv
from tqdm import tqdm

SENTINEL_CSV = Path("sentinel2.csv")  # CSV columns: projection, col, row, index, scene
SENTINEL_TILES = list(Path("./raw_data/sentinel2").glob("*.tif"))
# Each tile filename is expected to be like "32610_898_-6475.tif": UTM zone, col, row.
TILE_IDS = [tuple(map(int, tile.name.split("_")[:-1])) for tile in SENTINEL_TILES]
tile_ids_set = set(TILE_IDS)
max_utm_proj = max(tile_id[0] for tile_id in TILE_IDS)

# Prepare a dictionary to hold scene->sequence mapping for each tile.
tile_scene_map = {tile_id: {} for tile_id in tile_ids_set}

# Process CSV line by line to avoid memory issues.
with open(SENTINEL_CSV, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in tqdm(reader):
        # Expecting: projection, col, row, index, scene
        try:
            proj = int(row[0])
            if proj > max_utm_proj:
                break
            col = int(row[1])
            row_num = int(row[2])
            seq = int(row[3])
            scene = row[4]
        except Exception:
            continue
        tile_key = (proj, col, row_num)
        if tile_key in tile_ids_set:
            # If duplicate scenes exist, keep the smallest sequence number.
            if scene in tile_scene_map[tile_key]:
                tile_scene_map[tile_key][scene] = min(tile_scene_map[tile_key][scene], seq)
            else:
                tile_scene_map[tile_key][scene] = seq

# Compute the intersection of scene IDs across all tiles.
common_scenes = None
for mapping in tile_scene_map.values():
    scenes = set(mapping.keys())
    if common_scenes is None:
        common_scenes = scenes.copy()
    else:
        common_scenes.intersection_update(scenes)
if common_scenes is None:
    common_scenes = set()

# Select the first 8 common scenes (sorted in ascending order).
selected_common_scenes = sorted(common_scenes)[:8]
print("Selected common scenes:", selected_common_scenes)

# Map selected common scenes to sequence numbers for each tile.
tile_seq_mapping = {}
for tile_id, scene_map in tile_scene_map.items():
    tile_seq_mapping[tile_id] = {scene: scene_map[scene] for scene in selected_common_scenes}

# Save the mapping to a file.
output_file = Path("sentinel2_common_sequences.txt")
with open(output_file, 'w') as f_out:
    for tile_id in sorted(tile_seq_mapping.keys()):
        tile_str = "_".join(map(str, tile_id))
        seq_numbers = [str(tile_seq_mapping[tile_id][scene]) for scene in selected_common_scenes]
        # Format: tileID: seq1, seq2, ..., seq8
        f_out.write(f"{tile_str}: {', '.join(seq_numbers)}\n")

print("Tile to common sequence mapping saved to:", output_file)
