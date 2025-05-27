from pathlib import Path
import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm
import json

def get_tile_center_coords(transform, row, col, tile_size):
    # Compute the center pixel of the tile
    center_row = row * tile_size + tile_size // 2
    center_col = col * tile_size + tile_size // 2
    # Use rasterio's transform to get (x, y) in CRS
    x, y = rasterio.transform.xy(transform, center_row, center_col, offset='center')
    return y, x  # Return as (lat, lon) if CRS is EPSG:4326, else (y, x)

def preprocess_and_tile(tci_path, tile_size=(32, 32), overlap=0):
    with rasterio.open(tci_path) as src:
        out_image = np.clip(src.read() / 10, 0, 255).astype(np.uint8)
        transform = src.transform
        crs = src.crs

    tiles = []
    stride = (tile_size[0] - overlap, tile_size[1] - overlap)
    for i in range(0, out_image.shape[1] - overlap, stride[0]):
        for j in range(0, out_image.shape[2] - overlap, stride[1]):
            tile = out_image[:, i:i + tile_size[0], j:j + tile_size[1]]
            if tile.shape[1] == tile_size[0] and tile.shape[2] == tile_size[1]:
                tiles.append((i // stride[0], j // stride[1], tile, transform, crs))
    return tiles

if __name__ == "__main__":
    OVERLAP_SIZE = 0
    TILE_SIZE = 32
    OUTPUT_ROOT = Path(f"./tiles_q18/s2-tiles-overlap-{OVERLAP_SIZE:02d}-{TILE_SIZE}/input/0_0")
    print(f"Output directory: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    S2_TCI_FILES = Path("./raw_data/sentinel2_rgb_cropped_adjusted").glob("*.tif")

    all_tiles = {}
    tile_coords = {}
    for tci_path in tqdm(S2_TCI_FILES, "Files", total=8):
        tiles = preprocess_and_tile(tci_path, tile_size=(TILE_SIZE, TILE_SIZE), overlap=OVERLAP_SIZE)
        for row, col, tile, transform, crs in tiles:
            # Save tile as before...
            lat, lon = get_tile_center_coords(transform, row, col, TILE_SIZE)
            tile_coords[f"{row}_{col}"] = [lat, lon]
            if (row, col) not in all_tiles:
                all_tiles[(row, col)] = []
            all_tiles[(row, col)].append(tile)

    for (row, col), tile_list in all_tiles.items():
        if len(tile_list) == 8:
            stacked_tiles = np.stack(tile_list, axis=0)
            stacked_tiles = np.moveaxis(stacked_tiles, 1, -1)
            q18_tile = np.quantile(stacked_tiles, 0.18, axis=0).astype(np.uint8)
            output_image = Image.fromarray(q18_tile)
            output_image.save(OUTPUT_ROOT / f"{row}_{col}.png")
        else:
            raise ValueError(f"Unexpected number of tiles: {len(tile_list)}")

    with open(OUTPUT_ROOT / "tile_coords.json", "w") as f:
        json.dump(tile_coords, f, indent=2)

    print("Processing complete. Tile coordinates saved to tile_coords.json.")