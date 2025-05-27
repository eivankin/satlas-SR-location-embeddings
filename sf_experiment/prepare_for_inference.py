from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from tqdm import tqdm


# Function to preprocess and tile the TCI
def preprocess_and_tile(tci_path, tile_size=(32, 32), overlap=0):
    with rasterio.open(tci_path) as src:
        # Magic numbers for scaling taken from
        # https://huggingface.co/datasets/allenai/s2-naip/blob/main/visualize_tile.py
        out_image = np.clip(src.read() / 10, 0, 255).astype(np.uint8)

    # Split into tiles with overlap
    tiles = []
    stride = (tile_size[0] - overlap, tile_size[1] - overlap)
    for i in range(0, out_image.shape[1] - overlap, stride[0]):
        for j in range(0, out_image.shape[2] - overlap, stride[1]):
            tile = out_image[:, i:i + tile_size[0], j:j + tile_size[1]]
            if tile.shape[1] == tile_size[0] and tile.shape[2] == tile_size[1]:
                tiles.append((i // stride[0], j // stride[1], tile))

    return tiles

if __name__ == "__main__":
    OVERLAP_SIZE = 16
    TILE_SIZE = 32
    OUTPUT_ROOT = Path(f"./tiles/s2-tiles-overlap-{OVERLAP_SIZE:02d}-{TILE_SIZE}/input/0_0")
    print(f"Output directory: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    S2_TCI_FILES = Path("./raw_data/sentinel2_rgb_cropped_adjusted").glob("*.tif")

    all_tiles = {}
    # Process each downloaded file
    for tci_path in tqdm(S2_TCI_FILES, "Files", total=8):
        tiles = preprocess_and_tile(tci_path, tile_size=(TILE_SIZE, TILE_SIZE), overlap=OVERLAP_SIZE)

        # Collect tiles with the same indices
        for row, col, tile in tiles:
            if (row, col) not in all_tiles:
                all_tiles[(row, col)] = []
            all_tiles[(row, col)].append(tile)

    # Save tiles with the same indices
    for (row, col), tile_list in all_tiles.items():
        if len(tile_list) == 8:  # Ensure we have exactly 8 tiles for each index
            # Stack tiles along the first axis to create [8, 32, 32, 3]
            stacked_tiles = np.stack(tile_list, axis=0)
            stacked_tiles = np.moveaxis(stacked_tiles, 1, -1)  # Move channels to the last dimension

            # Save as PNG
            output_image = Image.fromarray(stacked_tiles.astype(np.uint8).reshape((8 * TILE_SIZE, TILE_SIZE, 3)))
            output_image.save(OUTPUT_ROOT / f"{row}_{col}.png")
        else:
            raise ValueError(f"Unexpected number of tiles: {len(tile_list)}")

    print("Processing complete.")
