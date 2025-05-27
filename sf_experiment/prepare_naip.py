"""
So we prepared S2 data in 'prepare_s2.py' script. Now we need to do it with NAIP data.
The data is packed in .TAR files. Firstly, we need to extract tiles with the same names as S2 data.
Each NAIP tile is an RGBA .png, where Alpha channel corresponds for NIR channel. We will drop it, we are interested in RGB only.
As an extra step, we will compute georeference from tile IDs. 
Each NAIP tile is 512x512 with 1.25 m/pixel.
Filename example: 32610_898_-6475
UTM zone is the first part; second and third parts are tile column and row.
"""
from pathlib import Path
import tarfile
import numpy as np
from PIL import Image
import rasterio
from rasterio.merge import merge
import tempfile
import os
from affine import Affine
from rasterio.crs import CRS

def remove_last_suffix(file_name: str) -> str:
    return file_name.rsplit('_', 1)[0]

NAIP_TAR_FILES = [Path("32610_27_-203_naip.tar"), Path("32610_28_-203_naip.tar")]

# Output directory for processed NAIP tiles and mosaic
OUTPUT_DIR = Path("./naip_rgb")
OUTPUT_DIR.mkdir(exist_ok=True)

if __name__ == '__main__':
    # Collect tile names from S2 data (filtering)
    s2_tiles = list(Path("./sentinel2").glob("*.tif"))
    tile_names = set(remove_last_suffix(tile.name) for tile in s2_tiles)
    print("Processing NAIP tiles corresponding to S2 tiles...")
    print("S2 tile bases:", list(tile_names)[:5])
    
    temp_files = []       # List of temporary file paths
    datasets_to_merge = []  # List of open rasterio datasets for merging

    # Process each TAR file and extract NAIP tiles as temporary files if the base name is in S2
    for tar_path in NAIP_TAR_FILES:
        print(f"Processing {tar_path.name}")
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                base_name = Path(member.name).stem
                # Expect base_name format: "32610_898_-6475"
                parts = base_name.split('_')
                if len(parts) != 3:
                    continue
                # Filter: only process NAIP tiles that have matching S2 tile names
                if base_name not in tile_names:
                    continue
                utm_zone = parts[0]
                try:
                    tile_col = int(parts[1])
                    tile_row = int(parts[2])
                except ValueError:
                    continue

                # Extract the PNG file into memory and drop alpha channel
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                with Image.open(extracted) as img:
                    rgb_img = img.convert("RGB")
                    naip_array = np.array(rgb_img)

                # Compute geotransform from tile id:
                # Each tile is 512x512 pixels at 1.25 m/pixel -> extent = 640 m.
                resolution = 1.25
                tile_size = 512
                x_origin = tile_col * tile_size * resolution
                y_origin = (-tile_row) * tile_size * resolution
                transform = Affine.translation(x_origin, y_origin) * Affine.scale(resolution, -resolution)

                # Build profile with computed transform and proper CRS from the UTM zone.
                profile = {
                    'driver': 'GTiff',
                    'height': naip_array.shape[0],
                    'width': naip_array.shape[1],
                    'count': 3,
                    'dtype': naip_array.dtype,
                    'crs': CRS.from_epsg(int(utm_zone)),
                    'transform': transform
                }
                # Write NAIP tile to a temporary file
                temp = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
                temp_files.append(temp.name)
                with rasterio.open(temp.name, 'w', **profile) as dst:
                    for i in range(3):
                        dst.write(naip_array[:, :, i], i+1)
                # Open temporary file as dataset for merging
                ds = rasterio.open(temp.name)
                datasets_to_merge.append(ds)
                print(f"Processed {base_name}")

    # Merge all NAIP tiles into a single mosaic
    if datasets_to_merge:
        merged_array, merged_transform = merge(datasets_to_merge)
        out_profile = datasets_to_merge[0].profile.copy()
        out_profile.update({
            'count': 3,
            'height': merged_array.shape[1],
            'width': merged_array.shape[2],
            'transform': merged_transform,
            'dtype': merged_array.dtype
        })
        output_file = OUTPUT_DIR / "merged_naip.tif"
        with rasterio.open(output_file, 'w', **out_profile) as dst:
            dst.write(merged_array)
        print(f"Merged NAIP mosaic saved as {output_file.name}")

    # Clean up temporary files and datasets
    for ds in datasets_to_merge:
        ds.close()
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except Exception:
            pass
