"""
I have multi-temporal georeferenced sentinel-2 64x64 tiles in ./sentinel2 dir made of channels B02, B03, B04, B08 (B, G, R, NIR).
Multiple images from different timestamps are stacked together in a single file, so there are more than 4 channels in each.
I want to stitch them together & take only first 8 images from them & only RGB,
also getting rid of multi-temporality, so each image is a separate file.

So it works now.
The data is kind of flawed: i-th temporal image from one tile often is not from the same timestamp as i-th temporal image from another tile, resulting in inconsistencies.
But for multi-temporal model it should be fine & NAIP data should not have the same problems.
_t6 file is perfect!
"""
from pathlib import Path
import rasterio
import numpy as np
from rasterio.merge import merge
import tempfile
import os

def process_and_merge_tiles(files_list, output_dir):
    if not files_list:
        return

    # Read first file to get number of timestamps and basic profile
    with rasterio.open(files_list[0]) as src:
        num_timestamps = src.count // 4
        timestamps_to_process = min(8, num_timestamps)
        base_profile = src.profile.copy()

    # Process each timestamp
    for t in range(timestamps_to_process):
        temp_files = []
        datasets_to_merge = []

        try:
            for file_path in files_list:
                with rasterio.open(file_path) as src:
                    data = src.read()
                    rgb_bands = np.stack([
                        data[t*4 + 2],  # B04 (R)
                        data[t*4 + 1],  # B03 (G)
                        data[t*4 + 0],  # B02 (B)
                    ])

                    # Create temporary file with original tile's profile
                    temp_profile = src.profile.copy()  # Use each tile's own profile
                    temp_profile.update(count=3, dtype=rgb_bands.dtype)
                    
                    temp_file = tempfile.NamedTemporaryFile(suffix='.tif', delete=False)
                    temp_files.append(temp_file.name)
                    
                    with rasterio.open(temp_file.name, 'w', **temp_profile) as tmp:
                        tmp.write(rgb_bands)
                    
                    datasets_to_merge.append(rasterio.open(temp_file.name))

            # Merge all tiles for this timestamp
            merged_array, merged_transform = merge(datasets_to_merge)

            # Create output profile with merged dimensions
            out_profile = base_profile.copy()
            out_profile.update(
                count=3,
                height=merged_array.shape[1],
                width=merged_array.shape[2],
                transform=merged_transform,
                dtype=merged_array.dtype
            )

            # Save merged image for this timestamp
            output_file = output_dir / f"merged_t{t}.tif"
            with rasterio.open(output_file, 'w', **out_profile) as dst:
                dst.write(merged_array)

        finally:
            # Clean up: close datasets and remove temporary files
            for dataset in datasets_to_merge:
                dataset.close()
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except:
                    pass

if __name__ == '__main__':
    files_to_merge = sorted(Path("./sentinel2").glob("*.tif"))
    print("# tiles to merge:", len(files_to_merge))

    # Create output directory
    output_dir = Path("./sentinel2_rgb")
    output_dir.mkdir(exist_ok=True)

    # Group files by their position if needed
    # If all files are already adjacent, we can process them directly
    print("Merging tiles for each timestamp...")
    process_and_merge_tiles(files_to_merge, output_dir)
