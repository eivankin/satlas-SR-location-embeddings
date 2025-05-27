"""We need to downscale NAIP from 1.25 m/pixel to 2.5m/pixel (x2) to match x4 upscale factor from Sentinel-2 10m/pixel."""
from pathlib import Path
import rasterio
from rasterio.enums import Resampling

NAIP_MOSAIC = Path("naip_rgb_cropped_adjusted/merged_naip_adj.tif")
OUTPUT_DIR = Path("naip_rgb_cropped_adjusted_downscaled")

if __name__ == '__main__':
    OUTPUT_DIR.mkdir(exist_ok=True)
    with rasterio.open(NAIP_MOSAIC) as src:
        # Calculate new dimensions (downscale factor of 2)
        new_height = src.height // 2
        new_width = src.width // 2

        # Resample data using average resampling
        data_downscaled = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.average
        )

        # Update transform: scale pixel size accordingly
        new_transform = src.transform * src.transform.scale(
            src.width / new_width,
            src.height / new_height
        )
        out_profile = src.profile.copy()
        out_profile.update({
            'height': new_height,
            'width': new_width,
            'transform': new_transform
        })

    out_file = OUTPUT_DIR / NAIP_MOSAIC.name.replace(".tif", "_downscaled.tif")
    with rasterio.open(out_file, 'w', **out_profile) as dst:
        dst.write(data_downscaled)
    print(f"Downscaled NAIP mosaic saved as: {out_file}")
