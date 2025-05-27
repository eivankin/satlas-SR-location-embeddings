import rasterio
from rasterio.transform import Affine
import numpy as np
from PIL import Image

def create_georeferenced_png(stitched_png_path, cropped_tif_path, output_geotiff_path, upscale_factor=1):
    """
    Creates a geo-referenced GeoTIFF from a stitched PNG file using metadata from a cropped TIFF file.
    Supports upscaled PNG files with a known upscale factor.

    Args:
        stitched_png_path (str): Path to the stitched PNG file.
        cropped_tif_path (str): Path to the original cropped TIFF file (for metadata).
        output_geotiff_path (str): Path to save the output geo-referenced GeoTIFF.
        upscale_factor (int): Upscale factor of the stitched PNG (default is 1, no upscaling).
    """
    # Open the stitched PNG file
    with Image.open(stitched_png_path) as img:
        stitched_data = np.array(img)  # Read the PNG data as a numpy array

    # Open the original cropped TIFF file to extract metadata
    with rasterio.open(cropped_tif_path) as src:
        # Get the metadata from the cropped TIFF
        meta = src.meta.copy()
        transform = src.transform  # Geotransform of the cropped TIFF
        crs = src.crs  # Coordinate reference system

        # Calculate the new transform for the stitched PNG
        # The stitched PNG may be upscaled, so we adjust the transform to account for the upscale factor
        new_transform = Affine(
            transform.a / upscale_factor, transform.b, transform.c,
            transform.d, transform.e / upscale_factor, transform.f
        )

        # Update metadata for the output GeoTIFF
        meta.update({
            "driver": "GTiff",
            "height": stitched_data.shape[0],
            "width": stitched_data.shape[1],
            "count": 3,  # Assuming RGB (3 bands)
            "dtype": "uint8",
            "transform": new_transform,
            "crs": crs
        })

    # Save the stitched data as a geo-referenced GeoTIFF
    with rasterio.open(output_geotiff_path, 'w', **meta) as dst:
        # Write the RGB bands to the GeoTIFF
        dst.write(stitched_data[:, :, 0], 1)  # Red band
        dst.write(stitched_data[:, :, 1], 2)  # Green band
        dst.write(stitched_data[:, :, 2], 3)  # Blue band

    print(f"Geo-referenced GeoTIFF saved to {output_geotiff_path}")

# Example usage
stitched_png_path = "testing_infer_grid/0_0/stitched_sr_2.png"
cropped_tif_path = "s2-images/S2B_MSIL1C_20240725T080609_N0511_R078_T39UUB_20240725T090127/GRANULE/L1C_T39UUB_A038571_20240725T081238/IMG_DATA/T39UUB_20240725T080609_TCI_rep.jp2"
output_geotiff_path = stitched_png_path.replace(".png", "_georef.tif")
create_georeferenced_png(stitched_png_path, cropped_tif_path, output_geotiff_path, upscale_factor=4)
