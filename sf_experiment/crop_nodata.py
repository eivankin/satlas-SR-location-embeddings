"""
A few first rows of NAIP mosaic are covered by NODATA values (zero pixels).
We crop them to keep only valid data, then crop Sentinel-2 mosaics to the same extent.
"""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

# Input files and directories
NAIP_MOSAIC = Path("naip_rgb/merged_naip.tif")
SENTINEL_FOLDER = Path("sentinel2_rgb")
# Output directories for cropped files
NAIP_OUT_DIR = Path("naip_rgb_cropped")
SENTINEL_OUT_DIR = Path("sentinel2_rgb_cropped")
NAIP_OUT_DIR.mkdir(exist_ok=True)
SENTINEL_OUT_DIR.mkdir(exist_ok=True)

def crop_raster_by_nodata(ds, nodata_value=0):
    """
    Finds the first row (from the top) where at least one pixel is non-zero across all bands,
    then returns a cropping window from that row to the end.
    """
    arr = ds.read()  # shape: (bands, height, width)
    # Compute a mask for valid (non-zero) pixels across bands
    valid_mask = (arr != nodata_value).any(axis=0)  # shape: (height, width)
    # For each row, check if any pixel is valid
    valid_rows = valid_mask.any(axis=1)  # shape: (height,)
    if not valid_rows.any():
        return Window(0, 0, ds.width, ds.height)  # no valid data; return full window
    first_valid_row = int(np.argmax(valid_rows))
    # Define window: full width, from first_valid_row to bottom.
    new_height = ds.height - first_valid_row
    return Window(col_off=0, row_off=first_valid_row, width=ds.width, height=new_height), first_valid_row

def main():
    # Original processing: crop NAIP mosaic by nodata
    with rasterio.open(NAIP_MOSAIC) as naip_ds:
        crop_window, row_offset = crop_raster_by_nodata(naip_ds, nodata_value=0)
        cropped_naip = naip_ds.read(window=crop_window)
        new_transform = naip_ds.window_transform(crop_window)
        cropped_bounds = rasterio.windows.bounds(crop_window, transform=naip_ds.transform)
        out_profile = naip_ds.profile.copy()
        out_profile.update({
            'height': int(crop_window.height),
            'width': int(crop_window.width),
            'transform': new_transform
        })
    naip_outfile = NAIP_OUT_DIR / NAIP_MOSAIC.name.replace(".tif", "_cropped.tif")
    with rasterio.open(naip_outfile, 'w', **out_profile) as dst:
        dst.write(cropped_naip)
    print(f"Cropped NAIP mosaic saved as: {naip_outfile}")

    # Original processing: crop each Sentinel mosaic using NAIP cropped bounds
    sentinel_files = list(SENTINEL_FOLDER.glob("*.tif"))
    for sf in sentinel_files:
        with rasterio.open(sf) as s2_ds:
            window = from_bounds(
                cropped_bounds[0], cropped_bounds[1],
                cropped_bounds[2], cropped_bounds[3],
                transform=s2_ds.transform,
                width=s2_ds.width,
                height=s2_ds.height
            )
            window = Window(
                col_off=int(np.floor(window.col_off)),
                row_off=int(np.floor(window.row_off)),
                width=int(np.ceil(window.width)),
                height=int(np.ceil(window.height))
            )
            cropped_s2 = s2_ds.read(window=window)
            new_s2_transform = s2_ds.window_transform(window)
            out_profile_s2 = s2_ds.profile.copy()
            out_profile_s2.update({
                'height': int(window.height),
                'width': int(window.width),
                'transform': new_s2_transform
            })
        out_file = SENTINEL_OUT_DIR / sf.name.replace(".tif", "_cropped.tif")
        with rasterio.open(out_file, 'w', **out_profile_s2) as dst:
            dst.write(cropped_s2)
        print(f"Cropped Sentinel mosaic saved as: {out_file}")

    # NEW STEPS: Adjust mosaics to remove the upper offset by first cropping the first row from S2,
    # then using the resulting bounds to crop NAIP accordingly.
    NAIP_OUT_ADJ_DIR = Path("naip_rgb_cropped_adjusted")
    SENTINEL_OUT_ADJ_DIR = Path("sentinel2_rgb_cropped_adjusted")
    NAIP_OUT_ADJ_DIR.mkdir(exist_ok=True)
    SENTINEL_OUT_ADJ_DIR.mkdir(exist_ok=True)

    new_sentinel_bounds = None
    for sf in SENTINEL_OUT_DIR.glob("*.tif"):
        with rasterio.open(sf) as s2_ds:
            # Crop the first row: remove row_off=0 by starting at row 1.
            adj_window = Window(col_off=0, row_off=1, width=s2_ds.width, height=s2_ds.height - 1)
            cropped_adj = s2_ds.read(window=adj_window)
            adj_transform = s2_ds.window_transform(adj_window)
            # Use the bounds from the first adjusted Sentinel mosaic as reference.
            if new_sentinel_bounds is None:
                new_sentinel_bounds = rasterio.windows.bounds(adj_window, transform=s2_ds.transform)
            adj_profile = s2_ds.profile.copy()
            adj_profile.update({
                'height': int(adj_window.height),
                'width': int(adj_window.width),
                'transform': adj_transform
            })
        out_adj_file = SENTINEL_OUT_ADJ_DIR / sf.name.replace(".tif", "_adj.tif")
        with rasterio.open(out_adj_file, 'w', **adj_profile) as dst:
            dst.write(cropped_adj)
        print(f"Adjusted Sentinel mosaic saved as: {out_adj_file}")

    if new_sentinel_bounds is not None:
        with rasterio.open(NAIP_MOSAIC) as naip_ds:
            new_window = from_bounds(
                new_sentinel_bounds[0], new_sentinel_bounds[1],
                new_sentinel_bounds[2], new_sentinel_bounds[3],
                transform=naip_ds.transform,
                width=naip_ds.width,
                height=naip_ds.height
            )
            new_window = Window(
                col_off=int(np.floor(new_window.col_off)),
                row_off=int(np.floor(new_window.row_off)),
                width=int(np.ceil(new_window.width)),
                height=int(np.ceil(new_window.height))
            )
            cropped_naip_adj = naip_ds.read(window=new_window)
            new_naip_transform = naip_ds.window_transform(new_window)
            adj_profile_naip = naip_ds.profile.copy()
            adj_profile_naip.update({
                'height': int(new_window.height),
                'width': int(new_window.width),
                'transform': new_naip_transform
            })
        naip_adj_file = NAIP_OUT_ADJ_DIR / NAIP_MOSAIC.name.replace(".tif", "_adj.tif")
        with rasterio.open(naip_adj_file, 'w', **adj_profile_naip) as dst:
            dst.write(cropped_naip_adj)
        print(f"Adjusted NAIP mosaic saved as: {naip_adj_file}")

if __name__ == '__main__':
    main()
