import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import reproject, calculate_default_transform
from rasterio.mask import mask
from eodag import EODataAccessGateway
from eodag.api.search_result import SearchResult
from rasterio.plot import show
from PIL import Image
import base64
from tqdm import tqdm
import xml.etree.ElementTree as ET
import shapely
import pyproj
from shapely.ops import transform


os.environ["EODAG__PEPS__AUTH__CREDENTIALS__USERNAME"] = base64.b64decode("dC5raGF6aGlldkBpbm5vcG9saXMucnU=").decode()
os.environ["EODAG__PEPS__AUTH__CREDENTIALS__PASSWORD"] = base64.b64decode("RzdZVWtQNzROQ3Z3V0Fl").decode()

# Define the ROI (small region of interest) and time period
roi_geometry = { "type": "Polygon", "coordinates": [ [ [ 48.728744277796949, 55.739372091252925 ], [ 48.766599283869745, 55.739372091252925 ], [ 48.766599283869745, 55.763426546309461 ], [ 48.728744277796949, 55.763426546309461 ], [ 48.728744277796949, 55.739372091252925 ] ] ] }
time_period = ("2024-06-01", "2024-07-31")

# Initialize EODAG
# dag = EODataAccessGateway()
#dag.set_preferred_provider("peps")
# Search for Sentinel-2 products
#search_results = dag.search(
#    productType="S2_MSI_L1C",
#    geometry=roi_geometry,
#    start=time_period[0],
#    end=time_period[1],
#    provider="peps",
#    cloudCover=100  # Adjust cloud cover as needed
#)

#print(len(search_results))
#exit()

# Download the first 8 products
# products = search_results[:8]
# print(products)

# downloaded_files = []
#for product in tqdm(products, desc="Products download"):
#    product.download(output_dir="./s2-images/")
#    downloaded_files.append(product.location)
downloaded_files = ['file:///home/eugen/coding/thesis/experiments/satlas-super-resolution/s2-images/S2A_MSIL1C_20240730T080611_N0511_R078_T39UUB_20240730T100830', 'file:///home/eugen/coding/thesis/experiments/satlas-super-resolution/s2-images/S2A_MSIL1C_20240727T075611_N0511_R035_T39UUB_20240727T094916', 'file:///home/eugen/coding/thesis/experiments/satlas-super-resolution/s2-images/S2B_MSIL1C_20240725T080609_N0511_R078_T39UUB_20240725T090127', 'file:///home/eugen/coding/thesis/experiments/satlas-super-resolution/s2-images/S2B_MSIL1C_20240722T075609_N0510_R035_T39UUB_20240722T084442', 'file:///home/eugen/coding/thesis/experiments/satlas-super-resolution/s2-images/S2A_MSIL1C_20240720T080611_N0510_R078_T39UUB_20240720T085319', 'file:///home/eugen/coding/thesis/experiments/satlas-super-resolution/s2-images/S2A_MSIL1C_20240717T075611_N0510_R035_T39UUB_20240717T084606', 'file:///home/eugen/coding/thesis/experiments/satlas-super-resolution/s2-images/S2B_MSIL1C_20240715T080609_N0510_R078_T39UUB_20240715T085722', 'file:///home/eugen/coding/thesis/experiments/satlas-super-resolution/s2-images/S2B_MSIL1C_20240712T075609_N0510_R035_T39UUB_20240712T085141']
   
print(downloaded_files)

def reproject_geometry(geometry, src_crs, dst_crs):
    """
    Reprojects a Shapely geometry from one CRS to another.

    :param geometry: Shapely geometry to reproject.
    :param src_crs: Source CRS (e.g., 'EPSG:4326').
    :param dst_crs: Target CRS (e.g., 'EPSG:3857').
    :return: Reprojected Shapely geometry.
    """
    project = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True).transform
    return transform(project, geometry)

def reproject_and_crop_geotiff(input_path, output_path, target_crs, resolution, geojson_geometry, geometry_crs='EPSG:4326'):
    """
    Reprojects a GeoTIFF raster to a given CRS and resolution, then crops it by a GeoJSON geometry.

    :param input_path: Path to the input GeoTIFF file.
    :param output_path: Path to save the output GeoTIFF file.
    :param target_crs: Target CRS (e.g., 'EPSG:4326').
    :param resolution: Target resolution in the units of the target CRS (e.g., (10, 10) for 10x10 units).
    :param geojson_geometry: GeoJSON geometry to crop the raster.
    """
    # Open the input raster
    with rasterio.open(input_path) as src:
        # Calculate the transform and dimensions for the reprojected raster
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds, resolution=resolution
        )

        # Create metadata for the output raster
        metadata = src.meta.copy()
        metadata.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        # Reproject the raster
        with rasterio.open(output_path, 'w', **metadata) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=rasterio.warp.Resampling.bilinear
                )

    # Crop the reprojected raster by the GeoJSON geometry
    with rasterio.open(output_path) as src:
        # Convert GeoJSON geometry to Shapely geometry
        geometry = shapely.geometry.shape(geojson_geometry)
        reprojected_geometry = reproject_geometry(geometry, geometry_crs, target_crs)

        # Mask the raster using the geometry
        out_image, out_transform = mask(src, [reprojected_geometry], crop=True)

        # Update metadata for the cropped raster
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Save the cropped raster
        with rasterio.open(output_path, 'w', **out_meta) as dest:
            dest.write(out_image)
        return out_image


# Function to preprocess and tile the TCI
def preprocess_and_tile(tci_jp2_path, roi_geometry, tile_size=(32, 32), overlap=0, use_cache=True):
    rep_path = tci_jp2_path.replace('.jp2', '_rep.jp2')
    if use_cache and os.path.exists(rep_path):
        with rasterio.open(rep_path) as src:
            out_image = src.read()
    else:
        out_image = reproject_and_crop_geotiff(tci_jp2_path, rep_path, rasterio.crs.CRS.from_epsg(3857), (9.555, 9.555),
                                               roi_geometry)
    # print("Cropped shape:", out_image.shape)

    # Split into tiles with overlap
    tiles = []
    stride = (tile_size[0] - overlap, tile_size[1] - overlap)
    for i in range(0, out_image.shape[1] - overlap, stride[0]):
        for j in range(0, out_image.shape[2] - overlap, stride[1]):
            tile = out_image[:, i:i + tile_size[0], j:j + tile_size[1]]
            if tile.shape[1] == tile_size[0] and tile.shape[2] == tile_size[1]:
                tiles.append((i // stride[0], j // stride[1], tile))

    return tiles
       
def get_tci_path(manifest_path):
    root = ET.parse(manifest_path).getroot()
    return (root.findall(".//IMAGE_FILE")[-1]).text

if __name__ == "__main__":
    OVERLAP_SIZE = 0
    TILE_SIZE = 32
    OUTPUT_ROOT = Path(f"s2-tiles-overlap-{OVERLAP_SIZE:02d}-{TILE_SIZE}/input/0_0")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    all_tiles = {}
    # Process each downloaded file
    for file_path in tqdm(downloaded_files, "Files"):
        file_path = file_path[len("file://"):]
        manifest_path = os.path.join(file_path, "MTD_MSIL1C.xml")
        tci_jp2_path = os.path.join(file_path, get_tci_path(manifest_path) + '.jp2')
        # print(tci_jp2_path)
        tiles = preprocess_and_tile(tci_jp2_path, roi_geometry, overlap=OVERLAP_SIZE, tile_size=(TILE_SIZE, TILE_SIZE))

        # Collect tiles with the same indices
        for row, col, tile in tiles:
            if (row, col) not in all_tiles:
                all_tiles[(row, col)] = []
            all_tiles[(row, col)].append(tile)

    # exit()

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
