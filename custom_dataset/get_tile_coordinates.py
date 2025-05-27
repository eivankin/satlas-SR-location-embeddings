"""Convert tile IDs back to geographical coordinates and save as points in GPKG format."""

import pandas as pd
import geopandas as gpd
from pathlib import Path
import shapely
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from tqdm import tqdm

INPUT_FILES = {
    'groups': Path("./selected_tiles.csv"),
    'images': Path("./selected_images.csv")
}
OUTPUT_DIR = Path("./gis")
OUTPUT_DIR.mkdir(exist_ok=True)

def tile_to_point(utm_zone: int, row: int, col: int, size: int = 512, pixel_size: float = 1.25):
    """Convert tile coordinates to center point in UTM coordinates"""
    # Convert from tile coordinates to meters, get center point
    x = (-col * size * pixel_size) - (size * pixel_size / 2)
    y = (row * size * pixel_size) + (size * pixel_size / 2)  # Negative because row increases southward
    return shapely.Point(y, x)

def utm_to_wgs84(geometry, utm_zone: int):
    """Transform geometry from UTM to WGS84"""
    try:
        src_crs = CRS.from_epsg(utm_zone)
        dst_crs = CRS.from_epsg(4326)
        return shapely.geometry.shape(transform_geom(src_crs, dst_crs, geometry))
    except Exception as e:
        print(f"Warning: Failed to transform point {geometry} in zone {utm_zone}: {str(e)}")
        return None

def process_group_tiles(df: pd.DataFrame):
    """Process tile groups from selected_tiles.csv"""
    geometries = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing tile groups"):
        utm_zone = int(row['utm_zone'])
        point = tile_to_point(
            utm_zone=utm_zone,
            row=int(row['row']),
            col=int(row['col']),
            size=512 * 32
        )
        wgs84_point = utm_to_wgs84(point, utm_zone)
        if wgs84_point is not None:
            geometries.append({
                'geometry': wgs84_point,
                'utm_zone': utm_zone,
                'row': row['row'],
                'col': row['col'],
                'urban_coverage': row['urban_coverage'],
                'n_images': row['n_images']
            })
    return gpd.GeoDataFrame(geometries, crs="EPSG:4326")

def process_image_tiles(df: pd.DataFrame):
    """Process individual images from selected_images.csv"""
    geometries = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing image tiles"):
        utm_zone = int(row['utm_zone'])
        point = tile_to_point(
            utm_zone=utm_zone,
            row=int(row['image_row']),
            col=int(row['image_col'])
        )
        wgs84_point = utm_to_wgs84(point, utm_zone)
        if wgs84_point is not None:
            geometries.append({
                'geometry': wgs84_point,
                'utm_zone': utm_zone,
                'group_row': row['row'],
                'group_col': row['col'],
                'image_row': row['image_row'],
                'image_col': row['image_col'],
                'urban_coverage': row['urban_coverage']
            })
    return gpd.GeoDataFrame(geometries, crs="EPSG:4326")

def process_image_groups(df: pd.DataFrame):
    """Process image groups from selected_images.csv"""
    geometries = []
    grouped_by_group = df.groupby(['utm_zone', 'row', 'col'])
    for (utm_zone, row, col), group in tqdm(grouped_by_group, total=len(grouped_by_group), desc="Processing image groups"):
        point = tile_to_point(utm_zone=utm_zone, row=row, col=col, size=512 * 32)
        wgs84_point = utm_to_wgs84(point, utm_zone)
        if wgs84_point is not None:
            geometries.append({
                'geometry': wgs84_point,
                'utm_zone': utm_zone,
                'row': row,
                'col': col,
                'n_images': len(group)
            })
    return gpd.GeoDataFrame(geometries, crs="EPSG:4326")

if __name__ == '__main__':
    # tile: 32610_895_-6475
    # point = tile_to_point(3610, 895, -6475)
    # print(point)
    # print(utm_to_wgs84(point, 32610))
    for name, file_path in INPUT_FILES.items():
        if not file_path.exists():
            print(f"Skipping {name}, file {file_path} does not exist")
            continue

        print(f"Processing {name}...")
        df = pd.read_csv(file_path)

        # Process based on file type
        if name == 'groups':
            continue
            gdf = process_group_tiles(df)
        else:
            # gdf = process_image_tiles(df)
            gdf = process_image_groups(df)

        # Save as GeoPackage
        output_file = OUTPUT_DIR / f"{name}_groups.gpkg"
        gdf.to_file(output_file, driver="GPKG")
        print(f"Saved {len(gdf)} points to {output_file}")
