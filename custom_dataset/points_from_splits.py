import pandas as pd
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
import shapely
from rasterio.crs import CRS
from rasterio.warp import transform_geom
from sklearn.model_selection import train_test_split

SPLIT_DIR = Path("./splits")
OUTPUT_DIR = Path("./splits")
OUTPUT_DIR.mkdir(exist_ok=True)
SPLIT_FILES = ["train_updated.csv", "val_updated.csv", "test_updated.csv"]

# Coordinate conversion utilities (same as in prepare_split.py)
def tile_to_point(utm_zone: int, row: int, col: int, size: int = 512, pixel_size: float = 1.25):
    x = (-col * size * pixel_size) - (size * pixel_size / 2)
    y = (row * size * pixel_size) + (size * pixel_size / 2)
    return shapely.Point(y, x)

def utm_to_wgs84(geometry, utm_zone: int):
    try:
        src_crs = CRS.from_epsg(utm_zone)
        dst_crs = CRS.from_epsg(4326)
        return shapely.geometry.shape(transform_geom(src_crs, dst_crs, geometry))
    except Exception as e:
        print(f"Warning: Failed to transform point {geometry} in zone {utm_zone}: {str(e)}")
        return None

def df_to_geodf(df: pd.DataFrame) -> gpd.GeoDataFrame:
    geoms = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting to GeoDataFrame"):
        utm_zone = int(row['utm_zone'])
        point = tile_to_point(
            utm_zone=utm_zone,
            row=int(row['image_row']),
            col=int(row['image_col'])
        )
        wgs84_point = utm_to_wgs84(point, utm_zone)
        if wgs84_point is not None:
            geoms.append({**row, 'geometry': wgs84_point})
    return gpd.GeoDataFrame(geoms, crs="EPSG:4326")

def main():
    # Merge all updated splits
    dfs = []
    for split_file in SPLIT_FILES:
        split_path = SPLIT_DIR / split_file
        if split_path.exists():
            print(f"Reading {split_file}...")
            dfs.append(pd.read_csv(split_path))
    if not dfs:
        print("No split files found.")
        return
    merged = pd.concat(dfs, ignore_index=True)
    print(f"Merged {len(merged)} images from all splits.")
    # Identify unique groups
    groups = merged[['utm_zone', 'row', 'col']].drop_duplicates()
    # Stratified sampling of groups by UTM zone
    TEST_VAL_UTM_ZONES = [zone + 32600 for zone in [13, 18, 19]]
    TEST_FRACTION = 0.1
    VAL_FRACTION = 0.05
    # Assign splits by image UTM zone (not by group)
    test_mask = merged['utm_zone'].astype(int).isin(TEST_VAL_UTM_ZONES)
    test_val = merged[test_mask]
    train = merged[~test_mask]
    # Split test_val into test and val at image level
    test_val_images = test_val.copy()
    test, val = train_test_split(
        test_val_images,
        test_size=VAL_FRACTION/(VAL_FRACTION+TEST_FRACTION),
        random_state=42,
        stratify=test_val_images['utm_zone']
    )
    # Save splits as CSV
    train.to_csv(OUTPUT_DIR / "train_merged.csv", index=False)
    val.to_csv(OUTPUT_DIR / "val_merged.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test_merged.csv", index=False)
    # Save splits as GPKG
    for split_name, split_df in zip(["train_merged", "val_merged", "test_merged"], [train, val, test]):
        gdf = df_to_geodf(split_df)
        gdf.to_file(OUTPUT_DIR / f"{split_name}.gpkg", driver="GPKG")
        print(f"Saved {len(gdf)} points to {split_name}.gpkg and {split_name}.csv")

if __name__ == "__main__":
    main()

