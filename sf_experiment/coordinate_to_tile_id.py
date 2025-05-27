from rasterio.crs import CRS
from rasterio.warp import transform_geom
import shapely
import utm
# Define source location.
src_crs = CRS.from_epsg(4326)
"""
upper left: -122.17661,37.45180
32610_895_-6475
32610_27_-203
"""
"""lower right: -122.13341,37.39376
32610_901_-6468
32610_28_-203
"""
src_point = shapely.Point(-122.13341,37.39376)
# Get UTM zone.
_, _, zone_suffix, _ = utm.from_latlon(src_point.y, src_point.x)
epsg_code = 32600 + zone_suffix
dst_crs = CRS.from_epsg(epsg_code)
# Transform to UTM CRS.
dst_point = transform_geom(src_crs, dst_crs, src_point)
dst_point = shapely.geometry.shape(dst_point)
# dst_point is in projection coordinates (meters).
# Now convert to pixel coordinates at 1.25 m/pixel.
col = int(dst_point.x/1.25)
row = int(dst_point.y/-1.25)
# Print the prefix for the image filenames.
print(f"{epsg_code}_{col//512}_{row//512}")
# Print the prefix for the tar filenames to know which one to download.
# These group together many 1.25 m/pixel 512x512 tiles into one tar file.
print(f"{epsg_code}_{col//512//32}_{row//512//32}")
