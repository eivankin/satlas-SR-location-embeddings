import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import contextily as cx
import pandas as pd

# Load the GPKG files
train = gpd.read_file("splits/train_merged.gpkg")
val = gpd.read_file("splits/val_merged.gpkg")
test = gpd.read_file("splits/test_merged.gpkg")

# Set to common CRS if needed
train = train.to_crs("EPSG:4326")
val = val.to_crs("EPSG:4326")
test = test.to_crs("EPSG:4326")

source = cx.providers.OpenStreetMap.Mapnik # cx.providers.CartoDB.VoyagerNoLabels

# Plotting
# fig, ax = plt.subplots()
plot_all = False
if plot_all:
    # Plot each set
    ax = train.plot(markersize=40, color='blue', label='Train Set', figsize=(14, 7))
    test.plot(ax=ax, markersize=40, color='orange', label='Test Set')
    val.plot(ax=ax, markersize=20, color='red', label='Validation Set')


    # Draw UTM zones for CONUS (Zones 10–19)
    for zone in range(10, 20):
        lon_min = -186 + 6 * zone
        lon_max = lon_min + 6
        rect = patches.Rectangle(
            (lon_min, 24),  # bottom-left corner (lon, lat)
            6,  # width
            26,  # height to roughly cover the U.S.
            linewidth=1,
            edgecolor='black',
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text((lon_min + lon_max) / 2, 51, f"ZONE {zone}N", ha='center')

    # Styling
    ax.set_xlim([-126, -66])
    ax.set_ylim([24, 50])
    ax.legend()
    ax.axis('off')

    cx.add_basemap(ax, crs="EPSG:4326", source=source, zoom=4)

    plt.tight_layout()
    # plt.show()
    plt.savefig('all_points.png')

group_cols = ['utm_zone', 'row', 'col']
if all(col in train.columns for col in group_cols):
    group_sizes = train.groupby(group_cols).size()
    biggest_group_key = group_sizes.idxmax()
    first_group = pd.Series(biggest_group_key, index=group_cols)
    group_mask = (train[group_cols] == first_group).all(axis=1)
    group_points = train[group_mask]

    fig2, ax2 = plt.subplots(figsize=(8, 8))
    group_points.plot(ax=ax2, markersize=20, color='blue', label='Tile centers')
    ax2.set_title(f"Group: UTM {int(first_group['utm_zone'])}, Row {int(first_group['row'])}, Col {int(first_group['col'])}")
    ax2.legend()
    ax2.axis('off')
    cx.add_basemap(ax2, crs="EPSG:4326", source=source, zoom=12)
    plt.tight_layout()
    plt.savefig('single_group_points.png')