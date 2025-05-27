"""
Implementation of the tile selection procedure proposed by Panangian & Bittner in "Can Location Embeddings Enhance Super-Resolution of Satellite Imagery?".
They use WorldCover data from S2-NAIP dataset to select tiles with >30% urban coverage.

Steps:
1. (Done in 'download_worldcover.py') Downloads all the WorldCover data from S2-NAIP dataset using huggingface.
2. Extracts the data from .tar files.
3. Selects tiles with >30% urban coverage.
4. Saves projection, row & col of tiles in a csv file.

Takes 15 minutes instead of 2 hours thanks to multiprocessing & 8 CPU cores
"""

from pathlib import Path
import tarfile
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm
import shutil
import multiprocessing as mp

MIN_URBAN_COV = 30
BUILT_UP_CLASS = 50  # Class reference: https://worldcover2021.esa.int/data/docs/WorldCover_PUM_V2.0.pdf
WORLDCOVER_DIR = Path("./worldcover")  # Structure: ./worldcover/worldcover_{utm_zone}/{utm_zone}_{row}_{col}.tar/worldcover/{utm_zone}_{image_row}_{image_col}.png
SELECTED_TILES_OUTPUT = Path("./selected_tiles.csv")

def extract_tar(tar_path: Path) -> Path:
    """Extract tar file and return path to extracted directory"""
    with tarfile.open(tar_path, 'r') as tar:
        extract_dir = tar_path.parent / tar_path.stem
        tar.extractall(path=extract_dir)
    return extract_dir

def cleanup_extracted(extract_dir: Path):
    """Remove extracted directory and its contents"""
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

def calculate_urban_coverage(image_path: Path) -> float:
    """Calculate percentage of urban coverage in a WorldCover tile"""
    with Image.open(image_path) as img:
        data = np.array(img)
        total_pixels = data.size
        urban_pixels = np.sum(data == BUILT_UP_CLASS)
        return (urban_pixels / total_pixels) * 100

def process_single_tar(tar_file: Path):
    """Process a single tar file and return its metadata if it meets urban coverage criteria"""
    parts = tar_file.stem.split('_')
    utm_zone, row, col = parts

    extract_dir = extract_tar(tar_file)
    total_coverage = 0
    n_images = 0

    try:
        # Calculate average urban coverage for all images in the tar
        for png_file in extract_dir.glob("worldcover/*.png"):
            urban_coverage = calculate_urban_coverage(png_file)
            total_coverage += urban_coverage
            n_images += 1

        if n_images > 0:
            avg_urban_coverage = total_coverage / n_images

            # Return group metadata if it meets the threshold
            if avg_urban_coverage >= MIN_URBAN_COV:
                return {
                    'utm_zone': utm_zone,
                    'row': row,
                    'col': col,
                    'urban_coverage': avg_urban_coverage,
                    'n_images': n_images
                }
    finally:
        cleanup_extracted(extract_dir)

    return None

def init_worker():
    """Initialize worker process"""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def process_tiles(max_workers=None):
    """Process all WorldCover tiles using multiprocessing"""
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 16)

    tar_files = list(WORLDCOVER_DIR.rglob("*.tar"))
    selected_groups = []

    with mp.Pool(processes=max_workers, initializer=init_worker) as pool:
        try:
            results_iter = pool.imap_unordered(process_single_tar, tar_files)

            for result in tqdm(results_iter,
                             total=len(tar_files),
                             desc="Processing WorldCover tiles"):
                if result is not None:
                    selected_groups.append(result)

        except KeyboardInterrupt:
            print("\nCaught KeyboardInterrupt, terminating workers...")
            pool.terminate()
            raise
        finally:
            pool.close()
            pool.join()

    return pd.DataFrame(selected_groups)

if __name__ == '__main__':
    try:
        n_workers = min(mp.cpu_count(), 16)
        print(f"Processing with {n_workers} workers...")

        selected_df = process_tiles(max_workers=n_workers)
        selected_df.to_csv(SELECTED_TILES_OUTPUT, index=False)
        print(f"Selected {len(selected_df)} tile groups with average >{MIN_URBAN_COV}% urban coverage")  # Selected 108 tile groups
        print(f"Results saved to {SELECTED_TILES_OUTPUT}")
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        exit(1)
