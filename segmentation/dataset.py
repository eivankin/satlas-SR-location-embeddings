"""
A dataset class such that:
1. It accepts a tile size to crop OSM masks
2. it accepts a name of split to select OSM masks from
3. For each mask it selects a corresponding image. If mode = 'NAIP', then it selects a NAIP image from the OSM_ROOT,
otherwise it should select a file from a given dir such that it starts with a chip name
"""

import os
from pathlib import Path

import rasterio
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import scipy.ndimage

class OSMMaskSegmentationDataset(Dataset):
    def __init__(self,
                 osm_root,
                 split,
                 tile_size=256,
                 mode='NAIP',
                 image_root=None,
                 transforms=None):
        """
        osm_root: root directory containing 'osm_masks_{utm_zone}' folders per split
        split: split name ('train', 'val', 'test')
        tile_size: crop size for mask and image
        mode: 'NAIP' or 'custom'
        image_root: if mode != 'NAIP', directory to search for images
        transforms: torchvision transforms to apply
        """
        self.tile_size = tile_size
        self.transforms = transforms
        self.mode = mode
        self.split = split
        self.osm_root = Path(osm_root)
        self.image_root = Path(image_root) if image_root else None

        # Find all mask files for the split
        self.mask_files = []
        for group_dir in (self.osm_root / self.split / f"osm_masks").glob("*"):
            self.mask_files += list(group_dir.glob("*.png"))

        # Build mapping from chip name to mask path
        self.chip_to_mask = {f.stem: f for f in self.mask_files}

        # Build list of (mask_path, image_path) pairs
        self.pairs = []
        for chip, mask_path in self.chip_to_mask.items():
            if self.mode == 'NAIP':
                # NAIP images are in osm_root/split/naip/{group_id}/{chip}.png
                group_id = mask_path.parent.name
                img_path = self.osm_root / self.split / "naip" / group_id / f"{chip}.png"
            elif self.mode == 'LR':
                group_id = mask_path.parent.name
                img_path = self.osm_root / self.split / "sentinel2" / group_id / f"{chip}.tif"
            else:
                # Find image in image_root that starts with chip name
                img_candidates = list(self.image_root.rglob(f"{chip}*"))
                img_path = img_candidates[0] if img_candidates else None
            if img_path and img_path.exists():
                self.pairs.append((mask_path, img_path))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        mask_path, img_path = self.pairs[idx]
        mask = np.array(Image.open(mask_path).convert("L"))
        if self.mode == "LR":
            with rasterio.open(img_path) as src:
                img = src.read([39, 38, 37]).transpose(1, 2, 0)  # Read RGB channels

            img = scipy.ndimage.zoom(img, (4, 4, 1), order=3)
        else:
            img = np.array(Image.open(img_path).convert("RGB"))

        th, tw = self.tile_size, self.tile_size
        i, j = 0, 0  # upper left corner
        mask = mask[i:i + th, j:j + tw]
        img = img[i:i + th, j:j + tw, :]
        # Normalize to [0,1]
        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        else:
            img = torch.from_numpy(img).permute(2, 0, 1)
            mask = torch.from_numpy(mask).unsqueeze(0)
        return img, mask


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Example usage
    dataset = OSMMaskSegmentationDataset(
        osm_root="./custom_dataset/prepared",
        split="train",
        tile_size=128,
        mode="custom",
        image_root="/home/eugen/coding/thesis/experiments/satlas-super-resolution/results/osm-obj-esrgan/visualization/test",
    )

    # Visualize a sample
    img, mask = dataset[0]
    img_np = img.permute(1, 2, 0).numpy()
    mask_np = mask.squeeze().numpy()

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title("NAIP Image")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(mask_np, cmap="gray")
    plt.title("OSM Mask")
    plt.axis("off")
    plt.tight_layout()
    plt.show()