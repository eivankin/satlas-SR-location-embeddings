import torch
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import random
from segmentation.dataset import OSMMaskSegmentationDataset

# --- Config ---
random.seed(123)
DATA_ROOT = "../custom_dataset/prepared"
SPLIT = "val"
TILE_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 4  # Now 4 random samples
MODEL_TYPE = "Unet++" # Unet++ or DeepLabV3+
PLOT_TYPE = "mask"  # "image" or "mask"


MODELS_TO_COMPARE = {
    "NAIP": ("best_model_naip_upp.pth", "NAIP", None),
    "Sentinel-2": ("best_model_lr_upp.pth", "LR", None),
    "Object-Aware": ("best_model_osm_upp.pth", "custom", "/home/eugen/coding/thesis/experiments/satlas-super-resolution/results/osm-obj-esrgan-val/visualization/test"),
    "Location Embeddings": ("best_model_loc_emb_upp.pth", "custom", "/home/eugen/coding/thesis/experiments/satlas-super-resolution/results/loc-match-esrgan-val/visualization/test"),
    "Compression-Guided": ("best_model_bpp_upp.pth", "custom", "/home/eugen/coding/thesis/experiments/satlas-super-resolution/results/bpp-loss-esrgan-val/visualization/test"),
    "Baseline Single-Image": ("best_model_baseline_upp.pth", "custom", "/home/eugen/coding/thesis/experiments/satlas-super-resolution/results/bpp-loss-esrgan-val/visualization/test"),
    "EDiffSR": ("best_model_ediffsr_upp.pth", "custom", "/home/eugen/coding/thesis/experiments/satlas-super-resolution/results/ediffsr-val/visualization/test"),
}

# --- Prepare datasets for each model ---
datasets = {}
for name, (_, mode, image_root) in MODELS_TO_COMPARE.items():
    datasets[name] = OSMMaskSegmentationDataset(
        osm_root=DATA_ROOT,
        split=SPLIT,
        tile_size=TILE_SIZE,
        mode=mode,
        image_root=image_root,
    )

base_dataset = datasets["NAIP"]
all_indices = list(range(len(base_dataset)))
indices = random.sample(all_indices, min(N_SAMPLES, len(base_dataset)))
model_names = list(MODELS_TO_COMPARE.keys())

if PLOT_TYPE == "image":
    n_cols = len(model_names)
    col_titles = model_names
elif PLOT_TYPE == "mask":
    n_cols = 2 + len(model_names)  # GT image, GT mask, predictions
    col_titles = ["NAIP Image", "GT Mask"] + [f"{name} Pred" for name in model_names]
else:
    raise ValueError("PLOT_TYPE must be 'image' or 'mask'")

n_rows = N_SAMPLES
fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))

if n_rows == 1:
    axes = axes[None, :]  # Ensure axes is 2D

for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=12)

for row, idx in enumerate(indices):
    if PLOT_TYPE == "image":
        for col, name in enumerate(model_names):
            img, _ = datasets[name][idx]
            img_np = img.permute(1, 2, 0).numpy()
            axes[row, col].imshow(img_np)
            axes[row, col].axis("off")
    elif PLOT_TYPE == "mask":
        # GT image and mask
        img_gt, mask_gt = base_dataset[idx]
        img_gt_np = img_gt.permute(1, 2, 0).numpy()
        mask_gt_np = mask_gt.squeeze().numpy()
        axes[row, 0].imshow(img_gt_np)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(mask_gt_np, cmap="gray")
        axes[row, 1].axis("off")
        # Model predictions
        for col, name in enumerate(model_names, start=2):
            img, _ = datasets[name][idx]
            # Load model only once
            if row == 0:
                if MODEL_TYPE == "DeepLabV3+":
                    model = smp.DeepLabV3Plus(
                        encoder_name="mit_b5",
                        encoder_weights=None,
                        in_channels=3,
                        classes=1,
                        activation=None
                    ).to(DEVICE)
                else:
                    model = smp.UnetPlusPlus(
                        encoder_name="timm-efficientnet-b8",
                        encoder_weights=None,
                        in_channels=3,
                        classes=1,
                        activation=None
                    ).to(DEVICE)
                model.load_state_dict(torch.load(MODELS_TO_COMPARE[name][0], map_location=DEVICE))
                model.eval()
                datasets[name + "_model"] = model
            else:
                model = datasets[name + "_model"]
            img_input = img.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = torch.sigmoid(model(img_input)).cpu().squeeze().numpy()
            axes[row, col].imshow(pred > 0.5, cmap="gray")
            axes[row, col].axis("off")

plt.tight_layout()
plt.show()