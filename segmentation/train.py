import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import segmentation_models_pytorch as smp
from segmentation.dataset import OSMMaskSegmentationDataset
import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


# --- Config ---
EXPERIMENT_NAME = "ediffsr_upp"
DATA_ROOT = "../custom_dataset/prepared"
SPLIT = "test"
VAL_SPLIT = 0.1
BATCH_SIZE = 4
NUM_WORKERS = 4
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = None

# --- Dataset & Dataloaders ---
full_dataset = OSMMaskSegmentationDataset(
    osm_root=DATA_ROOT,
    split=SPLIT,
    tile_size=128,
    mode="custom",
    image_root="/home/eugen/coding/thesis/experiments/satlas-super-resolution/results/ediffsr/visualization/test",
)
val_len = int(len(full_dataset) * VAL_SPLIT)
train_len = len(full_dataset) - val_len
train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# --- Model ---
model = smp.UnetPlusPlus(
    encoder_name="timm-efficientnet-b8",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
).to(DEVICE)
if CKPT_PATH is not None:
    model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))

# --- Loss & Optimizer ---
loss_fn = smp.losses.DiceLoss(mode="binary")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# --- Training Loop with Early Stopping ---
best_val_iou = 0
patience = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = []
    for imgs, masks in tqdm(train_loader, desc="Train batches"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    # --- Validation ---
    model.eval()
    val_loss = []
    val_iou = []
    val_f1 = []
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Val batches"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            logits = model(imgs)
            loss = loss_fn(logits, masks)
            val_loss.append(loss.item())
            preds = torch.sigmoid(logits)
            tp, fp, fn, tn = smp.metrics.get_stats(preds, masks.int(), mode='binary', threshold=0.5)
            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
            f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()
            val_iou.append(iou_score)
            val_f1.append(f1_score)
    mean_val_iou = np.mean(val_iou)
    mean_val_f1 = np.mean(val_f1)
    print(f"Epoch {epoch+1}: Train loss={np.mean(train_loss):.4f}, Val loss={np.mean(val_loss):.4f}, Val IoU={mean_val_iou:.4f}, Val F1={mean_val_f1:.4f}")

    # --- Early Stopping ---
    if mean_val_iou > best_val_iou:
        best_val_iou = mean_val_iou
        patience = 0
        torch.save(model.state_dict(), f"best_model_{EXPERIMENT_NAME}.pth")

        model.eval()
        idx = random.randint(0, len(val_dataset) - 1)
        img, mask = val_dataset[idx]
        img_input = img.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = torch.sigmoid(model(img_input)).cpu().squeeze().numpy()
        img_np = img.permute(1, 2, 0).numpy()
        mask_np = mask.squeeze().numpy()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(img_np)
        plt.title("Input")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(mask_np, cmap="gray")
        plt.title("Ground Truth")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(pred > 0.5, cmap="gray")
        plt.title("Prediction")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(f"best_examples/best_model_example_{epoch+1}.png")
        plt.close()
    else:
        patience += 1
        if patience >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

print("Training complete. Best val IoU:", best_val_iou)