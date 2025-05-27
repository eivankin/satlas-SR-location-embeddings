import torch
import segmentation_models_pytorch as smp
from segmentation.dataset import OSMMaskSegmentationDataset
import numpy as np
from tqdm import tqdm

"""
--- DEEPLABV3 ---

NAIP:
Results on split 'val':
IoU:      0.3294
F1 score: 0.4956
Accuracy: 0.8599
Precision:0.4129
Recall:   0.6198

Object-aware:
Results on split 'val':
IoU:      0.3120
F1 score: 0.4756
Accuracy: 0.8734
Precision:0.4404
Recall:   0.5168

EDiffSR:
Results on split 'val':
IoU:      0.3113
F1 score: 0.4748
Accuracy: 0.8624
Precision:0.4121
Recall:   0.5600

Loc embeddings:
Results on split 'val':
IoU:      0.3193
F1 score: 0.4840
Accuracy: 0.8785
Precision:0.4580
Recall:   0.5132

BPP:
Results on split 'val':
IoU:      0.2743
F1 score: 0.4305
Accuracy: 0.7917
Precision:0.3091
Recall:   0.7088

NAIP model on SR data (osm):
Results on split 'val':
IoU:      0.2825
F1 score: 0.4406
Accuracy: 0.8640
Precision:0.4054
Recall:   0.4824

NAIP model on SR data (loc emb):
Results on split 'val':
IoU:      0.2708
F1 score: 0.4262
Accuracy: 0.8676
Precision:0.4109
Recall:   0.4428

Baseline (No OSM):
Results on split 'val':
IoU:      0.2343
F1 score: 0.3797
Accuracy: 0.8507
Precision:0.3525
Recall:   0.4114

LR (cubic):
Results on split 'val':
IoU:      0.2856
F1 score: 0.4444
Accuracy: 0.8727
Precision:0.4312
Recall:   0.4584

--- UNET++ ---

NAIP (Unet++):
Results on split 'val':
IoU:      0.3860
F1 score: 0.5570
Accuracy: 0.8997
Precision:0.5467
Recall:   0.5678

Obj-aware (Unet++):
Results on split 'val':
IoU:      0.3367
F1 score: 0.5038
Accuracy: 0.8593
Precision:0.4141
Recall:   0.6432

Bpp (Unet++):
Results on split 'val':
IoU:      0.3308
F1 score: 0.4971
Accuracy: 0.8393
Precision:0.3809
Recall:   0.7152

Loc match (Unet++):
Results on split 'val':
IoU:      0.3524
F1 score: 0.5211
Accuracy: 0.8813
Precision:0.4721
Recall:   0.5815

NAIP on loc match output (Unet++):
Results on split 'val':
IoU:      0.2216
F1 score: 0.3627
Accuracy: 0.8869
Precision:0.4848
Recall:   0.2898

NAIP on obj-aware output (Unet++):
Results on split 'val':
IoU:      0.2977
F1 score: 0.4588
Accuracy: 0.8889
Precision:0.4999
Recall:   0.4240

Baseline - no osm (Unet++)
Results on split 'val':
IoU:      0.2371
F1 score: 0.3834
Accuracy: 0.7348
Precision:0.2584
Recall:   0.7423

LR (Unet++)
Results on split 'val':
IoU:      0.3325
F1 score: 0.4990
Accuracy: 0.8733
Precision:0.4448
Recall:   0.5684

EDiffSR (Unet++):
Results on split 'val':
IoU:      0.3075
F1 score: 0.4703
Accuracy: 0.8249
Precision:0.3541
Recall:   0.7001
"""

# --- Config ---
DATA_ROOT = "../custom_dataset/prepared"
SPLIT = "val"  # Change to "val" or "test" as needed
BATCH_SIZE = 16
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_model_ediffsr_upp.pth"

# --- Dataset & Dataloader ---
dataset = OSMMaskSegmentationDataset(
    osm_root=DATA_ROOT,
    split=SPLIT,
    tile_size=128,
    mode="custom",
    image_root="/home/eugen/coding/thesis/experiments/satlas-super-resolution/results/ediffsr-val/visualization/test",
)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# --- Model ---
model = smp.UnetPlusPlus(
    encoder_name="timm-efficientnet-b8",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation=None
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- Metrics ---
all_tp, all_fp, all_fn, all_tn = [], [], [], []

with torch.no_grad():
    for imgs, masks in tqdm(loader, desc=f"Evaluating {SPLIT}"):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        logits = model(imgs)
        preds = torch.sigmoid(logits)
        tp, fp, fn, tn = smp.metrics.get_stats(preds, masks.int(), mode='binary', threshold=0.5)
        all_tp.append(tp)
        all_fp.append(fp)
        all_fn.append(fn)
        all_tn.append(tn)

# Concatenate stats
tp = torch.cat(all_tp)
fp = torch.cat(all_fp)
fn = torch.cat(all_fn)
tn = torch.cat(all_tn)

# Calculate metrics
mean_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
mean_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro").item()
mean_acc = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro").item()
mean_precision = smp.metrics.precision(tp, fp, fn, tn, reduction="micro").item()
mean_recall = smp.metrics.recall(tp, fp, fn, tn, reduction="micro").item()

print(f"Results on split '{SPLIT}':")
print(f"IoU:      {mean_iou:.4f}")
print(f"F1 score: {mean_f1:.4f}")
print(f"Accuracy: {mean_acc:.4f}")
print(f"Precision:{mean_precision:.4f}")
print(f"Recall:   {mean_recall:.4f}")