"""
Train a YOLO segmentation model on the golf-club-tracking dataset.

Dataset : golf-club-tracking.v2i.yolo26 (CC BY 4.0, Roboflow)
          9 456 train / 1 347 val / 674 test images (640x640, pre-augmented)
          3 classes — 0:club (shaft), 1:clubhead, 2:hand

Base model : yolo26x-seg.pt — auto-downloaded by Ultralytics on first run.

Output : runs_train/golf_club_seg_v1/
         weights/best.pt   ← use this for inference

Training runs until val mAP stops improving for PATIENCE=50 epochs, then stops
automatically. MAX_EPOCHS=1000 is just a safety ceiling.
Set BATCH=8 if VRAM allows (>= 12 GB).
"""

import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO, settings

# ---------------------------------------------------------------------------
# Paths — absolute so the script works from any working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR     = PROJECT_ROOT / "runs_train"
MODELS_DIR   = PROJECT_ROOT / "models"
DATA_DIR     = PROJECT_ROOT / "data" / "golf-club-tracking.v2i.yolo26"

MODELS_DIR.mkdir(exist_ok=True)
settings.update({"weights_dir": str(MODELS_DIR)})

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_MODEL  = "yolo26x-seg.pt"   # auto-downloaded by Ultralytics on first run
OUTPUT_NAME = "golf_club_seg_v1"

MAX_EPOCHS  = 1000   # ceiling only — early stopping will kick in long before this
IMG_SIZE    = 640    # dataset is already 640x640; no gain from going larger
BATCH       = 4      # safe for 8 GB VRAM; try 8 if you have more headroom
WORKERS     = 0      # must be 0 on Windows
PATIENCE    = 50     # stop when val mAP hasn't improved for 50 consecutive epochs
SAVE_PERIOD = 10     # save checkpoint every 10 epochs


def main():
    # -----------------------------------------------------------------------
    # GPU check
    # -----------------------------------------------------------------------
    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA not available. Run: uv sync")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU : {gpu_name} ({vram_gb:.1f} GB VRAM)")
    print(f"Output : {RUNS_DIR / OUTPUT_NAME}\n")

    # -----------------------------------------------------------------------
    # Build data.yaml with absolute paths and correct class names.
    # The original data.yaml from Roboflow has wrong relative paths
    # (../train/images) and unnamed classes ('0','1','3').
    # -----------------------------------------------------------------------
    cfg = {
        "train": str(DATA_DIR / "train" / "images"),
        "val":   str(DATA_DIR / "valid" / "images"),
        "test":  str(DATA_DIR / "test"  / "images"),
        "nc":    3,
        "names": ["club", "clubhead", "hand"],
    }
    fixed_yaml = DATA_DIR / "data_fixed.yaml"
    with open(fixed_yaml, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"Classes : {cfg['names']}")
    print(f"Train   : {len(list((DATA_DIR / 'train' / 'images').glob('*')))} images")
    print(f"Val     : {len(list((DATA_DIR / 'valid' / 'images').glob('*')))} images\n")

    model = YOLO(BASE_MODEL)

    model.train(
        data          = str(fixed_yaml),
        epochs        = MAX_EPOCHS,
        imgsz         = IMG_SIZE,
        batch         = BATCH,
        device        = 0,
        name          = OUTPUT_NAME,
        project       = str(RUNS_DIR),
        amp           = True,
        cache         = False,
        workers       = WORKERS,
        patience      = PATIENCE,
        save_period   = SAVE_PERIOD,
        plots         = True,
        lr0           = 0.01,
        lrf           = 0.01,
        warmup_epochs = 3,
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    best = RUNS_DIR / OUTPUT_NAME / "weights" / "best.pt"
    if best.exists():
        print(f"\nTraining complete.")
        print(f"Best model  : {best}")
        print(f"All results : {RUNS_DIR / OUTPUT_NAME}/")
    else:
        print(f"\nDone. Check {RUNS_DIR / OUTPUT_NAME}/ for weights.")


if __name__ == "__main__":
    main()
