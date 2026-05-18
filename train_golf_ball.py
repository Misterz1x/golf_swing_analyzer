"""
Train a YOLO detection model on the Roboflow golf ball dataset.

Base model : yolo26x.pt  (auto-downloaded by ultralytics if absent)
Output     : golf_ball_yolo26x.pt  (copied from runs/train/.../weights/best.pt)
GPU        : CUDA device 0 (RTX 5060).  Requires PyTorch with CUDA 12.x.
             If CUDA is missing, run:  uv sync
"""

import shutil
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_MODEL  = "yolo26x.pt"
DATA_DIR    = Path("data/golfballtracking.v1i.yolo26")
OUTPUT_NAME = "golf_ball_yolo26x"

EPOCHS      = 100
IMG_SIZE    = 640
BATCH       = 8    # safe for yolo26x at 640px on 8 GB VRAM; increase to 16 if no OOM
WORKERS     = 0    # must be 0 on Windows to avoid multiprocessing errors
PATIENCE    = 30
SAVE_PERIOD = 10


def main():
    # -----------------------------------------------------------------------
    # GPU check — abort immediately if CUDA is not available
    # -----------------------------------------------------------------------
    if not torch.cuda.is_available():
        sys.exit(
            "ERROR: CUDA not available. Training on CPU is not allowed.\n"
            "Fix: run  uv sync  — pyproject.toml is configured to pull the CUDA 12.8 build."
        )

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU detected : {gpu_name}  ({vram_gb:.1f} GB VRAM)")

    # -----------------------------------------------------------------------
    # Fix data.yaml — Roboflow exports use ../train/images which is relative
    # to a parent directory that doesn't exist here. Rewrite with absolute paths.
    # -----------------------------------------------------------------------
    original_yaml = DATA_DIR / "data.yaml"
    if not original_yaml.exists():
        sys.exit(f"ERROR: data.yaml not found at {original_yaml}")

    with open(original_yaml) as f:
        cfg = yaml.safe_load(f)

    data_root = DATA_DIR.resolve()
    cfg["train"] = str(data_root / "train" / "images")
    cfg["val"]   = str(data_root / "valid" / "images")
    cfg["test"]  = str(data_root / "test"  / "images")

    fixed_yaml = DATA_DIR / "data_fixed.yaml"
    with open(fixed_yaml, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

    print(f"Dataset root : {data_root}")
    print(f"Classes      : {cfg['names']}")
    print(f"Fixed yaml   : {fixed_yaml}\n")

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    model = YOLO(BASE_MODEL)

    model.train(
        data        = str(fixed_yaml),
        epochs      = EPOCHS,
        imgsz       = IMG_SIZE,
        batch       = BATCH,
        device      = 0,
        name        = OUTPUT_NAME,
        project     = "runs/train",
        amp         = True,
        cache       = False,
        workers     = WORKERS,
        patience    = PATIENCE,
        save_period = SAVE_PERIOD,
        plots       = True,
    )

    # -----------------------------------------------------------------------
    # Copy best weights to project root
    # -----------------------------------------------------------------------
    best_weights = Path("runs/train") / OUTPUT_NAME / "weights" / "best.pt"
    output_path  = Path(f"{OUTPUT_NAME}.pt")

    if best_weights.exists():
        shutil.copy(best_weights, output_path)
        print(f"\nTraining complete.")
        print(f"Best model   : {output_path}")
        print(f"All results  : runs/train/{OUTPUT_NAME}/")
    else:
        print(f"\nTraining finished but weights not found at {best_weights}.")
        print("Check runs/train/ manually for your model.")


if __name__ == "__main__":
    main()
