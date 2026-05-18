"""
Fine-tune golf_ball_yolo26x on the GolfBallDetector dataset.

v1 results (80 epochs): mAP@50=0.915, mAP@50-95=0.693 — no overfitting,
still improving at final epoch. Training stopped at the epoch limit, not at
a plateau. Running more epochs will improve the model further.

Set RESUME = True to continue from the last v2 checkpoint (recommended).
Set RESUME = False to start a fresh run from best.pt (new LR schedule).
"""

import shutil
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO, settings

# ---------------------------------------------------------------------------
# Paths — all absolute so the runs folder always lands in this project,
# regardless of which directory you launch the script from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_DIR     = PROJECT_ROOT / "runs_train"   # output folder for all training runs
MODELS_DIR   = PROJECT_ROOT / "models"

# Redirect all ultralytics model downloads to the local models/ folder.
# This affects the nano model ultralytics downloads internally for its AMP
# sanity check, as well as any other auto-downloaded weights.
MODELS_DIR.mkdir(exist_ok=True)
settings.update({"weights_dir": str(MODELS_DIR)})

PREV_BEST    = PROJECT_ROOT / "runs2/train/golf_ball_yolo26x_v2/weights/best.pt"
PREV_LAST    = PROJECT_ROOT / "runs2/train/golf_ball_yolo26x_v2/weights/last.pt"
DATA_DIR     = PROJECT_ROOT / "data/GolfBallDetector.v10i.yolo26"
OUTPUT_NAME  = "golf_ball_yolo26x_v2"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESUME      = False   # True = continue from last.pt; False = fresh run from best.pt
EPOCHS      = 150     # total epochs for this run (was 80 before — 150 gives ~70 more)
IMG_SIZE    = 1280    # matches inference resolution; much better for small objects in HD video
BATCH       = 4       # 1280px needs more VRAM — drop batch from 8 to 4
WORKERS     = 0       # must be 0 on Windows
PATIENCE    = 30
SAVE_PERIOD = 10
FREEZE      = 0       # 0 = unfreeze all layers (backbone already adapted; let it refine)


def remap_labels_inplace(label_dir: Path) -> int:
    """Set all class IDs in every .txt label file to 0. Idempotent."""
    if not label_dir.exists():
        return 0
    count = 0
    for label_file in label_dir.glob("*.txt"):
        lines = label_file.read_text(encoding="utf-8").strip().splitlines()
        remapped = []
        for line in lines:
            parts = line.split()
            if parts:
                parts[0] = "0"
                remapped.append(" ".join(parts))
        label_file.write_text("\n".join(remapped), encoding="utf-8")
        count += 1
    return count


def main():
    # -----------------------------------------------------------------------
    # GPU check
    # -----------------------------------------------------------------------
    if not torch.cuda.is_available():
        sys.exit("ERROR: CUDA not available. Run: uv sync")

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU : {gpu_name} ({vram_gb:.1f} GB VRAM)")
    print(f"Runs will be saved to: {RUNS_DIR}\n")

    # -----------------------------------------------------------------------
    # Pick model to load
    # -----------------------------------------------------------------------
    if RESUME:
        model_path = PREV_LAST
        if not model_path.exists():
            sys.exit(f"ERROR: last.pt not found at {model_path}")
        print(f"Resuming from : {model_path}")
    else:
        model_path = PREV_BEST
        if not model_path.exists():
            sys.exit(f"ERROR: best.pt not found at {model_path}")
        print(f"Fine-tuning from : {model_path}")

    # -----------------------------------------------------------------------
    # Remap labels: 15 messy class names → 1 class 'golfball' (idempotent)
    # -----------------------------------------------------------------------
    print("\nRemapping labels (15 classes → 1 class 'golfball')...")
    for split in ["train", "valid", "test"]:
        n = remap_labels_inplace(DATA_DIR / split / "labels")
        print(f"  {split}: {n} files")

    # -----------------------------------------------------------------------
    # Build data.yaml with absolute paths and single class
    # -----------------------------------------------------------------------
    cfg = {
        "train": str(DATA_DIR / "train" / "images"),
        "val":   str(DATA_DIR / "valid" / "images"),
        "test":  str(DATA_DIR / "test"  / "images"),
        "nc":    1,
        "names": ["golfball"],
    }
    fixed_yaml = DATA_DIR / "data_fixed.yaml"
    with open(fixed_yaml, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
    print(f"\nFixed yaml : {fixed_yaml}")
    print(f"Classes    : {cfg['names']}\n")

    # -----------------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------------
    model = YOLO(str(model_path))

    model.train(
        data          = str(fixed_yaml),
        epochs        = EPOCHS,
        imgsz         = IMG_SIZE,
        batch         = BATCH,
        device        = 0,
        name          = OUTPUT_NAME,
        project       = str(RUNS_DIR),   # absolute path — always saves here
        amp           = True,
        cache         = False,
        workers       = WORKERS,
        patience      = PATIENCE,
        save_period   = SAVE_PERIOD,
        plots         = True,
        resume        = RESUME,
        freeze        = FREEZE,
        lr0           = 0.0005,  # slightly lower than before (0.001) for continued fine-tuning
        lrf           = 0.01,
        warmup_epochs = 2,
        copy_paste    = 0.1,
    )

    # -----------------------------------------------------------------------
    # Copy best weights to project root
    # -----------------------------------------------------------------------
    best_weights = RUNS_DIR / OUTPUT_NAME / "weights" / "best.pt"
    output_path  = PROJECT_ROOT / f"{OUTPUT_NAME}.pt"

    if best_weights.exists():
        shutil.copy(best_weights, output_path)
        print(f"\nTraining complete.")
        print(f"Best model  : {output_path}")
        print(f"All results : {RUNS_DIR / OUTPUT_NAME}/")
    else:
        print(f"\nDone. Check {RUNS_DIR / OUTPUT_NAME}/ for weights.")


if __name__ == "__main__":
    main()
