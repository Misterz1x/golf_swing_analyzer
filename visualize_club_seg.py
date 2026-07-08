"""
Visualize the golf club segmentation model on a video.

For each frame the model detects:
  class 0 — shaft      → cyan semi-transparent mask + box
  class 1 — clubhead   → orange semi-transparent mask + box + centroid dot

Output videos are written to result_videos_images/videos_club_seg/.

Usage
-----
  # single video
  python visualize_club_seg.py --video sample_videos/IMG_5887.MOV

  # all videos in sample_videos/
  python visualize_club_seg.py

  # lower confidence threshold to see more detections
  python visualize_club_seg.py --video sample_videos/IMG_5887.MOV --conf 0.15
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "runs_train/golf_club_seg_v1/weights/best.pt"
VIDEO_DIR  = Path("sample_videos")
OUTPUT_DIR = Path("result_videos_images/videos_club_seg")
IMG_SIZE   = 1920
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
CONF       = 0.25

# BGR colours per class
CLASS_COLORS = {
    0: (255, 220,  50),   # shaft   — cyan/yellow
    1: ( 30, 140, 255),   # clubhead — orange
}
CLASS_NAMES = {0: "shaft", 1: "clubhead"}
MASK_ALPHA  = 0.40   # mask transparency


def _draw_mask(frame: np.ndarray, mask_xy: np.ndarray, color: tuple, alpha: float):
    """Fill the polygon defined by mask_xy onto frame with transparency."""
    pts = mask_xy.astype(np.int32).reshape((-1, 1, 2))
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)


def process_video(video_path: Path, model: YOLO, conf: float) -> Path:
    cap    = cv2.VideoCapture(str(video_path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n{video_path.name}  ({width}x{height} @ {fps:.1f} fps, {total} frames)")

    out_path = OUTPUT_DIR / (video_path.stem + "_club_seg.mp4")
    writer   = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frame_idx     = 0
    det_counts    = {0: 0, 1: 0}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, imgsz=IMG_SIZE, conf=conf, verbose=False)
        r = results[0]

        has_masks = r.masks is not None
        boxes     = r.boxes

        for i in range(len(boxes)):
            cls_id   = int(boxes.cls[i].item())
            conf_val = float(boxes.conf[i].item())
            color    = CLASS_COLORS.get(cls_id, (200, 200, 200))
            label    = CLASS_NAMES.get(cls_id, f"cls{cls_id}")

            # Segmentation mask
            if has_masks:
                xy = r.masks.xy[i]
                if len(xy) >= 3:
                    _draw_mask(frame, xy, color, MASK_ALPHA)

            # Bounding box
            x1, y1, x2, y2 = map(int, boxes.xyxy[i].cpu().numpy())
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Centroid dot for clubhead
            if cls_id == 1:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 8, color, -1)
                cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 2)

            # Label
            text = f"{label} {conf_val:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            ty = max(y1 - 6, th + 4)
            cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
            cv2.putText(frame, text, (x1 + 2, ty - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            det_counts[cls_id] = det_counts.get(cls_id, 0) + 1

        # Frame counter overlay
        cv2.putText(frame, f"frame {frame_idx}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"  shaft detections : {det_counts.get(0, 0)}")
    print(f"  clubhead detections: {det_counts.get(1, 0)}")
    print(f"  Saved → {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize golf club segmentation on video(s)")
    parser.add_argument("--video", help="Path to a specific video (omit to process all in sample_videos/)")
    parser.add_argument("--conf", type=float, default=CONF,
                        help=f"Confidence threshold (default {CONF})")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(MODEL_PATH)
    print(f"Loaded model: {MODEL_PATH}")
    print(f"Classes: {model.names}")

    if args.video:
        video_paths = [Path(args.video)]
    else:
        video_paths = sorted(p for p in VIDEO_DIR.glob("**/*") if p.suffix.lower() in VIDEO_EXTS)

    if not video_paths:
        print(f"No videos found.")
        raise SystemExit

    for vp in video_paths:
        process_video(vp, model, args.conf)

    print("\nDone.")
