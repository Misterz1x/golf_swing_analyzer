"""
Run the trained golf ball model on all videos in data/our_videos/.
Annotated output videos are saved to output_ballv2/.

False-positive filtering
------------------------
Three geometric filters are applied after the model's raw detections to cut
objects like nets, hats, shoes and shoulders that the model confuses with balls:

  CONF          — raise to reject low-confidence detections
  MAX_AREA_FRAC — reject boxes that cover more than this fraction of the frame
                  (a real golf ball is tiny; a hat or net patch is much bigger)
  ASPECT_RATIO  — reject boxes that aren't roughly square (golf balls are round)
"""

import cv2
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH     = "runs2/train/golf_ball_yolo26x_v2/weights/best.pt"
VIDEO_DIR      = Path("data/our_videos")
OUTPUT_DIR     = Path("output_ballv2")
# 1920 = native video resolution, zero downscaling. The ball stays exactly the
# size it appears in the original frame. Fine for inference on RTX 5060 (single frame).
IMG_SIZE       = 1920
VIDEO_EXTS     = {".mp4", ".avi", ".mov", ".mkv"}

# --- Tune these to trade off false positives vs missed detections ---
CONF           = 0.30   # raise if still seeing false positives; lower if missing the ball
MAX_AREA_FRAC  = 0.05   # reject if box covers >5% of frame area (golf ball is small)
ASPECT_MIN     = 0.4    # reject if width/height < 0.4  (too tall/thin to be a ball)
ASPECT_MAX     = 2.5    # reject if width/height > 2.5  (too wide to be a ball)


def is_valid_detection(x1, y1, x2, y2, frame_w, frame_h) -> bool:
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return False
    area_frac  = (w * h) / (frame_w * frame_h)
    aspect     = w / h
    if area_frac > MAX_AREA_FRAC:
        return False   # too large — net, hat, shoulder, etc.
    if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
        return False   # not round enough
    return True


if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    OUTPUT_DIR.mkdir(exist_ok=True)

    video_paths = [p for p in VIDEO_DIR.glob("**/*") if p.suffix.lower() in VIDEO_EXTS]
    if not video_paths:
        print(f"No videos found in {VIDEO_DIR}/")
        raise SystemExit

    for video_path in video_paths:
        cap    = cv2.VideoCapture(str(video_path))
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\nProcessing {video_path.name}  ({width}x{height} @ {fps:.1f} fps, {total} frames)")

        out_path = OUTPUT_DIR / (video_path.stem + "_ball.mp4")
        writer   = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        frame_count  = 0
        raw_count    = 0
        kept_count   = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, imgsz=IMG_SIZE, conf=CONF, verbose=False)

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_val = float(box.conf[0])
                raw_count += 1

                if not is_valid_detection(x1, y1, x2, y2, width, height):
                    # Draw filtered-out detections in red so you can see what got rejected
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(frame, f"filtered {conf_val:.2f}", (x1, max(y1 - 6, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ball {conf_val:.2f}", (x1, max(y1 - 6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                kept_count += 1

            writer.write(frame)
            frame_count += 1

        cap.release()
        writer.release()
        filtered = raw_count - kept_count
        print(f"  Done — {frame_count} frames")
        print(f"  Raw detections : {raw_count}  |  kept: {kept_count}  |  filtered out: {filtered}")
        print(f"  Saved -> {out_path}")

    print("\nAll videos processed.")
