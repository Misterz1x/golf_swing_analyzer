"""
Run the trained golf ball model on all videos in data/our_videos/.
Annotated output videos are saved to output_ballv2/.

False-positive filtering
------------------------
Four filters are applied after the model's raw detections:

  CONF          — reject low-confidence detections
  MAX_AREA_FRAC — reject boxes covering >5% of frame (golf ball is tiny)
  ASPECT_RATIO  — reject non-square boxes (golf balls are round)
  Y_MIN_FRAC    — reject if ball centre is in the top N% of the frame
                  (sky, trees, nets are above the playing surface)

Temporal consistency
--------------------
After geometry filtering, a detection must appear within TRACK_RADIUS pixels
in at least TRACK_REQUIRE of the last TRACK_WINDOW frames to be drawn green
("confirmed ball").  Detections that passed geometry but are not yet confirmed
are drawn yellow ("?") — single-frame noise stays yellow and disappears.
"""

import argparse
import math
from collections import deque
import cv2
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH  = "runs2/train/golf_ball_yolo26x_v2/weights/best.pt"
VIDEO_DIR   = Path("data/our_videos")
OUTPUT_DIR  = Path("output_ballv2")
IMG_SIZE    = 1920
VIDEO_EXTS  = {".mp4", ".avi", ".mov", ".mkv"}

# --- Geometric filters ---
CONF          = 0.30
MAX_AREA_FRAC = 0.05   # reject if box > 5% of frame area
ASPECT_MIN    = 0.4    # reject if width/height < 0.4
ASPECT_MAX    = 2.5    # reject if width/height > 2.5
Y_MIN_FRAC    = 0.25   # reject if ball centre is in the top 25% of the frame

# --- Temporal consistency ---
TRACK_WINDOW  = 5    # how many past frames to look at
TRACK_RADIUS  = 80   # pixel distance that counts as "same ball"
TRACK_REQUIRE = 2    # must appear in >= this many of the past TRACK_WINDOW frames


def _center(x1, y1, x2, y2):
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def geometry_reject_reason(x1, y1, x2, y2, frame_w, frame_h) -> str:
    """Return a short reason string if rejected, empty string if valid."""
    w = x2 - x1
    h = y2 - y1
    if w <= 0 or h <= 0:
        return "zero-size"
    if (w * h) / (frame_w * frame_h) > MAX_AREA_FRAC:
        return "too-large"
    aspect = w / h
    if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
        return f"aspect={aspect:.1f}"
    _, cy = _center(x1, y1, x2, y2)
    if cy < frame_h * Y_MIN_FRAC:
        return f"too-high(y={cy/frame_h:.2f})"
    return ""


def is_temporally_confirmed(cx, cy, history: deque) -> bool:
    """True if (cx,cy) is within TRACK_RADIUS of a detection in >= TRACK_REQUIRE past frames."""
    hits = 0
    for frame_centers in history:
        for (px, py) in frame_centers:
            if math.hypot(cx - px, cy - py) <= TRACK_RADIUS:
                hits += 1
                break  # each past frame counts at most once
    return hits >= TRACK_REQUIRE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Golf ball detection on video(s)")
    parser.add_argument("--video", help="Path to a specific video file (omit to process all in data/our_videos/)")
    parser.add_argument("--tag", default="",
                        help="Label appended to the output filename, e.g. --tag v2_yfilter  →  video_ball_v2_yfilter.mp4")
    parser.add_argument("--conf", type=float, default=CONF,
                        help=f"Confidence threshold (default {CONF})")
    parser.add_argument("--y-min", type=float, default=Y_MIN_FRAC,
                        help=f"Top-of-frame exclusion zone as fraction of height (default {Y_MIN_FRAC})")
    parser.add_argument("--track-radius", type=int, default=TRACK_RADIUS,
                        help=f"Pixel radius for temporal consistency (default {TRACK_RADIUS})")
    args = parser.parse_args()

    CONF         = args.conf
    Y_MIN_FRAC   = args.y_min
    TRACK_RADIUS = args.track_radius

    model = YOLO(MODEL_PATH)
    OUTPUT_DIR.mkdir(exist_ok=True)

    if args.video:
        video_paths = [Path(args.video)]
    else:
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
        print(f"  Filters: conf={CONF}  y_min={Y_MIN_FRAC}  track_radius={TRACK_RADIUS}px  require={TRACK_REQUIRE}/{TRACK_WINDOW} frames")

        tag      = f"_{args.tag}" if args.tag else ""
        out_path = OUTPUT_DIR / (video_path.stem + f"_ball{tag}.mp4")
        writer   = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        frame_count     = 0
        raw_count       = 0
        geo_passed      = 0
        confirmed_count = 0

        # Per-frame list of (cx, cy) centers that passed geometry — used for temporal check
        history: deque[list] = deque(maxlen=TRACK_WINDOW)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, imgsz=IMG_SIZE, conf=CONF, verbose=False)

            frame_geo_hits = []

            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf_val = float(box.conf[0])
                raw_count += 1

                reason = geometry_reject_reason(x1, y1, x2, y2, width, height)

                if reason:
                    # Red — rejected by geometry
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(frame, f"{reason} {conf_val:.2f}", (x1, max(y1 - 6, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                    continue

                geo_passed += 1
                cx, cy = _center(x1, y1, x2, y2)
                frame_geo_hits.append((cx, cy))

                if is_temporally_confirmed(cx, cy, history):
                    # Green — confirmed ball
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"ball {conf_val:.2f}", (x1, max(y1 - 6, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    confirmed_count += 1
                else:
                    # Yellow — passed geometry, awaiting temporal confirmation
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 1)
                    cv2.putText(frame, f"? {conf_val:.2f}", (x1, max(y1 - 6, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

            history.append(frame_geo_hits)
            writer.write(frame)
            frame_count += 1

        cap.release()
        writer.release()
        print(f"  Done — {frame_count} frames")
        print(f"  Raw: {raw_count}  |  geometry ok: {geo_passed}  |  confirmed: {confirmed_count}  |  filtered: {raw_count - confirmed_count}")
        print(f"  Saved -> {out_path}")

    print("\nAll videos processed.")
