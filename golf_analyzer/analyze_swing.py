"""
Task 4 — Main pipeline + CLI.

Usage:
    uv run python golf_analyzer/analyze_swing.py --video path/to/swing.mp4
    uv run python golf_analyzer/analyze_swing.py --video path/to/swing.mp4 --output ./results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from annotator      import annotate_phase_image
from config         import POSE_MODEL_PATH
from metrics        import get_raw_values
from phase_detector import detect_swing_phases

# Order for the 2×2 summary grid (row-major)
_PHASES = ["address", "top_backswing", "impact", "follow_through"]

# Inference resolution for pose extraction on 1920×1080 footage
_POSE_IMGSZ = 1280


def analyze_swing(
    video_path: str | Path,
    output_dir: str | Path,
    stride:     int   = 3,
    conf:       float = 0.3,
) -> dict[str, dict[str, float]]:
    """
    Full pipeline: detect phases → extract keypoints → compute metrics
    → annotate → save images + summary sheet.

    Returns a dict of {phase_name: {metric_name: fault_score}} for all phases.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model (relative to project root, one level above golf_analyzer/) ---
    model_path = Path(__file__).resolve().parent.parent / POSE_MODEL_PATH
    if not model_path.exists():
        sys.exit(f"ERROR: pose model not found at {model_path}")
    print(f"Loading pose model: {model_path}")
    model = YOLO(str(model_path))

    # --- Phase detection -------------------------------------------------------
    print(f"Detecting swing phases in: {video_path.name}")
    phase_frames = detect_swing_phases(
        video_path, model, stride=stride, conf=conf, imgsz=_POSE_IMGSZ
    )
    print("Detected frames:")
    for phase, frame_idx in phase_frames.items():
        print(f"  {phase:<20} frame {frame_idx:>5}")

    # --- Extract raw frames + keypoints at each phase -------------------------
    frames: dict[str, np.ndarray]  = {}
    kpts_map: dict[str, np.ndarray] = {}

    cap = cv2.VideoCapture(str(video_path))
    for phase in _PHASES:
        idx = phase_frames[phase]
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"  WARNING: could not read frame {idx} for {phase}")
            continue

        results   = model(frame, imgsz=_POSE_IMGSZ, verbose=False, conf=conf)
        kp        = results[0].keypoints
        if kp is not None and kp.xy is not None and len(kp.xy) > 0:
            kpts_map[phase] = kp.xy[0].cpu().numpy()  # (17, 2)
        else:
            print(f"  WARNING: no keypoints detected at {phase} (frame {idx})")
            kpts_map[phase] = np.zeros((17, 2), dtype=float)

        frames[phase] = frame
    cap.release()

    address_kpts = kpts_map.get("address", np.zeros((17, 2), dtype=float))

    # --- Compute raw values + annotate each phase -----------------------------
    all_raw:   dict[str, dict[str, float]] = {}
    annotated: dict[str, np.ndarray]       = {}

    for phase in _PHASES:
        if phase not in frames:
            continue
        kpts = kpts_map[phase]
        raw  = get_raw_values(kpts, address_kpts)
        all_raw[phase] = raw

        annotated[phase] = annotate_phase_image(
            frames[phase], kpts, address_kpts, {}, phase, raw_values=raw
        )
        out_path = output_dir / f"{phase}.jpg"
        cv2.imwrite(str(out_path), annotated[phase])
        print(f"  Saved {out_path}")

    # --- 2×2 summary sheet ----------------------------------------------------
    _save_summary(annotated, output_dir / "summary.jpg")

    # --- Console text summary -------------------------------------------------
    _print_summary(all_raw)

    return all_raw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_summary(annotated: dict[str, np.ndarray], out_path: Path) -> None:
    """Build a 2×2 grid from the four annotated frames and save it."""
    target_h = 540   # each tile height in the summary (2×540 = 1080)
    tiles: list[np.ndarray] = []

    for phase in _PHASES:
        if phase not in annotated:
            # Placeholder for missing phases
            tiles.append(np.zeros((target_h, int(target_h * 16 / 9), 3), dtype=np.uint8))
            continue
        img = annotated[phase]
        h, w = img.shape[:2]
        new_w = int(w * target_h / h)
        tiles.append(cv2.resize(img, (new_w, target_h)))

    # Make all tiles the same width (use the minimum)
    min_w = min(t.shape[1] for t in tiles)
    tiles = [t[:, :min_w] for t in tiles]

    top    = cv2.hconcat([tiles[0], tiles[1]])
    bottom = cv2.hconcat([tiles[2], tiles[3]])
    summary = cv2.vconcat([top, bottom])
    cv2.imwrite(str(out_path), summary)
    print(f"  Summary saved to {out_path}")


_RAW_UNITS = {
    "spine_angle":                 "deg",
    "spine_delta":                 "deg",
    "shoulder_plane_angle":        "deg",
    "hip_plane_angle":             "deg",
    "hip_lateral_displacement":    "(norm)",
    "shoulder_hip_rotation_delta": "deg",
}


def _print_summary(all_raw: dict[str, dict[str, float]]) -> None:
    """Print a formatted table of raw biomechanical values to the console."""
    if not all_raw:
        return

    metrics = list(next(iter(all_raw.values())).keys())
    col_w   = 18

    header = f"\n{'Metric':<28}" + "".join(f"{p:<{col_w}}" for p in _PHASES)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for metric in metrics:
        unit = _RAW_UNITS.get(metric, "")
        row  = f"{metric} ({unit}):"
        row  = f"{row:<28}"
        for phase in _PHASES:
            val = all_raw.get(phase, {}).get(metric, float("nan"))
            if np.isnan(val):
                row += f"{'—':<{col_w}}"
            else:
                row += f"{val:>8.3f}{'':>{col_w - 9}}"
        print(row)
    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Golf swing analyzer — extracts phases, computes biomechanical metrics."
    )
    parser.add_argument("--video",  required=True, help="Path to the golf swing video")
    parser.add_argument("--output", default="./results", help="Output directory (default: ./results)")
    parser.add_argument("--stride", type=int,   default=3,   help="Frame sampling stride (default 3)")
    parser.add_argument("--conf",   type=float, default=0.3, help="Pose detection confidence (default 0.3)")
    args = parser.parse_args()

    analyze_swing(args.video, args.output, stride=args.stride, conf=args.conf)


if __name__ == "__main__":
    main()
