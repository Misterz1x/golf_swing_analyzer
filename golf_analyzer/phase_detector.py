"""
Swing phase detector.

Analyses average wrist Y-position over time to locate the four key frames:
  address, top_backswing, impact, follow_through
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter

# Allow running this file directly from inside golf_analyzer/
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import L_WRIST, R_WRIST

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
_PEAK_PROMINENCE    = 15    # minimum pixel prominence for a real wrist peak
_STABILITY_WINDOW   = 5     # sampled frames used for rolling-std address check
_FOLLOW_THROUGH_FRAMES = 25 # actual frames after impact for follow-through


def detect_swing_phases(
    video_path: str | Path,
    model,
    stride: int = 3,
    conf: float = 0.3,
    imgsz: int = 640,
    debug: bool = False,
) -> dict[str, int]:
    """
    Detect the four key golf swing phase frames in a video.

    Parameters
    ----------
    video_path  path to the video file
    model       loaded YOLO26x-pose model (ultralytics YOLO instance)
    stride      process every Nth frame (default 3)
    conf        pose detection confidence threshold
    imgsz       inference image size
    debug       if True returns (phases, smoothed_signal, sampled_frame_indices)
                instead of just phases — useful for plotting the wrist curve

    Returns
    -------
    dict with keys 'address', 'top_backswing', 'impact', 'follow_through'
    whose values are original video frame numbers (not sampled indices).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    actual_indices: list[int] = []
    wrist_y_raw:    list[float] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            actual_indices.append(frame_idx)
            wrist_y_raw.append(_extract_wrist_y(frame, model, conf, imgsz))
        frame_idx += 1
    cap.release()

    n_samples = len(actual_indices)
    if n_samples < 10:
        raise RuntimeError(
            f"Only {n_samples} sampled frames — "
            "check that the pose model detects a person or reduce stride."
        )

    wrist_y  = _interpolate_nans(np.array(wrist_y_raw, dtype=float))
    smoothed = _smooth(wrist_y)
    n        = len(smoothed)

    # ---- Impact: most prominent wrist Y peak (wrists lowest = largest Y) -----
    # Image Y increases downward, so lowest wrist position = largest Y value.
    peaks, p_props = find_peaks(
        smoothed,
        prominence=_PEAK_PROMINENCE,
        distance=max(3, n // 10),
    )
    if len(peaks) > 0:
        impact = int(peaks[np.argmax(p_props["prominences"])])
    else:
        impact = int(np.argmax(smoothed))
    impact = min(impact, n - 1)

    # ---- Top of backswing: last significant valley BEFORE impact -------------
    pre_impact = smoothed[:impact]
    valleys, v_props = find_peaks(
        -pre_impact,
        prominence=_PEAK_PROMINENCE // 2,
        distance=max(3, impact // 8),
    )
    if len(valleys) > 0:
        top_bs = int(valleys[-1])
    elif len(pre_impact) > 0:
        top_bs = int(np.argmin(pre_impact))
    else:
        top_bs = max(0, impact - 5)

    # ---- Address: last stable frame before backswing begins ------------------
    address = _find_address(wrist_y, top_bs)

    # ---- Follow-through: fixed actual-frame offset after impact --------------
    ft_samples   = max(1, _FOLLOW_THROUGH_FRAMES // stride)
    follow_through = min(impact + ft_samples, n - 1)

    phases: dict[str, int] = {
        "address":        actual_indices[address],
        "top_backswing":  actual_indices[top_bs],
        "impact":         actual_indices[impact],
        "follow_through": actual_indices[follow_through],
    }

    if debug:
        return phases, smoothed, actual_indices
    return phases


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_wrist_y(frame: np.ndarray, model, conf: float, imgsz: int) -> float:
    """Run pose inference on one frame and return average wrist Y, or NaN."""
    results = model(frame, imgsz=imgsz, verbose=False, conf=conf)
    kp = results[0].keypoints
    if kp is None or kp.xy is None or len(kp.xy) == 0:
        return float("nan")

    kpts = kp.xy[0].cpu().numpy()          # (17, 2) for the most confident person
    if kpts.shape[0] <= max(L_WRIST, R_WRIST):
        return float("nan")

    ly, ry = float(kpts[L_WRIST, 1]), float(kpts[R_WRIST, 1])

    # YOLO returns (0, 0) for undetected keypoints
    if ly == 0 and ry == 0:
        return float("nan")
    if ly == 0:
        return ry
    if ry == 0:
        return ly
    return (ly + ry) / 2.0


def _interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """Replace NaN values with linear interpolation from neighbours."""
    nans = np.isnan(arr)
    if not nans.any():
        return arr
    x = np.arange(len(arr))
    arr[nans] = np.interp(x[nans], x[~nans], arr[~nans])
    return arr


def _smooth(y: np.ndarray) -> np.ndarray:
    """Savitzky-Golay smoothing with auto-sized window."""
    n      = len(y)
    window = min(max(5, n // 6), n - 1)
    if window % 2 == 0:
        window -= 1
    poly = min(3, window - 1)
    return savgol_filter(y, window_length=window, polyorder=poly)


def _find_address(wrist_y: np.ndarray, top_bs: int) -> int:
    """
    Find the address sample index within the pre-backswing segment.

    Only searches the first half of the pre-backswing window — address must
    happen before the swing commits, not near the top-of-backswing deceleration
    zone (which also looks 'stable' and caused the original backwards walk to
    return the wrong frame).

    Within that window, returns the frame with the lowest absolute wrist
    velocity (most still), which is the genuine setup moment.
    """
    pre = wrist_y[:top_bs] if top_bs > 0 else wrist_y[:1]
    n   = len(pre)
    if n < _STABILITY_WINDOW:
        return 0

    # Restrict search to first 50% of the pre-backswing segment
    search_n = max(_STABILITY_WINDOW, n // 2)
    search   = pre[:search_n]

    # Velocity (absolute discrete derivative) — address = most still frame
    vel = np.abs(np.gradient(search))
    return int(np.argmin(vel))


# ---------------------------------------------------------------------------
# Quick test — run as:  uv run python golf_analyzer/phase_detector.py <video>
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description="Test swing phase detection on a video.")
    parser.add_argument("video",  help="Path to the golf video")
    parser.add_argument("--model", default="models/yolo26x-pose.pt", help="Pose model path")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--conf",   type=float, default=0.3)
    parser.add_argument("--imgsz",  type=int, default=640)
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    pose_model = YOLO(args.model)

    print(f"Analysing: {args.video}  (stride={args.stride})")
    phases, smoothed, frame_nums = detect_swing_phases(
        args.video, pose_model,
        stride=args.stride, conf=args.conf, imgsz=args.imgsz,
        debug=True,
    )

    print("\n── Detected swing phases ──────────────────────")
    for phase, frame in phases.items():
        print(f"  {phase:<20} frame {frame:>5}")

    print(f"\n── Wrist Y signal stats ───────────────────────")
    print(f"  Sampled frames : {len(smoothed)}")
    print(f"  Y range        : {smoothed.min():.1f} – {smoothed.max():.1f} px")

    # Optional: plot the signal if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(frame_nums, smoothed, label="Wrist Y (smoothed)", color="steelblue")
        colors = {"address": "green", "top_backswing": "purple",
                  "impact": "red", "follow_through": "orange"}
        for phase, frame in phases.items():
            idx = frame_nums.index(frame)
            ax.axvline(frame, color=colors[phase], linestyle="--", label=phase)
            ax.annotate(phase, (frame, smoothed[idx]), textcoords="offset points",
                        xytext=(4, 6), fontsize=8, color=colors[phase])
        ax.invert_yaxis()          # flip so "high in frame" = up on plot
        ax.set_xlabel("Video frame")
        ax.set_ylabel("Avg wrist Y (px, inverted)")
        ax.legend(fontsize=8)
        ax.set_title("Golf swing wrist trajectory")
        plt.tight_layout()
        plt.savefig("phase_debug.png", dpi=120)
        print("\n  Debug plot saved to phase_debug.png")
        plt.show()
    except ImportError:
        print("  (install matplotlib to get a debug plot)")
