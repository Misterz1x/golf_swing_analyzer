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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    L_WRIST, R_WRIST,
    L_SHOULDER, R_SHOULDER,
    L_HIP, R_HIP,
    L_ANKLE, R_ANKLE,
)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
_PEAK_PROMINENCE       = 15   # minimum pixel prominence for a real wrist peak
_FOLLOW_THROUGH_FRAMES = 30   # fallback: actual frames after impact if no valley found

# Keypoints used for the multi-joint body-stillness signal (address detection)
_BODY_KP_INDICES = [
    L_SHOULDER, R_SHOULDER,
    L_HIP,      R_HIP,
    L_WRIST,    R_WRIST,
    L_ANKLE,    R_ANKLE,
]


def detect_swing_phases(
    video_path:  str | Path,
    model,
    stride:      int   = 3,
    conf:        float = 0.3,
    imgsz:       int   = 640,
    debug:       bool  = False,
    ball_model         = None,
    ball_conf:   float = 0.25,
) -> dict[str, int]:
    """
    Detect the four key golf swing phase frames in a video.

    Parameters
    ----------
    video_path  path to the video file
    model       loaded YOLO pose model (ultralytics YOLO instance)
    stride      process every Nth frame (default 3)
    conf        pose detection confidence threshold
    imgsz       inference image size
    debug       if True returns (phases, smoothed_signal, sampled_frame_indices)
    ball_model  optional loaded YOLO ball-detection model
    ball_conf   confidence threshold for ball detections

    Returns
    -------
    dict with keys 'address', 'top_backswing', 'impact', 'follow_through'
    whose values are original video frame numbers (not sampled indices).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    actual_indices: list[int]                        = []
    wrist_y_raw:    list[float]                      = []
    kpts_seq:       list[np.ndarray]                 = []
    ball_xy:        list[tuple[float, float] | None] = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            actual_indices.append(frame_idx)
            wy, kp = _extract_frame_data(frame, model, conf, imgsz)
            wrist_y_raw.append(wy)
            kpts_seq.append(kp)
            if ball_model is not None:
                ball_xy.append(_detect_ball_xy(frame, ball_model, ball_conf, imgsz))
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

    # ---- Top of backswing: most prominent valley in the first 70 % ----------
    # Require top_bs >= 10 % so there is always room for address before it.
    # Without this floor, a flat/short backswing whose minimum is at index 0
    # causes all phases to stack on the first frame.
    search_n   = max(5, int(n * 0.70))
    min_top_bs = max(3, int(n * 0.10))

    t_valleys, t_props = find_peaks(
        -smoothed[:search_n],
        prominence=_PEAK_PROMINENCE,
        distance=max(3, search_n // 8),
    )
    valid = t_valleys >= min_top_bs
    if np.any(valid):
        vv, vp = t_valleys[valid], t_props["prominences"][valid]
        top_bs = int(vv[np.argmax(vp)])
    elif len(t_valleys) > 0:
        top_bs = int(t_valleys[np.argmax(t_props["prominences"])])
    else:
        top_bs = min_top_bs + int(np.argmin(smoothed[min_top_bs:search_n]))

    # ---- Address: most-still frame in first half of pre-backswing ------------
    address = _find_address(top_bs, kpts_seq)

    # ---- Impact ---------------------------------------------------------------
    # Hard constraint: impact must be within the first 65 % of the post-top_bs
    # segment so follow-through is never mistaken for impact.
    impact_cap = top_bs + 1 + max(3, int((n - top_bs - 1) * 0.65))
    impact_cap = min(impact_cap, n - 1)

    a_lo = max(0, address - 3)
    a_hi = min(n, address + 4)

    # Impact: the deepest point of the wrists in the downswing.
    # In image coordinates Y increases downward, so the argmax of the smoothed
    # wrist Y signal finds the frame where hands are furthest down — which
    # corresponds to club-ball contact.  After impact the wrists rise again into
    # the follow-through, so the maximum is a natural anchor point.
    # The search is capped at impact_cap so follow-through cannot be selected.
    post_y = smoothed[top_bs + 1 : impact_cap + 1]
    if len(post_y) > 0:
        impact = top_bs + 1 + int(np.argmax(post_y))
    else:
        impact = top_bs

    impact = min(int(impact), n - 1)

    # ---- Optionally refine impact with ball position signal ------------------
    if ball_model is not None and ball_xy:
        impact = _find_impact_from_ball(ball_xy, impact)

    # ---- Follow-through -------------------------------------------------------
    # At the finish the wrists have swung to the left side and risen back up
    # (back facing the camera, club over the left shoulder).  Wrist Y is low
    # again (wrists high), like the top of backswing but on the other side.
    #
    # Videos are cut to one swing, so the finish is near the end of the video.
    # Search only the last 28 % of the video (or everything after impact + 3
    # samples, whichever starts later) and pick the frame where wrists are
    # highest (argmin of wrist Y).  This avoids the intermediate mid-rotation
    # position that looks like the backswing.
    ft_zone_start = max(impact + 3, int(n * 0.72))
    if ft_zone_start < n:
        follow_through = ft_zone_start + int(np.argmin(smoothed[ft_zone_start:]))
    else:
        follow_through = n - 1
    follow_through = min(follow_through, n - 1)

    print(f"  [phase_detector] n={n}  addr={address}  top_bs={top_bs}"
          f"  impact={impact}  follow_through={follow_through}  (impact_cap={impact_cap})")

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

def _extract_frame_data(
    frame: np.ndarray, model, conf: float, imgsz: int
) -> tuple[float, np.ndarray]:
    """
    Run pose inference on one frame.
    Returns (avg_wrist_y, keypoints_17x2).
    wrist_y is NaN and keypoints are zeros on detection failure.
    """
    results = model(frame, imgsz=imgsz, verbose=False, conf=conf)
    kp = results[0].keypoints
    if kp is None or kp.xy is None or len(kp.xy) == 0:
        return float("nan"), np.zeros((17, 2), dtype=float)

    kpts = kp.xy[0].cpu().numpy()   # (17, 2) for the most confident person
    if kpts.shape[0] <= max(L_WRIST, R_WRIST):
        return float("nan"), np.zeros((17, 2), dtype=float)

    ly, ry = float(kpts[L_WRIST, 1]), float(kpts[R_WRIST, 1])
    if ly == 0 and ry == 0:
        wrist_y = float("nan")
    elif ly == 0:
        wrist_y = ry
    elif ry == 0:
        wrist_y = ly
    else:
        wrist_y = (ly + ry) / 2.0

    return wrist_y, kpts


def _body_velocity(kpts_seq: list[np.ndarray]) -> np.ndarray:
    """
    Per-frame body motion: mean displacement of the tracked keypoints relative
    to the previous frame. Keypoints at (0, 0) are treated as undetected and
    excluded from the mean. Frame 0 is assigned the value of frame 1.
    """
    n   = len(kpts_seq)
    vel = np.zeros(n, dtype=float)
    for i in range(1, n):
        prev, curr = kpts_seq[i - 1], kpts_seq[i]
        total, count = 0.0, 0
        for idx in _BODY_KP_INDICES:
            px, py = float(prev[idx, 0]), float(prev[idx, 1])
            cx, cy = float(curr[idx, 0]), float(curr[idx, 1])
            if (px == 0 and py == 0) or (cx == 0 and cy == 0):
                continue
            total += np.hypot(cx - px, cy - py)
            count += 1
        vel[i] = total / count if count > 0 else 0.0
    if n > 1:
        vel[0] = vel[1]
    return vel


def _find_address(top_bs: int, kpts_seq: list[np.ndarray]) -> int:
    """
    Find the address sample index within the pre-backswing segment.

    Uses multi-keypoint body velocity (8 joints: shoulders, hips, wrists,
    ankles). Only searches the first half of the pre-backswing window so the
    deceleration zone at the top of backswing cannot be mistaken for address.
    """
    n = min(top_bs, len(kpts_seq))
    if n < 2:
        return 0
    search_n = max(2, n // 2)
    vel = _body_velocity(kpts_seq[:search_n])
    return int(np.argmin(vel))


def _detect_ball_xy(
    frame: np.ndarray, model, conf: float, imgsz: int
) -> tuple[float, float] | None:
    """
    Run ball detection on one frame.
    Returns (cx, cy) of the most confident valid detection, or None.
    Boxes that are too large or non-round are rejected as false positives.
    """
    results = model(frame, imgsz=imgsz, verbose=False, conf=conf)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    h, w = frame.shape[:2]
    frame_area = float(h * w)

    best_conf, best_xy = -1.0, None
    for box, c in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw * bh > 0.04 * frame_area:
            continue
        aspect = bw / max(bh, 1e-6)
        if not (0.4 < aspect < 2.5):
            continue
        if float(c) > best_conf:
            best_conf = float(c)
            best_xy = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    return best_xy


def _find_impact_from_ball(
    ball_xy: list[tuple[float, float] | None],
    wrist_impact: int,
) -> int:
    """
    Refine the impact sample index using ball position data.

    Resting position is estimated from the first 40 % of the pre-wrist segment.
    Falls back to wrist_impact if fewer than 3 early detections are available.
    """
    n = len(ball_xy)

    rest_end  = max(5, int(wrist_impact * 0.4))
    early_det = [(i, xy) for i, xy in enumerate(ball_xy[:rest_end]) if xy is not None]
    if len(early_det) < 3:
        return wrist_impact

    xs = np.array([xy[0] for _, xy in early_det])
    ys = np.array([xy[1] for _, xy in early_det])
    rest_x, rest_y = float(np.median(xs)), float(np.median(ys))

    dists     = np.hypot(xs - rest_x, ys - rest_y)
    threshold = max(20.0, float(np.percentile(dists, 90)) * 1.5)

    search_end  = min(n, wrist_impact + 8)
    last_still  = None
    seen_ball   = False
    move_streak = 0

    for i in range(search_end):
        xy = ball_xy[i]
        if xy is None:
            move_streak = 0
            continue
        dist = np.hypot(xy[0] - rest_x, xy[1] - rest_y)
        if dist < threshold:
            last_still  = i
            move_streak = 0
            seen_ball   = True
        elif seen_ball:
            move_streak += 1
            if move_streak >= 2:
                break

    return last_still if last_still is not None else wrist_impact


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
        ax.invert_yaxis()
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
