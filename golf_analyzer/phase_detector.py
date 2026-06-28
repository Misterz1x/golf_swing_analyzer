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

# Head keypoint indices (COCO 17-point order)
_NOSE  = 0
_L_EYE = 1
_R_EYE = 2
_L_EAR = 3
_R_EAR = 4


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

    _fps        = cap.get(cv2.CAP_PROP_FPS)
    _total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

    # Baseline: blend of inflection and argmax of smoothed wrist-Y.
    # argmax(d/dt) = inflection = peak downward velocity → fires slightly early.
    # argmax(wrist_y) = deepest point → fires too late (post-contact descent).
    # The true impact lies between: we take 40 % of the way from inflection
    # toward argmax, which empirically centres on ball contact.
    _seg_end = min(impact_cap + 2, n)
    _dy      = np.diff(smoothed[top_bs : _seg_end])
    if len(_dy) > 0:
        _k_infl  = int(np.argmax(_dy))
        _infl    = min(top_bs + _k_infl + 1, impact_cap)
    else:
        _infl = top_bs + 1

    _post_y  = smoothed[top_bs + 1 : impact_cap + 1]
    _argmax  = (top_bs + 1 + int(np.argmax(_post_y))) if len(_post_y) > 0 else _infl

    # Degenerate guard: if argmax is at the very start of the window the
    # wrist-Y signal is not helping (wrists rise immediately, no clear peak).
    # Fall back to 40 % through the downswing window as a neutral estimate.
    _ds_window = impact_cap - top_bs
    if _argmax <= top_bs + max(3, int(_ds_window * 0.08)):
        impact = top_bs + max(3, int(_ds_window * 0.40))
    else:
        impact = _infl + int((_argmax - _infl) * 0.60)
    impact = min(max(int(impact), top_bs + 1), impact_cap)

    # Minimum floor: impact cannot be in the first 10 % of the downswing window.
    # This prevents top-of-backswing mis-detection when crossing signals fire
    # immediately after top_bs due to degenerate wrist-Y or wrong crossing direction.
    _min_impact = top_bs + max(5, int(_ds_window * 0.10))

    _blend = impact   # record blend for debug; used as fallback

    # --- Primary: wrist directly over ball rest X --------------------------------
    # Most accurate when ball is detected: the first frame where wrist X crosses
    # ball_rest_x is physically the contact moment, adapting to each golfer's
    # ball placement (e.g. forward in the stance for a driver).
    _wb = (_find_impact_wrist_over_ball(kpts_seq, ball_xy, top_bs, impact_cap)
           if (ball_model is not None and ball_xy) else None)

    # --- Secondary: wrist over address X (proxy when ball is not detected) ------
    # address X is the best available proxy for ball X when ball detection fails.
    # Called with _blend as fallback so _w == _blend signals "no crossing found".
    _w_raw = _refine_impact_wrist_x(kpts_seq, _blend, top_bs, impact_cap, address)
    _w = _w_raw if _w_raw != _blend else None   # None → no crossing found

    # Priority: wrist_over_ball beats wrist_over_address beats blend.
    if _wb is not None:
        impact = _wb
    elif _w is not None:
        impact = _w
    # else: impact = _blend already set above

    _baseline = impact

    # --- Head rotation: hard upper-bound cap -------------------------------------
    # Head starts rotating only after impact — cap the estimate there.
    _h = _refine_impact_head_stability(kpts_seq, impact, top_bs, impact_cap)
    if _h < impact:
        impact = _h

    impact = min(max(impact, _min_impact), impact_cap)

    print(f"  [impact_debug] blend={_blend}  wrist_over_ball={_wb}  wrist_x={_w}"
          f"  head={_h}  min_floor={_min_impact}  final={impact}")

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

    _ts = lambda i: f"{actual_indices[i] / _fps:.2f}s" if _fps > 0 else "?s"
    print(f"  [phase_detector] n={n}  fps={_fps:.1f}  total={_total_frames / _fps:.1f}s"
          f"  addr={address}({_ts(address)})"
          f"  top_bs={top_bs}({_ts(top_bs)})"
          f"  impact={impact}({_ts(impact)})"
          f"  follow_through={follow_through}({_ts(follow_through)})"
          f"  (impact_cap={impact_cap})")

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
    ball_xy:      list[tuple[float, float] | None],
    wrist_impact: int,
    top_bs:       int = 0,
) -> int:
    """
    Refine the impact sample index using ball position data.

    Resting position is estimated from all frames before the backswing (top_bs),
    where the ball is guaranteed to be stationary. Falls back to the
    40%-of-wrist_impact heuristic if top_bs yields fewer than 3 detections,
    and ultimately to wrist_impact if still not enough data.
    """
    n = len(ball_xy)

    # Prefer pre-backswing frames for the rest-position estimate.
    rest_end  = max(top_bs, max(5, int(wrist_impact * 0.4)))
    early_det = [(i, xy) for i, xy in enumerate(ball_xy[:rest_end + 1]) if xy is not None]
    if len(early_det) < 3:
        return wrist_impact

    xs = np.array([xy[0] for _, xy in early_det])
    ys = np.array([xy[1] for _, xy in early_det])
    rest_x, rest_y = float(np.median(xs)), float(np.median(ys))

    dists     = np.hypot(xs - rest_x, ys - rest_y)
    threshold = max(20.0, float(np.percentile(dists, 90)) * 1.5)

    # Start from top_bs — the ball cannot be hit during the backswing.
    search_start = max(0, top_bs)
    search_end   = min(n, wrist_impact + 8)
    last_still   = None
    seen_ball    = False
    move_streak  = 0

    for i in range(search_start, search_end):
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


def _find_impact_wrist_over_ball(
    kpts_seq:   list[np.ndarray],
    ball_xy:    list[tuple[float, float] | None],
    top_bs:     int,
    impact_cap: int,
) -> int | None:
    """
    Find the first downswing frame where the wrist X crosses the ball's resting X.

    At impact the hands are directly over the ball horizontally (face-on view).
    This is more accurate than using the address wrist X as the crossing target
    because it accounts for each golfer's ball placement — forward stance placement
    means the ball is to the camera-left of where the hands were at address, so
    the wrist reaches ball_rest_x earlier and the crossing fires closer to actual
    contact.

    Returns None if ball detections are insufficient or no crossing is found.
    """
    n = len(ball_xy)

    # Estimate ball resting position from a window centred on top_bs.
    # At and just after top_bs the golfer has rotated away, making the ball
    # visible for the first time even if it was occluded during address.
    # Frames in this window are still well before impact so the ball is static.
    rest_start = max(0, top_bs - 10)
    rest_end   = min(n - 1, top_bs + 10)
    window_det = [(i, xy) for i, xy in enumerate(ball_xy[rest_start : rest_end + 1]) if xy is not None]
    if len(window_det) < 3:
        return None

    ball_rest_x = float(np.median([xy[0] for _, xy in window_det]))

    def _avg_wrist_x(kp: np.ndarray) -> float | None:
        lx, rx = float(kp[L_WRIST, 0]), float(kp[R_WRIST, 0])
        if lx > 0 and rx > 0:
            return (lx + rx) / 2.0
        if lx > 0:
            return lx
        if rx > 0:
            return rx
        return None

    top_wx = _avg_wrist_x(kpts_seq[top_bs])
    if top_wx is None or abs(top_wx - ball_rest_x) < 20:
        return None  # not enough displacement for a reliable crossing

    going_right = top_wx < ball_rest_x  # most common in face-on right-handed view

    # Don't allow the crossing to fire in the first 10 % of the downswing
    # window — a crossing there is almost certainly a detection artifact.
    search_lo = top_bs + max(3, int((impact_cap - top_bs) * 0.10))

    # First two consecutive frames where the wrist has crossed ball_rest_x.
    for i in range(search_lo, impact_cap):
        if i + 1 >= len(kpts_seq):
            break
        wx      = _avg_wrist_x(kpts_seq[i])
        wx_next = _avg_wrist_x(kpts_seq[i + 1])
        if wx is None or wx_next is None:
            continue
        if going_right and wx >= ball_rest_x and wx_next >= ball_rest_x:
            return i
        if not going_right and wx <= ball_rest_x and wx_next <= ball_rest_x:
            return i

    return None  # no crossing found


def _refine_impact_wrist_x(
    kpts_seq:   list[np.ndarray],
    impact:     int,
    top_bs:     int,
    impact_cap: int,
    address:    int,
) -> int:
    """
    Refine impact using the wrist-X crossing point.

    During the downswing the hands sweep back through the address horizontal
    position as the club meets the ball. This function finds the FIRST frame
    where the wrists have confirmed crossed back to address X (two consecutive
    frames on the address side), which corresponds to the moment of contact.

    Unlike argmin-of-distance (which finds the closest approach, possibly late
    in the follow-through arc), the first crossing is anchored to the actual
    transition point and is always early rather than late.

    Skipped if wrist horizontal displacement during backswing is < 30 px.
    """
    if address >= len(kpts_seq):
        return impact

    def _avg_wrist_x(kp: np.ndarray) -> float | None:
        lx, rx = float(kp[L_WRIST, 0]), float(kp[R_WRIST, 0])
        if lx > 0 and rx > 0:
            return (lx + rx) / 2.0
        if lx > 0:
            return lx
        if rx > 0:
            return rx
        return None

    # Stable address reference: median over a small window around address.
    addr_xs = [_avg_wrist_x(kpts_seq[ai])
               for ai in range(max(0, address - 2), min(len(kpts_seq), address + 3))]
    addr_xs = [x for x in addr_xs if x is not None]
    if not addr_xs:
        return impact
    addr_x = float(np.median(addr_xs))

    top_x = _avg_wrist_x(kpts_seq[top_bs])
    if top_x is None or abs(top_x - addr_x) < 30:
        return impact  # too little horizontal motion — crossing signal unreliable

    # Direction of crossing: hands went left during backswing → cross right going down.
    going_right = top_x < addr_x

    # Don't allow the crossing to fire in the first 10 % of the downswing window.
    search_lo = top_bs + max(3, int((impact_cap - top_bs) * 0.10))

    # Find the first two consecutive frames where the wrists have crossed addr_x.
    # Requiring two consecutive frames filters single-frame keypoint noise.
    for i in range(search_lo, impact_cap):
        if i + 1 >= len(kpts_seq):
            break
        wx      = _avg_wrist_x(kpts_seq[i])
        wx_next = _avg_wrist_x(kpts_seq[i + 1])
        if wx is None or wx_next is None:
            continue
        if going_right and wx >= addr_x and wx_next >= addr_x:
            return i   # first confirmed crossing → impact
        if not going_right and wx <= addr_x and wx_next <= addr_x:
            return i

    return impact  # no crossing found in window


def _refine_impact_head_stability(
    kpts_seq:   list[np.ndarray],
    impact:     int,
    top_bs:     int,
    impact_cap: int,
) -> int:
    """
    Refine impact by detecting the onset of head rotation at contact.

    During the swing the golfer looks DOWN at the ball — the nose is below the
    ear midpoint in pixel space (positive nose-to-ear tilt). At impact the body
    extends and the head rotates upward: the nose rises RELATIVE to the ears,
    so the tilt decreases. Tracking this relative rotation filters out whole-head
    translation and catches only the genuine chin-up rotation.

    Falls back to the current estimate if the signal is too weak or absent.
    """
    def _head_tilt(kp: np.ndarray) -> float | None:
        """
        nose_y − ear_mid_y  (or nose_y − eye_mid_y as fallback).
        Positive  = nose below reference = looking down.
        Decreasing = head rotating upward = post-impact signal.
        """
        nose_y = float(kp[_NOSE, 1])
        if nose_y <= 0:
            return None

        lear_y = float(kp[_L_EAR, 1])
        rear_y = float(kp[_R_EAR, 1])
        ley_y  = float(kp[_L_EYE, 1])
        rey_y  = float(kp[_R_EYE, 1])

        # Prefer ears (better rotation indicator); fall back to eye midpoint.
        if lear_y > 0 and rear_y > 0:
            ref_y = (lear_y + rear_y) / 2.0
        elif lear_y > 0:
            ref_y = lear_y
        elif rear_y > 0:
            ref_y = rear_y
        elif ley_y > 0 and rey_y > 0:
            ref_y = (ley_y + rey_y) / 2.0
        elif ley_y > 0:
            ref_y = ley_y
        elif rey_y > 0:
            ref_y = rey_y
        else:
            return None

        return nose_y - ref_y   # positive = looking down

    # Reference tilt around top_bs — golfer is definitely looking down here.
    ref_start  = max(0, top_bs - 2)
    ref_end    = min(len(kpts_seq), top_bs + 3)
    ref_tilts  = [_head_tilt(kpts_seq[i]) for i in range(ref_start, ref_end)]
    ref_tilts  = [t for t in ref_tilts if t is not None]
    if len(ref_tilts) < 2:
        return impact

    stable_tilt = float(np.median(ref_tilts))
    spread      = max(ref_tilts) - min(ref_tilts)
    # Tighter threshold than a raw-position check because this is already
    # a relative (rotation) measurement.
    threshold   = max(10.0, spread * 2.0)

    # Cap search at 80 % of the downswing window. The head also rotates up
    # naturally during the follow-through; limiting the search prevents that
    # later motion from being mistaken for impact.
    search_end = top_bs + int((impact_cap - top_bs) * 0.80)

    # Find the first 2 consecutive frames where the tilt has decreased past
    # the threshold (head is rotating upward = chin coming up off the ball).
    rise_start = None
    streak     = 0

    for i in range(top_bs, search_end + 1):
        tilt = _head_tilt(kpts_seq[i])
        if tilt is None:
            streak     = 0
            rise_start = None
            continue
        if tilt < stable_tilt - threshold:   # tilt decreased = nose rose relative to ears
            if streak == 0:
                rise_start = i
            streak += 1
            if streak >= 2:
                break
        else:
            streak     = 0
            rise_start = None

    if streak < 2 or rise_start is None:
        return impact  # no clear head rotation detected

    # Impact is the frame just before the rotation started.
    candidate = max(rise_start - 1, top_bs + 1)
    if top_bs + 1 <= candidate <= impact_cap:
        return candidate
    return impact


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
