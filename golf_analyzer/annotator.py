"""
Task 3 — Frame annotation.

annotate_phase_image(frame, kpts, address_kpts, metrics, phase_name)
  -> annotated copy of the frame (original is not modified)
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_WRIST, R_WRIST

# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------
_C_SPINE_GHOST = (180,  60,  20)   # blue-ish (for semi-transparent ghost)
_C_SPINE_LIVE  = (  0, 255, 255)   # yellow
_C_SHOULDER    = (255, 255,   0)   # cyan
_C_HIP         = (  0, 165, 255)   # orange
_C_WRIST       = (  0, 255,   0)   # bright green
_C_ARC         = (  0, 215, 255)   # gold
_C_TEXT        = (255, 255, 255)   # white

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def annotate_phase_image(
    frame:        np.ndarray,
    kpts:         np.ndarray,
    address_kpts: np.ndarray,
    metrics:      dict[str, float],
    phase_name:   str,
    raw_values:   dict[str, float] | None = None,
) -> np.ndarray:
    """
    Draw biomechanical annotations onto a copy of *frame*.

    Parameters
    ----------
    frame         raw video frame (H x W x 3 BGR)
    kpts          current phase keypoints (17, 2)
    address_kpts  address/setup keypoints for the ghost spine (17, 2)
    metrics       unused (kept for API compatibility)
    phase_name    label string drawn bottom-left
    raw_values    dict of raw physical values (degrees, displacement) for HUD

    Returns
    -------
    Annotated copy of the frame.
    """
    out   = frame.copy()
    h, w  = out.shape[:2]
    scale = h / 1080.0   # normalise all sizes to a 1080-tall reference frame

    # ---- 1. Ghost spine (address reference, semi-transparent dashed blue) ---
    sh_mid_addr  = _mid(address_kpts, L_SHOULDER, R_SHOULDER)
    hip_mid_addr = _mid(address_kpts, L_HIP,      R_HIP)
    if _valid(sh_mid_addr) and _valid(hip_mid_addr):
        overlay = out.copy()
        _draw_dashed_line(
            overlay,
            _pt(sh_mid_addr), _pt(hip_mid_addr),
            _C_SPINE_GHOST, max(1, int(2 * scale)),
            dash_len=int(14 * scale), gap_len=int(8 * scale),
        )
        cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)

    # ---- 2. Live spine (solid yellow) ----------------------------------------
    sh_mid  = _mid(kpts, L_SHOULDER, R_SHOULDER)
    hip_mid = _mid(kpts, L_HIP,      R_HIP)
    if _valid(sh_mid) and _valid(hip_mid):
        cv2.line(out, _pt(sh_mid), _pt(hip_mid), _C_SPINE_LIVE, max(2, int(2 * scale)))

    # ---- 3. Shoulder plane (cyan, extended 30 px each side, dots) -----------
    if _valid(kpts[L_SHOULDER]) and _valid(kpts[R_SHOULDER]):
        _draw_plane_line(out, kpts[L_SHOULDER], kpts[R_SHOULDER],
                         _C_SHOULDER, scale)

    # ---- 4. Hip plane (orange, same style) -----------------------------------
    if _valid(kpts[L_HIP]) and _valid(kpts[R_HIP]):
        _draw_plane_line(out, kpts[L_HIP], kpts[R_HIP], _C_HIP, scale)

    # ---- 5. Spine angle arc + label ------------------------------------------
    if _valid(sh_mid) and _valid(hip_mid):
        _draw_spine_arc(out, sh_mid, hip_mid, scale)

    # ---- 6. Wrist dots (bright green) ----------------------------------------
    for idx in (L_WRIST, R_WRIST):
        if _valid(kpts[idx]):
            cv2.circle(out, _pt(kpts[idx]), max(5, int(7 * scale)),
                       _C_WRIST, -1)

    # ---- 7. Phase label (bottom-left) ----------------------------------------
    label_y = h - int(20 * scale)
    _text_with_bg(out, phase_name.replace("_", " ").upper(),
                  (int(16 * scale), label_y),
                  scale * 0.9, _C_TEXT, thickness=max(1, int(2 * scale)))

    # ---- 8. Raw values HUD (top-right) ---------------------------------------
    _draw_hud(out, raw_values, scale)

    return out


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_plane_line(
    img:   np.ndarray,
    left:  np.ndarray,
    right: np.ndarray,
    color: tuple[int, int, int],
    scale: float,
    ext:   int = 30,
) -> None:
    """Draw a line through two keypoints extended `ext` px past each endpoint."""
    d = right - left
    length = np.linalg.norm(d)
    if length < 1:
        return
    d_unit = d / length
    ext_px = int(ext * scale)
    p1 = _pt(left  - d_unit * ext_px)
    p2 = _pt(right + d_unit * ext_px)
    cv2.line(img, p1, p2, color, max(2, int(2 * scale)))
    r = max(4, int(5 * scale))
    cv2.circle(img, _pt(left),  r, color, -1)
    cv2.circle(img, _pt(right), r, color, -1)


def _draw_spine_arc(
    img:     np.ndarray,
    sh_mid:  np.ndarray,
    hip_mid: np.ndarray,
    scale:   float,
) -> None:
    """Draw a small arc at shoulder midpoint showing spine deviation from vertical."""
    spine_vec = hip_mid - sh_mid
    if np.linalg.norm(spine_vec) < 1:
        return

    # Angle of spine direction measured clockwise from east (OpenCV convention)
    spine_from_east = float(np.degrees(np.arctan2(spine_vec[1], spine_vec[0])))
    vertical = 90.0   # straight down = 90° from east in image coords

    start_a = int(min(vertical, spine_from_east))
    end_a   = int(max(vertical, spine_from_east))
    angle_deg = abs(spine_from_east - vertical)

    if angle_deg < 0.5:
        return

    radius = max(25, int(35 * scale))
    center = _pt(sh_mid)
    cv2.ellipse(img, center, (radius, radius), 0, start_a, end_a, _C_ARC, max(1, int(2 * scale)))

    # Label near arc
    mid_angle = np.radians((start_a + end_a) / 2)
    lx = int(center[0] + (radius + int(8 * scale)) * np.cos(mid_angle))
    ly = int(center[1] + (radius + int(8 * scale)) * np.sin(mid_angle))
    _text_with_bg(img, f"{angle_deg:.1f}deg", (lx, ly),
                  scale * 0.55, _C_ARC, thickness=1)


def _draw_dashed_line(
    img:       np.ndarray,
    pt1:       tuple[int, int],
    pt2:       tuple[int, int],
    color:     tuple[int, int, int],
    thickness: int,
    dash_len:  int = 14,
    gap_len:   int = 8,
) -> None:
    d      = np.array(pt2, float) - np.array(pt1, float)
    length = np.linalg.norm(d)
    if length < 1:
        return
    unit   = d / length
    pos    = 0.0
    draw   = True
    while pos < length:
        if draw:
            end = min(pos + dash_len, length)
            p1  = tuple((np.array(pt1, float) + unit * pos).astype(int))
            p2  = tuple((np.array(pt1, float) + unit * end).astype(int))
            cv2.line(img, p1, p2, color, thickness)
            pos += dash_len
        else:
            pos += gap_len
        draw = not draw


def _draw_hud(
    img:        np.ndarray,
    raw_values: dict[str, float] | None,
    scale:      float,
) -> None:
    """Draw a semi-transparent raw-values panel in the top-right corner."""
    if not raw_values:
        return

    h, w = img.shape[:2]

    margin  = max(8, int(10 * scale))
    row_h   = max(18, int(22 * scale))
    label_w = int(180 * scale)
    val_w   = int(110 * scale)
    title_h = int(26 * scale)
    panel_w = label_w + val_w + 2 * margin
    panel_h = len(raw_values) * row_h + title_h + 2 * margin

    x0 = w - panel_w - margin
    y0 = margin

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)

    # Title
    cv2.putText(img, "BIOMECHANICS",
                (x0 + margin, y0 + margin + int(16 * scale)),
                _FONT, scale * 0.52, (180, 180, 180), 1, cv2.LINE_AA)

    _LABELS = {
        "spine_angle":                 "Spine Angle",
        "spine_delta":                 "Spine Delta",
        "shoulder_plane_angle":        "Shoulder Tilt",
        "hip_plane_angle":             "Hip Tilt",
        "hip_lateral_displacement":    "Hip Lateral",
        "shoulder_hip_rotation_delta": "Sh/Hip Diff",
    }

    for i, (metric, val) in enumerate(raw_values.items()):
        y    = y0 + title_h + margin + i * row_h
        text_y = y + row_h - max(4, int(5 * scale))

        cv2.putText(img, _LABELS.get(metric, metric),
                    (x0 + margin, text_y),
                    _FONT, scale * 0.42, _C_TEXT, 1, cv2.LINE_AA)

        if metric == "hip_lateral_displacement":
            direction = "sway" if val >= 0 else "slide"
            val_str = f"{abs(val):.3f} {direction}"
        elif metric in ("spine_angle", "spine_delta",
                        "shoulder_plane_angle", "hip_plane_angle",
                        "shoulder_hip_rotation_delta"):
            val_str = f"{val:.1f} deg"
        else:
            val_str = f"{val:.3f}"

        cv2.putText(img, val_str,
                    (x0 + margin + label_w, text_y),
                    _FONT, scale * 0.42, (100, 220, 255), 1, cv2.LINE_AA)


def _text_with_bg(
    img:    np.ndarray,
    text:   str,
    org:    tuple[int, int],
    scale:  float,
    color:  tuple[int, int, int],
    thickness: int = 1,
    pad:    int = 4,
) -> None:
    """Draw text with a small dark rectangle behind it for readability."""
    (tw, th), bl = cv2.getTextSize(text, _FONT, scale, thickness)
    x, y = org
    cv2.rectangle(img,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + bl + pad),
                  (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), _FONT, scale, color, thickness, cv2.LINE_AA)


def _mid(kpts: np.ndarray, i: int, j: int) -> np.ndarray:
    return (kpts[i] + kpts[j]) / 2.0


def _pt(arr: np.ndarray) -> tuple[int, int]:
    return (int(arr[0]), int(arr[1]))


def _valid(pt: np.ndarray) -> bool:
    return not (pt[0] == 0 and pt[1] == 0)
