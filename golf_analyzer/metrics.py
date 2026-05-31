"""
Task 2 — Biomechanical metric computation.

compute_metrics(kpts, address_kpts) -> dict[str, float]
Returns fault scores 0–100 (higher = more fault) for each metric.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    L_SHOULDER, R_SHOULDER, L_HIP, R_HIP,
    METRIC_THRESHOLDS,
)


def compute_metrics(
    kpts: np.ndarray,
    address_kpts: np.ndarray,
) -> dict[str, float]:
    """
    Compute biomechanical fault scores from a (17, 2) keypoint array.

    Parameters
    ----------
    kpts          current phase keypoints, shape (17, 2)
    address_kpts  address/setup keypoints as reference, shape (17, 2)

    Returns
    -------
    dict mapping metric name → fault score 0–100 (higher = more fault).
    """
    sh_mid       = _mid(kpts, L_SHOULDER, R_SHOULDER)
    hip_mid      = _mid(kpts, L_HIP, R_HIP)
    sh_mid_addr  = _mid(address_kpts, L_SHOULDER, R_SHOULDER)
    hip_mid_addr = _mid(address_kpts, L_HIP, R_HIP)

    spine_angle      = _angle_from_vertical(sh_mid, hip_mid)
    spine_angle_addr = _angle_from_vertical(sh_mid_addr, hip_mid_addr)

    sh_width_addr = float(np.linalg.norm(
        address_kpts[L_SHOULDER] - address_kpts[R_SHOULDER]
    ))
    sh_width_addr = max(sh_width_addr, 1.0)

    hip_lateral_signed = (hip_mid[0] - hip_mid_addr[0]) / sh_width_addr

    raw: dict[str, float] = {
        "spine_angle":                 spine_angle,
        "spine_delta":                 abs(spine_angle - spine_angle_addr),
        "shoulder_plane_angle":        _plane_angle(kpts[L_SHOULDER], kpts[R_SHOULDER]),
        "hip_plane_angle":             _plane_angle(kpts[L_HIP],      kpts[R_HIP]),
        "hip_lateral_displacement":    abs(hip_lateral_signed),
        "shoulder_hip_rotation_delta": abs(
            _plane_angle(kpts[L_SHOULDER], kpts[R_SHOULDER])
            - _plane_angle(kpts[L_HIP],   kpts[R_HIP])
        ),
    }

    return {
        name: _fault_score(val, *METRIC_THRESHOLDS[name])
        for name, val in raw.items()
    }


def get_raw_values(
    kpts: np.ndarray,
    address_kpts: np.ndarray,
) -> dict[str, float]:
    """
    Same computation as compute_metrics but returns the raw physical values
    (degrees / normalised displacement) instead of fault scores.
    Useful for display labels in the annotator.
    """
    sh_mid       = _mid(kpts, L_SHOULDER, R_SHOULDER)
    hip_mid      = _mid(kpts, L_HIP, R_HIP)
    sh_mid_addr  = _mid(address_kpts, L_SHOULDER, R_SHOULDER)
    hip_mid_addr = _mid(address_kpts, L_HIP, R_HIP)

    spine_angle      = _angle_from_vertical(sh_mid, hip_mid)
    spine_angle_addr = _angle_from_vertical(sh_mid_addr, hip_mid_addr)

    sh_width_addr = max(float(np.linalg.norm(
        address_kpts[L_SHOULDER] - address_kpts[R_SHOULDER]
    )), 1.0)

    hip_lateral_signed = (hip_mid[0] - hip_mid_addr[0]) / sh_width_addr

    return {
        "spine_angle":                 round(spine_angle, 1),
        "spine_delta":                 round(abs(spine_angle - spine_angle_addr), 1),
        "shoulder_plane_angle":        round(_plane_angle(kpts[L_SHOULDER], kpts[R_SHOULDER]), 1),
        "hip_plane_angle":             round(_plane_angle(kpts[L_HIP], kpts[R_HIP]), 1),
        "hip_lateral_displacement":    round(hip_lateral_signed, 3),   # signed: + sway, − slide
        "shoulder_hip_rotation_delta": round(abs(
            _plane_angle(kpts[L_SHOULDER], kpts[R_SHOULDER])
            - _plane_angle(kpts[L_HIP],   kpts[R_HIP])
        ), 1),
    }


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _mid(kpts: np.ndarray, i: int, j: int) -> np.ndarray:
    return (kpts[i] + kpts[j]) / 2.0


def _angle_from_vertical(top: np.ndarray, bottom: np.ndarray) -> float:
    """Angle in degrees of the top→bottom vector from vertical (straight down)."""
    dx = float(bottom[0] - top[0])
    dy = float(bottom[1] - top[1])
    return float(np.degrees(np.arctan2(abs(dx), max(dy, 1e-6))))


def _plane_angle(left: np.ndarray, right: np.ndarray) -> float:
    """Angle in degrees of the left→right segment from horizontal."""
    dx = float(right[0] - left[0])
    dy = float(right[1] - left[1])
    return float(np.degrees(np.arctan2(abs(dy), max(abs(dx), 1e-6))))


def _fault_score(raw: float, good: float, bad: float) -> float:
    if bad == good:
        return 0.0
    return float(np.clip((raw - good) / (bad - good) * 100.0, 0.0, 100.0))
