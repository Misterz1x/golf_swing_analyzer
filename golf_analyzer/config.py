# ---------------------------------------------------------------------------
# Keypoint indices — COCO 17-point skeleton, YOLO26 order
# ---------------------------------------------------------------------------
L_SHOULDER = 5
R_SHOULDER = 6
L_HIP      = 11
R_HIP      = 12
L_WRIST    = 9
R_WRIST    = 10
L_ANKLE    = 15
R_ANKLE    = 16

# ---------------------------------------------------------------------------
# Model paths (relative to project root, i.e. one level above golf_analyzer/)
# ---------------------------------------------------------------------------
POSE_MODEL_PATH = "models/yolo26x-pose.pt"
BALL_MODEL_PATH = "runs2/train/golf_ball_yolo26x_v2/weights/best.pt"

# ---------------------------------------------------------------------------
# Phase-specific metric thresholds used by the HUD zone bar.
#
# Each entry: (good_val, bad_val, higher_is_better)
#
#   higher_is_better=False  →  low values preferred
#       fault = 0   when raw ≤ good_val
#       fault = 100 when raw ≥ bad_val
#
#   higher_is_better=True   →  high values preferred  (good_val > bad_val)
#       fault = 0   when raw ≥ good_val
#       fault = 100 when raw ≤ bad_val
#       The _fault_score formula handles this via double-negative cancellation.
#
# Raw values: degrees for all angle metrics;
#             fraction of shoulder-width for hip_lateral_displacement.
#
# Sources: Bourgain et al. 2022 systematic review (paper in project root)
#          + standard coaching cues for 2D face-on analysis.
# ---------------------------------------------------------------------------
PHASE_METRIC_THRESHOLDS: dict[str, dict[str, tuple[float, float, bool]]] = {

    # ------------------------------------------------------------------
    # ADDRESS — static setup, all planes level, spine laterally upright
    # ------------------------------------------------------------------
    "address": {
        "spine_angle":                 ( 5.0,  15.0, False),  # lateral tilt from vertical; small is correct
        "spine_delta":                 ( 2.0,   8.0, False),  # reference phase, delta ≈ 0 by definition
        "shoulder_plane_angle":        ( 5.0,  12.0, False),  # shoulders should be level
        "hip_plane_angle":             ( 3.0,   8.0, False),  # hips should be level
        "hip_lateral_displacement":    ( 0.05,  0.15, False), # weight centred; too much lean = fault
        "shoulder_hip_rotation_delta": ( 5.0,  12.0, False),  # both planes level → small delta
    },

    # ------------------------------------------------------------------
    # TOP OF BACKSWING — full rotation; shoulder tilt and X-factor should
    # be LARGE, hip sway should be small
    # ------------------------------------------------------------------
    "top_backswing": {
        "spine_angle":                 ( 5.0,  22.0, False),  # some lateral tilt natural; >22° = sway
        "spine_delta":                 ( 5.0,  28.0, False),  # paper: ~28° lateral-bend amplitude normal
        "shoulder_plane_angle":        (35.0,  20.0, True),   # ≥35° = good turn; <20° = incomplete — INVERTED
        "hip_plane_angle":             ( 5.0,  20.0, False),  # 5-12° normal; >20° = excessive hip tilt
        "hip_lateral_displacement":    ( 0.05,  0.22, False), # slight trail shift ok; >0.22 = sway fault
        "shoulder_hip_rotation_delta": (25.0,  10.0, True),   # X-factor proxy: ≥25° good; <10° = poor turn — INVERTED
    },

    # ------------------------------------------------------------------
    # IMPACT — club meets ball; spine close to address, hips open,
    # shoulders squaring to target line
    # ------------------------------------------------------------------
    "impact": {
        "spine_angle":                 ( 5.0,  18.0, False),  # some tilt normal at impact
        "spine_delta":                 ( 3.0,  20.0, False),  # ideally close to address spine angle
        "shoulder_plane_angle":        (10.0,  28.0, False),  # trail shoulder dropping; 10-22° typical
        "hip_plane_angle":             ( 8.0,  25.0, False),  # hips open, lead hip higher; 8-20° typical
        "hip_lateral_displacement":    ( 0.05,  0.28, False), # some lateral slide toward target is normal
        "shoulder_hip_rotation_delta": (12.0,   5.0, True),   # hips ahead of shoulders; <5° = no separation — INVERTED
    },

    # ------------------------------------------------------------------
    # FOLLOW-THROUGH — deceleration / finish; full rotation,
    # weight transferred to lead side
    # ------------------------------------------------------------------
    "follow_through": {
        "spine_angle":                 (10.0,  30.0, False),  # significant tilt into finish is expected
        "spine_delta":                 (10.0,  35.0, False),  # large deviation from address is normal
        "shoulder_plane_angle":        (25.0,  12.0, True),   # ≥25° = full rotation; <12° = blocked — INVERTED
        "hip_plane_angle":             (10.0,  28.0, False),  # hips fully rotated; some tilt expected
        "hip_lateral_displacement":    ( 0.10,  0.38, False), # weight on lead side; too much = over-slide
        "shoulder_hip_rotation_delta": ( 8.0,  25.0, False),  # hips and shoulders mostly aligned at finish
    },
}

# ---------------------------------------------------------------------------
# Legacy thresholds kept for compute_metrics() fault-score function.
# The main pipeline uses get_raw_values() + PHASE_METRIC_THRESHOLDS instead.
# ---------------------------------------------------------------------------
METRIC_THRESHOLDS: dict[str, tuple[float, float]] = {
    "spine_angle":                 (5.0,  25.0),
    "spine_delta":                 (3.0,  25.0),
    "shoulder_plane_angle":        (5.0,  20.0),
    "hip_plane_angle":             (3.0,  15.0),
    "hip_lateral_displacement":    (0.05, 0.25),
    "shoulder_hip_rotation_delta": (30.0, 10.0),
}
