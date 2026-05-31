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

# ---------------------------------------------------------------------------
# Metric fault-score thresholds: (good_threshold, bad_threshold)
# fault score = 0 at good_threshold, 100 at bad_threshold, clamped outside.
# Raw values are in degrees unless noted.
# ---------------------------------------------------------------------------
METRIC_THRESHOLDS: dict[str, tuple[float, float]] = {
    "spine_angle":                 (5.0,  25.0),   # degrees from vertical
    "spine_delta":                 (3.0,  15.0),   # degrees vs address
    "shoulder_plane_angle":        (5.0,  20.0),   # degrees from horizontal
    "hip_plane_angle":             (3.0,  15.0),   # degrees from horizontal
    "hip_lateral_displacement":    (0.05, 0.25),   # fraction of shoulder width
    "shoulder_hip_rotation_delta": (10.0, 40.0),   # degrees
}
