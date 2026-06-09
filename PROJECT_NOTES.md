# Golf Swing Analyzer — Project Notes

## What This Project Does

Analyses a golf swing video and produces four annotated still frames:
**Address → Top of Backswing → Impact → Follow-Through**

Each frame is annotated with the live spine line, ghost address spine, shoulder plane, hip plane, wrist dots, a spine-angle arc, and a HUD panel showing six raw biomechanical values. A 2×2 summary image (`summary.jpg`) is also saved.

---

## Environment

- **Python**: 3.12+, managed with `uv`
- **GPU**: NVIDIA RTX 5060 (Blackwell), CUDA 12.8
- **PyTorch**: installed from `https://download.pytorch.org/whl/cu128` (see `pyproject.toml`)
- **Key packages**: `ultralytics`, `torch`, `torchvision`, `scipy`, `opencv-python`

Run the pipeline:
```
uv run python golf_analyzer/analyze_swing.py --video path/to/swing.mp4 --output ./results
```

Optional flags: `--stride 3` (sample every Nth frame), `--conf 0.3` (pose confidence threshold).

---

## Project Structure

```
golf_swing_analyzer/
├── golf_analyzer/
│   ├── analyze_swing.py      # Main pipeline + CLI entry point
│   ├── phase_detector.py     # Phase detection (most complex, most edited)
│   ├── annotator.py          # Frame annotation / drawing
│   ├── metrics.py            # Biomechanical value computation
│   └── config.py             # Keypoint indices + model paths + thresholds
├── models/
│   ├── yolo26x-pose.pt       # Primary pose model (YOLO v2 6x, 17-pt COCO)
│   └── yolo26x.pt            # General detection model (unused in pipeline)
├── runs2/train/
│   └── golf_ball_yolo26x_v2/weights/best.pt   # Custom ball detection model (mAP@50=0.915)
├── data/                     # Training data for ball model
├── pyproject.toml
└── PROJECT_NOTES.md          # This file
```

Legacy scripts at root level (`yolo.py`, `train_golf_ball.py`, `gb_detection_improvement.py`, `detect_golf_ball.py`) were training/testing scripts and are not part of the current pipeline.

---

## Models

### Pose Model: `models/yolo26x-pose.pt`
- YOLO v2 6x variant with pose estimation head
- 17-point COCO skeleton
- Inference size: `imgsz=1280` for 1920×1080 footage (set in `analyze_swing.py`)
- Confidence: `conf=0.3` default

### Ball Detection Model: `runs2/train/golf_ball_yolo26x_v2/weights/best.pt`
- Custom-trained on two datasets
- mAP@50 = 0.915
- Used optionally in `phase_detector.py` to refine impact detection
- Auto-loaded in `analyze_swing.py` if the file exists
- Ball detection filters: box area < 4% of frame, aspect ratio 0.4–2.5

### COCO Keypoint Indices (from `config.py`)
```
L_SHOULDER = 5    R_SHOULDER = 6
L_WRIST    = 9    R_WRIST    = 10
L_HIP      = 11   R_HIP      = 12
L_ANKLE    = 15   R_ANKLE    = 16
```

---

## Phase Detection Algorithm (`phase_detector.py`)

### Input
Every `stride`-th frame is sampled. For each sampled frame:
- Pose inference → average wrist Y pixel coordinate (NaN if no detection)
- Full 17-point keypoint array stored in `kpts_seq`
- Optional ball detection → `(cx, cy)` or `None`

NaN values are linearly interpolated. The wrist Y signal is smoothed with a Savitzky-Golay filter (auto window = `n // 6`, order 3).

### Detection Order

#### 1. Top of Backswing (`top_bs`)
- Search first 70% of the signal for wrist-Y valleys using `scipy.find_peaks`
- **Key constraint**: `min_top_bs = max(3, int(n * 0.10))` — top_bs must be at least 10% into the signal to leave room for address before it
- Valley selection: most prominent valley that satisfies the floor; if none pass the floor, use the most prominent overall; if no valleys at all, use `argmin` from the floor onwards

#### 2. Address (`address`)
- Uses **multi-keypoint body velocity** over 8 joints: both shoulders, hips, wrists, ankles
- Only searches the **first 50% of the pre-top_bs window** — this prevents the deceleration zone at the top of backswing from being mistaken for address
- Returns `argmin` of body velocity = most-still frame
- **Status: working well.** Confirmed good on 3+ test videos.

#### 3. Impact (`impact`)
- **Hard cap**: `impact_cap = top_bs + 1 + max(3, int((n - top_bs - 1) * 0.65))` — impact must be in the first 65% of the post-top_bs segment (prevents follow-through from being selected)
- Primary: `argmax` of smoothed wrist Y in `[top_bs+1, impact_cap]`
  - Wrist Y is highest (wrists furthest down in frame) at the moment of ball contact
  - After impact, wrists rise into follow-through, so this is a natural anchor
- Optional refinement: `_find_impact_from_ball()` if ball model is loaded
  - Estimates ball resting position from first 40% of pre-impact frames
  - Walks forward looking for first frame where ball leaves resting position
  - Falls back to wrist-based impact if fewer than 3 early ball detections
- **Status: sometimes a bit early or late** (varies by golfer and camera angle). The `argmax` is stable but biomechanically approximate.

#### 4. Follow-Through (`follow_through`)
- **Key insight**: videos are cut to one swing, so the finish is near the END of the video
- Search window: `max(impact + 3, int(n * 0.72))` to `n-1` (last 28% of video)
- Within that window: `argmin` of smoothed wrist Y = wrists at their highest = finish position (club over left shoulder, back to camera)
- **Status: recently fixed.** Previously used a fixed offset (too close to impact) or valley detection (found intermediate mid-rotation position that looked like backswing).

### Debug Output
Every run prints:
```
[phase_detector] n=120  addr=8  top_bs=22  impact=61  follow_through=103  (impact_cap=80)
```
All values are sampled-frame indices, not actual video frame numbers.

### Key Tuning Constants
```python
_PEAK_PROMINENCE       = 15   # min prominence for top_bs valley detection
_FOLLOW_THROUGH_FRAMES = 30   # fallback frames after impact (unused if ft_zone found)
_BODY_KP_INDICES = [5, 6, 11, 12, 9, 10, 15, 16]  # joints used for address velocity
```

---

## Biomechanical Metrics (`metrics.py`)

`get_raw_values(kpts, address_kpts)` returns a dict of physical values (used for annotation):

| Metric | Unit | Description |
|---|---|---|
| `spine_angle` | degrees | Angle of shoulder-mid → hip-mid line from vertical |
| `spine_delta` | degrees | Difference from address spine angle |
| `shoulder_plane_angle` | degrees | L_shoulder → R_shoulder tilt from horizontal |
| `hip_plane_angle` | degrees | L_hip → R_hip tilt from horizontal |
| `hip_lateral_displacement` | normalised | Lateral hip shift relative to address; + = sway, − = slide |
| `shoulder_hip_rotation_delta` | degrees | Shoulder tilt minus hip tilt (X-factor proxy) |

`compute_metrics()` converts these to 0–100 fault scores using thresholds from `config.py`. This function exists but is **not used** in the current pipeline — only `get_raw_values` is called.

---

## Annotation (`annotator.py`)

`annotate_phase_image(frame, kpts, address_kpts, metrics, phase_name, raw_values=None)`

Draws on a copy of the frame (original unchanged):
1. **Ghost spine** (dashed, semi-transparent blue) — address reference line
2. **Live spine** (solid yellow) — shoulder-mid to hip-mid
3. **Shoulder plane** (cyan, extended + endpoint dots)
4. **Hip plane** (orange, same style)
5. **Spine arc + label** at shoulder midpoint showing deviation from vertical
6. **Wrist dots** (bright green circles)
7. **Phase label** bottom-left (white text with dark background)
8. **BIOMECHANICS HUD** top-right — semi-transparent dark panel with all 6 raw values

**Important**: OpenCV's `FONT_HERSHEY_SIMPLEX` does not support Unicode. All labels use ASCII only — `deg` not `°`, `sway`/`slide` not `×`.

---

## Main Pipeline (`analyze_swing.py`)

```
load pose model
load ball model (optional, auto-detected from BALL_MODEL_PATH)
  ↓
detect_swing_phases() → {phase: actual_frame_number}
  ↓
for each phase:
  seek to frame number
  run pose inference
  get_raw_values(kpts, address_kpts)
  annotate_phase_image(...)
  save phase.jpg
  ↓
save summary.jpg (2×2 grid, 540px tile height)
print console table of all raw values
```

Inference for annotation uses `imgsz=1280`. Phase detection uses whatever `imgsz` is passed (default 640 in `phase_detector.py`, but 1280 is passed from `analyze_swing.py`).

---

## Known Issues and History

### What Worked Well
- **Address detection**: multi-keypoint body velocity (8 joints, first 50% of pre-top_bs window) works reliably across multiple golfers.
- **Top of backswing**: `find_peaks` + `min_top_bs` floor works well.
- **Follow-through**: now uses last 28% of video + `argmin` — recently fixed.

### Impact Detection (Main Open Problem)
Impact is the hardest phase to detect reliably. The current `argmax` of wrist Y within the capped window is stable but can be a few frames early or late depending on:
- Camera angle (side-on vs face-on vs down-the-line)
- Individual golfer's swing tempo
- Whether the backswing raises the wrists significantly above impact height

**Things tried and abandoned (don't repeat these without good reason):**
- Spine angle zero-crossing with `_SPINE_IMPACT_MARGIN`: worked for some videos but caused regressions in others. The spine starts returning to vertical before ball contact (hips lead), making the timing per-golfer.
- Wrist X horizontal crossing back to address position: also per-golfer, failed on flat swings.
- Combined spine + wrist X + wrist Y fallback chain: too many failure modes, got worse overall.
- Ball detection as primary impact signal: inconsistent. Ball model sometimes has false positives/negatives. Currently kept as optional refinement, but may cause instability if auto-loaded.

**Possible future improvements for impact:**
- Use ball detection model more carefully (it has mAP@50=0.915 — look for the last frame where ball is stationary before it disappears)
- Wrist X crossing: at impact, wrists cross back through the address horizontal position. Requires computing average wrist X at address and finding the first frame after top_bs where wrist X crosses through it.
- Spine angle check as a FILTER (not primary): if the argmax candidate has spine angle far from address spine angle, adjust slightly.

### Follow-Through Detection History
- Original: fixed frame offset from impact → always too close, showed mid-downswing
- First fix: find next wrist-Y valley after impact → found intermediate mid-rotation position that looked identical to top of backswing
- Current fix: `argmin` of wrist Y in last 28% of video → works correctly

### Stacking Bug (Fixed)
On videos with flat/short backswings or a different camera angle, `find_peaks` found no valleys and the `argmin` fallback returned index 0. All four phases stacked on the first frame. Fixed by: `min_top_bs = max(3, int(n * 0.10))` applied to both the valley filter and the `argmin` fallback.

---

## Video Requirements
- Videos should be **cut to a single swing** in slow motion — no long pre/post-swing idle time
- Slow motion footage works better because more frames are available per phase
- Camera should have a clear view of the golfer (any angle works in principle, but side-on or behind are most common)
- The pipeline processes every 3rd frame by default (`--stride 3`). For very fast cameras (240fps+) this is fine. For normal 30fps footage, consider `--stride 1`.

---

## Quick Diagnostic

If phases look wrong, check the debug line first:
```
[phase_detector] n=120  addr=8  top_bs=22  impact=61  follow_through=103  (impact_cap=80)
```
- `addr` and `top_bs` should not be the same (or very close)
- `impact` should be between `top_bs` and `impact_cap`
- `follow_through` should be >= 72% of `n` (i.e., >= `int(n * 0.72)`)
- If `top_bs` is very small (< `int(n * 0.10)`), the `min_top_bs` floor failed — check that the pose model is detecting the golfer properly
