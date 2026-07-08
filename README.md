# Golf Swing Analyzer

An AI-powered golf swing analysis tool that automatically detects the key phases of a golf swing from a standard video, overlays biomechanical metrics on each phase frame, and produces an annotated summary image.

Built as part of the Advanced Data Science & AI course at MCI Innsbruck.

---

## What it does

The pipeline takes a single face-on golf swing video and outputs four annotated phase images:

| Phase | What is detected |
|---|---|
| **Address** | Static setup position |
| **Top of Backswing** | Peak of the backswing rotation |
| **Impact** | Moment of club-ball contact |
| **Follow-Through** | Completion of the swing |

Each image is annotated with:
- Live spine line (yellow) and address ghost spine (dashed blue) for comparison
- Shoulder plane (cyan) and hip plane (orange)
- Spine angle arc with degree label
- **BIOMECHANICS HUD** — six metrics with numeric values and phase-specific zone bars (green = good, yellow = acceptable, red = fault)

The six metrics shown are:

| Metric | Description |
|---|---|
| Spine Angle | Lateral tilt of the spine from vertical |
| Spine Delta | Change in spine angle since address |
| Shoulder Tilt | Tilt of the shoulder line from horizontal |
| Hip Tilt | Tilt of the hip line from horizontal |
| Hip Lateral | Lateral hip displacement (sway / slide) |
| Sh/Hip Diff | Shoulder-to-hip rotation separation (X-factor proxy) |

Zone bar thresholds are phase-specific and informed by the biomechanics literature (Bourgain et al., *Sports* 2022).

A separate ball detection script runs the custom-trained YOLO model on a video and outputs an annotated video showing raw detections (red), geometry-filtered candidates (yellow), and temporally confirmed balls (green).

A club segmentation script visualizes the club segmentation model output — drawing semi-transparent masks and bounding boxes for the shaft (cyan) and clubhead (orange) on every frame.

---

## Requirements

- Python =3.12
- NVIDIA GPU with CUDA (tested on RTX 5060)
- [uv](https://docs.astral.sh/uv/) package manager

---

## Installation

```bash
git clone https://github.com/Misterz1x/golf_swing_analyzer.git
cd golf_swing_analyzer
uv sync
```

---

## Models

The pose model must be placed in `models/`. The two custom-trained models live in their training output folders (paths are already set in `golf_analyzer/config.py`).

| File | Purpose | Size | Release |
|---|---|---|---|
| `models/yolo26x-pose.pt` | YOLO pose model — detects 17-point body skeleton | ~120 MB | [v1.0](https://github.com/Misterz1x/golf_swing_analyzer/releases/tag/v1.0) |
| `runs2/train/golf_ball_yolo26x_v2/weights/best.pt` | Custom-trained golf ball detector (mAP@50 = 0.915) | ~113 MB | [v1.0](https://github.com/Misterz1x/golf_swing_analyzer/releases/tag/v1.0) |
| `runs_train/golf_club_seg_v1/weights/best.pt` | Custom-trained golf club segmentation (mAP@50 mask = 0.866) | ~404 MB | [v1.1](https://github.com/Misterz1x/golf_swing_analyzer/releases#release-v1.1) |

```
golf_swing_analyzer/
├── models/
│   └── yolo26x-pose.pt
├── runs2/train/golf_ball_yolo26x_v2/weights/
│   └── best.pt
└── runs_train/golf_club_seg_v1/weights/
    └── best.pt
```

> Model paths are configured in `golf_analyzer/config.py` as `BALL_MODEL_PATH` and `CLUB_SEG_MODEL_PATH`.

---

## Usage

### Swing analysis

Analyzes a video and saves four annotated phase images plus a 2×2 summary sheet.

```bash
uv run python golf_analyzer/analyze_swing.py --video "sample_videos/IMG_5887.MOV"
```

Output is saved to `results/` by default. You can change the output directory:

```bash
uv run python golf_analyzer/analyze_swing.py \
    --video "sample_videos/IMG_5887.MOV" \
    --output "results/my_swing"
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--video` | *(required)* | Path to golf swing video |
| `--output` | `./results` | Output directory |
| `--stride` | `3` | Sample every N-th frame (lower = slower, more accurate) |
| `--conf` | `0.3` | Pose detection confidence threshold |

### Ball detection

Runs the ball detection model on a video and saves an annotated output video showing each detection with its confidence score and filter status.

```bash
uv run python detect_golf_ball.py --video "sample_videos/IMG_5887.MOV" --tag test
```

Output is saved to `output_ballv2/IMG_5887_ball_test.mp4`.

Options:

| Flag | Default | Description |
|---|---|---|
| `--video` | *(all in data/our_videos/)* | Path to a specific video |
| `--tag` | *(none)* | Label appended to the output filename to avoid overwrites |
| `--conf` | `0.30` | Detection confidence threshold |
| `--y-min` | `0.25` | Top-of-frame exclusion zone (fraction of frame height) |
| `--track-radius` | `80` | Pixel radius for temporal consistency filter |

**Box colours in the output video:**
- **Red** — rejected by geometry filter (too large, wrong shape, or too high in frame)
- **Yellow** — passed geometry but not yet temporally confirmed (single-frame candidate)
- **Green** — confirmed ball (appeared in same location in ≥ 2 of the last 5 frames)

### Club segmentation visualization

Runs the club segmentation model on a video and saves an annotated output video with semi-transparent masks and bounding boxes for each detected class.

```bash
uv run python visualize_club_seg.py --video "sample_videos/IMG_5887.MOV"
```

Output is saved to `result_videos_images/videos_club_seg/IMG_5887_club_seg.mp4`.

Options:

| Flag | Default | Description |
|---|---|---|
| `--video` | *(all in sample_videos/)* | Path to a specific video |
| `--conf` | `0.25` | Detection confidence threshold |

**Overlay colours in the output video:**
- **Cyan** — shaft (class 0): semi-transparent filled polygon + bounding box
- **Orange** — clubhead (class 1): semi-transparent filled polygon + bounding box + centroid dot

---

## Sample videos

Four sample videos are included in `sample_videos/` for testing:

| File | Description |
|---|---|
| `IMG_5840.MOV` | Golfer 1 — driver, face-on view |
| `IMG_5884.MOV` | Golfer 2 — iron, face-on view |
| `IMG_5887.MOV` | Golfer 3 — driver, face-on view |
| `IMG_5897.MOV` | Golfer 4 — iron, face-on view |

Videos are slow-motion or real-time depending on the device used.

---

## Project structure

```
golf_swing_analyzer/
├── golf_analyzer/
│   ├── analyze_swing.py        # Main pipeline and CLI
│   ├── phase_detector.py       # Swing phase detection (address, top, impact, follow-through)
│   ├── metrics.py              # Biomechanical metric computation
│   ├── annotator.py            # Frame annotation and HUD rendering
│   └── config.py               # Model paths, keypoint indices, phase-specific thresholds
├── detect_golf_ball.py         # Standalone ball detection visualization script
├── visualize_club_seg.py       # Standalone club segmentation visualization script
├── train_golf_ball.py          # Ball detection model training script
├── gb_detection_improvement.py # Ball detection model fine-tuning script
├── train_golf_club_seg.py      # Club segmentation model training script
├── sample_videos/              # Four sample golf swing videos
├── result_videos_images/       # Output folder for annotated videos and images
├── docs/
│   ├── training_ball_detection/ # Ball detection training metrics and plots
│   └── training_club_seg/       # Club segmentation training metrics and plots
├── golf_biomechanics.pdf       # Reference: Bourgain et al. 2022 systematic review
├── PROJECT_NOTES.md            # Development notes and known issues
├── pyproject.toml
└── uv.lock
```

---

## Training

### Ball detection model

The ball detection model was trained in two runs on a combined dataset of ~2,600 images (GolfBallDetector dataset from Roboflow + custom frames), totalling approximately **7 hours** of GPU training.

Run 1 trained for 50 epochs at 640 px resolution on the base dataset. Run 2 fine-tuned on an augmented set for a further 80 epochs, giving the final weights.

**Best epoch metrics:**

| Metric | Value |
|---|---|
| mAP@50 | **0.915** |
| mAP@50-95 | 0.693 |
| Precision | 0.953 |
| Recall | 0.842 |

Training plots are in [`docs/training_ball_detection/`](docs/training_ball_detection/).

![Ball detection training results](docs/training_ball_detection/results.png)

![Ball detection validation predictions](docs/training_ball_detection/val_batch0_pred.jpg)

---

### Club segmentation model

The club segmentation model was trained from scratch on a custom dataset of **11,500 images** with polygon masks for two classes — shaft and clubhead. Training ran for 190 epochs at 1920 px resolution and took approximately **48 hours** of GPU time.

**Best epoch metrics (epoch 178):**

| Metric | Box | Mask |
|---|---|---|
| mAP@50 | **0.932** | **0.866** |
| mAP@50-95 | 0.675 | 0.466 |
| Precision | 0.921 | 0.883 |
| Recall | 0.882 | 0.828 |

Training plots are in [`docs/training_club_seg/`](docs/training_club_seg/).

![Club segmentation training results](docs/training_club_seg/results.png)

![Club segmentation training batch](docs/training_club_seg/train_batch0.jpg)

---

## Phase detection approach

Phase detection runs in three steps:

1. **Pose estimation** — YOLO pose model extracts 17-point skeleton on every N-th frame (default stride = 3).
2. **Wrist-Y signal** — the vertical wrist trajectory is smoothed (Savitzky-Golay) and peak-detected to locate address and top of backswing.
3. **Impact refinement** — a priority chain refines impact: wrist-over-ball X crossing (using the ball rest position detected near top of backswing) → wrist-X proxy → velocity-inflection blend fallback.

---

## Reference

Bourgain, M., Rouch, P., Rouillon, O., Thoreux, P., & Sauret, C. (2022). *Golf Swing Biomechanics: A Systematic Review and Methodological Recommendations for Kinematics.* Sports, 10(6), 91. https://doi.org/10.3390/sports10060091

## Contributors

Guido Bäumer;
Luis Marrufo; 
Vilian Knap; 
Elias Zischg
