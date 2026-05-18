import csv
import cv2
import ultralytics
from pathlib import Path

MODEL_PATH = "yolo26x-pose.pt"
BALL_MODEL_PATH = "golf_ball_yolo11n.pt"
DATA_DIR = "data/our_videos"
OUTPUT_DIR = "output"

# Lower this to catch more ball detections (range 0.0–1.0, default YOLO is 0.25)
BALL_CONF_THRESHOLD = 0.10

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

video_paths = [
    p for p in Path(DATA_DIR).glob("**/*")
    if p.suffix.lower() in VIDEO_EXTENSIONS
]

if not video_paths:
    print(f"No video files found in '{DATA_DIR}/'")
    exit(1)

model = ultralytics.YOLO(MODEL_PATH)
ball_model = ultralytics.YOLO(BALL_MODEL_PATH)
Path(OUTPUT_DIR).mkdir(exist_ok=True)

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

for video_path in video_paths:
    cap = cv2.VideoCapture(str(video_path))

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if native_fps and native_fps > 0:
        fps = native_fps
        print(f"Processing {video_path.name} ({width}x{height} @ {fps:.3f} fps — native)")
    else:
        fps = 30.0
        print(f"Processing {video_path.name} ({width}x{height} @ {fps:.1f} fps — fallback, native FPS unreadable)")

    output_path = Path(OUTPUT_DIR) / video_path.name
    csv_path = Path(OUTPUT_DIR) / (video_path.stem + "_keypoint_conf.csv")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Run ball detection at max 30 fps; reuse last result on skipped frames
    ball_frame_interval = max(1, round(fps / 30))
    last_ball_boxes = []

    frame_count = 0
    with open(csv_path, "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(["frame", "person_id"] + KEYPOINT_NAMES)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, imgsz=160, verbose=False)
            annotated_frame = results[0].plot()

            if frame_count % ball_frame_interval == 0:
                ball_results = ball_model(frame, imgsz=256, conf=BALL_CONF_THRESHOLD, verbose=False)
                last_ball_boxes = ball_results[0].boxes

            for box in last_ball_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(annotated_frame, f"ball {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

            writer.write(annotated_frame)

            kp = results[0].keypoints
            if kp is not None and kp.conf is not None:
                for person_id, confs in enumerate(kp.conf):
                    row = [frame_count, person_id] + [round(float(c), 4) for c in confs]
                    csv_writer.writerow(row)
            else:
                csv_writer.writerow([frame_count, -1] + [""] * len(KEYPOINT_NAMES))

            frame_count += 1

    cap.release()
    writer.release()
    print(f"  Done — {frame_count} frames | video -> {output_path} | csv -> {csv_path}")

print("\nAll videos processed.")
