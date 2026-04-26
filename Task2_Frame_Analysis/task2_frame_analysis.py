"""
Task 2: Data Preprocessing and Analysis
Author: Anil Kumar
Video URL: https://www.pexels.com/video/busy-city-highway-traffic-at-rush-hour-31115112/
"""

from itertools import count
import cv2
import os
import csv
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
import os
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "..", "traffic_video.mp4")
FRAME_INTERVAL = 30                          # extract every 30 frames
MAX_FRAMES     = None                        # same as Task 1
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output_frames")
REPORT_CSV = os.path.join(BASE_DIR, "detection_report.csv")
CONFIDENCE     = 0.4                         # minimum detection confidence

# ─────────────────────────────────────────────
# STEP 1: Extract & Preprocess Frames
# ─────────────────────────────────────────────
def extract_and_preprocess(video_path, interval, max_frames):
    """
    Extracts frames and applies 3 preprocessing steps:

    1. Resize to 640x640 — standard YOLO input size
    2. Gaussian Blur (3x3) — mild denoising without losing edges
    3. CLAHE on L channel — adaptive contrast enhancement
       useful for frames with uneven lighting (shadows, glare)
       Applied in LAB color space to avoid distorting colors
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return []

    frames = []
    frame_idx = 0

    # CLAHE — Contrast Limited Adaptive Histogram Equalization
    # clipLimit=2.0 prevents over-amplifying noise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    print(f"[INFO] Extracting and preprocessing frames...")

    while cap.isOpened() and (MAX_FRAMES is None or count < MAX_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            # Step 1: Resize to 640x640
            resized = cv2.resize(frame, (640, 640))

            # Step 2: Gaussian blur — removes pixel noise
            blurred = cv2.GaussianBlur(resized, (3, 3), 0)

            # Step 3: CLAHE for contrast enhancement
            # Convert BGR → LAB, apply CLAHE to L (luminance) only
            lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = clahe.apply(l)
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            preprocessed = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            frames.append((frame_idx, preprocessed))
            print(f"  → Frame {frame_idx} preprocessed")

        frame_idx += 1

    cap.release()
    print(f"[INFO] Total frames ready: {len(frames)}")
    return frames

# ─────────────────────────────────────────────
# STEP 2: Load YOLOv8 Model
# ─────────────────────────────────────────────
def load_yolo():
    """
    Loads YOLOv8 nano — lightest and fastest variant.
    
    Why YOLOv8n?
    - Only 6MB — fast download on any machine
    - COCO-pretrained: car, person, truck, bus, bicycle,
      motorcycle, traffic light, stop sign (80 classes)
    - Cross-platform: runs on Mac MPS, Windows CUDA, or CPU
    - Ultralytics automatically selects the best device

    Device selection is handled automatically by Ultralytics:
    - Mac M-series → MPS
    - NVIDIA GPU → CUDA
    - No GPU → CPU
    """
    print("[INFO] Loading YOLOv8 nano model...")
    print("[INFO] Device will be selected automatically (MPS/CUDA/CPU)")
    model = YOLO("yolov8n.pt")  # auto-downloads on first run (~6MB)
    print("[INFO] YOLOv8 loaded ")
    return model

# ─────────────────────────────────────────────
# STEP 3: Run Detection on One Frame
# ─────────────────────────────────────────────
def detect(model, frame, confidence):
    """
    Runs YOLOv8 on a single frame.
    Returns list of detections: {class_name, confidence, bbox}

    confidence threshold = 0.4
    Filters out weak/uncertain detections
    """
    results = model(frame, conf=confidence, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            class_id   = int(box.cls[0])
            class_name = model.names[class_id]
            conf_score = float(box.conf[0])
            bbox       = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

            detections.append({
                "class_name": class_name,
                "confidence": round(conf_score, 3),
                "bbox": bbox
            })

    return detections

# ─────────────────────────────────────────────
# STEP 4: Draw Boxes & Save Frame
# ─────────────────────────────────────────────
def annotate_and_save(frame, detections, frame_idx, output_dir):
    """
    Draws green bounding boxes and labels on the frame.
    Saves the annotated image as a JPEG file.
    """
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = f"{det['class_name']} {det['confidence']:.2f}"

        # Green bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)

        # Label background for readability
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            annotated, (x1, y1 - th - 6), (x1 + tw, y1), (0, 200, 0), -1
        )
        cv2.putText(
            annotated, label, (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
        )

    save_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
    cv2.imwrite(save_path, annotated)
    return save_path

# ─────────────────────────────────────────────
# STEP 5: Save CSV Report
# ─────────────────────────────────────────────
def save_csv(all_detections, path):
    """
    Saves all detections to CSV for analysis.
    One row per detected object per frame.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_index", "class_name", "confidence",
            "x1", "y1", "x2", "y2"
        ])
        for frame_idx, detections in all_detections:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                writer.writerow([
                    frame_idx, det["class_name"], det["confidence"],
                    round(x1), round(y1), round(x2), round(y2)
                ])
    print(f"[INFO] CSV report saved: {path}")

# ─────────────────────────────────────────────
# STEP 6: Print Summary
# ─────────────────────────────────────────────
def print_summary(all_detections):
    """
    Counts total detections per class across all frames.
    Helps identify what dominates the scene.
    """
    class_counts = defaultdict(int)
    for _, detections in all_detections:
        for det in detections:
            class_counts[det["class_name"]] += 1

    print(f"\n{'='*45}")
    print("DETECTION SUMMARY (all frames combined):")
    print(f"{'='*45}")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:<20}: {count} detections")
    print(f"{'='*45}\n")

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    import platform
    print(f"[INFO] Running on: {platform.system()} {platform.machine()}")

    # Create output folder for annotated frames
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Extract and preprocess frames
    frames = extract_and_preprocess(VIDEO_PATH, FRAME_INTERVAL, MAX_FRAMES)

    if not frames:
        print("[ERROR] No frames extracted. Check video path.")
        return

    # Step 2: Load YOLO
    model = load_yolo()

    # Step 3, 4: Detect + annotate each frame
    print("\n[INFO] Running object detection...")
    all_detections = []

    for frame_idx, frame in frames:
        detections = detect(model, frame, CONFIDENCE)
        save_path  = annotate_and_save(frame, detections, frame_idx, OUTPUT_DIR)
        all_detections.append((frame_idx, detections))

        classes_found = ", ".join(set(d["class_name"] for d in detections)) or "none"
        print(f"  Frame {frame_idx:>5}: {len(detections):>2} detections "
              f"[{classes_found}] → saved")

    # Step 5: Save CSV
    save_csv(all_detections, REPORT_CSV)

    # Step 6: Print summary
    print_summary(all_detections)

if __name__ == "__main__":
    main()