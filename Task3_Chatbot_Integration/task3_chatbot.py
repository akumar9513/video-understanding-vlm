"""
Task 3: Video Analysis Chatbot — Pipeline Orchestrator
Author: Anil Kumar

This script is the master controller that:
- Runs Task1 (BLIP video captioning) on demand
- Runs Task2 (YOLOv8 object detection) on demand
- Performs color detection using HSV color space analysis
- Performs lane detection using bounding box coordinates
- Uses GPT-4o to answer natural language queries about the video

Usage:
    python3 task3_chatbot.py
"""

import cv2
import os
import sys
import json
import subprocess
import numpy as np
from ultralytics import YOLO
from openai import OpenAI
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, "..", ".env")
load_dotenv(dotenv_path=ENV_PATH)
API_KEY = os.getenv("OPENAI_API_KEY")

VIDEO_PATH   = os.path.join(BASE_DIR, "..", "traffic_video.mp4")
TASK1_SCRIPT = os.path.join(BASE_DIR, "..", "Task1_Video_Captioning", "task1_captioning.py")
TASK2_SCRIPT = os.path.join(BASE_DIR, "..", "Task2_Frame_Analysis", "task2_frame_analysis.py")
TASK1_OUTPUT = os.path.join(BASE_DIR, "..", "Task1_Video_Captioning", "task1_captions.txt")
TASK2_CSV    = os.path.join(BASE_DIR, "..", "Task2_Frame_Analysis", "detection_report.csv")

CONFIDENCE     = 0.4
FRAME_INTERVAL = 15
MAX_FRAMES     = None
VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]

# ─────────────────────────────────────────────
# COLOR DETECTION
# ─────────────────────────────────────────────
def detect_color(frame, bbox):
    """
    Detects dominant color of a vehicle using HSV color space.

    Why HSV?
    - Separates color (Hue) from brightness (Value)
    - More robust than RGB under different lighting conditions
    - Standard approach for color detection in computer vision
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return "unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    med_h = int(np.median(h))
    med_s = int(np.median(s))
    med_v = int(np.median(v))

    if med_v < 50:
        return "black"
    elif med_s < 50 and med_v > 200:
        return "white"
    elif med_s < 50:
        return "silver/gray"
    elif med_h < 10 or med_h > 160:
        return "red"
    elif 15 <= med_h <= 40:
        return "yellow/orange"
    elif 35 <= med_h <= 85:
        return "green"
    elif 85 <= med_h <= 130:
        return "blue"
    else:
        return "other"

# ─────────────────────────────────────────────
# LANE DETECTION
# ─────────────────────────────────────────────
def detect_lane(bbox, frame_width, num_lanes=4):
    """
    Estimates lane position from horizontal bounding box center.

    Divides frame into equal lane segments and determines
    which segment the vehicle center falls into.

    Note: This is a simplified estimation. A production system
    would use perspective transform and Hough line detection
    for accurate lane boundary detection.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    center_x = (x1 + x2) // 2

    lane_width  = frame_width / num_lanes
    lane_number = min(int(center_x / lane_width) + 1, num_lanes)

    if lane_number == 1:
        position = "leftmost lane"
    elif lane_number == num_lanes:
        position = "rightmost lane"
    else:
        position = f"lane {lane_number} from left"

    return lane_number, position

# ─────────────────────────────────────────────
# VIDEO ANALYSIS
# ─────────────────────────────────────────────
def analyze_video(video_path):
    """
    Processes video frames and builds structured scene data:
    - Detected vehicle class
    - Vehicle color (HSV analysis)
    - Lane position (bounding box center)
    - Timestamp per frame

    This structured data is passed to GPT-4o for reasoning.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return []

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    model       = YOLO("yolov8n.pt")
    scene_data  = []
    frame_idx   = 0
    count       = 0

    print("[INFO] Analyzing video for color & lane data...")

    while cap.isOpened() and (MAX_FRAMES is None or count < MAX_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_INTERVAL == 0:
            timestamp = round(frame_idx / 30, 1)
            results   = model(frame, conf=CONFIDENCE, verbose=False)
            objects   = []

            for result in results:
                for box in result.boxes:
                    class_id   = int(box.cls[0])
                    class_name = model.names[class_id]

                    if class_name not in VEHICLE_CLASSES:
                        continue

                    bbox = box.xyxy[0].tolist()
                    conf = round(float(box.conf[0]), 2)

                    color    = detect_color(frame, bbox)
                    _, lane  = detect_lane(bbox, frame_width)

                    objects.append({
                        "class"     : class_name,
                        "color"     : color,
                        "lane"      : lane,
                        "confidence": conf
                    })

            scene_data.append({
                "frame"    : frame_idx,
                "timestamp": timestamp,
                "objects"  : objects
            })

            print(f"  → Frame {frame_idx} ({timestamp}s): "
                  f"{len(objects)} vehicles")
            count += 1

        frame_idx += 1

    cap.release()
    print("[INFO] Color & lane analysis complete ✅\n")
    return scene_data

# ─────────────────────────────────────────────
# RUN A TASK SCRIPT
# ─────────────────────────────────────────────
def run_task(script_path, task_name):
    """
    Runs a task Python script as a subprocess.
    Shows live output to the user.
    """
    print(f"\n[INFO] Running {task_name}...")
    print("-" * 45)

    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=False,
        text=True
    )

    if result.returncode == 0:
        print(f"\n[INFO] {task_name} completed successfully ✅")
    else:
        print(f"\n[ERROR] {task_name} failed ❌")

    return result.returncode == 0

# ─────────────────────────────────────────────
# LOAD TASK RESULTS
# ─────────────────────────────────────────────
def load_task1_results():
    if not os.path.exists(TASK1_OUTPUT):
        return None
    with open(TASK1_OUTPUT, "r") as f:
        return f.read()

def load_task2_results():
    if not os.path.exists(TASK2_CSV):
        return None
    with open(TASK2_CSV, "r") as f:
        return f.read()

# ─────────────────────────────────────────────
# DETECT USER INTENT
# ─────────────────────────────────────────────
def detect_intent(client, user_input):
    """
    Uses GPT to classify user intent into:
    - run_task1: run video captioning
    - run_task2: run object detection
    - run_all:   run all tasks
    - query:     ask about results
    - exit:      quit
    """
    prompt = f"""
Classify this input into exactly one of these intents:
run_task1, run_task2, run_all, query, exit

Examples:
"run task1" -> run_task1
"run captioning" -> run_task1
"run task2" -> run_task2
"run detection" -> run_task2
"run all" -> run_all
"run everything" -> run_all
"how many cars?" -> query
"quit" -> exit

Input: "{user_input}"
Reply with ONLY the intent word.
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10
    )
    return response.choices[0].message.content.strip().lower()

# ─────────────────────────────────────────────
# ANSWER QUERY
# ─────────────────────────────────────────────
def answer_query(client, question, scene_data=None):
    """
    Answers user questions using all available data:
    - Task 1 captions
    - Task 2 detection CSV
    - Color & lane analysis from scene_data
    """
    context = ""

    task1 = load_task1_results()
    if task1:
        context += f"\n=== TASK 1 - VIDEO CAPTIONS ===\n{task1}\n"

    task2 = load_task2_results()
    if task2:
        context += f"\n=== TASK 2 - OBJECT DETECTION ===\n{task2}\n"

    if scene_data:
        summary = "\n=== COLOR & LANE ANALYSIS ===\n"
        for i, scene in enumerate(scene_data):
            summary += f"\nFrame {i+1} (at {scene['timestamp']}s):\n"
            for obj in scene["objects"]:
                summary += (f"  - {obj['color']} {obj['class']} "
                           f"in {obj['lane']} "
                           f"(confidence: {obj['confidence']})\n")
        context += summary

    if not context:
        return "No results available yet. Please run Task 1 or Task 2 first."

    prompt = f"""
You are a video analysis assistant.
Here is the data from the video analysis pipeline:

{context}

Answer this question based ONLY on the data above.
Be specific about frame numbers, colors, and lane positions.
If something is not in the data, say so clearly.

Question: {question}
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a precise video analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400
    )
    return response.choices[0].message.content

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Video Analysis Chatbot")
    print("=" * 55)
    print("\nCommands:")
    print("  'run task1'  → Run video captioning")
    print("  'run task2'  → Run object detection")
    print("  'run all'    → Run all tasks")
    print("  Any question → Query the results")
    print("  'quit'       → Exit")
    print("-" * 55)

    if not API_KEY:
        print("[ERROR] Missing OPENAI_API_KEY in .env file")
        return

    client = OpenAI(api_key=API_KEY)

    # Pre-load color & lane analysis at startup
    scene_data = analyze_video(VIDEO_PATH)

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        intent = detect_intent(client, user_input)
        print(f"[INFO] Intent: {intent}")

        if intent == "exit":
            print("Goodbye!")
            break
        elif intent == "run_task1":
            run_task(TASK1_SCRIPT, "Task 1 - Video Captioning")
        elif intent == "run_task2":
            run_task(TASK2_SCRIPT, "Task 2 - Object Detection")
        elif intent == "run_all":
            run_task(TASK1_SCRIPT, "Task 1 - Video Captioning")
            run_task(TASK2_SCRIPT, "Task 2 - Object Detection")
            print("\n[INFO] All tasks done! Ask your questions.")
        else:
            print("Bot: thinking...")
            answer = answer_query(client, user_input, scene_data)
            print(f"\nBot: {answer}")

if __name__ == "__main__":
    main()