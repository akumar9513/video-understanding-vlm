# Video Understanding with Vision-Language Models
**Author:** Anil Kumar  
**Date:** April 2026

---

## Overview

This project implements a video understanding pipeline that combines Vision-Language Models (VLM) and object detection to extract meaningful information from video content. The solution is divided into three tasks:

- **Task 1:** Video captioning using BLIP (Vision-Language Model)
- **Task 2:** Frame extraction, preprocessing, and object detection using YOLOv8
- **Task 3:** Chatbot integration strategy using RAG architecture

---

## Video Used

- **Source:** Pexels (royalty-free)
- **URL:** https://www.pexels.com/video/busy-city-highway-traffic-at-rush-hour-31115112/
- **Content:** Busy city street with cars, buses, and pedestrians

---

## Project Structure
_Task_Solution_AnilKumar/
├── Task1_Video_Captioning/
│   ├── task1_captioning.py       ← VLM video captioning pipeline
│   └── description.md            ← Observations and decisions
├── Task2_Frame_Analysis/
│   ├── task2_frame_analysis.py   ← Frame preprocessing + YOLOv8 detection
│   ├── output_frames/            ← Annotated frame images (generated on run)
│   ├── detection_report.csv      ← Per-frame detection results (generated on run)
│   └── description.md            ← Findings and difficulties
├── Task3_Chatbot_Integration/
│   └── description.md            ← RAG-based chatbot integration strategy
└── README.md                     ← This file

---

## Setup & Installation

### Requirements
- Python 3.9 or higher
- Works on Mac (Apple Silicon or Intel), Windows, and Linux

### Step 1 — Create virtual environment
```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### Step 2 — Install dependencies
```bash
pip install transformers torch torchvision pillow opencv-python ultralytics accelerate
```

### Step 3 — Place the video file
Download the video from the URL above and place it as: _Task_Solution_AnilKumar/../../traffic_video.mp4 Or update the VIDEO_PATH variable in both scripts to point to your local video file.

---

## Running the Solutions

### Task 1 — Video Captioning
```bash
cd Task1_Video_Captioning
python3 task1_captioning.py
```

**What happens:**
- Downloads BLIP model on first run (~990MB, cached after)
- Extracts 8 frames from the video at 1 second intervals
- Generates a natural language caption for each frame
- Saves all captions and a video summary to task1_captions.txt

**Output:**
[INFO] Extracting frames every 30 frames...
[INFO] Using Apple M4 MPS backend
[INFO] Generating captions...
Frame     0: a city street filled with lots of traffic
Frame    30: a city street filled with lots of traffic
...
VIDEO SUMMARY: The video predominantly features: city, street, traffic...

---

### Task 2 — Frame Analysis & Object Detection
```bash
cd Task2_Frame_Analysis
python3 task2_frame_analysis.py
```

**What happens:**
- Extracts and preprocesses 8 frames (resize, blur, CLAHE)
- Downloads YOLOv8 nano model on first run (~6MB, cached after)
- Runs object detection on each preprocessed frame
- Saves annotated frames with bounding boxes to output_frames/
- Saves full detection data to detection_report.csv

**Output:**
[INFO] Running object detection...
Frame     0:  8 detections [car, person, bus] → saved
Frame    30:  9 detections [car, truck, bus] → saved
...
DETECTION SUMMARY:
car                 : 48 detections
bus                 : 15 detections
person              :  4 detections
truck               :  1 detections

---

### Task 3 — Chatbot Integration
No code to run. See `Task3_Chatbot_Integration/description.md` for the full RAG-based integration strategy, architecture diagram, example interactions, and potential improvements.

---

## Technology Stack

| Component | Technology | Reason |
|---|---|---|
| VLM Captioning | BLIP (Salesforce) | Lightweight, cross-platform, HuggingFace native |
| Object Detection | YOLOv8 nano | Fast, COCO-pretrained, only 6MB |
| Frame Processing | OpenCV | Industry standard for computer vision |
| Chatbot Strategy | LangChain + RAG | Modular, LLM-agnostic, scalable |

---

## Cross-Platform Device Support

Both scripts automatically detect and use the best available device:

| Platform | Device Used |
|---|---|
| Apple M-series (M1/M2/M3/M4) | MPS (Metal Performance Shaders) |
| Windows / Linux with NVIDIA GPU | CUDA |
| Any machine without GPU | CPU |

No manual configuration required.

---

## Notes
- All code is fully commented with explanations of every decision
- Each task folder contains a description.md with observations, findings, and potential improvements
- Task 2 generates output_frames/ and detection_report.csv automatically on first run
- Models are cached after first download subsequent runs are instant