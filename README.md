# Video Understanding with Vision-Language Models
**Author:** Anil Kumar  
**Date:** April 2026

---

## Overview

This project implements a complete video understanding pipeline
combining Vision-Language Models and object detection to extract
meaningful information from video content.

- **Task 1:** Video captioning using BLIP Vision-Language Model
- **Task 2:** Frame preprocessing and object detection using YOLOv8
- **Task 3:** Interactive chatbot with color detection, lane detection and GPT-4o reasoning with a pipeline orchestrator to run all tasks

---

## Video Used

- **Source:** Pexels (royalty-free)
- **URL:** https://www.pexels.com/video/busy-city-highway-traffic-at-rush-hour-31115112/
- **Content:** Busy city street with cars, buses, and pedestrians

---

## Project Structure

```
Video-Understanding-VLM/
├── traffic_video.mp4             ← Place your video file here
├── .env                          ← Add your OpenAI API key here (see step4 Option 2)
├── Task1_Video_Captioning/
│   ├── task1_captioning.py       ← VLM video captioning pipeline
│   ├── task1_captions.txt        ← Generated on run (captions output)
│   └── description.md            ← Observations and decisions
├── Task2_Frame_Analysis/
│   ├── task2_frame_analysis.py   ← Frame preprocessing + YOLOv8
│   ├── output_frames/            ← Generated on run (annotated frames)
│   ├── detection_report.csv      ← Generated on run (detection data)
│   └── description.md            ← Findings and difficulties
├── Task3_Chatbot_Integration/
│   ├── task3_chatbot.py          ← Master chatbot + pipeline orchestrator
│   └── description.md            ← Integration strategy
└── README.md                     ← This file
```

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
Download the video from the URL below and place it as: BMW_Task_Solution_AnilKumar/traffic_video.mp4

Video URL: https://www.pexels.com/video/cars-on-road-during-daytime-2103099/

### Step 4 — Set your OpenAI API key (only for Task 3)

You can set your API key in two ways:

**Option 1 — Shell export (quickest):**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```
Note: This is temporary and will be lost when the terminal is closed.
You will need to run this command again in each new terminal session.

**Option 2 — .env file (recommended, persistent):**
Create a `.env` file in the root folder:
OPENAI_API_KEY=your-openai-api-key-here
This persists across terminal sessions and is loaded automatically.

Note: Tasks 1 and 2 run fully locally and do not require an API key.
Only Task 3 chatbot requires the OpenAI API key.

---

## Running the Solutions

## Option A — Run Each Task Individually

Use this approach if you want to run and inspect each task
separately without the chatbot.

### Run Task 1 — Video Captioning
```bash
cd Task1_Video_Captioning
python3 task1_captioning.py
```
What it does:
- Extracts frames from the video every 30 frames
- Generates a natural language caption for each frame using BLIP
- Saves all captions and a video summary to task1_captions.txt

Expected output:
```
[INFO] Extracting frames every 30 frames...
[INFO] Using Apple MPS backend
[INFO] Generating captions...
Frame     0: a city street filled with lots of traffic
Frame    30: a city street filled with lots of traffic
...
VIDEO SUMMARY: The video predominantly features: city, street, traffic...
[INFO] Results saved to task1_captions.txt
```
---

### Run Task 2 — Object Detection
```bash
cd Task2_Frame_Analysis
python3 task2_frame_analysis.py
```
What it does:
- Extracts and preprocesses frames (resize, blur, CLAHE)
- Runs YOLOv8 nano object detection on each frame
- Saves annotated frames with bounding boxes to output_frames/
- Saves full detection data to detection_report.csv

Expected output:
```
[INFO] Running object detection...
Frame     0:  8 detections [car, person, bus] → saved
Frame    30:  9 detections [car, truck, bus] → saved
...
DETECTION SUMMARY:
car                 : 108 detections
bus                 :  32 detections
person              :   5 detections
```
---

### Run Task 3 — Chatbot (standalone, no orchestrator)
Task 3 is designed to be run through the chatbot interface. See Option B below.

---

## Option B — Run Everything Through the Chatbot

Use this approach to run all tasks and ask questions about
the video using natural language — all from one interface.

### Step 1 — Start the chatbot
```bash
cd Task3_Chatbot_Integration
python3 task3_chatbot.py
```

The chatbot will automatically pre-load color and lane
analysis at startup, then wait for your commands.

### Step 2 — Run tasks using natural language commands

To run Task 1 (video captioning):
You: run task1

To run Task 2 (object detection):
You: run task2

To run all tasks in sequence:
You: run all

### Step 3 — Ask questions about the video

```
Once tasks have been run, ask anything about the video:
You: what do you see in frame 1?
Bot: In Frame 1 (at 0.0 seconds):
- yellow/orange bus in rightmost lane (confidence: 0.88)
- black car in lane 2 from left (confidence: 0.82)
- blue car in lane 3 from left (confidence: 0.77)
...
You: in which frame is there a yellow bus?
Bot: Yellow/orange bus detected in frames 1-7 in rightmost lane
You: how many cars were detected in total?
Bot: 108 car detections across all 16 frames
You: in which frames were pedestrians detected?
Bot: Persons detected in frames 1, 6, 7, 8, 9
You: which lane has the most traffic?
Bot: Lane 3 from left has the highest concentration of vehicles

```
### Step 4 — Exit the chatbot
You: quit

---

## Technology Stack

| Component | Technology | Reason |
|---|---|---|
| VLM Captioning | BLIP (Salesforce) | Lightweight, cross-platform |
| Object Detection | YOLOv8 nano | Fast, COCO-pretrained, 6MB |
| Color Detection | OpenCV HSV | Robust under varying lighting |
| Lane Estimation | Bounding box center | Fast, no extra model needed |
| Chatbot Reasoning | GPT-4o | Natural language understanding |
| Frame Processing | OpenCV | Industry standard |

---

## Cross-Platform Device Support

| Platform | Device Used |
|---|---|
| Apple M-series (M1/M2/M3/M4) | MPS (Metal Performance Shaders) |
| Windows / Linux with NVIDIA GPU | CUDA |
| Any machine without GPU | CPU |

No manual configuration required — detected automatically.

---

## Notes
- Tasks 1 and 2 run fully locally — no API key needed
- OpenAI API key is only required for Task 3 chatbot
- Models download automatically on first run and are cached for subsequent runs
- All code is fully commented
- Each task folder contains a description.md with observations, implementation decisions, and potential improvements
