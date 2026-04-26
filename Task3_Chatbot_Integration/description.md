# Task 3: Integration with Chatbot System

## Overview

This document describes the implementation of a chatbot that integrates the video captioning system (Task 1) and object detection system (Task 2) to allow users to query video content through natural language text prompts.

The chatbot also serves as a pipeline orchestrator, it can run Task 1 and Task 2 on demand and answer questions about the results using GPT-4o reasoning over structured data.

---

## How to Run

```bash
cd Task3_Chatbot_Integration
python3 task3_chatbot.py
```

### Available Commands

| Command | What it does |
|---|---|
| `run task1` | Runs video captioning pipeline |
| `run task2` | Runs object detection pipeline |
| `run all` | Runs both tasks in sequence |
| Any question | Queries results using GPT-4o |
| `quit` | Exits the chatbot |

### Example Interactions
```
You: run all
→ Runs Task 1 and Task 2 automatically
You: what do you see in frame 1?
Bot: In Frame 1 (0.0s):
- yellow/orange bus in rightmost lane (confidence: 0.88)
- black car in lane 2 from left (confidence: 0.82)
- blue car in lane 3 from left (confidence: 0.77)
You: in which frame is there a yellow bus?
Bot: Yellow/orange bus detected in frames 1-7 in rightmost lane
You: how many cars were detected in total?
Bot: 108 car detections across all 16 frames
You: in which frames were pedestrians detected?
Bot: Persons detected in frames 1, 6, 7, 8, 9
```

---

## Integration Architecture
```
User Query
↓
GPT-4o Intent Classifier
↓ (run task / query)
├── Run Task 1 → BLIP captions saved to task1_captions.txt
├── Run Task 2 → YOLOv8 detections saved to detection_report.csv
└── Query → GPT-4o reasons over:
- Task 1 captions
- Task 2 detection CSV
- Color & lane analysis (HSV + bbox position)
↓
Natural language answer
```

---

## Implementation Details

### Intent Detection
User input is classified by GPT-4o into one of five intents: run_task1, run_task2, run_all, query, or exit.
This allows flexible natural language commands the user does not need to type exact commands.

### Color Detection (HSV Analysis)
Vehicle colors are detected using HSV color space analysis:
- HSV separates color (Hue) from brightness (Value)
- More robust than RGB under different lighting conditions
- Detects: black, white, silver/gray, red, yellow/orange, green, blue

### Lane Detection (Bounding Box Position)
Lane position is estimated from the horizontal center of each vehicle's bounding box:
- Frame width divided into equal lane segments
- Vehicle center determines which lane it occupies
- Labels: leftmost lane, lane N from left, rightmost lane

Note: This is a simplified estimation. A production system would use perspective transform and Hough line detection for accurate lane boundary detection.

### GPT-4o Reasoning
All structured data (captions, detections, colors, lanes) is formatted into a readable summary and passed to GPT-4o as context. GPT-4o then reasons over this data to answer natural language questions accurately.

---

## Potential Improvements

### 1. Multimodal Chatbot
Allow image-based queries — user uploads a screenshot, chatbot identifies the matching timestamp and provides full context. Requires a multimodal LLM (GPT-4o vision).

### 2. Temporal Query Understanding
Add timestamp metadata filters to handle queries like "what happened at the beginning" or "after 30 seconds" more accurately.

### 3. Real-Time Streaming
For live camera feeds:
- Process frames in a sliding window as they arrive
- Update data store incrementally in real time
- Chatbot answers queries about what is happening right now
This is directly applicable to real-world automotive deployment where systems must respond to a dynamic driving environment.

### 4. On-Device Deployment
Replace cloud LLM with a quantized local model
(Llama 3.2 / Phi-3 Mini via Ollama) for:
- Privacy-preserving processing
- Low-latency responses without internet connectivity
- Deployment on edge devices

### 5. Voice Interface
Add Whisper (speech-to-text) at input and TTS at output for a fully hands-free querying experience.

### 6. Cloud Storage for Scalable Data Management
Currently annotated frames, captions, and detection results are stored locally. For production deployment, these should be pushed to cloud object storage (AWS S3, Google Cloud Storage, or Azure Blob) so data is accessible across multiple devices without local disk dependency. This also enables large-scale dataset collection from multiple camera sources running simultaneously.