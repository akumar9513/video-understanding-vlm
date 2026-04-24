# Task 3: Integration with Chatbot System

## Overview

This document describes a strategy for integrating the video captioning system (Task 1) with a chatbot, enabling users to query video content through natural language text prompts. 
The proposed architecture uses RAG (Retrieval Augmented Generation) a proven pattern for grounding LLM responses in specific data.

---

## Integration Architecture
User types a question
↓
Chatbot LLM (GPT-4o / Claude / local Llama)
↓
RAG Retriever searches caption + detection store
↓
Top-k relevant captions + detections retrieved
↓
LLM generates grounded answer with timestamp references
↓
Response returned to user

---

## Step-by-Step Strategy

### Step 1: Build the Caption & Detection Store
After running Tasks 1 and 2, store all outputs structured per frame:
- Frame index and timestamp
- BLIP caption text
- YOLOv8 detections (class, confidence, bounding box)
- Vector embedding of caption for semantic search

### Step 2: Embed Captions for Semantic Search
Convert captions to vector embeddings using a sentence transformer (all-MiniLM-L6-v2). Store in a lightweight local vector database — ChromaDB or FAISS. Both run entirely locally with no cloud dependency.

### Step 3: Query Pipeline
When user sends a question:
1. Embed the query using the same sentence transformer
2. Retrieve top-k most relevant frame captions via vector similarity search
3. Inject retrieved captions and detections into LLM prompt
4. LLM generates a grounded answer referencing specific timestamps and detected objects

### Step 4: LangChain Orchestration
Use LangChain to connect:
- Retriever (ChromaDB vector store)
- LLM (OpenAI / Anthropic / local Ollama)
- Memory (conversation history for follow-up questions)

---

## Example User Interactions

| User Query | Expected Response |
|---|---|
| "What was happening in the video?" | Summary synthesized from all captions |
| "Were there any pedestrians?" | Reports frames and timestamps with person detections |
| "How many cars were detected?" | Aggregates counts from detection CSV |
| "What happened around 3 seconds?" | Retrieves frames closest to that timestamp |
| "Was there anything unusual?" | Flags frames with unexpected or low-confidence detections |

---

## Potential Improvements

### 1. Multimodal Chatbot
Allow image-based queries — user uploads a screenshot, chatbot identifies the matching timestamp and provides full context. Requires a multimodal LLM (GPT-4o, Claude 3.5).

### 2. Temporal Query Understanding
Current RAG retrieves by semantic similarity only. Adding timestamp metadata filters would handle queries
like "what happened at the beginning" or "after 30 seconds" much more accurately.

### 3. Real-Time Streaming
For live in-vehicle video feeds:
- Process frames in a sliding window as they arrive
- Update vector store incrementally in real time
- Chatbot answers queries about what is happening right now
This is directly relevant to real-world automotive deployment where systems must respond to a dynamic driving environment.

### 4. On-Device Deployment
Replace cloud LLM with a quantized local model (Llama 3.2 / Phi-3 Mini via Ollama) for:
- Privacy-preserving processing (no data leaves the vehicle)
- Low-latency responses without internet connectivity
- Compliance with automotive data regulations

### 5. Voice Interface
Add Whisper (speech-to-text) at input and TTS at output creating a fully hands-free in-cabin video querying experience directly relevant to 's HMI and GenAI roadmap.

### 6. Real-Time Camera Feed Integration
- The current implementation processes a pre-recorded video file.
- The natural next step is connecting directly to a live camera feed, such as a vehicle's front-facing or cabin camera and processing frames in real time as they arrive.

This would involve:
- Replacing the video file input with a live stream OpenCV supports this with cv2.VideoCapture(0 for webcam or an RTSP stream URL for vehicle cameras)
- Processing frames in a sliding window pipeline
- Updating the caption and detection store incrementally
- Allowing the chatbot to answer queries about what is happening right now in the live feed

This architecture is directly applicable to real-world automotive deployment where the system needs to understand and respond to a dynamic driving environment in real time rather than analyzing footage after the fact.

### 7. Cloud Storage for Scalable Data Management
Currently annotated frames, captions, and detection results are stored locally. For production deployment, these should be pushed to cloud object storage (AWS S3, Google Cloud Storage, or Azure Blob) so data is accessible across multiple devices and users without local disk dependency. This also enables large-scale dataset collection from multiple test vehicles running simultaneously.

---

## Summary

The core integration pattern is RAG:
**Video → Captions + Detections → Vector Store → LLM → Answer**

This architecture is modular, scalable, and directly applicable to real-world automotive AI development — enabling natural language interfaces over multimodal vehicle sensor data.