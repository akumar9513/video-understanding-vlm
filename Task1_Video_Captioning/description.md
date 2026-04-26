# Task 1: Vision-Language Model Exploration – Video Captioning

## Video Used
- **Source:** Pexels (royalty-free, no restrictions)
- **URL:** https://www.pexels.com/video/busy-city-highway-traffic-at-rush-hour-31115112/
- **Content:** Busy city street with cars, buses, and pedestrians
- **Duration:** ~8 seconds sampled (30fps source)

---

## Model Choice: BLIP (Salesforce/blip-image-captioning-base)

I evaluated several Vision-Language Models before selecting BLIP base:

| Model | Decision | Reason |
|---|---|---|
| BLIP base | Selected | Lightweight, runs on MPS/CUDA/CPU, HuggingFace native |
| BLIP-2 | Not selected | Requires significantly more VRAM, impractical without a discrete GPU |
| LLaVA | Not selected | Larger model footprint, slower inference on CPU/MPS |
| CLIP | Not selected | Produces vector embeddings, not natural language captions |

---

## Implementation Decisions

**Frame sampling (interval=30)**
One frame per second at 30fps. Avoids redundant near-identical frames while capturing meaningful scene variation.

**Beam search (num_beams=4)**
Produces more coherent captions than greedy decoding. Minimal extra compute cost for noticeably better output quality.

**Cross-platform device selection**
Automatically detects and uses the best available backend:
- Apple M-series chips → MPS (Metal Performance Shaders)
- NVIDIA GPU → CUDA
- Any other machine → CPU
This ensures the script runs on Mac, Windows, and Linux without any manual configuration.

**float32 precision**
Used instead of float16 for maximum compatibility across all platforms including CPU-only machines.

**Absolute output path**
Output file `task1_captions.txt` is saved using an absolute path derived from the script location — ensures the file is always saved in the correct folder regardless of where the script is called from.

---

## Results

All 16 sampled frames produced the caption:
> "a city street filled with lots of traffic"

One frame at 300 produced:
> "a busy city street filled with lots of traffic"

**Video Summary:**
The video predominantly features: city, street, filled, traffic, lots. Across 16 sampled frames the scene consistently depicts an urban traffic environment with various moving subjects.

**Output file:** `task1_captions.txt`
Contains all frame captions and the video-level summary.

---

## Observations & Challenges

### 1. Repetitive captions
All frames produced identical captions because the video is a short static shot, the camera barely moves and the scene does not change significantly between seconds. BLIP correctly identifies the same scene each time. A longer, more varied video (highway → intersection → parking lot) would produce different captions per frame. This is expected model behavior, not a bug.

### 2. No temporal reasoning
BLIP processes each frame independently with no awareness of what came before or after. This is a fundamental limitation of image-level VLMs applied to video. A video-native model like Video-LLaVA would handle temporal context natively.

### 3. Background model download during inference
A safetensors file download ran in the background during inference on first run — cosmetic terminal noise only, no impact on caption quality or results.

### 4. Cross-platform compatibility
Tested on Apple M4 (MPS backend). The code automatically detects the best available device — MPS, CUDA, or CPU — ensuring it runs consistently across Mac, Windows, and Linux without any manual configuration required.

---

## Potential Improvements
- Use a video-native VLM (Video-LLaVA) for temporal understanding
- Apply scene change detection to sample frames more intelligently
- Feed all captions into an LLM summarizer via LangChain
- Use quantized models (4-bit GGUF) for faster edge deployment