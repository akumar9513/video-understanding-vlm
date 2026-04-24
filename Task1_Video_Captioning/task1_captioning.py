"""
Task 1: Vision-Language Model Exploration - Video Captioning
Author: Anil Kumar
Video URL: https://www.pexels.com/video/busy-city-highway-traffic-at-rush-hour-31115112/
"""

import cv2
import torch
import os
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from collections import Counter

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
VIDEO_PATH     = "../../traffic_video.mp4"  # video sits in _Task root
FRAME_INTERVAL = 30                          # extract one frame every 30 frames
MAX_FRAMES     = 8                           # max frames to caption
OUTPUT_FILE    = "task1_captions.txt"        # results saved here

# ─────────────────────────────────────────────
# STEP 1: Extract Frames
# ─────────────────────────────────────────────
def extract_frames(video_path, interval, max_frames):
    """
    Opens the video and extracts frames at regular intervals.
    Converts BGR (OpenCV default) to RGB (required by BLIP).
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return []

    frames = []
    frame_idx = 0

    print(f"[INFO] Extracting frames every {interval} frames...")

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % interval == 0:
            # OpenCV reads BGR → convert to RGB for BLIP
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            frames.append((frame_idx, pil_image))
            print(f"  → Frame {frame_idx} extracted")

        frame_idx += 1

    cap.release()
    print(f"[INFO] Total frames extracted: {len(frames)}")
    return frames

# ─────────────────────────────────────────────
# STEP 2: Load BLIP Model
# ─────────────────────────────────────────────
def load_blip_model():
    """
    Loads BLIP image captioning model from HuggingFace.
    
    Why BLIP?
    - Lightweight enough to run without a discrete GPU
    - Produces natural language captions (not just labels)
    - Cross-platform: works on Mac M-series, Windows, Linux
    
    Device selection (automatic):
    - Apple Silicon (M1/M2/M3/M4) → MPS (Metal) backend
    - Windows/Linux with NVIDIA GPU → CUDA backend  
    - Any machine without GPU → CPU (slower but works)
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] Device: Apple MPS (Metal) — M-series chip detected")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("[INFO] Device: CUDA GPU detected")
    else:
        device = torch.device("cpu")
        print("[INFO] Device: CPU — no GPU detected, inference will be slower")

    print("[INFO] Loading BLIP model... (first run downloads ~990MB)")
    print("[INFO] Subsequent runs use cached model instantly")

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        torch_dtype=torch.float32   # float32 works on ALL platforms
    ).to(device)

    model.eval()
    print("[INFO] BLIP model loaded")
    return processor, model, device

# ─────────────────────────────────────────────
# STEP 3: Generate Caption for One Frame
# ─────────────────────────────────────────────
def generate_caption(image, processor, model, device):
    """
    Feeds one PIL image into BLIP and returns a natural language caption.
    
    num_beams=4 → beam search produces better captions than greedy decoding
    max_new_tokens=50 → limits caption length to keep it concise
    torch.no_grad() → disables gradient tracking (not needed for inference)
    """
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=4,
            early_stopping=True
        )

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

# ─────────────────────────────────────────────
# STEP 4: Summarize All Captions
# ─────────────────────────────────────────────
def summarize_captions(captions):
    """
    Finds the most frequent meaningful words across all frame captions
    to produce a simple video-level summary description.
    """
    stopwords = {"a", "an", "the", "of", "in", "on", "at", "is",
                 "with", "and", "are", "to", "some", "many"}
    all_words = []
    for caption in captions:
        words = [w.lower() for w in caption.split()
                 if w.lower() not in stopwords and len(w) > 3]
        all_words.extend(words)

    top_words = [word for word, _ in Counter(all_words).most_common(5)]
    return (
        f"The video predominantly features: {', '.join(top_words)}. "
        f"Across {len(captions)} sampled frames, the scene depicts "
        f"an urban traffic environment with various moving subjects."
    )

# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def main():
    # Step 1: Extract frames from video
    frames = extract_frames(VIDEO_PATH, FRAME_INTERVAL, MAX_FRAMES)

    if not frames:
        print("[ERROR] No frames extracted. Check video path.")
        return

    # Step 2: Load BLIP model
    processor, model, device = load_blip_model()

    # Step 3: Generate caption for each frame
    print("\n[INFO] Generating captions...")
    captions = []
    results = []

    for frame_idx, image in frames:
        caption = generate_caption(image, processor, model, device)
        captions.append(caption)
        results.append((frame_idx, caption))
        print(f"  Frame {frame_idx:>5}: {caption}")

    # Step 4: Summarize all captions
    summary = summarize_captions(captions)
    print(f"\n{'='*60}")
    print("VIDEO SUMMARY:")
    print(summary)
    print(f"{'='*60}\n")

    # Step 5: Save everything to text file
    with open(OUTPUT_FILE, "w") as f:
        f.write("VIDEO CAPTIONS\n")
        f.write("=" * 40 + "\n")
        for frame_idx, caption in results:
            f.write(f"Frame {frame_idx:>5}: {caption}\n")
        f.write("\nVIDEO SUMMARY\n")
        f.write("=" * 40 + "\n")
        f.write(summary + "\n")

    print(f"[INFO] Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()