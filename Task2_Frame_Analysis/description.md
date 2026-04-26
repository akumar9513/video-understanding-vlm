# Task 2: Data Preprocessing and Analysis

## Video Used
Same video as Task 1:
- **URL:** https://www.pexels.com/video/busy-city-highway-traffic-at-rush-hour-31115112/

---

## Preprocessing Pipeline

Each frame goes through 3 steps before detection:

### Step 1: Resize to 640×640
YOLOv8 is optimized for 640×640 input. Resizing ensures consistent inference behavior and eliminates aspect ratio issues from varying source video resolutions.

### Step 2: Gaussian Blur (Kernel 3×3)
Mild denoising step. A small 3×3 kernel removes high-frequency pixel noise without blurring object boundaries important for maintaining detection accuracy on edges.

### Step 3: CLAHE (Contrast Limited Adaptive Histogram Equalization)
Applied to the luminance (L) channel in LAB color space.

Why CLAHE over standard histogram equalization?
Standard HE applies a global transformation which can over-brighten already bright regions. CLAHE applies local contrast enhancement with a clip limit much more suitable for traffic scenes with mixed lighting (shadows, glare).

Why LAB color space?
Separating luminance from color channels means we enhance contrast without distorting hue or saturation.

---

## Object Detection: YOLOv8 Nano

**Model:** yolov8n.pt (COCO-pretrained, 80 classes)

**Why YOLOv8n?**
- Fastest and lightest YOLO variant (only 6MB)
- Pre-trained on COCO — includes all relevant classes:
  car, person, truck, bus, bicycle, motorcycle, traffic light
- Single-stage detector — fast enough for real-time use
- Cross-platform: Ultralytics auto-selects MPS/CUDA/CPU

**Confidence threshold: 0.4**
Filters weak detections while retaining meaningful ones.

**Absolute output paths**
Both `output_frames/` and `detection_report.csv` are saved using absolute paths derived from the script location ensures files are always saved in the correct Task 2 folder regardless of where the script is called from.


---

## Results

| Frame | Time | Detections | Classes |
|---|---|---|---|
| 0 | 0.0s | 8 | car, person, bus |
| 30 | 1.0s | 9 | car, truck, bus |
| 60 | 2.0s | 8 | car, bus |
| 90 | 3.0s | 8 | car, bus |
| 120 | 4.0s | 5 | car, bus |
| 150 | 5.0s | 6 | car, person |
| 180 | 6.0s | 12 | car, person, bus |
| 210 | 7.0s | 12 | car, person, bus |
| 240 | 8.0s | 9 | car, person, bus |
| 270 | 9.0s | 11 | car, truck, bus |
| 300 | 10.0s | 14 | car, bus |
| 330 | 11.0s | 9 | car, bus |
| 360 | 12.0s | 11 | car, truck, bus |
| 390 | 13.0s | 11 | car, bus |
| 420 | 14.0s | 9 | car, train, bus |
| 450 | 15.0s | 7 | car, bus |

**Overall Detection Summary:**
- car: 108 detections — dominant class across all frames
- bus: 32 detections — large vehicles consistently visible
- person: 5 detections — pedestrians near road edges
- truck: 3 detections — occasional heavy vehicle
- train: 1 detection — misclassification likely

**Annotated frames** 
saved as JPEG in `output_frames/` folder with green bounding boxes and confidence scores drawn on each detected object.

**Full detection data** 
saved in `detection_report.csv` with frame index, class, confidence, and bounding box coordinates.

---

## Observations & Findings

- **Cars dominate** — average 6 cars per frame, consistent
  with a busy urban traffic scene
- **Bus detection is strong** — large vehicles reliably detected
  even at 0.4 confidence threshold
- **Pedestrian count is low** — video is road-focused with
  few visible sidewalk areas
- **Detection count varies** — frames 180 and 210 had 12
  detections each, suggesting a busier section of the video
- **CLAHE visibly improved contrast** — especially noticeable
  in shadowed regions of the annotated output frames

---

## Difficulties Encountered

### 1. Motion blur on fast vehicles
Fast-moving cars create streaking artifacts at standard frame rates. Gaussian preprocessing reduces noise but cannot fully eliminate motion blur. A higher fps source would help.

### 2. Occlusion in dense traffic
Vehicles occlude each other in busy frames, cars behind buses are partially or fully hidden. Fundamental limitation of 2D single-camera detection without depth information.

### 3. Domain-specific class limitations
The COCO dataset does not include domain-specific classes like emergency vehicles or lane markings. Fine-tuning on automotive datasets like BDD100K or nuScenes would improve detection performance in real-world driving scenarios.

---

## Potential Improvements
- Add DeepSORT tracking to follow objects across frames
- Fine-tune on a domain-specific dataset for better performance
- Add MiDaS depth estimation for spatial context
- Add scene classifier (intersection / highway / parking lot)