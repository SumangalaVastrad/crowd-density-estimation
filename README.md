# Crowd Density Estimation Using Surveillance Camera

A computer vision pipeline developed in **Python** using **OpenCV**, **YOLOv8**, and **CNN**-based feature extraction to detect individuals in surveillance video streams, compute real-time crowd density distributions, and trigger automated safety thresholds.

---

## Overview

Crowd monitoring in public transit hubs, educational campuses, and event venues requires automated estimation to prevent overcrowding incidents. This project builds a lightweight, real-time video processing pipeline that:
1. Samples frames from live RTSP/webcam feeds or recorded video files.
2. Performs robust pedestrian detection and bounding-box tracking using YOLOv8.
3. Classifies frame-level and region-level crowd density into **LOW**, **MEDIUM**, and **HIGH** categories based on configurable area-density thresholds.
4. Generates visual overlay warnings on high-density events and appends structured timestamp logs to CSV for post-event audit.

---

## Pipeline Architecture

```
[ Video Source / RTSP ]
          │
          ▼
   Frame Preprocessing (Resizing, Normalization, Noise Reduction)
          │
          ▼
   YOLO Object Detection (Ultralytics yolov8n/yolov8s pretrained on COCO)
          │
          ▼
   Non-Maximum Suppression (NMS) & Centroid Extraction
          │
          ▼
   Spatial Density Estimation (Grid-based occupancy & threshold logic)
          │
    ┌─────┴─────────────────────────┐
    ▼                               ▼
Live Annotated Display      Structured CSV Logger
(Bounding Boxes + Alerts)   (`outputs/crowd_log.csv`)
```

---

## Key Features
- **Flexible Stream Inputs**: Seamlessly switches between live webcam devices (`--source 0`), RTSP IP camera streams, or local MP4/AVI video files.
- **Configurable Density Thresholds**: Customizable parameters for Low, Medium, and High crowd limits depending on the camera's field of view and focal distance.
- **Alert Triggering**: On-screen warning banners and immediate console alerts whenever headcounts exceed safety ceilings.
- **Historical Telemetry Logging**: Records `timestamp`, `frame_id`, `detected_count`, and `density_status` to `outputs/crowd_log.csv`.
- **Annotated Video Export**: Option to render and save the full processed stream with detection bounding boxes and crowd metrics.

---

## Repository Structure

```
crowd-density-estimation/
├── main.py                 # Core video processing loop, detection, and visualization
├── requirements.txt         # Dependencies (ultralytics, opencv-python, numpy)
├── outputs/                # Directory storing crowd_log.csv and annotated video exports
├── .gitattributes          # Linguist language definitions
└── README.md
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or 3.11
- Webcam or sample video file (e.g. `crowd_sample.mp4`)

### Setup Instructions
```bash
# Clone the repository
git clone https://github.com/SumangalaVastrad/crowd-density-estimation.git
cd crowd-density-estimation

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate      # Windows
# source .venv/bin/activate    # Linux / macOS

# Install required dependencies
pip install -r requirements.txt
```

---

## Running the Application

### 1. Run with live webcam
```bash
python main.py --source 0
```

### 2. Run with a video file
```bash
python main.py --source "sample_video.mp4"
```

### 3. Save the annotated video output
```bash
python main.py --source "sample_video.mp4" --save-video
```

### 4. Customizing density and alert thresholds
```bash
python main.py --source 0 --low-limit 8 --high-limit 18 --alert-limit 25
```

*Controls*: Press **Q** in the video display window to stop processing and save telemetry.

---

## Technical Details

| Parameter | Default Value | Description |
|---|---|---|
| Model Backbone | `yolov8n.pt` | Lightweight nano-model optimized for CPU/Edge inference |
| Confidence Threshold | `0.45` | Minimum detection confidence for person class |
| IoU Threshold (NMS) | `0.50` | Overlap threshold to eliminate duplicate bounding boxes |
| Log Output | `outputs/crowd_log.csv` | Output telemetry file with timestamp and headcount |

---

## Author
**Sumangala Vastrad**  
B.E. in Artificial Intelligence & Data Science, Nitte Meenakshi Institute of Technology (NMIT)  
Email: [sbvastrad2005@gmail.com](mailto:sbvastrad2005@gmail.com)  
GitHub: [@SumangalaVastrad](https://github.com/SumangalaVastrad)
