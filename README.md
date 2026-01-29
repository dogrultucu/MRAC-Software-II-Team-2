# Group Project Repository Submission Template 

## Index
  - [Overview](#overview) 
  - [Getting Started](#getting-started)
  - [Demo](#demo)
  - [Authors](#authors)
  - [References](#references)
  - [Credits](#credits)

## MRAC0X (25/26): Software II – CONFISCAN  
**Confidence-Aware Robotic Inspection for Post-Earthquake Structures**

This project explores a robotic inspection pipeline designed for post-earthquake scenarios, where human access is unsafe or restricted.  
The system focuses on **image-based damage detection**, supported by **point cloud validation**, to provide **risk assessment with explicit confidence estimation**.  
The work is developed within the scope of the MRAC Software II course at IAAC.

---
![Group 1 -Software II](https://github.com/user-attachments/assets/233c9217-9fac-4dab-95b8-88b80ba1eb54)


## Overview

After major earthquakes, many structures remain standing but suffer hidden or progressive damage.  
Entering these environments repeatedly poses a significant risk to first responders and engineers.

**CONFISCAN** proposes a **robot-assisted inspection system** that enables:
- rapid on-site scanning of damaged interiors,
- real-time crack detection using RGB images,
- confidence-aware decision support,
- re-inspection and change detection over time.

### Key Design Principles
- **Image-first analysis** for crack detection (photogrammetry-based)
- **Point cloud as a supporting layer**, not the primary detector
- **Hybrid autonomy**: human-guided navigation with perception-driven local decisions
- **Explainable outputs**: risk levels always paired with confidence

The system is designed to operate in partially collapsed, low-visibility, and communication-limited environments.

---

## Getting Started

### Prerequisites
To replicate this project, ensure the following environment:

* Ubuntu LTS 20.04+
* Python 3.8+
* ROS Noetic
* Docker (optional, for deployment)

---

### Dependencies
Main software dependencies include:

* **NumPy** – numerical computation
* **OpenCV** – image processing and QA metrics
* **ROS** – robot communication and state handling
* **Ultralytics YOLOv8** – crack segmentation (ML-based)
* **COLMAP / OpenMVG (optional)** – photogrammetry and SfM

Dependencies can be installed as follows:



```bash
# ROS Noetic and core dependencies
wget -c https://raw.githubusercontent.com/qboticslabs/ros_install_noetic/master/ros_install_noetic.sh
chmod +x ./ros_install_noetic.sh
./ros_install_noetic.sh

# Python dependencies
pip3 install numpy opencv-python ultralytics

# Dependencies

This document lists all library dependencies required for the crack detection and point cloud processing scripts.

## Quick Install

```bash
pip install opencv-python open3d numpy ultralytics scipy tqdm laspy torch torchvision
```

## Core Libraries

| Library | Version | Description |
|---------|---------|-------------|
| `opencv-python` | >= 4.8.0 | Image processing, feature detection, camera calibration |
| `open3d` | >= 0.17.0 | Point cloud processing, mesh reconstruction, 3D visualization |
| `numpy` | >= 1.24.0 | Numerical computing, array operations |
| `ultralytics` | >= 8.0.0 | YOLOv8 segmentation models for crack detection |
| `scipy` | >= 1.10.0 | Spatial algorithms (KDTree for coherence filtering) |
| `tqdm` | >= 4.65.0 | Progress bars for batch processing |
| `laspy` | >= 2.4.0 | LAS/LAZ point cloud file I/O |
| `torch` | >= 2.0.0 | PyTorch backend for YOLOv8 |
| `torchvision` | >= 0.15.0 | PyTorch vision utilities |

## Dependency Matrix

| Script | opencv | open3d | numpy | ultralytics | scipy | tqdm | laspy | torch |
|--------|:------:|:------:|:-----:|:-----------:|:-----:|:----:|:-----:|:-----:|
| `crack_detector_pcd_render.py` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `images_to_pcd.py` | ✓ | ✓ | ✓ | - | - | ✓ | - | - |
| `undistort_images.py` | ✓ | - | ✓ | - | - | - | - | - |
| `advanced_crack_inspector.py` | ✓ | - | ✓ | ✓* | - | - | - | ✓* |

*Optional - only required if using YOLOv8 detection mode

## Per-Script Dependencies

### crack_detector_pcd_render.py
Main crack detection with 3D point cloud analysis.
```bash
pip install opencv-python open3d numpy ultralytics scipy tqdm laspy torch
```

### images_to_pcd.py
Structure from Motion pipeline for generating point clouds from images.
```bash
pip install opencv-python open3d numpy tqdm
```

### undistort_images.py
Lens distortion correction using COLMAP camera data.
```bash
pip install opencv-python numpy
```

### advanced_crack_inspector.py
Real-time webcam crack inspection with QA metrics.
```bash
# Minimal (OpenCV detection only)
pip install opencv-python numpy

# Full (with YOLOv8 detection)
pip install opencv-python numpy ultralytics torch
```

## GPU Acceleration (Optional)

For faster YOLOv8 inference, install PyTorch with CUDA support:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Requirements File

You can also install all dependencies using the requirements.txt file:

```bash
pip install -r requirements.txt
```

## Troubleshooting

### Open3D Installation Issues
If Open3D fails to install, try:
```bash
pip install open3d --no-cache-dir
```

### CUDA/PyTorch Compatibility
Ensure your CUDA version matches the PyTorch installation. Check with:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### laspy for LAS/LAZ Files
For LAZ (compressed) file support:
```bash
pip install laspy[lazrs]
```



