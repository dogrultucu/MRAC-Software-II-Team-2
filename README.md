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
