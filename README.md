# Group Project Repository Submission Template 
## Index
  - [Overview](#overview) 
  - [Getting Started](#getting-started)
  - [Demo](#demo)
  - [Authors](#authors)
  - [References](#references)
  - [Credits](#credits)
<!--  Other options to write Readme
  - [Deployment](#deployment)
  - [Used or Referenced Projects](Used-or-Referenced-Projects)
-->
## MRAC0X(XX/XX): ClassName XX - Student Project Name
<!--Write a few sentences of academic context and project description -->  
This project aims to demonstrate a fantastic application using fascinating technologies, developed within the scope of the best class ever.   
## Overview
<!-- Write Overview about this project -->
The project's justification, state-of-the-art, and inspiration live in this section.

## Getting Started

### Prerequisites
Ensure that you fulfill the following criteria to replicate this project.
* Ubuntu LTS 20.04 <
* Python 3.7 <
* Docker

### Depencies
The project's dependencies include:
* Numpy - for matrix manipulation
* OpenCV - for image processing
* ROS - for interfacing with the robot

The dependencies are satisfied using the following sources:

```bash
# ROS Noetic and core dependencies
wget -c https://raw.githubusercontent.com/qboticslabs/ros_install_noetic/master/ros_install_noetic.sh && chmod +x ./ros_install_noetic.sh && ./ros_install_noetic.sh
# install numpy
pip3 install numpy setuptools
```

### Installing
A step by step series of examples that tell you how to get a development 
env running

```bash
cd ~/catkin_ws/src
git submodule init
git submodule update
cd ../
rosdep install --from-paths src --ignore-src -r -y
catkin_make -DCMAKE_BUILD_TYPE=Release
source ./devel/setup.bash
```
### Deployment
Add additional notes about how to deploy this on a live system
* Run the application with `.docker/run_user_nvidia.sh`
* Ensure that you are running the indicate command `sudo chmod -R <user_name> \dev_ws` for permitions
* Run `terminator`

## Demo
Here is what the project can do and what are the results.

The project can be launched with the following command:
* `roslaunch package_name package_name.launch`

This opens up `rviz` and shows the robot moving around

## Authors
  - [Name](insert linkedin/webpage link) - role

## References
- [K. Albee et al., “A robust observation, planning, and control pipeline for autonomous rendezvous with tumbling targets,” Frontiers in Robotics and AI, vol. 8, p. 234, 2021, doi: 10.3389/frobt.2021.641338.](https://www.frontiersin.org/articles/10.3389/frobt.2021.641338/full)

## Credits
  - [Name](insert linkedin/webpage link) - role

<!--  DO NOT REMOVE
-->
#### Acknowledgements

- Creation of GitHub template: [Marita Georganta](https://www.linkedin.com/in/marita-georganta/) - Robotic Sensing Expert
- Creation of MRAC-IAAC GitHub Structure: [Huanyu Li](https://www.linkedin.com/in/huanyu-li-457590268/) - Robotic Researcher


