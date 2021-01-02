# Experts in Team Project - Autonoumous Fence Inspection Robot

This project is about making a prototype fence inspection mobile robot to detect holes in fences along an airport. The mobile robot platform used is the SDU Frobit seen below.

| The prototype platform SDU Frobit |
|:------------------------:|
| ![](assets/frobit.jpg) |

The project can both run in ros-gazebo simulation and on the frobit itself in real life. 

## Real life

Due to LIDAR limits the frobit was not able to detect lines along an fence. However it could detect lines a more dense structure such as a building. The frobit driving using the PID controller as distance control is illustrated below. 

| Straight section | Corner section | Fence following | 
|:----------------:|:--------------:|:---------------:|
| ![Real life straight](assets/real_life_straight.gif) | ![Real life corner](assets/real_life_corner.gif) | ![Real life fence following](assets/real_life_fence_follow.gif)| 

## Simulation
|  Dog-bone map ICO |  Square map ICO | Airport map ICO |
|:-----------------:|:---------------:|:---------------:|
| ![Simulation ico dog-bone](assets/dog_bone_ico_optim.gif) | ![Simulation ico square](assets/sqare_map_ico.gif) | ![Simulation ico airport](assets/airport_sim.gif) |

## Vision
The vision folder contains the different algorithm used to detect holes in the fence. Below is a small example of one of the nueral network for hole detection.

|  YOLOv5 sim example |  YOLOv5 real example |
|:-------------------:|:--------------------:|
| ![](vision/assets/yolov5_sim.gif) | ![](vision/assets/yolov5_real_world.gif) |