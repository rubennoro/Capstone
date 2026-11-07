# Capstone Project - Codebase

This file contains a roadmap for navigating the codebase that supports our Northeastern undergraduate Capstone project. 

## Project Overview

### Previous Inspiration

Our project takes inspiration from Stanford's [UMI](https://github.com/real-stanford/universal_manipulation_interface) and [iPhUMI](https://github.com/real-stanford/iPhUMI). Each team 
constructed an end-effector style physical gripper for improving the efficiency of data collection 
to train robot manipulation tasks. 

Behavior cloning is a method of learning a policy where a robot can be manually controlled, or "teleoperated", by a human using it to complete a task. For a particular task, if repeated thousands of times, a policy can be learned to recreate the motion based on the sensor observations and actuator sequence
of teleoperation data. This comes at a cost, as manipulating entire robot arms is costly and time consuming, and for manipulation tasks, the most important data comes from the end-effector, the robot hand. 

Rather than collect data using an entire robot arm, all that's needed is the sensor information and action sequence of the hand, and inverse kinematics can be used to derive a 6DOF arm's position. And rather than teleoperate a robot hand, it's feasible to directly interface with it during training, which is exactly what the UMI and iPhUMI do. 

To collect data with these devices, a human holds the device, and since the manipulation interface has 2 fingers, can squeeze on a mechanical trigger to close and release them. This was shown to work well for diffusion and behavior cloning policies when tested in both papers, and we wanted to try it ourselves.

However, the barrier to entry is high. The UMI collects sensor data of surroundings with a ~$x00 GoPro, and the iPhUMI uses a bulky ~$x00 iPhone. Everyone has an iPhone, but their setup only works for a particular iPhone model, and it's not a device many would like to dedicate towards robot data collection.

## Our Solution

We evolve the designs of these two grippers to use a custom sensor rig, equipped with the same sensors as an iPhone but at a fraction of the price or size. 

### Sensors

The iPhone possesses an IMU, and two cameras, one wide-angle, and one regular lens. It processes this data onboard the iPhone chip, with proprietary ARKit software, which is very computationally intensive and causes the phone to heat quickly. We use:

- Arduino IMU
- Arducam IMX219 - widelens
- Arducam IMX519 - normal

### Mounting

We also add a custom CAD design that mounts to the gripper, housing all the sensors at the proper positions and angles to produce training-quality data. 

### Processing Power 

Onboard our gripper is a lightweight Raspberry Pi5. It collects per-episode data for a particular task we're training, and wirelessly packages and sends it to an Nvidia Jetson Orin edge compute board.

The Nvidia Jetson replicates the data processing done by the iPhone, with a custom software pipeline to turn raw data into clean policy model inputs. 

This is located in the **rnoro/data-work** branch.

## Simulation

We also validate our solution in simulation as well, under the **rnoro/main-sim** branch. Here we demonstrate a working simulation in MuJoCo, with our sensor setup, as shown below: 

<video src="docs/media/mujoco_pick_place.webm" controls width="100%"></video>

[MuJoCo pick-and-place demo (webm)](docs/media/mujoco_pick_place.webm)

### TBD - More to come soon!