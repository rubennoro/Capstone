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

## Pipeline: Data Collection → Training → Deployment

### Data Collection

Demonstrations recorded with the handheld rig and processed into training data. Located in the **rnoro/data-work**
branch.

### Training

Training a diffusion policy on the processed data, on the Northeastern cluster. Located in the **rnoro/training**
branch.

### Deployment

Running a policy on the real Franka Panda arm, with the sensor rig mounted on its hand. Located in
[`deployment/`](deployment/) on the **rnoro/final-deployment** branch.

Three machines work together:

```
  Sensor rig (Raspberry Pi 5) ──camera frames, Wi-Fi──► Laptop ──timed moves, Wi-Fi──► PC ──Ethernet──► Franka Panda
                                                        (policy)  ◄──arm state───────   (arm control)
```

- **Raspberry Pi 5:** on the robot's hand. It streams both cameras.
- **Laptop:** runs the policy and decides the next moves.
- **PC:** cabled to the arm. It's the only machine that drives it, and it checks every move first.

All three are on the same lab Wi-Fi router.

#### The docs

| Doc | Read it for |
|---|---|
| [deployment/DIRECTIONS.md](deployment/DIRECTIONS.md) | **Start here.** The physical setup, which machine's directions to follow in what order, and a trial run |
| [deployment/control_deploy/DIRECTIONS.md](deployment/control_deploy/DIRECTIONS.md) | Setting up and running the PC (the arm side) |
| [deployment/policy_deploy/DIRECTIONS.md](deployment/policy_deploy/DIRECTIONS.md) | Setting up and running the laptop (cameras and policy) |
| [deployment/policy_deploy/TEST_FRAMES.md](deployment/policy_deploy/TEST_FRAMES.md) | Saving camera images from the rig on any computer, to check them |
| [deployment/PLAN.md](deployment/PLAN.md) | Why it is built this way: the messages between machines, clocks, safety, the failure rules |
| [deployment/STRUCTURE.md](deployment/STRUCTURE.md) | How the code is laid out |

#### Initial directions

1. **Physical setup:**
   - The arm has the Franka Hand, the printed fingers and the sensor rig mounted.
   - The user stop button is within reach.
   - The PC is cabled straight to the Franka controller.
   - The PC, the laptop and the Pi are all on the lab router's Wi-Fi.
2. **One-time software setup:** follow the order table in [deployment/DIRECTIONS.md](deployment/DIRECTIONS.md). It
   alternates between the PC's and the laptop's directions. Both machines must be on the same commit, with the same
   `deployment/contracts/link.yaml`.
3. **A trial run:**
   1. In Desk (the arm's web page), unlock the brakes and activate FCI.
   2. On the PC: `control_deploy run`.
   3. On the rig: press the C button. The LED turns blue.
   4. On the laptop: `policy_deploy check`, then `policy_deploy run`.
   5. Drive it from the laptop's keys: `h` home, `c` start an episode, `y` / `n` end it.

   **Always start the PC before the laptop.**

Until a policy is trained on data from this rig, the laptop runs **`hold_still`**: the arm stays where each episode
started. This tests the cameras, the link and the timing without moving the arm.

#### What the code does at each step

1. **`control_deploy run` (PC).**
   - It first runs its checks:
     - the config;
     - the arm reachable at `172.16.0.2`;
     - the network addresses;
     - the ports.
   - It then starts `franka_control`, a C++ program, the only process connected to the arm. `franka_control`:
     - homes the fingers once;
     - holds the arm still with a 1 kHz Cartesian impedance loop;
     - broadcasts the arm's state (pose, joints, gripper width, mode) 100 times a second;
     - waits for commands from the laptop.
2. **The C button (Pi).** The Pi streams both cameras (640×480, 30 FPS) as JPEG pairs over ZeroMQ. Each pair carries
   the capture times of both images.
3. **`policy_deploy check` (laptop).** It moves nothing. It checks:
   - the Pi's stream arrives;
   - the PC answers;
   - both machines run the same commit and the same limits and TCP definition;
   - the laptop's map from its own clock to the PC's clock agrees.
4. **`policy_deploy run` (laptop).** Starts three separate processes:
   - **frames:** decodes and pairs the camera images. It keeps the last second of them on the laptop's clock and
     prepares them exactly as in training (resize, fisheye correction).
   - **policy:** loads the model and turns images plus the hand's recent motion into the next 16 relative steps.
   - **manager:** runs episodes and the keyboard. It is the only process that talks to the PC.
5. **`h`, then `c`.**
   - `h` asks the PC to move all 7 joints to the home pose.
   - `c` starts an episode. Each policy step then:
     1. takes the newest images;
     2. looks up where the hand was at the moments they were taken, in the PC's state history and on the PC's clock;
     3. asks the policy for its next steps;
     4. turns the first few into absolute fingertip poses, timed 50 ms apart.
   - The manager sends those poses to the PC, which checks each one before the arm moves:
     - its time is not already past;
     - it is inside the workspace box and the tilt cone;
     - its speed is under the limit.

     The PC's 1 kHz loop then moves smoothly between the accepted poses, and the next policy step has already
     started.
6. **Safety throughout.**
   - The laptop sends a heartbeat every 20 ms. If the PC hears nothing for 1 s, it stops and holds the arm.
   - `s` on either machine, or the user stop, stops the arm at any time.
   - Any failure ends the run with one line saying which process failed, what failed, and what to check.

## Simulation

We also validate our solution in simulation as well, under the **rnoro/main-sim** branch. Here we demonstrate a working simulation in MuJoCo, with our sensor setup, as shown below: 

<video src="docs/media/mujoco_pick_place.webm" controls width="100%"></video>

[MuJoCo pick-and-place demo (webm)](docs/media/mujoco_pick_place.webm)

### TBD - More to come soon!