# Alpcer3d_dualCam

## 1. Preparation

### 1.1 OS requirements

  * Ubuntu 20.04 for ROS2 Foxy;

  **Tips:**

  Colcon is a build tool used in ROS2.

  How to install colcon: [Colcon installation instructions](https://docs.ros.org/en/foxy/Tutorials/Beginner-Client-Libraries/Colcon-Tutorial.html)

### 1.2 Install ROS2

For ROS2 Foxy installation, please refer to:
[ROS Foxy installation instructions](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html)

Desktop-Full installation is recommend.

### 1.3 Other dependencies

  OpenCV(4.10.0)
  OpenMP
  
### 1.4 Camera params files

  Put the camera params files in the path: /camera/, include camera_params_le.xml/camera_params_re.xml/camera_stereo.xml

## 2. Build & Run Alpcer3d_dualCam

### 2.1 Clone Alpcer3d_dualCam source code:

```shell
git clone https://github.com/Alpcer/Alpcer3d_dualCam.git ws_livox/src/depth_camera
```

  **Note :**

  Be sure to clone the source code in a '[work_space]/src/' folder (as shown above), otherwise compilation errors will occur due to the compilation tool restriction.

### 2.3 Build the Alpcer3d_dualCam:

#### For ROS2 Foxy:
```shell
cd ws_livox
source /opt/ros/foxy/setup.sh
colcon build --packages-select depth_camera
```

### 2.4 Run Alpcer3d_dualCam:

#### For ROS2:
```shell
cd ws_livox
source install/setup.sh
ros2 run depth_camera depth_camera_node --ros-args -p resolution:=3840x1080 -p undistort:=true
```

Press Ctrl+C to stop at the end.

If you want to see point cloud in rviz2, in another shell:

```shell
source /opt/ros/foxy/setup.sh
rviz2
```
