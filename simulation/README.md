# Simulation Gazebo-ROS
This folder contains the Gazebo-ROS package to simulate the fence inspecting robot. 

## Installation Ubuntu 20.04
The following packages needs to be compiled/installed in order to simulate the robot.

* Hector Gazebo plugins: Used for simulating a GPS

    1) Start by clonning from the following repository into your catkin_ws:
    ```console
    foo@bar:~/path/to/catkin_ws$ git clone https://github.com/tu-darmstadt-ros-pkg/hector_gazebo/tree/melodic-devel
    ```
    
    2) Move the folder hector_gazebo_plugins out from the root folder and delete the root folder.

    3) Build/make your environment.

## Installation Ubuntu 18.04
The following packages needs to be compiled/installed in order to simulate the robot.

* Hector Gazebo plugins: Used for simulating a GPS
    1) Run the following package install command:
        ```console
        foo@bar:~$ sudo apt-get install ros-melodic-hector-gazebo-plugins
        ```

## Run the simulation
1) Put both mybot_description and mybot_gazebo in the src folder of your catkin workspace.
2) Build/make your environment.
3) Source your environment.
    ```console
    foo@bar:~$ source /path/to/catkin_ws/devel/setup.bash
    ```
4) Use this command to launch the simulation:
    ```console
    foo@bar:~/path/to/catkin_ws$ roslaunch mybot_gazebo mybot_world.launch  
    ```
5) Use this command to move the robot:

	rostopic pub /cmd_vel geometry_msgs/Twist "linear:
  x: 0.1
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.0"


Tip:  Use "Tab Completion" while launching the simulation and giving the movement command.

