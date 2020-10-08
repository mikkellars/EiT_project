# Experts in Team - Project
Gazebo Simulation:
1) Put both (mybot_description and mybot_gazebo) in the src folder of your catkin workspace.
2) Build/make your environment.
3) Source your environment.
4) use this command to launch the simulation:

	roslaunch mybot_gazebo mybot_world.launch  

5)open a new terminal in the same directory and source the enviornment.
6) Use this command to move the robot:

	rostopic pub /cmd_vel geometry_msgs/Twist "linear:
  x: 0.1
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.0"


Tip:  Use "Tab Completion" while launching the simulation and giving the movement command.
