# FROM ros:melodic-robot
FROM ros:melodic

RUN apt update && apt upgrade -y
RUN apt update && apt install -y apt-utils build-essential git cmake python3-pip nano
RUN apt update && apt install -y ros-melodic-hector-gazebo-plugins

RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install sklearn
RUN pip3 install matplotlib
RUN pip3 install catkin-tools

RUN mkdir -p /catkin_ws/src
COPY simulation/mybot_description /catkin_ws/src/mybot_description/
COPY simulation/mybot_gazebo /catkin_ws/src/mybot_gazebo/
# COPY ROS_RANSAC_PID /catkin_ws/src/ROS_RANSAC_PID/

WORKDIR /catkin_ws
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; catkin_init_workspace /catkin_ws/src'

RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; catkin_make'
RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc