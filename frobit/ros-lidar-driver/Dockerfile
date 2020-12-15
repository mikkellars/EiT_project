FROM ros:melodic

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    python3-pip \
    python-catkin-tools \
    ros-melodic-roslint \
    ros-melodic-driver-common

RUN pip3 install --upgrade pip
RUN pip3 install \
    catkin_pkg \
    serial \
    rospkg

WORKDIR /catkin_ws

RUN git clone https://github.com/ros-drivers/hokuyo_node.git /catkin_ws/src/hokuyo_node
RUN /bin/bash -c "source /opt/ros/melodic/setup.bash; catkin_make"
# RUN /bin/bash -c "source /catkin_ws/devel/setup.bash"

RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc

COPY hokuyo.launch /catkin_ws/src/launch/hokuyo.launch
COPY run_driver.sh /catkin_ws/run_driver.sh
