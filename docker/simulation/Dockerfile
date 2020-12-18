FROM ros:melodic-perception
#FROM px4io/px4-dev-ros-melodic

#Update and install
RUN apt-get update && apt-get install -y \
	git \
	cmake \
	python3-pip \
	python-catkin-tools \
	wget \
	curl \
	libbullet-dev \
	libsdl-image1.2-dev \
	libsdl-dev \
	ros-melodic-desktop-full \
	#ros-melodic-gazebo-ros \
	ros-melodic-xacro \
	ros-melodic-tf2-sensor-msgs \
	ros-melodic-controller-manager \
	ros-melodic-diagnostic-updater \
	ros-melodic-roslint \
	ros-melodic-map-msgs \ 
	ros-melodic-move-base-msgs \
	ros-melodic-joint-state-publisher \
	ros-melodic-robot-state-publisher \
	nano

RUN pip3 install --upgrade pip
RUN pip3 install catkin_pkg pyyaml rospkg 

# Copying the nodes to catkin
COPY nodes/chain_link_fence /catkin_ws/src/chain_link_fence 
COPY nodes/hole_1_fence /catkin_ws/src/hole_1_fence
COPY nodes/hole_2_fence /catkin_ws/src/hole_2_fence
COPY nodes/hole_3_fence /catkin_ws/src/hole_3_fence
COPY nodes/hole_4_fence /catkin_ws/src/hole_4_fence
COPY nodes/hole_5_fence /catkin_ws/src/hole_5_fence
COPY nodes/husky /catkin_ws/src/husky
COPY nodes/mybot_control /catkin_ws/src/mybot_control
COPY nodes/mybot_description /catkin_ws/src/mybot_description
COPY nodes/mybot_gazebo /catkin_ws/src/mybot_gazebo
COPY nodes/navigation /catkin_ws/src/navigation

# Copying models to .gazebo models folder
RUN mkdir -p /.gazebo/models
RUN cp -ar /catkin_ws/src/chain_link_fence /.gazebo/models
RUN cp -ar /catkin_ws/src/hole_1_fence /.gazebo/models
RUN cp -ar /catkin_ws/src/hole_2_fence /.gazebo/models
RUN cp -ar /catkin_ws/src/hole_3_fence /.gazebo/models
RUN cp -ar /catkin_ws/src/hole_4_fence /.gazebo/models
RUN cp -ar /catkin_ws/src/hole_5_fence /.gazebo/models

WORKDIR /catkin_ws
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; catkin_init_workspace /catkin_ws/src'

RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; catkin_make'
RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc

#RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; rosrun detection detection.py'

