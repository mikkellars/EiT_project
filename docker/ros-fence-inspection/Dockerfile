FROM ros:melodic

#Update and install
RUN apt-get update && apt-get install -y \
	git \
	cmake \
	python3-pip \
	python-catkin-tools \
	ros-melodic-vision-opencv \
	ros-melodic-cv-bridge \
	python3-opencv \
	wget \
	curl \
	zip \
	apt-transport-https \
	nano

RUN apt-get update && apt-get install -y \
	python3-matplotlib \
	python3-sklearn 
	#python3-pillow

RUN pip3 install --upgrade pip
RUN pip3 install catkin_pkg pyyaml rospkg # pillow==5.3.0 

# Install Tensorflow Edge CPU libraries
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list \
&& curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update
RUN apt-get install -y python3-pycoral


# ARMv7 PLATFORM Tensorflow lite runtime
# RUN pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp36-cp36m-linux_armv7l.whl
# RUN pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/pycoral-1.0.0-cp36-cp36m-linux_armv7l.whl


# x86 PLATFORM Tensorflow lite runtime
RUN pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp36-cp36m-linux_x86_64.whl 
RUN pip3 install https://github.com/google-coral/pycoral/releases/download/release-frogfish/pycoral-1.0.0-cp36-cp36m-linux_x86_64.whl

# Copying the nodes
COPY nodes/detection /catkin_ws/src/detection
COPY nodes/ROS_RANSAC_PID /catkin_ws/src/ROS_RANSAC_PID
COPY nodes/ico_fence_following /catkin_ws/src/ico_fence_following

WORKDIR /catkin_ws
RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; catkin_init_workspace /catkin_ws/src'

RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; catkin_make'
RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc

#RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; rosrun detection detection.py'

