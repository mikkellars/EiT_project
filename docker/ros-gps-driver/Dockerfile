FROM ros:melodic

# Update and install
RUN apt-get update && apt-get install -y \
        git \
        cmake \
        python3-pip \
        python-catkin-tools \
        ros-melodic-roslint \
	ros-melodic-nmea-navsat-driver \
	ros-melodic-serial

RUN pip3 install --upgrade pip
RUN pip3 install \
	catkin_pkg \
	rospkg

# RUN git clone https://github.com/ros-drivers/nmea_navsat_driver /catkin_ws/src/nmea_navsat_driver
# WORKDIR /catkin_ws

RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
# RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; catkin_init_workspace /catkin_ws/src'

# RUN /bin/bash -c '. /opt/ros/melodic/setup.bash; catkin_make --pkg nmea_navsat_driver'
# RUN echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc

RUN groups root
RUN usermod -aG dialout root

# RUN dmesg | grep tty
# RUN rosrun nmea_navsat_driver nmea_serial_driver _port:=/dev/ttyUSB0

COPY scripts/ /scripts/
