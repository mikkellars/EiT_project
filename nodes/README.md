# ROS Nodes
This folder contains all the ROS nodes used by the fence inspecting robot.

## Installation
The following packages needs to be compiled/installed in order to run the fence inspecting robot in real life.

* Hokuyo laser: Follow the guide [Integrating the Hokuyo URG Lidar with ROS](https://idorobotics.com/2018/11/02/integrating-the-hokuyo-urg-lidar-with-ros/)

* GPS module: 

    1) Start by clonning from the following repository into your catkin_ws:
    ```console
    foo@bar:~/path/to/catkin_ws$ git clone https://github.com/ros-drivers/nmea_navsat_driver 
    ```

    2) Build/make your environment.

    3) Check if your account can access "dialout" by:
    ```console
    foo@bar:~$ groups <username>
    ```

    4) If "dailout" is not included in the output run the following and restart your PC:
    ```console
    foo@bar:~$ sudo usermod -aG dialout <username>
    ```
    
    5) Check which USB port the GPS is connected to by connecting the GPS and run the command below:
    ```console
    foo@bar:~$ dmesg | grep tty
    ```

    6) The port should be "ttyUSBx" where "x" is the USB port connect to.
    

    7) Test by connecting the GPS and run the following:
    ```console
    foo@bar:~/path/to/catkin_ws$ roscore
    foo@bar:~/path/to/catkin_ws$ rosrun nmea_navsat_driver nmea_serial_driver _port:=/dev/ttyUSBx
    foo@bar:~/path/to/catkin_ws$ rostopic echo /fix
    ```

    8) The out should contain the output from the GPS, e.g. latitude, longitude and altitude, which are NaN if there are no GPS signal.
