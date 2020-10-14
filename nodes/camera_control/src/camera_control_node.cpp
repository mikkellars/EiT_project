#include <ros/ros.h>
#include "camera_control/camera_control.hpp"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "camera_control");
  ros::NodeHandle nodeHandle("~");
  camera_control::camera_control camera_control(nodeHandle);

  ros::spin();
  cv::destroyWindow("cam1");
  return 0;
}
