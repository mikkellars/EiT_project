#include <ros/ros.h>
#include "gps_control/gps_control.hpp"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "gps_control");
  ros::NodeHandle nodeHandle("~");
  gps_control::gps_control gps_control(nodeHandle);

  ros::spin();

  return 0;
}
