#include "gps_control/gps_control.hpp"

namespace gps_control {

gps_control::gps_control(ros::NodeHandle& nodeHandle) :
      nodeHandle_(nodeHandle)
{
  gps_sub_ = nodeHandle_.subscribe("/fix", 10, &gps_control::gps_callback, this);
}

void gps_control::gps_callback(const sensor_msgs::NavSatFixConstPtr& msg)
{
  ROS_INFO("Altitude [%f]", msg->altitude);
  ROS_INFO("Altitude [%f]", msg->latitude);
  ROS_INFO("Altitude [%f]", msg->longitude);
}

gps_control::~gps_control()
{
}

} /* namespace */
