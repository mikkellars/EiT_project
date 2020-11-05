#include "gps_control/gps_control.hpp"

namespace gps_control {

gps_control::gps_control(ros::NodeHandle& nodeHandle) :
      nodeHandle_(nodeHandle)
{
  gps_sub_ = nodeHandle_.subscribe("/fix", 10, &gps_control::gps_callback, this);
}

gps_control::~gps_control()
{
}

sensor_msgs::NavSatStatus gps_control::get_status()
{
  return status;
}

float gps_control::get_altitude()
{
  return altitude;
}

float gps_control::get_latitude()
{
  return latitude;
}

float gps_control::get_longitude()
{
  return longitude;
}

/* private functions */
void gps_control::gps_callback(const sensor_msgs::NavSatFixConstPtr& msg)
{
  status = msg->status;
  altitude = msg->altitude;
  latitude = msg->latitude;
  longitude = msg->longitude;
//  ROS_INFO("Status [%d]", msg->status);
//  ROS_INFO("Altitude [%f]", msg->altitude);
//  ROS_INFO("Altitude [%f]", msg->latitude);
//  ROS_INFO("Altitude [%f]", msg->longitude);
}


} /* namespace */
