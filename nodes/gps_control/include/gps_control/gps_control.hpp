#pragma once

#include <ros/ros.h>
#include <sensor_msgs/NavSatFix.h>


namespace gps_control {

/*!
 * Class containing the camera control
 */
class gps_control {
public:
	/*!
	 * Constructor.
	 */
  gps_control(ros::NodeHandle& nodeHandle);
	/*!
	 * Destructor.
	 */
  virtual ~gps_control();

private:
	ros::NodeHandle nodeHandle_;
  ros::Subscriber gps_sub_;

  void gps_callback(const sensor_msgs::NavSatFixConstPtr& msg);
};

} /* namespace */
