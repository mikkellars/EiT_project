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
   * Get methods.
   */
  sensor_msgs::NavSatStatus get_status();
  float get_altitude();
  float get_latitude();
  float get_longitude();

	/*!
	 * Destructor.
	 */
  virtual ~gps_control();

private:
  sensor_msgs::NavSatStatus status;
  float altitude, latitude, longitude;
	ros::NodeHandle nodeHandle_;
  ros::Subscriber gps_sub_;

  void gps_callback(const sensor_msgs::NavSatFixConstPtr& msg);
};

} /* namespace */
