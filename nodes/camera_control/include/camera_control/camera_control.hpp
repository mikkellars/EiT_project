#pragma once

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include "image_save.hpp"
#include "gps_control/gps_control.hpp"

namespace camera_control {

/*!
 * Class containing the camera control
 */
class camera_control {
public:
	/*!
	 * Constructor.
	 */
	camera_control(ros::NodeHandle& nodeHandle);
	/*!
	 * Destructor.
	 */
	virtual ~camera_control();

private:
	ros::NodeHandle nodeHandle_;
	ros::Subscriber cam_sub_;
  cv_bridge::CvImagePtr cv_ptr_;

  void cam_img_callback(const sensor_msgs::Image& msg);
};

} /* namespace */
