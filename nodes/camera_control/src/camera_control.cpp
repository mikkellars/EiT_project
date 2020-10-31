#include "camera_control/camera_control.hpp"

namespace camera_control {

camera_control::camera_control(ros::NodeHandle& nodeHandle) :
      nodeHandle_(nodeHandle)
{
  cam_sub_ = nodeHandle_.subscribe("/rrbot/camera1/image_raw", 10, &camera_control::cam_img_callback, this);
}

void camera_control::cam_img_callback(const sensor_msgs::Image& msg)
{
    try
    {
      // Pointer used for the conversion from a ROS message to an OpenCV-compatible image
      cv_bridge::CvImagePtr cv_ptr_;
      cv_ptr_ = cv_bridge::toCvCopy(msg, "bgr8");

      // Store the values of the OpenCV-compatible image into the current_frame variable
      cv::Mat current_frame = cv_ptr_->image;

      // Display the current frame
      cv::imshow("cam1", current_frame);

      // Display frame for 30 milliseconds, because of 30 fps
      cv::waitKey(30);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

camera_control::~camera_control()
{
}

} /* namespace */
