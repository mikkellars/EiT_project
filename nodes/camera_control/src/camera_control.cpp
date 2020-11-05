#include "camera_control/camera_control.hpp"

namespace camera_control {

camera_control::camera_control(ros::NodeHandle& nodeHandle) :
      nodeHandle_(nodeHandle)
{
  cam_sub_ = nodeHandle_.subscribe("/rrbot/camera1/image_raw", 10, &camera_control::cam_img_callback, this);
  gps_control::gps_control gps(nodeHandle);
  ROS_INFO("[%d]",gps.get_status().status);
  if (gps.get_status().status == -1) // STATUS PROBERLY ALSO ALWAYS 0 !!
    ROS_ERROR("No GPS Signal");
  else if (gps.get_status().status == 0) // GPS signal present
  {
    cv::Mat img = imread("/home/mikkel/Downloads/coral.jpg", cv::IMREAD_COLOR);
    std::string s_f("/home/mikkel/Downloads/test");
    image_save::image_save img_save(s_f);
    for (size_t var = 0; var < 5; ++var) {
      ROS_INFO("[%f]", gps.get_latitude());
      img_save.save_data(img, gps.get_latitude(), gps.get_longitude(), gps.get_altitude()); // WRITES OUT 0.00000 !!
    }
  }



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
