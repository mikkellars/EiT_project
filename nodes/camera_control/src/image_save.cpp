#include "camera_control/image_save.hpp"

namespace image_save {

image_save::image_save(cv::Mat img)
{
  in_img = img;
}

image_save::~image_save()
{
}

} /* namespace */
