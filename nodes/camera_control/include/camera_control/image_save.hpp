#pragma once

#include <opencv2/highgui/highgui.hpp>

namespace image_save {
/*!
 * Class containing the image save to save images comming from the camera
 */
class image_save {
public:
  /*!
   * Constructor.
   */
  image_save(cv::Mat img);
  /*!
   * Destructor.
   */
  virtual ~image_save();

private:
  cv::Mat in_img;
};

} /* namespace */
