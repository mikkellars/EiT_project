#pragma once

#include <opencv2/highgui/highgui.hpp>
#include <sys/stat.h>
#include <unistd.h>
#include <string>
#include <fstream>
#include <string>

namespace image_save {
/*!
 * Class containing the image save to save images comming from the camera
 */
class image_save {
public:
  /*!
   * Constructor.
   */
  image_save(std::string save_folder);
  void save_data(cv::Mat img, float latitude, float longitude, float altitude);
  /*!
   * Destructor.
   */
  virtual ~image_save();

private:
  void write_metadata(float latitude, float longitude, float altitude);
  void write_image(cv::Mat img);
  inline bool file_exists(const std::string& name);
  std::string s_folder;
  int img_number = 0;
};

} /* namespace */
