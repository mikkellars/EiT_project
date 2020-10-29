#include "camera_control/image_save.hpp"
#include "gps_control/gps_control.hpp"

namespace image_save {

/* Constructer and deconstructer */
image_save::image_save(std::string save_folder)
{
  s_folder = save_folder;
}

image_save::~image_save()
{
}

/* public functions */
void image_save::save_data(cv::Mat img, float latitude, float longitude, float altitude)
{
  write_image(img);
  write_metadata(latitude, longitude, altitude);
  img_number++;
}

/* private functions */
void image_save::write_metadata(float latitude, float longitude, float altitude)
{
  std::ofstream outfile;
  ROS_INFO("[%s]", s_folder.c_str());
  std::string file_path(s_folder + "/test.txt");
  if (file_exists(file_path))
  {
    ROS_INFO("Appending to [%s]", file_path.c_str());
    outfile = std::ofstream(file_path, std::ios_base::app);
  }
  else
  {
    ROS_INFO("Creating and writing to [%s]", file_path.c_str());
    outfile = std::ofstream(file_path);
    outfile << "NAME\t|\tlatitude\t|\tlongitude\t|\taltitude\t" << std::endl;
  }

  outfile << "img_" + std::to_string(img_number) + ":" + std::to_string(latitude) + ":" + std::to_string(longitude) + ":" + std::to_string(altitude) << std::endl;
  outfile.close();
  ROS_INFO("Writing metadata");
}


void image_save::write_image(cv::Mat img)
{
  std::string file_path = s_folder + "/img_" + std::to_string(img_number) + ".jpg";
  bool isSuccess = imwrite(file_path, img); //write the image to a file as JPEG
  if (isSuccess == false)
  {
    ROS_ERROR("Could save image");
  }
}


inline bool image_save::file_exists(const std::string& name)
{
  struct stat buffer;
  return (stat (name.c_str(), &buffer) == 0);
}

} /* namespace */
