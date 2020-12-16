#! /usr/bin/env python3
import rospy
from image_save.image_save import ImageSave


def main():
    rospy.init_node('camera_control', anonymous=True)
    
    imgsave = ImageSave("/home/mikkel/Downloads/gps_test")

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()