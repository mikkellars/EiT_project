"""Python file for writing geotag to image.
Inspiration from https://answers.ros.org/question/332491/geotag-image-opencv-mavros/ 
"""

import message_filters
import cv2
import rospy
import os
import piexif
from sensor_msgs.msg import NavSatFix, Image
from cv_bridge import CvBridge, CvBridgeError
from datetime import datetime
from fractions import Fraction

class ImageSave:
    def __init__(self, save_dir):
        self.save_dir = save_dir

        # GPS attributes
        self.status = None
        self.altitude = None
        self.latitude = None
        self.longitude = None
        
        # Image attributes
        self.bridge = CvBridge()
        self.img = None

        MSG_QUEUE_MAXLEN = 50

        # Subscribers
        image_sub = message_filters.Subscriber("/rrbot/camera1/image_raw", Image)
        gps_sub = message_filters.Subscriber("/fix", NavSatFix)

        # Ensures to only syncronize message from multiple sources with the same timestamp. 
        ts = message_filters.ApproximateTimeSynchronizer([gps_sub, image_sub], queue_size=MSG_QUEUE_MAXLEN, slop=1)
        ts.registerCallback(self.__callback)

       


    def __callback(self, gps_data, cam_data):
        # log some info about the image topic
        rospy.loginfo("img time_ %i", cam_data.header.stamp.secs)
        rospy.loginfo("gps time: %i", gps_data.header.stamp.secs)


        # GPS data
        self.status = gps_data.status
        self.altitude = gps_data.altitude
        self.latitude = gps_data.latitude
        self.longitude = gps_data.longitude
        self.gps_data_header_stamp = gps_data.header.stamp

        # Img data
        try:
            self.img = self.bridge.imgmsg_to_cv2(cam_data, "bgr8")
        except CvBridgeError as e:
            print(e)

        self.__geotag_and_save()

        


    def __geotag_and_save(self):
        # Exif data geotag
        picGpsTimeFlt = self.__rostime2floatSecs(self.gps_data_header_stamp)
        picNameTimeString = datetime.utcfromtimestamp(picGpsTimeFlt).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        picSaveName = picNameTimeString + ".jpg"

        self.file_name = os.path.join(self.save_dir, picSaveName)
        cv2.imwrite(self.file_name , self.img) # saving image for afterward writing geotag

      #  picGpsTimeFlt = self.rostime2floatSecs(gps_data.header.stamp)
        gpsTimeString = datetime.utcfromtimestamp(picGpsTimeFlt).strftime('%Y:%m:%d %H:%M:%S')

        self.__set_gps_location(gpsTime=gpsTimeString)
    


    def __rostime2floatSecs(self, rostime):
        return rostime.secs + (rostime.nsecs / 1000000000.0)

    def __set_gps_location(self, gpsTime):
        """Adds GPS position as EXIF metadata
        Keyword arguments:
        file_name -- image file
        lat -- latitude (as float)
        lng -- longitude (as float)
        altitude -- altitude (as float)
        """
        lat_deg = self.__to_deg(self.latitude, ["S", "N"])
        lng_deg = self.__to_deg(self.longitude, ["W", "E"])
        
        print(self.latitude)
        print(self.longitude)

        exiv_lat = (self.__change_to_rational(lat_deg[0]), self.__change_to_rational(lat_deg[1]), self.__change_to_rational(lat_deg[2]))
        exiv_lng = (self.__change_to_rational(lng_deg[0]), self.__change_to_rational(lng_deg[1]), self.__change_to_rational(lng_deg[2]))

        gps_ifd = {
            piexif.GPSIFD.GPSVersionID: (2, 0, 0, 0),
            piexif.GPSIFD.GPSAltitudeRef: 0,
            piexif.GPSIFD.GPSAltitude: self.__change_to_rational(round(self.altitude)),
            piexif.GPSIFD.GPSLatitudeRef: lat_deg[3],
            piexif.GPSIFD.GPSLatitude: exiv_lat,
            piexif.GPSIFD.GPSLongitudeRef: lng_deg[3],
            piexif.GPSIFD.GPSLongitude: exiv_lng,
        }

        exif_dict = {"GPS": gps_ifd}
        exif_bytes = piexif.dump(exif_dict) # BUG here in simulation maybe working in real life. 
        piexif.insert(exif_bytes, self.file_name)


    @staticmethod
    def __to_deg(value, loc):
        """convert decimal coordinates into degrees, munutes and seconds tuple
        Keyword arguments: value is float gps-value, loc is direction list ["S", "N"] or ["W", "E"]
        return: tuple like (25, 13, 48.343 ,'N')
        """
        if value < 0:
            loc_value = loc[0]
        elif value > 0:
            loc_value = loc[1]
        else:
            loc_value = ""

        abs_value = abs(value)
        deg = int(abs_value)
        t1 = (abs_value - deg) * 60
        min = int(t1)
        sec = round((t1 - min) * 60, 5)
        return (deg, min, sec, loc_value)

    @staticmethod
    def __change_to_rational(number):
        """convert a number to rantional
        Keyword arguments: number
        return: tuple like (1, 2), (numerator, denominator)
        """
        f = Fraction(str(number))
        return (f.numerator, f.denominator)

