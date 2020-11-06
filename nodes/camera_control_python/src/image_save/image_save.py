import message_filters
import cv2
from sensor_msgs.msg import NavSatFix, Image
from cv_bridge import CvBridge, CvBridgeError


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

        # Subscribers
        image_sub = message_filters.Subscriber("/rrbot/camera1/image_raw", Image)
        gps_sub = message_filters.Subscriber("/fix", NavSatFix)

        # Ensures to only syncronize message from multiple sources with the same timestamp. 
        ts = message_filters.ApproximateTimeSynchronizer([gps_sub, image_sub], 1, 1)
        ts.registerCallback(self.__callback)

       


    def __callback(self, gps_data, cam_data):
        # GPS data
        self.status = gps_data.status
        self.altitude = gps_data.altitude
        self.latitude = gps_data.latitude
        self.longitude = gps_data.longitude

        # Img data
        self.img = self.bridge.imgmsg_to_cv2(cam_data, "bgr8")
        

    def __geotag_and_save(self):
    # Exif data geotag