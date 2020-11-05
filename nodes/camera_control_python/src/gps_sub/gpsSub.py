import cv2
import rospy
from sensor_msgs.msg import NavSatFix

class gpsSub():
    def __init__(self): 
        
        self.image_sub = rospy.Subscriber("/fix", NavSatFix, self.callback)
        
    def callback(self, data):
        print('t')