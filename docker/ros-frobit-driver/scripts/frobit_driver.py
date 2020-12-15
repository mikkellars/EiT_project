#!/usr/bin/env python
import rospy
from numpy import sign
from std_msgs.msg import Float64
from hardware import * 

class FrobitDriver():
    def __init__(self):
        rospy.init_node('frobit_node', anonymous=True)

        self.left_subscriber = rospy.Subscriber("/frobit/left_pwm", Float64, self.leftCallback)
        self.right_subscriber = rospy.Subscriber("/frobit/right_pwm", Float64, self.rightCallback)

        self.left_publisher = rospy.Publisher("/frobit/left_encoder", Float64, queue_size = 1)
        self.right_publisher = rospy.Publisher("/frobit/right_encoder", Float64, queue_size = 1)
        self.left_msg = Float64()
        self.right_msg = Float64()

        self.left_pwm = 0.0 # percent
        self.right_pwm = 0.0 # percent
        self.left_previous_pwm = 0.0
        self.right_previous_pwm = 0.0
        self.max_change_per_update = 10.0

        self.left_encoder_value = 0
        self.right_encoder_value = 0
        self.left_previous_encoder_value = 0
        self.right_previous_encoder_value = 0
        self.meters_per_tick = (0.22*3.14) / 980 # wheel_circumference (0.22 m * pi) / 980 ticks per revolution

        self.left_drive = None
        self.right_drive = None
        self.left_encoder = None
        self.right_encoder = None

        self.update_rate = 10

        self.initHardware()

    def initHardware(self):
        self.left_encoder = Encoder(20,21)
        self.right_encoder = Encoder(12,16)
        self.left_drive = Drive(6,1,1)
        self.right_drive = Drive(5,0,0)
        self.left_drive.setSpeed( self.left_pwm )
        self.right_drive.setSpeed( self.left_pwm )
        self.left_drive.forward()
        self.right_drive.forward()

    def leftCallback(self,msg):
        self.left_pwm = msg.data
    
    def rightCallback(self,msg):
        self.right_pwm = msg.data

    def updateEncoders(self):
        #TODO: Handle overflow
        self.left_encoder_value = - self.left_encoder.getValue()
        self.right_encoder_value = self.right_encoder.getValue()

    def setSpeeds(self):
        left_pwm = self.left_pwm# self.left_previous_pwm
        right_pwm = self.right_pwm#self.right_previous_pwm
        
        #left_diff = self.left_previous_pwm - self.left_pwm 
        #if left_diff > self.max_change_per_update:
        #    left_pwm += self.max_change_per_update
        #elif left_diff < self.max_change_per_update:
        #    left_pwm -= self.max_change_per_update
        #
        #right_diff = self.right_previous_pwm - self.right_pwm 
        #if right_diff > self.max_change_per_update:
        #    right_pwm += self.max_change_per_update
        #elif right_diff < self.max_change_per_update:
        #    right_pwm -= self.max_change_per_update

        self.left_drive.setSpeed( self.left_pwm )
        self.right_drive.setSpeed( self.right_pwm )

        self.left_previous_pwm = left_pwm
        self.right_previous_pwm = right_pwm


    def spin(self):
        r = rospy.Rate(self.update_rate)
        while not rospy.is_shutdown(): 
            self.updateEncoders()
           
            self.left_msg.data = (self.left_encoder_value - self.left_previous_encoder_value) * self.meters_per_tick
            self.right_msg.data = (self.right_encoder_value - self.right_previous_encoder_value) * self.meters_per_tick
            
            self.left_publisher.publish(self.left_msg)
            self.right_publisher.publish(self.right_msg)
            
            self.left_previous_encoder_value = self.left_encoder_value
            self.right_previous_encoder_value = self.right_encoder_value

            self.setSpeeds()

            r.sleep()

