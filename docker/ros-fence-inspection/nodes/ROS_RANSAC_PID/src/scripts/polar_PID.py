#!/usr/bin/env python3
import numpy as np
from sklearn import linear_model
import rospy
import cv2 as cv
import geometry_msgs
from geometry_msgs.msg import TwistStamped
from geometry_msgs.msg import Twist
from dist_ransac.msg import Polar_dist
import matplotlib.pyplot as plt
from time import time

TARGET_DIST = 1.75
RATE = 50

class polar_PID():
    def __init__(self, simulate):
        self.simulate = simulate

        if not self.simulate:
            self.P = 0.015
            self.I = 0.001
            self.D = 0.01
            self.ang_vel_max = 0.04
            self.vel = 0.5
            topic_out = "/frobit/twist"
            self.msgType = TwistStamped

        if self.simulate:
            self.P = 12
            self.I = 0.05
            self.D = 15
            self.ang_vel_max = 8
            self.vel = 0.5
            topic_out = "/cmd_vel"
            self.msgType = Twist
        

        
        topic_in = "laser/dist_to_wall"
        self.subscription = rospy.Subscriber(topic_in, Polar_dist, self.PID)
        self.publisher = rospy.Publisher(topic_out, self.msgType, queue_size=1)
        
        self.last_err = 0
        self.integral_err = 0

        #for recording:
        self.dists = []
        self.times = []
        self.time = time()
        #self.showgraph = 1000

        self.num = 0


    def PID(self, msg):
        dist = msg.dist
        angle = msg.angle

        time_diff = time() - self.time
        self.time = time()

        def right_or_left(ang):
            if (ang > 0):
                return "left"
            return "right"

        # update error terms
        dist_diff = TARGET_DIST - dist
        #if angle > 0: 
        #    dist_diff = (-dist_diff)
        self.integral_err += dist_diff * time_diff
        dist_deriv = (dist_diff - self.last_err) / time_diff

        # calculate control value
        ctrl = self.P * dist_diff
        ctrl += self.I * self.integral_err 
        ctrl += self.D * dist_deriv

        # limit to max/min values
        if ctrl > self.ang_vel_max:
            ctrl = self.ang_vel_max
        elif ctrl < -self.ang_vel_max:
            ctrl = -self.ang_vel_max

        self.last_err = dist_diff

        #skip sending on the first iteration
        if self.num == 0:
            self.num += 1
            return

        #make and send the message
        if self.simulate:
            rmsg = Twist()
            rmsg.linear.x = self.vel
            rmsg.angular.z = ctrl
            self.publisher.publish(rmsg)

        if not self.simulate:
            rmsg = TwistStamped()
            rmsg.twist.linear.x = self.vel
            rmsg.twist.angular.z = ctrl
            rmsg.header.stamp = rospy.Time.now()
            self.publisher.publish(rmsg)

        self.dists.append(dist_diff)
        self.times.append(self.time)

        #print 
        if self.num % 10 == 0:
            print("In: ", np.round(dist_diff, 5), "    out: ", np.round(ctrl, 5), "	P:", np.round(self.P * dist_diff, 5), "	I:", np.round(self.I * self.integral_err, 5), "	P:", np.round(self.D * dist_deriv, 5))
        self.num += 1
        # if self.time > self.showgraph:
        #     self.showgraph += 1000
        #     plt.plot(self.times, self.dists)
        #     plt.show()
        # if self.simulate and self.num % 10 == 0:
        #    with open('/assets/images/laser_scan/dist_err.npy', 'wb') as f:
        #       np.save(f, self.dists, allow_pickle=True)

# def main(args=None):
#     PID_node = polar_PID()
#     rospy.spin()

# if __name__ == '__main__':
#     main()
