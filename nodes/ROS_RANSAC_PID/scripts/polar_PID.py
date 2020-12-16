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

TARGET_DIST = 3
RATE = 50

class polar_PID():
    def __init__(self):
        print("STARTING POLAR PID NODE")
    
        topic_out = ""
        self.simulate = rospy.get_param('~simulate', True)
        if not self.simulate:
            self.P = 0.01
            self.I = 0.01
            self.D = 0.01
            self.ang_vel_max = 0.04
            self.vel = 0.4
            topic_out = "/frobit/twist"
            self.msgType = TwistStamped

        if self.simulate:
            self.P = 5
            self.I = 0.2
            self.D = 0
            self.ang_vel_max = 8
            self.vel = 0.5
            topic_out = "/cmd_vel"
            self.msgType = Twist
        

        rospy.init_node("wall_distance_PID_controller", anonymous=False)
        topic_in = "laser/dist_to_wall"
        self.subscription = rospy.Subscriber(topic_in, Polar_dist, self.PID)
        self.publisher = rospy.Publisher(topic_out, self.msgType, queue_size=1)
        self.rate = RATE
        rospy.Rate(self.rate)  # or whatever
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
        ctrl += self.I * self.integral_err * 1.0/self.rate
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
            print("In:", dist_diff, "    out:", ctrl)
        self.num += 1
        # if self.time > self.showgraph:
        #     self.showgraph += 1000
        #     plt.plot(self.times, self.dists)
        #     plt.show()
        if self.simulation and self.num % 1000 == 0:
            with open('/media/PID_results/err.npy', 'wb') as f:
               np.save(f, self.dist_diff, allow_pickle=True, fix_imports=True)[source]

def main(args=None):
    PID_node = polar_PID()
    rospy.spin()

if __name__ == '__main__':
    main()
