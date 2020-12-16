#!/usr/bin/env python3
import numpy as np
from sklearn import linear_model
import rospy
import cv2 as cv
import geometry_msgs
from geometry_msgs.msg import TwistStamped
from dist_ransac.msg import Polar_dist
import matplotlib.pyplot as plt
from time import time

TARGET_DIST = 3
RATE = 50

class polar_PID():
    def __init__(self):
        print("STARTING POLAR PID NODE")
    
        topic_out = ""
        simulate = rospy.get_param('~simulate', True)
        if not simulate:
            self.P = 0.01
            self.I = 0.01
            self.D = 0.01
            self.ang_vel_max = 0.04
            self.vel = 0.4
            topic_out = "/frobit/twist"

        if simulate:
            self.P = 4
            self.I = 0
            self.D = 0
            self.ang_vel_max = 2
            self.vel = 0.2
            topic_out = "/velocity_controller/cmd_vel"
        

        rospy.init_node("wall_distance_PID_controller", anonymous=False)
        topic_in = "laser/dist_to_wall"
        self.subscription = rospy.Subscriber(topic_in, Polar_dist, self.PID)
        self.publisher = rospy.Publisher(topic_out, TwistStamped, queue_size=1)
        self.rate = RATE
        rospy.Rate(self.rate)  # or whatever
        self.last_err = 0
        self.integral_err = 0

        #for recording:
        self.dists = []
        self.times = []
        self.time = time()
        #self.showgraph = 1000

        self.print_num = 0

    def PID(self, msg):
        dist = msg.dist
        angle = msg.angle

        time_diff = time() - self.time
        self.time = time()

        def right_or_left(ang):
            if (ang > 0):
                return "left"
            return "right"

        dist_diff = TARGET_DIST - dist
        if right_or_left(angle) == "left":  #account for differnece in direction
            dist_diff = (-dist_diff)

        self.integral_err += dist_diff * time_diff

        dist_deriv = (dist_diff - self.last_err) / time_diff

        ctrl = self.P * dist_diff
        ctrl += self.I * self.integral_err * 1.0/self.rate
        ctrl += self.D * dist_deriv

        if ctrl > self.ang_vel_max:
            ctrl = self.ang_vel_max
        elif ctrl < -self.ang_vel_max:
            ctrl = -self.ang_vel_max

        self.last_err = dist_diff

        rmsg = TwistStamped()
        rmsg.twist.linear.x = self.vel
        # rmsg.linear.y = 0
        # rmsg.linear.z = 0

        # rmsg.angular.x = 0
        # rmsg.angular.y = 0
        rmsg.twist.angular.z = ctrl
        rmsg.header.stamp = rospy.Time.now()

        self.publisher.publish(rmsg)

        self.dists.append(dist_diff)
        self.times.append(self.time)

        if self.print_num % 10 == 0:
            print("In:", dist_diff, "    out:", ctrl)
        self.print_num += 1
        # if self.time > self.showgraph:
        #     self.showgraph += 1000
        #     plt.plot(self.times, self.dists)
        #     plt.show()


def main(args=None):
    PID_node = polar_PID()
    rospy.spin()


if __name__ == '__main__':
    main()
