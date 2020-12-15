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

VEL = 0.4
AV_MAX = 0.04
TARGET_DIST = 1
P = 0.01
I = 0.01
D = 0.01
RATE = 10

class polar_PID():
    def __init__(self):
        print("STARTING POLAR PID NODE")
        rospy.init_node("wall_distance_PID_controller", anonymous=False)
        topic_in = "laser/dist_to_wall"
        self.subscription = rospy.Subscriber(topic_in, Polar_dist, self.PID)
        topic_out = "/frobit/twist"
        self.publisher = rospy.Publisher(topic_out, TwistStamped, queue_size=1)
        self.rate = RATE
        rospy.Rate(self.rate)  # or whatever
        self.P = P
        self.I = I
        self.D = D
        self.last_err = 0
        self.integral_err = 0

        #add timing

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

        if ctrl > AV_MAX:
            ctrl = AV_MAX
        elif ctrl < -AV_MAX:
            ctrl = -AV_MAX

        self.last_err = dist_diff

        rmsg = TwistStamped()
        rmsg.twist.linear.x = VEL
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
