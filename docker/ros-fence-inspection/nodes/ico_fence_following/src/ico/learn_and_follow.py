#! /usr/bin/env python3

import rospy
import random
from ico.ico import ICO
from ico.datalogger import DataLogger
from dist_ransac.msg import Polar_dist
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

class LearnFollow():
    def __init__(self, sub_name, pub_name, target_dist, learn_inteval, simulate:bool = False):
        self.sub_name = sub_name
        self.simulate = simulate
        self.target_dist = target_dist
        self.learn_inteval = learn_inteval

        # Init ICO's
        weight_init = random.uniform(0.5, 1.0)
        self.left_obs_ico = ICO(lr=0.1, weight_predic = weight_init)
        self.right_obs_ico = ICO(lr=0.1, weight_predic = weight_init) 
        self.ico = ICO(lr=0.1, weight_predic = weight_init) 

        # Init ICO loggers
        self.log_ico     = DataLogger('/assets/ico_logs/ico.txt')
        self.log_ico_col = DataLogger('/assets/ico_logs/ico_col.txt')
        self.log_idx = 0

        # Publisher to motors
        if self.simulate:
            self.pub_name = pub_name[0]
            self.publisher_twist = rospy.Publisher(self.pub_name, Twist, queue_size=1)
            self.subscription = rospy.Subscriber(self.sub_name, Polar_dist, self.__callback_sim, queue_size=1)
        else:
            self.pub_name_left = pub_name[0]
            self.pub_name_right = pub_name[1]
            self.publisher_left = rospy.Publisher(self.pub_name_left, Float64, queue_size=1)
            self.publisher_right = rospy.Publisher(self.pub_name_right, Float64, queue_size=1)

            # Subscriber to wall info
            self.subscription = rospy.Subscriber(self.sub_name, Polar_dist, self.__callback, queue_size=1)
        
    def __detect_relfex(self, cur_dist):
        error = self.target_dist - cur_dist
        predictive = error
        reflex = 0

        if error > self.learn_inteval: # Turn left
            reflex = 1
        elif error < (-1*self.learn_inteval): # Turn right
            reflex = -1

        return reflex, predictive

    @staticmethod
    def __convert_to_twist(vel_left, vel_right):
        wheel_dist = 0.14 # in simulation 
        vel_lin = (vel_right + vel_left)/2.0 # [m/s]
        vel_ang = (vel_right - vel_left)/wheel_dist # [rad/s]
        return (vel_lin, vel_ang)


    def __callback_sim(self, msg):
        cur_dist = msg.dist

        # Run and learning
        reflex, predictive = self.__detect_relfex(cur_dist)
        mc_val = self.ico.run_and_learn(reflex, predictive)
        right_mc_val = 0.15
        left_mc_val = 0.15

        msg = Twist()
        msg.linear.y = 0
        msg.linear.z = 0

        msg.angular.x = 0
        msg.angular.y = 0

        if reflex == -1:
            lin, ang = self.__convert_to_twist(vel_left = 0.20, vel_right = 0)
            msg.linear.x = lin
            msg.angular.z = ang
            self.publisher_twist.publish(msg)
        elif reflex == 1:
            lin, ang = self.__convert_to_twist(vel_left = 0, vel_right = 0.20)
            msg.linear.x = lin
            msg.angular.z = ang
            self.publisher_twist.publish(msg)
        elif mc_val < 0:
            # Publish to PWM values
            left_mc_val += mc_val * (-1)
            lin, ang = self.__convert_to_twist(vel_left = left_mc_val, vel_right = right_mc_val)
            msg.linear.x = lin
            msg.angular.z = ang
            self.publisher_twist.publish(msg)
        elif mc_val > 0:
            right_mc_val += mc_val
            lin, ang = self.__convert_to_twist(vel_left = left_mc_val, vel_right = right_mc_val)
            msg.linear.x = lin
            msg.angular.z = ang
            self.publisher_twist.publish(msg)
        else:
            lin, ang = self.__convert_to_twist(vel_left = left_mc_val, vel_right = right_mc_val)
            msg.linear.x = lin
            msg.angular.z = ang
            self.publisher_twist.publish(msg)

       # print(f"Lin: {msg.linear.x:0.3f}, Ang: {msg.angular.z:0.3f}")

    def __callback(self, msg):
        cur_dist = msg.dist

        # Run and learning
        reflex, predictive = self.__detect_relfex(cur_dist)
        mc_val = self.ico.run_and_learn(reflex, predictive)
        right_mc_val = 15
        left_mc_val = 15

        mc_val *= 100

        if reflex == -1:
            self.publisher_right.publish(0)
            self.publisher_left.publish(20)
            # print('Learning right with right val: ', left_mc_val, "Error: ", predictive)
        elif reflex == 1:
            self.publisher_right.publish(20)
            self.publisher_left.publish(0)
            # print('Learning left with left val: ', right_mc_val, "Error: ", predictive)
        elif mc_val < 0:
            # Publish to PWM values
            left_mc_val += mc_val * (-1)
            self.publisher_right.publish(right_mc_val)
            self.publisher_left.publish(left_mc_val)
        elif mc_val > 0:
            right_mc_val += mc_val
            self.publisher_right.publish(right_mc_val)
            self.publisher_left.publish(left_mc_val)      
        else:
            self.publisher_right.publish(right_mc_val)
            self.publisher_left.publish(left_mc_val) 

        print(f"Reflex {reflex}, Input {predictive:0.3f}, weight {self.ico.weight_predic:0.3f}, output {mc_val:0.3f}, left_mc {left_mc_val:0.3f}, right_mc {right_mc_val:0.3f}")

        # # Logging data from the ICO
        # self.log_ico.write_data(self.log_idx, [predictive, self.ico.weight_predic, mc_val])
        # self.log_ico_col.write_data(self.log_idx, [reflex])
        # self.log_idx += 1


    def stop_mc(self):
        self.publisher_right.publish(0)
        self.publisher_left.publish(0)