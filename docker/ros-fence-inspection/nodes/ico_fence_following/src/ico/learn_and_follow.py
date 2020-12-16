#! /usr/bin/env python3

import rospy
import random
from ico.ico import ICO
from ico.datalogger import DataLogger
from dist_ransac.msg import Polar_dist
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64

class LearnFollow():
    def __init__(self, sub_name, pub_name, target_dist, learn_inteval, simulate:bool = False, learn_type:str = 'one'):
        self.sub_name = sub_name
        self.simulate = simulate
        self.target_dist = target_dist
        self.learn_inteval = learn_inteval
        self.learn_type = learn_type

        # Init ICO's
        weight_init = random.uniform(0.5, 1.0)
        # Used if two ico learning
        self.left_ico = ICO(lr=0.1, weight_predic = weight_init, activation_func = 'sigmoid')
        self.right_ico = ICO(lr=0.1, weight_predic = weight_init, activation_func = 'sigmoid') 
        # Used if one ico learning
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

    def one_ico_learning(self, reflective, predictive, sim:bool = True, mc_scaling:float = 1.0):
        """One ico for learning to follow a fence. 
        If it is negative its driving to the right, if its postive it drive to the left.

        Args:
            reflective ([type]): [description]
            predictive ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Run and learning
        mc_val = self.ico.run_and_learn(reflective, predictive) 
        right_mc_val = 0.15
        left_mc_val = 0.15

        if reflective == -1:
            right_mc_val = 0.0
            if sim:
                left_mc_val = 0.2
            else:
                left_mc_val = 20.0 
        elif reflective == 1:
            left_mc_val = 0.0
            if sim:
                right_mc_val = 0.2
            else:
                right_mc_val = 20.0 
        elif mc_val < 0:
            left_mc_val += mc_val * (-1)
            right_mc_val *= mc_scaling
            left_mc_val *= mc_scaling
        elif mc_val > 0:
            right_mc_val += mc_val
            right_mc_val *= mc_scaling
            left_mc_val *= mc_scaling

        if sim:
            lin, ang = self.__convert_to_twist(vel_left = left_mc_val, vel_right = right_mc_val)

            msg = Twist()
            msg.linear.x = lin
            msg.linear.y = 0
            msg.linear.z = 0

            msg.angular.x = 0
            msg.angular.y = 0
            msg.angular.z = ang

            return msg

        else:
            return right_mc_val, left_mc_val

    def two_ico_learning(self, reflective, predictive, sim:bool = True, mc_scaling:float = 1.0):
        """Two ico's one for each wheel. 
        If the reflective signal is negative the left motor is learning to drive closer to the fence
        If the reflective signal is positive the right motor is learning to drive away from the fence

        Args:
            reflective ([type]): [description]
            predictive ([type]): [description]

        Returns:
            [type]: [description]
        """
        # Run and learning
        left_mc_val = self.left_ico.run_and_learn(1 if reflective is -1 else 0, (-1)*predictive)
        right_mc_val = self.right_ico.run_and_learn(1 if reflective is 1 else 0, predictive)

        if reflective == -1:
            right_mc_val = 0.0
            if sim:
                left_mc_val = 0.2
            else:
                left_mc_val = 20.0 
        elif reflective == 1:
            left_mc_val = 0.0
            if sim:
                right_mc_val = 0.2
            else:
                right_mc_val = 20.0 
        else:
            right_mc_val *= mc_scaling
            left_mc_val *= mc_scaling

        if sim:
            lin, ang = self.__convert_to_twist(vel_left = left_mc_val, vel_right = right_mc_val)

            msg = Twist()

            msg.linear.x = lin
            msg.linear.y = 0
            msg.linear.z = 0

            msg.angular.x = 0
            msg.angular.y = 0
            msg.angular.z = ang
            
            return msg

        else:
            return right_mc_val, left_mc_val
            

    @staticmethod
    def __convert_to_twist(vel_left, vel_right):
        wheel_dist = 0.14 # in simulation 
        vel_lin = (vel_right + vel_left)/2.0 # [m/s]
        vel_ang = (vel_right - vel_left)/wheel_dist # [rad/s]
        return (vel_lin, vel_ang)


    def __callback_sim(self, msg):
        cur_dist = msg.dist
        if cur_dist == float('inf') or cur_dist == -float('inf'):
            return

        # Finding out the reflex and predictive signal
        reflex, predictive = self.__detect_relfex(cur_dist)

        if self.learn_type is 'one':
            msg = self.one_ico_learning(reflex, predictive)
        elif self.learn_type is 'two':
            msg = self.two_ico_learning(reflex, predictive)
        else:
            ValueError('Learning type only supports one or two')

        self.publisher_twist.publish(msg)
        print(f"Reflex {reflex}, Input {predictive:0.3f}, weight {self.ico.weight_predic:0.3f}, Lin: {msg.linear.x:0.3f}, Ang: {msg.angular.z:0.3f}")
       # print(f"Lin: {msg.linear.x:0.3f}, Ang: {msg.angular.z:0.3f}")

    def __callback(self, msg):
        cur_dist = msg.dist
        if cur_dist == float('inf') or cur_dist == -float('inf'):
            return

        # Run and learning
        reflex, predictive = self.__detect_relfex(cur_dist)

        if self.learn_type is 'one':
            right_mc_val, left_mc_val = self.one_ico_learning(reflex, predictive, sim=False, mc_scaling=50.0)
        elif self.learn_type is 'two':
            right_mc_val, left_mc_val = self.two_ico_learning(reflex, predictive, sim=False, mc_scaling=50.0)
        else:
            ValueError('Learning type only supports one or two')

        self.publisher_right.publish(right_mc_val)
        self.publisher_left.publish(left_mc_val)

        print(f"Reflex {reflex}, Input {predictive:0.3f}, weight {self.ico.weight_predic:0.3f}, left_mc {left_mc_val:0.3f}, right_mc {right_mc_val:0.3f}")

        # # Logging data from the ICO
        # self.log_ico.write_data(self.log_idx, [predictive, self.ico.weight_predic, mc_val])
        # self.log_ico_col.write_data(self.log_idx, [reflex])
        # self.log_idx += 1


    def stop_mc(self):
        self.publisher_right.publish(0)
        self.publisher_left.publish(0)