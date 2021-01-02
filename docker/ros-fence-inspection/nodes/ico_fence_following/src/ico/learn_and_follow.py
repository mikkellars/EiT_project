#! /usr/bin/env python3

import math
import rospy
import random
import numpy as np
from ico.ico import ICO
from ico.datalogger import DataLogger
from dist_ransac.msg import Polar_dist
# from laser_line_extraction.msg import LineSegmentList, LineSegment
from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import Float64

class LearnFollow():
    def __init__(self, sub_name, pub_name, target_dist, learn_inteval, simulate:bool = False, learn_type:str = 'one', log:bool = False, stabilize_ang:bool = True):
        self.sub_name            = sub_name
        self.simulate            = simulate
        self.target_dist         = target_dist
        self.learn_inteval       = learn_inteval
        self.learn_type          = learn_type
        self.stabilize_ang       = stabilize_ang

        # Init ICO's519.0450
        weight_init              = random.uniform(0.1, 0.2)
        # Used if two ico learning
        self.left_ico            = ICO(lr=0.1, weight_predic = weight_init, activation_func = 'sigmoid')
        self.right_ico           = ICO(lr=0.1, weight_predic = weight_init, activation_func = 'sigmoid') 
        # Used if one ico learning
        self.ico                 = ICO(lr=0.0, weight_predic = 1.36)

        # Angle of attack ICO
        self.ico_ang             = ICO(lr=0.1, weight_predic = random.uniform(0.0, 0.1)) 

        # Init ICO loggers
        self.log                 = log
        if self.log: 
            # MC ICO
            self.log_ico_one_mc  = DataLogger('/assets/ico_logs/ico_one_mc.txt')
            self.log_ico_two_mc  = DataLogger('/assets/ico_logs/ico_two_mc.txt')
            self.log_ico_mc_col  = DataLogger('/assets/ico_logs/ico_mc_col.txt')
            self.log_mc_idx      = 0

            # Angle ico
            self.log_ico_ang     = DataLogger('/assets/ico_logs/ico_ang.txt')
            self.log_ico_ang_col = DataLogger('/assets/ico_logs/ico_ang_col.txt')
            self.log_ang_idx     = 0
            
            self.time_log        = rospy.get_time()

        # Publisher to motors
        if self.simulate:
            self.pub_name        = pub_name
            self.publisher_twist = rospy.Publisher(self.pub_name, Twist, queue_size=1)
            self.subscription    = rospy.Subscriber(self.sub_name, Polar_dist, self.__callback_sim, queue_size=1)
            #self.subscription = rospy.Subscriber("/line_segments", LineSegmentList, self.__callback_sim, queue_size=1)
        else:
            self.pub_name        = pub_name
            self.publisher_twist = rospy.Publisher(self.pub_name, TwistStamped, queue_size=1)
            # Subscriber to wall info
            self.subscription    = rospy.Subscriber(self.sub_name, Polar_dist, self.__callback, queue_size=1)

        self.sq_error = 0.0

    def __detect_relfex(self, cur_dist):
        error = self.target_dist - cur_dist
        predictive = error
        reflex = 0

        if error > self.learn_inteval: # Turn left
            reflex = 1
        elif error < (-1*self.learn_inteval): # Turn right
            reflex = -1

        return reflex, predictive

    def __detect_relfex_ang(self, cur_dist, cur_ang):
        error = self.target_dist - cur_dist
        predictive = cur_ang + 90.0
        predictive /= 180
        reflex = 0

        if cur_ang < -120: #and error > self.learn_inteval: # Turn left
            reflex = -1
        elif cur_ang > -60:# and error < (-1*self.learn_inteval): # Turn right
            reflex = 1
        
        return reflex, predictive

    def angle_ico_learning(self, cur_dist, cur_ang, sim:bool = True):
        reflex, predictive = self.__detect_relfex_ang(cur_dist, cur_ang)

        mc_val = self.ico_ang.run_and_learn(reflex, predictive)

        right_mc_val = 0.0
        left_mc_val = 0.0

        if mc_val < 0:
           right_mc_val -= mc_val * (-1)
        elif mc_val > 0:
           left_mc_val -= mc_val

        if self.log:
            cur_time = rospy.get_time() 
            self.log_ico_ang.write_data(self.log_ang_idx, [cur_time - self.time_log, predictive, self.ico_ang.weight_predic, mc_val])
            self.log_ico_ang_col.write_data(self.log_ang_idx, [cur_time - self.time_log, reflex])
            self.log_ang_idx += 1

       # print(f'reflex {reflex}, input: {predictive:0.3f}, weight {self.ico_ang.weight_predic:0.3f}, output {mc_val:0.3f}, Left: {left_mc_val:0.3f}, Right {right_mc_val:0.3f}')
        return left_mc_val, right_mc_val

    def one_ico_learning(self, cur_ang, reflective, predictive, sim:bool = True, mc_scaling:float = 1.0, upper_thresh:float = 1.0, ang_right=0.0, ang_left=0.0):
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
        if mc_val < 0.1 and mc_val > -0.1:
            right_mc_val = 0.5
            left_mc_val = 0.5
        else:
            right_mc_val = 0.25
            left_mc_val = 0.25

        left_mc_val += ang_left
        right_mc_val += ang_right

        if reflective == -1: #(cur_ang < -100) and (reflective == -1): #
            # print('right turn')
            right_mc_val = 0.0
            left_mc_val = 0.2
        elif  reflective == 1:# (cur_ang > -80) and (reflective == 1):
            # print('left turn')
            left_mc_val = 0.0
            right_mc_val = 0.2
        elif mc_val < 0:
            left_mc_val += mc_val * (-1)
            right_mc_val *= mc_scaling
            left_mc_val *= mc_scaling
        elif mc_val > 0:
            right_mc_val += mc_val
            right_mc_val *= mc_scaling
            left_mc_val *= mc_scaling

        # Upper threshold
        if right_mc_val > upper_thresh:
            right_mc_val = upper_thresh
        if left_mc_val > upper_thresh:
            left_mc_val = upper_thresh

        if self.log:
            cur_time = rospy.get_time() 
            r_ico_in = predictive
            l_ico_in = (-1)*predictive
            self.log_ico_one_mc.write_data(self.log_mc_idx, [cur_time - self.time_log, predictive, self.ico.weight_predic, mc_val])
            self.log_ico_mc_col.write_data(self.log_mc_idx, [cur_time - self.time_log, reflective])
            self.log_mc_idx += 1


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
             # Twist
            lin, ang = self.__convert_to_twist(vel_left = left_mc_val, vel_right = right_mc_val, wheel_dist=39.0)
           
            msg = TwistStamped()

            msg.twist.linear.x = lin
            msg.twist.linear.y = 0
            msg.twist.linear.z = 0

            msg.twist.angular.x = 0
            msg.twist.angular.y = 0
            msg.twist.angular.z = ang
            msg.header.stamp = rospy.Time.now()
            return msg
            

    def two_ico_learning(self, cur_ang, reflective, predictive, sim:bool = True, mc_scaling:float = 1.0, upper_thresh:float = 1.0, ang_right=0.0, ang_left=0.0):
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
        # Normalize
        predictive /= self.learn_inteval


        left_mc_val = self.left_ico.run_and_learn(1 if reflective is -1 else 0, (-1)*predictive)
        right_mc_val = self.right_ico.run_and_learn(1 if reflective is 1 else 0, predictive)
        
        left_mc_val += ang_left
        right_mc_val += ang_right

       # print(f'Error: {predictive:0.3f}, Decrease left: {ang_left:0.3f}, Decrease Right: {ang_right:0.3f}, Left: {left_mc_val:0.3f}, Right {right_mc_val:0.3f}, Weihgt {self.ico_ang.weight_predic:0.3f}')

        if reflective == -1:# (cur_ang < -100) and (reflective == -1): #reflective == -1:
            #print('right turn')
            right_mc_val = 0.0
            left_mc_val = 0.2

        elif reflective == 1: # (cur_ang > -80) and (reflective == 1): #reflective == 1:
            #print('left turn')
            left_mc_val = 0.0
            right_mc_val = 0.2 
        else:
            right_mc_val *= mc_scaling
            left_mc_val *= mc_scaling
        
        # Upper threshold
        if right_mc_val > upper_thresh:
            right_mc_val = upper_thresh
        if left_mc_val > upper_thresh:
            left_mc_val = upper_thresh

        if self.log:
            cur_time = rospy.get_time() 
            r_ico_in = predictive
            l_ico_in = (-1)*predictive
            self.log_ico_two_mc.write_data(self.log_mc_idx, [cur_time - self.time_log, l_ico_in, r_ico_in, self.left_ico.weight_predic, self.right_ico.weight_predic, left_mc_val, right_mc_val])
            self.log_ico_mc_col.write_data(self.log_mc_idx, [cur_time - self.time_log, 1 if reflective is -1 else 0, 1 if reflective is 1 else 0])
            self.log_mc_idx += 1

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
            lin, ang = self.__convert_to_twist(vel_left = left_mc_val, vel_right = right_mc_val, wheel_dist=39.0)
           
            msg = TwistStamped()

            msg.twist.linear.x = lin
            msg.twist.linear.y = 0
            msg.twist.linear.z = 0

            msg.twist.angular.x = 0
            msg.twist.angular.y = 0
            msg.twist.angular.z = ang# / 100 # because not in rad/s on frobit
            msg.header.stamp = rospy.Time.now()
            return msg
            

    @staticmethod
    def __convert_to_twist(vel_left, vel_right, wheel_dist:float = 0.14):
        vel_lin = (vel_right + vel_left)/2.0 # [m/s]
        vel_ang = (vel_right - vel_left)/wheel_dist # [rad/s]
        return (vel_lin, vel_ang)


    def __callback_sim(self, msg):

        cur_dist = msg.dist
        cur_ang = np.rad2deg(msg.angle)

        # print(cur_ang)
        # TEST #
        #cur_dist = float('inf')
        
        # for lineseg in msg.line_segments:
        #     point = self.nearest_point_on_line(lineseg.start, lineseg.end)
        #     dist = np.linalg.norm(point)
        #     if cur_dist > dist:
        #         cur_dist = dist

        
        if cur_dist == float('inf') or cur_dist == -float('inf'):
            return

        # Finding out the reflex and predictive signal
        reflex, predictive = self.__detect_relfex(cur_dist)

        if self.learn_type is 'one':
            if self.stabilize_ang:
                left, right = self.angle_ico_learning(cur_dist, cur_ang)
                msg = self.one_ico_learning(cur_ang, reflex, predictive, mc_scaling=0.3, upper_thresh=0.5, ang_right=right, ang_left=left)
            else:
                msg = self.one_ico_learning(cur_ang, reflex, predictive, mc_scaling=0.3, upper_thresh=0.5)
         #   print(f"Reflex {reflex}, Input {predictive:0.2f}, Weight {self.ico.weight_predic:0.2f}, Output {self.ico.output:0.2f}, Lin: {msg.linear.x:0.2f}, Ang: {msg.angular.z:0.2f}")
        elif self.learn_type is 'two':
            if self.stabilize_ang:
                left, right = self.angle_ico_learning(cur_dist, cur_ang)
                msg = self.two_ico_learning(cur_ang, reflex, predictive, mc_scaling=0.3, upper_thresh=0.5, ang_right=right, ang_left=left)
            else:
                msg = self.two_ico_learning(cur_ang, reflex, predictive, mc_scaling=0.3, upper_thresh=0.5)
        #    print(f"Reflex {reflex}, Input {predictive:0.2f}, Weight Left {self.left_ico.weight_predic:0.2f}, Output Left {self.left_ico.output:0.2f}, Weight Right {self.right_ico.weight_predic:0.2f}, Output Right {self.right_ico.output:0.2f}, Lin: {msg.linear.x:0.2f}, Ang: {msg.angular.z:0.2f}")
            print(f"Weight Left {self.left_ico.weight_predic:0.4f}, Weight Right {self.right_ico.weight_predic:0.4f}, Weight angle {self.ico_ang.weight_predic:0.4f}")
        else:
            ValueError('Learning type only supports one or two')

        # self.log_ico.write_data(self.log_idx, [predictive, self.ico.weight_predic, mc_val])

       # print(f'diff lin: {diff_msg.linear.x:0.2f}, diff ang {diff_msg.angular.z:0.2f}')
        

        self.publisher_twist.publish(msg)

        # Squared error
        # self.sq_error += math.pow(predictive, 2)
        # print(f'Squared error {self.sq_error:0.4f}, Time {rospy.get_time() - self.time_log:0.4f}')
        # if reflex != 0:
        #     print(f'reflex : {reflex}')


    def __callback(self, msg):
        cur_dist = msg.dist
        cur_ang = msg.angle
        cur_ang = np.rad2deg(cur_ang)

        if cur_dist == float('inf') or cur_dist == -float('inf'):
            return

        # Run and learning
        reflex, predictive = self.__detect_relfex(cur_dist)


        if self.learn_type is 'one':
            msg = self.one_ico_learning(cur_ang, reflex, predictive, sim=False, mc_scaling=0.3, upper_thresh=0.5)
            print(f"Reflex {reflex}, Input {predictive:0.2f}, Weight {self.ico.weight_predic:0.2f} Lin: {msg.twist.linear.x:0.2f}, Ang: {msg.twist.angular.z:0.5f}")
        elif self.learn_type is 'two':
          # msg = self.two_ico_learning(cur_ang, reflex, predictive, sim=False, mc_scaling=0.3, upper_thresh=0.5)
          # print(f"Reflex {reflex}, Input {predictive:0.2f}, Weight Left {self.left_ico.weight_predic:0.2f} Weight Right {self.right_ico.weight_predic:0.2f} Lin: {msg.twist.linear.x:0.2f}, Ang: {msg.twist.angular.z:0.5f}")
            left, right = self.angle_ico_learning(cur_dist, cur_ang)
            msg = self.two_ico_learning(cur_ang, reflex, predictive, sim=False, mc_scaling=0.5, upper_thresh=0.5, ang_right=right, ang_left=left)
        else:
            ValueError('Learning type only supports one or two')

        
        self.publisher_twist.publish(msg)


        # # Logging data from the ICO
        # self.log_ico.write_data(self.log_idx, [predictive, self.ico.weight_predic, mc_val])
        # self.log_ico_col.write_data(self.log_idx, [reflex])
        # self.log_idx += 1


    # # TEST
    # @staticmethod
    # def nearest_point_on_line(line_start, line_end, point=np.array((0,0))):
    #     line_start = np.array(line_start)
    #     line_end = np.array(line_end)
    #     a_to_p = -line_start
    #     a_to_b = line_end - line_start

    #     a_to_b_magnitude = np.linalg.norm(a_to_b)

    #     if (a_to_b_magnitude == 0):
    #         return line_start

    #     a_to_b_unit = a_to_b/a_to_b_magnitude

    #     a_to_p_scaled = a_to_p * (1.0 / a_to_b_magnitude)

    #     #find how far along the line the point is
    #     t = np.dot(a_to_b_unit, a_to_p_scaled)
    #     if t < 0.0:
    #         t = 0
    #     elif t > 1.0:
    #         t = 1
                
    #     nearest = a_to_b * t + line_start

    #     return nearest

