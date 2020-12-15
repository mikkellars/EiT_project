#! /usr/bin/env python3

import rospy
from ico.learn_and_follow import *#LearnFollow 

def main():

    rospy.init_node('ico_fence_following', anonymous=True)
    rospy.Rate(10)

  
    sub_name = 'laser/dist_to_wall' 
   # pub_name = '/frobit/twist' # Simulation

    pub_name_left_mc = '/frobit/left_pwm' # Frobit
    pub_name_right_mc = '/frobit/right_pwm' # Frobit

    target_dist = 1.0 # Meters
    learn_inteval = 0.2 # Meters: Accepted interval without learning

    learn_follow = LearnFollow(sub_name, pub_name_left_mc, pub_name_right_mc, target_dist, learn_inteval)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        learn_follow.stop_mc()
        print("Shutting down")


if __name__ == '__main__':
    main()

