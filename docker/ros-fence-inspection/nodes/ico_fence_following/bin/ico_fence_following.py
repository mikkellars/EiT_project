#! /usr/bin/env python3

import rospy
from ico.learn_and_follow import LearnFollow 

def main():

    rospy.init_node('ico_fence_following', anonymous=True)
    rospy.Rate(50)

    if not rospy.has_param('~simulate'):
        ValueError('Need to set simulate param')

    simulate = rospy.get_param('~simulate')
    
    if simulate: # Simulate
        pub_name_mc = ['/cmd_vel']
    else: # Frobit
        pub_name_mc = ['/frobit/left_pwm', '/frobit/right_pwm']

    sub_name = 'laser/dist_to_wall' 

    target_dist = 1.0 # Meters
    learn_inteval = 0.2 # Meters: Accepted interval without learning

    learn_follow = LearnFollow(sub_name, pub_name_mc, target_dist, learn_inteval, simulate, learn_type = 'two')

    try:
        rospy.spin()
    except KeyboardInterrupt:
        learn_follow.stop_mc()
        print("Shutting down")


if __name__ == '__main__':
    main()

