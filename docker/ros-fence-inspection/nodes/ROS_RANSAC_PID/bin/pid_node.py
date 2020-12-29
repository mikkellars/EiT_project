#! /usr/bin/env python3

import rospy
from scripts.polar_PID import polar_PID

def main():

    print("STARTING POLAR PID NODE")
    topic_out = ""
    simulate = True #rospy.get_param('~simulate', False)
    rospy.init_node("wall_distance_PID_controller", anonymous=True)

    PID_node = polar_PID(simulate)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        learn_follow.stop_mc()
        print("Shutting down")


if __name__ == '__main__':
    main()
