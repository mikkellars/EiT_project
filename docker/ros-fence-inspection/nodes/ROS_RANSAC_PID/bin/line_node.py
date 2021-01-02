#! /usr/bin/env python3

import rospy
from scripts.ransac_sub import RANSAC_subscriber

def main():

    # if not rospy.has_param('~simulate'):
    #     ValueError('Need to set simulate param')
    simulate = True
    rospy.init_node("ransac_wall_dist_pub", anonymous=True)

    RANSAC_node = RANSAC_subscriber(simulate)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()
