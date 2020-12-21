#! /usr/bin/env python3

"""Script for spawning and evaluating ico learning in simulation
"""

import rospy

from sim_spawn.spawn_frobit import SpawnFrobit
from ico.learn_and_follow import LearnFollow 

def main():

    rospy.init_node("ico_fence_follow_eval", anonymous=True)
    
   # rospy.sleep(5)
    rospy.wait_for_service("/gazebo/spawn_urdf_model")

    spawn = SpawnFrobit()
    spawn.spawn_model("frobit", x=0.0, y=0.0, z=0.0, angle=-90)
    # pub_name_mc = ['/frobit/cmd_vel']
    # sub_name = 'laser/dist_to_wall' 

    # target_dist = 1.0 # Meters
    # learn_inteval = 0.2 # Meters: Accepted interval without learning

    # learn_follow = LearnFollow(sub_name, pub_name_mc, target_dist, learn_inteval, simulate=True, learn_type = 'two')

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")


if __name__ == '__main__':
    main()

