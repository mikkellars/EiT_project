#! /usr/bin/env python3

"""Script for spawning and evaluating ico learning in simulation
"""

import rospy
import random
import numpy as np

import roslaunch

from sim_spawn.spawn_frobit import SpawnFrobit
from ico.learn_and_follow import LearnFollow 

random.seed(0)

# Sqaure map spawn area
MAP_1 = np.array(([(9, -5), (8, -4)],[(9, -0.9),(8, 0.9)])) # [ [x-outer bound, x-inner bound], [y-outer bound, y-inner bound]]

def gen_spawn_pos(map_bound, num_of_pos):
    positions = list()
    pos_x = list()
    pos_y = list()
    cur_num = 0

    x_bound = map_bound[0]
    y_bound = map_bound[1]

    for i in range(num_of_pos):
        square_select = random.randint(0, 4)

        if square_select == 0:
            gen_x = random.uniform(x_bound[0][0], x_bound[1][0])
            gen_y = random.uniform(y_bound[0][0], y_bound[1][1])
        elif square_select == 1:
            gen_x = random.uniform(x_bound[0][0], x_bound[1][1])
            gen_y = random.uniform(y_bound[1][1], y_bound[0][1])
        elif square_select == 2:
            gen_x = random.uniform(x_bound[1][1], x_bound[0][1])
            gen_y = random.uniform(y_bound[1][0], y_bound[0][1])
        elif square_select == 3:
            gen_x = random.uniform(x_bound[1][0], x_bound[0][1])
            gen_y = random.uniform(y_bound[0][0], y_bound[1][0])

        positions.append((gen_x, gen_y))


    return positions

    

def main():
    poses = gen_spawn_pos(MAP_1, 1)

    rospy.init_node("ico_fence_follow_eval", anonymous=True)
    
    rospy.wait_for_service("/gazebo/spawn_urdf_model")

    spawn = SpawnFrobit()
    
    # Running ico learning on the different generate poses
    for x, y in poses:
        spawn.spawn_model('frobit', x, y)
      #  spawn.delete_model('frobit')

 
    

        
    # while True:
        # if spawn.completed_one_lap(time_before_check=5):
        #     break

    # pub_name_mc = ['/frobit/cmd_vel']
    # sub_name = 'laser/dist_to_wall' 

    # target_dist = 1.0 # Meters
    # learn_inteval = 0.2 # Meters: Accepted interval without learning

    # learn_follow = LearnFollow(sub_name, pub_name_mc, target_dist, learn_inteval, simulate=True, learn_type = 'two')

    # try:
    #     rospy.spin()
    # except KeyboardInterrupt:
    #     print("Shutting down")


if __name__ == '__main__':
    main()

